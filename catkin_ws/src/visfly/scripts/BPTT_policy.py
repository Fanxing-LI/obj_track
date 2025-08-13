#!/usr/bin/env python3
#!/usr/bin/env python3
import sys, os

def remove_last_n_folders(path, n=5):
    path = path.rstrip('/\\')  # 去除末尾的斜杠
    for _ in range(n):
        path = path[:path.rfind('/')] if '/' in path else ''
    return path


add_path = remove_last_n_folders(os.path.dirname(os.path.abspath(__file__)), 4)
sys.path.append(add_path)
print(add_path)


# ONNX模型路径配置
ONNX_MODEL_PATH = "/home/lfx-desktop/files/obj_track/exps/real_world/saved/objTracking/\
SHAC_NoCaliHeadV_Pos_Dis3.0_spd3.4_lessNoise_2_policy.onnx"

# Action topic前缀配置
ACTION_TOPIC_PREFIX = "BPTT/drone_{}/action"

import rospy
from mav_msgs.msg import RateThrust
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Vector3
import threading
import numpy as np
import torch as th
import onnxruntime as ort
import os

# 导入VisFly的quaternion工具
from VisFly.utils.maths import Quaternion


class BPTTPolicy:
    """
    基于BPTT的policy，使用bodyrate和z轴加速度控制
    订阅多个drone的odom话题，发布PositionCommand
    """
    def __init__(self, num_agent=4):
        self.num_agent = num_agent

        # 初始化ROS节点
        rospy.init_node('bptt_policy', anonymous=True)

        # 加载ONNX模型
        self.onnx_session = self._load_onnx_model()

        self.pre_target_pos = None

        # 创建发布器和订阅器
        self.position_cmd_publishers = []
        self.env_status_publisher = None
        self.odom_subscribers = []
        self.action_subscribers = []
        self.target_odom_subscriber = None

        # 线程锁，确保线程安全
        self.lock = threading.Lock()

        # 存储最新的odom和action数据
        self.latest_odom = [None] * self.num_agent
        self.latest_actions = [None] * self.num_agent
        self.latest_target_odom = None  # 存储target位置信息

        # 环境状态信息
        self.env_status = {
            'positions': np.zeros((self.num_agent, 3)),
            'velocities': np.zeros((self.num_agent, 3)),
            'orientations': np.zeros((self.num_agent, 4)),  # quaternion (x,y,z,w)
            'angular_velocities': np.zeros((self.num_agent, 3)),
            'head_targets': np.zeros((self.num_agent, 3)),
            'head_targets_v': np.zeros((self.num_agent, 3)),
            'head_v': np.zeros((self.num_agent, 3)),
        }

        # BPTT policy参数
        self.device = th.device('cpu')
        self.setup_publishers_and_subscribers()

        self._count = 0

        rospy.loginfo(f"BPTT Policy 已启动，管理{self.num_agent}个drone")

    def _load_onnx_model(self):
        """加载ONNX模型"""
        try:
            if not os.path.exists(ONNX_MODEL_PATH):
                rospy.logwarn(f"ONNX模型文件不存在: {ONNX_MODEL_PATH}")
                return None

            # 创建ONNX Runtime推理会话
            providers = ['CPUExecutionProvider']  # 使用CPU推理
            session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)

            rospy.loginfo(f"成功加载ONNX模型: {ONNX_MODEL_PATH}")
            rospy.loginfo(f"模型输入: {[input.name for input in session.get_inputs()]}")
            rospy.loginfo(f"模型输出: {[output.name for output in session.get_outputs()]}")

            return session

        except Exception as e:
            rospy.logerr(f"加载ONNX模型失败: {e}")
            return None

    def _run_policy_inference(self, state):
        """运行ONNX模型推理"""
        if self.onnx_session is None:
            rospy.logwarn("ONNX模型未加载，返回零动作")
            return np.zeros(4)  # 返回默认动作 [vx, vy, vz, yaw_dot]

        try:
            # 准备输入数据
            input_name = self.onnx_session.get_inputs()[0].name
            input_data = {input_name: state.cpu().numpy().reshape(1, -1)}

            # 运行推理
            outputs = self.onnx_session.run(None, input_data)

            # 获取输出动作 - BPTT使用bodyrate和z轴加速度
            action = outputs[0].flatten()
            # print(action)

            self._count += 1
            if self._count % 20 == 0:
                rospy.loginfo(f"ONNX推理次数: {self._count}")
            return action

        except Exception as e:
            rospy.logwarn(f"ONNX推理失败: {e}")
            return np.zeros(4)  # 返回默认动作

    def setup_publishers_and_subscribers(self):
        """设置发布器和订阅器"""

        # 创建环境状态发布器
        self.env_status_publisher = rospy.Publisher('visfly/env_status', PoseStamped, queue_size=10)

        for i in range(self.num_agent):
            # 为每个drone创建Action发布器，使用新的topic格式
            pub = rospy.Publisher(ACTION_TOPIC_PREFIX.format(i), RateThrust, queue_size=10)
            self.position_cmd_publishers.append(pub)

            # 为每个drone创建Odometry订阅器
            odom_sub = rospy.Subscriber(f'visfly/drone_{i}/odom', Odometry, self._make_odom_callback(i))
            self.odom_subscribers.append(odom_sub)

            # 为每个drone创建Action订阅器
            action_sub = rospy.Subscriber(ACTION_TOPIC_PREFIX.format(i), RateThrust, self._make_action_callback(i))
            self.action_subscribers.append(action_sub)

        # 创建全局目标位姿订阅器
        self.target_odom_subscriber = rospy.Subscriber('visfly/target/odom', Odometry, self._target_odom_callback)

    def _make_odom_callback(self, drone_id):
        """创建特定drone的odom回调函数"""
        def odom_callback(odom_msg):
            with self.lock:
                self.latest_odom[drone_id] = odom_msg
                self._update_env_status(drone_id, odom_msg)
                self._publish_position_command(drone_id, odom_msg)
        return odom_callback

    def _make_action_callback(self, drone_id):
        """创建特定drone的action回调函数"""
        def action_callback(action_msg):
            with self.lock:
                self.latest_actions[drone_id] = action_msg
        return action_callback

    def _target_odom_callback(self, target_odom_msg):
        """全局目标位姿回调函数"""
        with self.lock:
            # 更新最新的目标位姿
            self.latest_target_odom = target_odom_msg
            # 处理目标位姿消息，更新预处理目标位置
            self._process_target_odom(target_odom_msg)

    def _process_target_odom(self, target_odom_msg):
        """处理目标位姿消息，更新预处理目标位置"""
        if target_odom_msg is not None:
            self.target_pos = th.tensor([
                target_odom_msg.pose.pose.position.x,
                target_odom_msg.pose.pose.position.y,
                target_odom_msg.pose.pose.position.z
            ], dtype=th.float32, device=self.device)
        else:
            rospy.logwarn("未收到目标位姿信息，使用默认位置")
            self.target_pos = th.tensor([8.0, 8.0, 1.0], dtype=th.float32, device=self.device)
        if self.pre_target_pos is None:
            self.pre_target_pos = self.target_pos.clone()
        self.target_v_world = (self.target_pos-self.pre_target_pos) / 0.03
        self.pre_target_pos = self.target_pos.clone()

    def _update_env_status(self, drone_id, odom_msg):
        """更新环境状态"""
        # 更新位置
        self.env_status['positions'][drone_id] = [
            odom_msg.pose.pose.position.x,
            odom_msg.pose.pose.position.y,
            odom_msg.pose.pose.position.z
        ]

        # 更新速度
        self.env_status['velocities'][drone_id] = [
            odom_msg.twist.twist.linear.x,
            odom_msg.twist.twist.linear.y,
            odom_msg.twist.twist.linear.z
        ]

        self.env_status['orientations'][drone_id] = [
            odom_msg.pose.pose.orientation.w,
            odom_msg.pose.pose.orientation.x,
            odom_msg.pose.pose.orientation.y,
            odom_msg.pose.pose.orientation.z,
        ]

        # 更新角速度
        self.env_status['angular_velocities'][drone_id] = [
            odom_msg.twist.twist.angular.x,
            odom_msg.twist.twist.angular.y,
            odom_msg.twist.twist.angular.z
        ]

    def preprocess_input(self, drone_id):
        """预处理policy输入，参考ObjectTrackingEnv的update_target和get_observation"""
        if self.latest_odom[drone_id] is None:
            return None

        # 获取当前状态
        position = th.tensor(self.env_status['positions'][drone_id], dtype=th.float32, device=self.device)
        velocity = th.tensor(self.env_status['velocities'][drone_id], dtype=th.float32, device=self.device)
        orientation_q = self.env_status['orientations'][drone_id]
        angular_velocity = th.tensor(self.env_status['angular_velocities'][drone_id], dtype=th.float32, device=self.device)

        # 转换ROS四元数格式(x,y,z,w)到VisFly格式(w,x,y,z)
        orientation = Quaternion(
            w=th.tensor(orientation_q[0], dtype=th.float32, device=self.device),
            x=th.tensor(orientation_q[1], dtype=th.float32, device=self.device),
            y=th.tensor(orientation_q[2], dtype=th.float32, device=self.device),
            z=th.tensor(orientation_q[3], dtype=th.float32, device=self.device)
        )

        # 获取目标位置和速度从订阅的topic（现在使用Odometry消息）
        if self.latest_target_odom is not None:
            target_world = th.tensor([
                self.latest_target_odom.pose.pose.position.x,
                self.latest_target_odom.pose.pose.position.y,
                self.latest_target_odom.pose.pose.position.z
            ], dtype=th.float32, device=self.device)
            # 从Odometry消息中获取目标速度
            if self.pre_target_pos is None:
                self.pre_target_pos = target_world.clone()
            target_v_world = self.target_v_world
        else:
            # 如果没有收到target信息，使用默认位置
            rospy.logwarn_throttle(1.0, "未收到target位置信息，使用默认位置")
            target_world = th.tensor([8.0, 8.0, 1.0], dtype=th.float32, device=self.device)
            target_v_world = th.zeros(3, dtype=th.float32, device=self.device)

        # 计算相对目标位置和速度
        rela_tar = target_world - position
        rela_v = target_v_world - velocity

        # 使用VisFly的quaternion方法计算head坐标系下的目标位置和速度
        head_targets = orientation.world_to_head(rela_tar.unsqueeze(0).T).T
        # if self._count % 20 == 0:
        #     print(orientation)
        head_targets_v = orientation.world_to_head(rela_v.unsqueeze(0).T).T
        head_v = orientation.world_to_head(velocity.unsqueeze(0).T).T

        # 获取四元数的4个分量作为orientation特征，参考ObjectTrackingEnv
        orientation_vec = th.atleast_2d(orientation.toTensor().to(self.device))  # 转换为tensor格式
        angular_velocity = th.atleast_2d(angular_velocity)
        # print all cat variable shape
        # 构建状态向量，参考ObjectTrackingEnv的get_observation
        state = th.hstack([
            head_targets,  # 3维 - head坐标系下的目标位置
            head_targets_v,  # 3维 - head坐标系下的目标速度
            orientation_vec,  # 4维 - 四元数(w,x,y,z)
            head_v / 10,  # 3维 - head坐标系下的速度(归一化)
            angular_velocity / 10,  # 3维 - 角速度(归一化)
        ])  # 总共16维

        # 更新环境状态用于发布
        self.env_status['head_targets'][drone_id] = head_targets.numpy()
        self.env_status['head_targets_v'][drone_id] = head_targets_v.numpy()
        self.env_status['head_v'][drone_id] = head_v.numpy()

        return state

    def _publish_position_command(self, drone_id, odom_msg):
        """基于BPTT policy发布PositionCommand，使用bodyrate和z轴加速度"""
        # 预处理输入
        state = self.preprocess_input(drone_id)
        if state is None:
            return

        # 使用ONNX模型进行推理
        action = self._run_policy_inference(state)

        # 创建PositionCommand消息
        cmd = RateThrust()

        # 设置消息头
        cmd.header.stamp = rospy.Time.now()
        cmd.header.frame_id = "world"

        # 获取当前位置和姿态
        current_pos = odom_msg.pose.pose.position
        current_ori = odom_msg.pose.pose.orientation

        # BPTT policy特有：ONNX模型输出格式为 [z_acc, bodyrate_x, bodyrate_y, bodyrate_z]

        cmd.thrust.x = 0.0
        cmd.thrust.y = 0.0
        cmd.thrust.z = action[0]  # z轴推力
        cmd.angular_rates.x = action[1]  # x轴滚转速率
        cmd.angular_rates.y = action[2]  # y轴俯仰速率
        cmd.angular_rates.z = action[3]  # z轴偏航速率

        # 发布消息
        self.position_cmd_publishers[drone_id].publish(cmd)
        # print(f"Drone{drone_id} - 发布RateThrust: ")
        # rospy.loginfo_throttle(1.0, f"Drone{drone_id} - BPTT ONNX控制: "
        #             f"z_thrust={cmd.thrust.z:.3f}, "
        #             f"angular_rates=[{cmd.angular_rates.x:.3f}, {cmd.angular_rates.y:.3f}, {cmd.angular_rates.z:.3f}]")
    def subscribe_action(self):
        """订阅action并提取推力和角速度，组成n*4的tensor"""
        with self.lock:
            actions = []
            for i in range(self.num_agent):
                if self.latest_actions[i] is not None:
                    action = self.latest_actions[i]
                    # 提取推力和角速度 (thrust.z, angular_rates.x, angular_rates.y, angular_rates.z)
                    thrust_rates = [
                        action.thrust.z,          # z轴推力
                        action.angular_rates.x,   # x轴角速度
                        action.angular_rates.y,   # y轴角速度
                        action.angular_rates.z    # z轴角速度
                    ]
                    actions.append(thrust_rates)
                else:
                    # 如果没有收到action，使用默认值
                    actions.append([0.0, 0.0, 0.0, 0.0])

        # 转换为tensor (n*4)
        action_tensor = th.tensor(actions, dtype=th.float32, device=self.device)

        return action_tensor

    def publish_env_status(self):
        """发布环境状态信息"""
        try:
            # 创建PoseStamped消息作为环境状态载体
            env_msg = PoseStamped()
            env_msg.header.stamp = rospy.Time.now()
            env_msg.header.frame_id = "world"

            # 这里可以将环境状态信息编码到消息中
            # 由于PoseStamped字段有限，这里只是示例
            if self.latest_odom[0] is not None:
                env_msg.pose.position.x = self.env_status['positions'][0][0]
                env_msg.pose.position.y = self.env_status['positions'][0][1]
                env_msg.pose.position.z = self.env_status['positions'][0][2]

            self.env_status_publisher.publish(env_msg)

        except Exception as e:
            rospy.logwarn(f"发布环境状态失败: {e}")

    def run(self):
        """运行BPTT Policy节点"""
        rate = rospy.Rate(30)  # 30Hz

        while not rospy.is_shutdown():
            # 定期发布环境状态
            self.publish_env_status()

            # 获取当前action tensor
            action_tensor = self.subscribe_action()

            rospy.loginfo_throttle(5.0, f"Action tensor shape: {action_tensor.shape}")

            rate.sleep()


def main():
    """
    主函数 - 创建并运行BPTTPolicy
    """
    try:
        # 从参数服务器获取agent数量，默认为4
        num_agent = rospy.get_param('~num_agent', 4)

        policy = BPTTPolicy(num_agent)
        policy.run()

    except rospy.ROSInterruptException:
        rospy.loginfo("BPTT Policy 已停止")


if __name__ == '__main__':
    main()
