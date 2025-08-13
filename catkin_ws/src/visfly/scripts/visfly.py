#!/usr/bin/env python3
import rospy
import sys, os

def remove_last_n_folders(path, n=5):
    path = path.rstrip('/\\')  # 去除末尾的斜杠
    for _ in range(n):
        path = path[:path.rfind('/')] if '/' in path else ''
    return path


add_path = remove_last_n_folders(os.path.dirname(os.path.abspath(__file__)), 4)
sys.path.append(add_path)
print(add_path)

import numpy as np
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from tf.transformations import quaternion_from_euler
import threading
import argparse
from exps.vary_v.run import main
from quadrotor_msgs.msg import PositionCommand
from mav_msgs.msg import RateThrust
import torch
from VisFly.utils.type import ACTION_TYPE

# Topic name definitions

ODOM_TOPIC_PREFIX = "visfly/drone_{}/odom"
TARGET_ODOM_TOPIC = "visfly/target/odom"
POINTCLOUD_TOPIC = "visfly/env/pointcloud"

def parse_args():
    parser = argparse.ArgumentParser(description='Run experiments', add_help=False)
    parser.add_argument('--comment', '-c', type=str, default="std")
    parser.add_argument("--train", "-t", type=int, default=1)
    parser.add_argument("--algorithm", "-a", type=str, default="SHAC")
    parser.add_argument("--env", "-e", type=str, default="objTracking")
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--weight", "-w", type=str, default=None, )
    parser.add_argument("--traj", "-tr", type=str, default="1", )
    parser.add_argument("--velocity", "-v", type=float, default=3.0, )
    parser.add_argument("--num_agent", "-n", type=int, default=4, )
    return parser


class ROSEnvWrapper:
    def __init__(self, env, comment="elastic"):
        self.envs = env
        self.num_agent = self.envs.num_envs
        self.action_type = self.envs.envs.dynamics.action_type
        if comment == "elastic":
            raise NotImplementedError
            ACTION_TOPIC_PREFIX = "visfly/drone_{}/action" # TODO: elastic的action订阅者需要改成PositionCommand
            assert self.action_type == ACTION_TYPE.POSITION, f"current action type is {self.action_type}, but it should be 'position' for elastic"
        elif comment == "BPTT":
            assert self.action_type == ACTION_TYPE.BODYRATE, f"current action type is {self.action_type}, but it should be 'bodyrate' for BPTT"
            ACTION_TOPIC_PREFIX = "BPTT/drone_{}/action"
        elif comment == "fsc":
            raise NotImplementedError
            ACTION_TOPIC_PREFIX = "visfly/drone_{}/action" # TODO: fsc的action订阅者需要改成fsc对应的action, fsc也需要对应修改
            assert self.action_type == ACTION_TYPE.BODYRATE, f"current action type is {self.action_type}, but it should be 'bodyrate' for fsc"
        self.comment = comment

        # Initialize ROS node
        rospy.init_node('visfly', anonymous=True)

        self._count = 0

        # Action data storage and lock for thread safety
        self.action_data = [None] * self.num_agent
        self.action_lock = threading.Lock()

        # Publishers
        self.drone_odom_pubs = []
        # Subscribers for action
        self.drone_action_subs = []

        for i in range(self.num_agent):
            # Publisher for odometry using defined topic prefix
            drone_odom_pub = rospy.Publisher(ODOM_TOPIC_PREFIX.format(i), Odometry, queue_size=1)
            self.drone_odom_pubs.append(drone_odom_pub)

            # 添加action订阅者
            if self.comment == "elastic":
                raise NotImplementedError
            elif self.comment == "BPTT":
                action_sub = rospy.Subscriber(ACTION_TOPIC_PREFIX.format(i), RateThrust, self._make_action_callback(i))
            elif self.comment == "fsc":
                raise NotImplementedError
            self.drone_action_subs.append(action_sub)

        self.target_odom_pub = rospy.Publisher(TARGET_ODOM_TOPIC, Odometry, queue_size=1)
        self.pointcloud_pub = rospy.Publisher(POINTCLOUD_TOPIC, PointCloud2, queue_size=1)

        # Frame IDs
        self.world_frame = "world"
        print("===========================================================================")
        rospy.loginfo(f"Visfly ROS Environment Wrapper initialized with {self.num_agent} agents")
        rospy.loginfo(f"Subscribing to action topics: {[ACTION_TOPIC_PREFIX.format(i) for i in range(self.num_agent)]}")
        rospy.loginfo(f"Publishing to topics: {[ODOM_TOPIC_PREFIX.format(i) for i in range(self.num_agent)]}")

    def reset(self, *args, **kwargs):
        """
        Reset the environment and clear action data.
        This method can be called to reset the environment state.
        """
        r=self.envs.reset(*args, **kwargs)
        # publish initial environment status
        self.publish_env_status()
        return r

    def predict(self, obs, deterministic=True):
        """
        publish current action
        """
        with self.action_lock:
            if self.comment == "elastic":
                raise NotImplementedError
            elif self.comment == "BPTT":
                # Extract z_acc and bodyrate from action_data
                action_tensor = torch.zeros(self.num_agent, 4)
                for i in range(self.num_agent):
                    if self.action_data[i] is not None:
                        action_tensor[i, 0] = self.action_data[i]['z_acc']
                        action_tensor[i, 1:4] = torch.tensor(self.action_data[i]['bodyrate'])
                self.action_data = [None] * self.num_agent  # Clear action data after use
                return action_tensor
            elif self.comment == "fsc":
                raise NotImplementedError

    def _make_action_callback(self, agent_id):
        def callback(msg):
            with self.action_lock:
                # Extract action data based on comment type
                if self.comment == "elastic":
                    raise NotImplementedError
                    # 保持现有的action预处理
                    self.action_data[agent_id] = {
                        'position': [msg.position.x, msg.position.y, msg.position.z],
                        'yaw': msg.yaw
                    }
                elif self.comment == "BPTT":
                    # 对于RateThrust消息：z_acc使用thrust，bodyrate使用angular_rates
                    self.action_data[agent_id] = {
                        'z_acc': msg.thrust.z,  # 使用z轴推力
                        'bodyrate': [msg.angular_rates.x, msg.angular_rates.y, msg.angular_rates.z]  # 使用角速度
                    }
                elif self.comment == "fsc":
                    # 暂时pass
                    raise NotImplementedError
                    pass
        return callback

    def subscribe_action(self):
        """
        订阅action并提取position和yaw，组成n*4的tensor并return
        """
        with self.action_lock:
            if self.comment == "elastic":
                raise NotImplementedError
                # 提取position和yaw组成n*4的tensor
                action_tensor = torch.zeros(self.num_agent, 4)
                for i in range(self.num_agent):
                    if self.action_data[i] is not None:
                        action_tensor[i, :3] = torch.tensor(self.action_data[i]['position'])
                        action_tensor[i, 3] = self.action_data[i]['yaw']  # yaw的dim是0而不是3
                return action_tensor
            elif self.comment == "BPTT":
                # 提取z_acc和bodyrate组成n*4的tensor
                action_tensor = torch.zeros(self.num_agent, 4)
                for i in range(self.num_agent):
                    if self.action_data[i] is not None:
                        action_tensor[i, 0] = self.action_data[i]['z_acc']
                        action_tensor[i, 1:4] = torch.tensor(self.action_data[i]['bodyrate'])
                return action_tensor
            elif self.comment == "fsc":
                # 暂时返回zeros
                raise NotImplementedError

    def publish_env_status(self):
        """
        发布所有环境信息
        """
        self.publish_drone_state()
        self.publish_target_odom()
        self.publish_pointcloud()
        # wait 0.03 s
        rospy.sleep(0.03)
        self._count += 1
        if self._count % 10 == 0:
            rospy.loginfo(f"Published environment status at count {self._count}")
        # print("Published environment status.")

    def publish_drone_state(self):
        """
        发布无人机状态信息
        从self.envs.state获取状态：num_agent*13 (pos, quaternion, vel, angular_vel)
        """
        # 获取当前环境状态
        if not hasattr(self.envs, 'state') or self.envs.state is None:
            rospy.logwarn("Environment state not available")
            return

        drone_states = self.envs.state  # shape: (num_agent, 13)

        for i in range(self.num_agent):
            if i < len(drone_states):
                state = drone_states[i]  # shape: (13,) [pos(3), quat_wxyz(4), vel(3), angular_vel(3)]
                odom_msg = Odometry()
                odom_msg.header.stamp = rospy.Time.now()
                odom_msg.header.frame_id = self.world_frame
                odom_msg.child_frame_id = f"drone_{i}"

                # 位置 (0:3)
                odom_msg.pose.pose.position.x = state[0]
                odom_msg.pose.pose.position.y = state[1]
                odom_msg.pose.pose.position.z = state[2]

                # 四元数姿态 (3:7) - 状态格式是wxyz，ROS格式是xyzw
                odom_msg.pose.pose.orientation.x = state[4]  # x from wxyz[1]
                odom_msg.pose.pose.orientation.y = state[5]  # y from wxyz[2]
                odom_msg.pose.pose.orientation.z = state[6]  # z from wxyz[3]
                odom_msg.pose.pose.orientation.w = state[3]  # w from wxyz[0]

                # 线速度 (7:10)
                odom_msg.twist.twist.linear.x = state[7]
                odom_msg.twist.twist.linear.y = state[8]
                odom_msg.twist.twist.linear.z = state[9]

                # 角速度 (10:13)
                odom_msg.twist.twist.angular.x = state[10]
                odom_msg.twist.twist.angular.y = state[11]
                odom_msg.twist.twist.angular.z = state[12]

                self.drone_odom_pubs[i].publish(odom_msg)

    def publish_target_odom(self):
        """
        发布目标位姿
        从self.envs.dynamic_object_position获取目标位置
        """
        # 获取目标位置
        if not hasattr(self.envs, 'dynamic_object_position') or self.envs.dynamic_object_position is None:
            rospy.logwarn("Dynamic object position not available")
            return

        # dynamic_object_position是一个len=num_agent的list，取第一个作为target位置
        if len(self.envs.dynamic_object_position) == 0:
            rospy.logwarn("Dynamic object position list is empty")
            return

        target_position = self.envs.dynamic_object_position[0][0]  # 取第一个作为target位置

        odom_msg = Odometry()
        odom_msg.header.stamp = rospy.Time.now()
        odom_msg.header.frame_id = self.world_frame
        odom_msg.child_frame_id = "target"

        odom_msg.pose.pose.position.x = target_position[0]
        odom_msg.pose.pose.position.y = target_position[1]
        odom_msg.pose.pose.position.z = target_position[2]

        # 默认朝向
        odom_msg.pose.pose.orientation.w = 1.0

        self.target_odom_pub.publish(odom_msg)

    def publish_pointcloud(self):
        """
        发布点云数据
        发布一个包含单个远点的示例点云
        """
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.world_frame

        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]

        # 创建一个包含单个远点的示例点云
        example_points = [[100.0, 100.0, 100.0]]  # 一个很远的点作为示例

        pc2_msg = pc2.create_cloud(header, fields, example_points)
        self.pointcloud_pub.publish(pc2_msg)

    @property
    def num_agent(self):
        return self._num_agent

    @num_agent.setter
    def num_agent(self, value):
        self._num_agent = value


if __name__ == '__main__':
    try:
        parser = parse_args()
        args = parser.parse_args(rospy.myargv()[1:])

        env_kwargs = {
            'traj': args.traj,
            'velocity': args.velocity,
            "comment": args.comment,
        }
        assert args.comment in ["BPTT", "elastic", "fsc"]

        env = main(traj=env_kwargs["traj"],
                   velocity=env_kwargs["velocity"],
                   comment=env_kwargs["comment"],
                   ROS_wrapper=ROSEnvWrapper,
                   debug=False,
                   )

        print(f"Environment created: {env}")
        # node = ROSEnvWrapper(env_kwargs)

        # Keep the node running
        # while True:
        #     rospy.sleep(0.01)
        #     env._publish_ros_data()

    except rospy.ROSInterruptException:
        pass
