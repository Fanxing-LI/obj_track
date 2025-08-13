#!/usr/bin/env python3

import rospy
from quadrotor_msgs.msg import PositionCommand
from nav_msgs.msg import Odometry
import math
import threading
import numpy as np


class DebugActorPublisher:
    """
    订阅多个drone的odom话题，每收到新的odom数据就发布PositionCommand
    """
    def __init__(self, num_agent=4):
        self.num_agent = num_agent

        # 初始化ROS节点
        rospy.init_node('debug_actor_publisher', anonymous=True)

        # 创建发布器和订阅器
        self.publishers = []
        self.subscribers = []

        # 线程锁，确保线程安全
        self.lock = threading.Lock()

        # 计数器，用于生成轨迹
        self.counters = [0] * self.num_agent

        # 高斯分布参数
        self.position_std = 0.1  # 位置的标准差
        self.velocity_std = 0.05  # 速度的标准差
        self.acceleration_std = 0.02  # 加速度的标准差
        self.yaw_std = 0.1  # 偏航角的标准差

        for i in range(self.num_agent):
            # 为每个drone创建PositionCommand发布器
            pub = rospy.Publisher(f'visfly/drone_{i}/position_cmd', PositionCommand, queue_size=10)
            self.publishers.append(pub)

            # 为每个drone创建Odometry订阅器
            sub = rospy.Subscriber(f'visfly/drone_{i}/odom', Odometry, self._make_odom_callback(i))
            self.subscribers.append(sub)

        rospy.loginfo(f"Debug Actor Publisher 已启动，订阅{self.num_agent}个drone的odom话题")

    def _make_odom_callback(self, drone_id):
        """创建特定drone的odom回调函数"""
        def odom_callback(odom_msg):
            with self.lock:
                self._publish_position_command(drone_id, odom_msg)
        return odom_callback

    def _publish_position_command(self, drone_id, odom_msg):
        """基于收到的odom数据发布PositionCommand"""
        try:
            # 创建PositionCommand消息
            cmd = PositionCommand()

            # 设置消息头
            cmd.header.stamp = rospy.Time.now()
            cmd.header.frame_id = "world"

            # 基于当前odom位置，使用均值为0的高斯分布生成随机动作
            current_pos = odom_msg.pose.pose.position

            # 设置目标位置 (当前位置 + 高斯噪声)
            cmd.position.x = 10
            cmd.position.y = 10
            cmd.position.z = 1

            # 设置速度 (均值为0的高斯分布)
            cmd.velocity.x = 0
            cmd.velocity.y = 0
            cmd.velocity.z = 0

            # 设置加速度 (均值为0的高斯分布)
            cmd.acceleration.x = 0
            cmd.acceleration.y = 0
            cmd.acceleration.z = 0

            # 设置偏航角 (均值为0的高斯分布)
            cmd.yaw = 0
            cmd.yaw_dot = 0

            # 设置控制增益
            cmd.kx = [5.0, 5.0, 5.0]  # 位置增益
            cmd.kv = [3.0, 3.0, 3.0]  # 速度增益

            # 发布消息
            self.publishers[drone_id].publish(cmd)

            # 更新计数器
            self.counters[drone_id] += 1

            # 打印调试信息 (每100次打印一次)
            # if self.counters[drone_id] % 100 == 0:
            rospy.loginfo(f"Drone{drone_id} - 发布随机位置命令: "
                        f"x={cmd.position.x:.2f}, y={cmd.position.y:.2f}, z={cmd.position.z:.2f}, "
                        f"vx={cmd.velocity.x:.2f}, vy={cmd.velocity.y:.2f}, vz={cmd.velocity.z:.2f}")

        except Exception as e:
            rospy.logwarn(f"发布drone{drone_id}位置命令失败: {e}")

    def run(self):
        """运行节点"""
        rospy.loginfo("等待odom消息...")
        rospy.spin()

    def set_noise_parameters(self, pos_std=0.1, vel_std=0.05, acc_std=0.02, yaw_std=0.1):
        """设置高斯噪声参数"""
        self.position_std = pos_std
        self.velocity_std = vel_std
        self.acceleration_std = acc_std
        self.yaw_std = yaw_std
        rospy.loginfo(f"噪声参数已更新: pos_std={pos_std}, vel_std={vel_std}, acc_std={acc_std}, yaw_std={yaw_std}")


def debug_actor_publisher():
    """
    主函数 - 创建并运行DebugActorPublisher
    """
    try:
        # 从参数服务器获取agent数量，默认为4
        num_agent = rospy.get_param('~num_agent', 4)

        publisher = DebugActorPublisher(num_agent)
        
        # 可选：从参数服务器读取噪声参数
        pos_std = rospy.get_param('~position_std', 0.1)
        vel_std = rospy.get_param('~velocity_std', 0.05)
        acc_std = rospy.get_param('~acceleration_std', 0.02)
        yaw_std = rospy.get_param('~yaw_std', 0.1)
        
        publisher.set_noise_parameters(pos_std, vel_std, acc_std, yaw_std)
        publisher.run()

    except rospy.ROSInterruptException:
        rospy.loginfo("Debug Actor Publisher 已停止")


if __name__ == '__main__':
    debug_actor_publisher()
