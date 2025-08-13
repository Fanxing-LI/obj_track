#!/usr/bin/env python3
import rospy
from std_msgs.msg import String

class MyNode:
    def __init__(self):
        # 初始化节点，并获取参数
        rospy.init_node('my_node')
        
        # 从 launch 文件读取参数（默认值：1 Hz）
        self.rate_val = rospy.get_param('~rate', 1.0)  # '~'表示私有参数
        self.message_prefix = rospy.get_param('~prefix', 'Hello')
        
        self.pub = rospy.Publisher('chatter', String, queue_size=10)
        self.rate = rospy.Rate(self.rate_val)

    def run(self):
        while not rospy.is_shutdown():
            msg = f"{self.message_prefix} ROS at {rospy.get_time()}"
            rospy.loginfo(msg)
            self.pub.publish(msg)
            self.rate.sleep()

if __name__ == '__main__':
    try:
        node = MyNode()
        node.run()
    except rospy.ROSInterruptException:
        pass