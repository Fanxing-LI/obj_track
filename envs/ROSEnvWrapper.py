import rospy
import numpy as np
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from tf.transformations import quaternion_from_euler
from VisFly.utils.maths import Quaternion


class ROSEnvWrapper:
    def __init__(self, drone_env):
        self.env = drone_env

        # Initialize ROS node
        rospy.init_node('visfly', anonymous=True)

        # Publishers
        self.drone_pose_pubs = []
        for i in range(self.num_agent):
            drone_pose_pub = rospy.Publisher(f'/drone_{i}/pose', PoseStamped, queue_size=1)
            self.drone_pose_pubs.append(drone_pose_pub)
        # self.drone_pose_pub = rospy.Publisher('/drone/pose', PoseStamped, queue_size=1)
        self.target_pose_pub = rospy.Publisher('/target/pose', PoseStamped, queue_size=1)
        self.pointcloud_pub = rospy.Publisher('/env/pointcloud', PointCloud2, queue_size=1)

        # Frame IDs
        self.world_frame = "world"

        rospy.loginfo("Visfly ROS Environment Wrapper initialized")

    @property
    def num_agent(self):
        return self.env.num_envs

    def _create_pose_msg(self, position, orientation=None, frame_id="world"):
        """Create a PoseStamped message from position and orientation"""
        pose_msg = PoseStamped()
        pose_msg.header = Header()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = frame_id

        # Set position
        pose_msg.pose.position.x = float(position[0])
        pose_msg.pose.position.y = float(position[1])
        pose_msg.pose.position.z = float(position[2])

        # Set orientation
        if orientation is not None:
            if len(orientation) == 3:  # Euler angles
                quat = quaternion_from_euler(orientation[0], orientation[1], orientation[2])
                pose_msg.pose.orientation.x = quat[0]
                pose_msg.pose.orientation.y = quat[1]
                pose_msg.pose.orientation.z = quat[2]
                pose_msg.pose.orientation.w = quat[3]
            elif len(orientation) == 4:  # Quaternion
                pose_msg.pose.orientation.x = float(orientation[0])
                pose_msg.pose.orientation.y = float(orientation[1])
                pose_msg.pose.orientation.z = float(orientation[2])
                pose_msg.pose.orientation.w = float(orientation[3])
        else:
            # Default orientation
            pose_msg.pose.orientation.w = 1.0

        return pose_msg

    def _create_pointcloud_msg(self, points):
        """Create a PointCloud2 message from point array"""
        if points is None or len(points) == 0:
            return None

        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.world_frame

        # Define point cloud fields
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)
        ]

        # Convert points to proper format
        if isinstance(points, np.ndarray):
            if points.shape[1] >= 3:
                points = points[:, :3]

        pointcloud_msg = pc2.create_cloud(header, fields, points)
        return pointcloud_msg

    def _publish_ros_data(self):
        """Publish current environment state to ROS topics"""
        try:
            # Publish drone pose
            drone_pos = getattr(self.env, 'drone_pos', None)
            drone_orient = getattr(self.env, 'drone_orient', None)

            if drone_pos is not None:
                drone_pose_msg = self._create_pose_msg(drone_pos, drone_orient)
                self.drone_pose_pub.publish(drone_pose_msg)

            # Publish target pose
            target_pos = getattr(self.env, 'target_pos', None)
            target_orient = getattr(self.env, 'target_orient', None)

            if target_pos is not None:
                target_pose_msg = self._create_pose_msg(target_pos, target_orient)
                self.target_pose_pub.publish(target_pose_msg)

            # Publish point cloud
            pointcloud_data = getattr(self.env, 'pointcloud', None)
            if pointcloud_data is None:
                pointcloud_data = getattr(self.env, 'point_cloud', None)

            if pointcloud_data is not None:
                pointcloud_msg = self._create_pointcloud_msg(pointcloud_data)
                if pointcloud_msg is not None:
                    self.pointcloud_pub.publish(pointcloud_msg)

        except Exception as e:
            rospy.logwarn(f"Failed to publish ROS data: {e}")

    def step(self, action):
        """Step the environment and publish ROS data"""
        result = self.env.step(action)
        self._publish_ros_data()
        return result

    def reset(self, **kwargs):
        """Reset the environment and publish initial ROS data"""
        result = self.env.reset(**kwargs)
        self._publish_ros_data()
        return result

    def close(self):
        """Close the environment and shutdown ROS node"""
        self.env.close()
        if not rospy.is_shutdown():
            rospy.signal_shutdown("Visfly wrapper closing")

    def __getattr__(self, name):
        """Delegate other attributes to the wrapped environment"""
        return getattr(self.env, name)