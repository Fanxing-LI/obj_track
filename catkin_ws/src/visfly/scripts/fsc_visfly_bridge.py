#!/usr/bin/env python3
"""
FSC-VisFly Bridge Node
Converts between FSC's ControlCommand format and VisFly's expected interfaces
"""

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32, Vector3
from scipy.spatial.transform import Rotation as R

# Import FSC message type
try:
    from vision_msgs.msg import ControlCommand
except ImportError:
    rospy.logerr("vision_msgs not found. Please build FSC_Aggressive workspace first.")
    raise


class FSCVisFlyBridge:
    """Bridge between FSC Aggressive autopilot and VisFly simulator"""
    
    def __init__(self):
        rospy.init_node('fsc_visfly_bridge', anonymous=False)
        
        # Configuration
        self.num_drones = rospy.get_param('~num_drones', 1)
        self.camera_offset = np.array([0.15, 0.0, 0.0])  # Camera 15cm forward of drone
        
        # State storage
        self.drone_state = None
        self.target_position = None
        
        # Setup subscribers
        self._setup_subscribers()
        
        # Setup publishers
        self._setup_publishers()
        
        rospy.loginfo("FSC-VisFly Bridge initialized")
        
    def _setup_subscribers(self):
        """Initialize all subscribers"""
        # Subscribe to VisFly odometry (single drone for FSC)
        self.visfly_odom_sub = rospy.Subscriber(
            'visfly/drone_0/odom',
            Odometry,
            self.visfly_odom_callback,
            queue_size=1
        )
        
        # Subscribe to VisFly target position
        self.visfly_target_sub = rospy.Subscriber(
            'visfly/target/odom',
            Odometry,
            self.target_odom_callback,
            queue_size=1
        )
        
    def _setup_publishers(self):
        """Initialize all publishers"""
        # Publish odometry in FSC format
        self.fsc_odom_pub = rospy.Publisher(
            '/hummingbird/ground_truth/odometry',
            Odometry,
            queue_size=1
        )
        
        # Publish target as point cloud (FSC expects camera-frame points)
        self.fsc_target_pub = rospy.Publisher(
            '/hummingbird/aprilfake/point',
            PointCloud,
            queue_size=1
        )
        
    def visfly_odom_callback(self, msg):
        """
        Forward VisFly odometry to FSC with proper frame naming
        
        Args:
            msg: Odometry message from VisFly
        """
        self.drone_state = msg
        
        # Create FSC-formatted odometry
        fsc_odom = Odometry()
        fsc_odom.header = msg.header
        fsc_odom.header.frame_id = "world"
        fsc_odom.child_frame_id = "hummingbird/base_link"
        
        # Copy pose and twist
        fsc_odom.pose = msg.pose
        fsc_odom.twist = msg.twist
        
        self.fsc_odom_pub.publish(fsc_odom)
        
    def target_odom_callback(self, msg):
        """
        Convert target position from world frame to camera frame for FSC
        
        Args:
            msg: Target odometry from VisFly
        """
        self.target_position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        
        if self.drone_state is None:
            return
            
        # Convert to camera frame point cloud
        point_cloud = self._create_camera_frame_pointcloud(
            self.target_position,
            self.drone_state
        )
        
        self.fsc_target_pub.publish(point_cloud)
        
    def _create_camera_frame_pointcloud(self, target_world_pos, drone_odom):
        """
        Transform target position from world frame to camera frame
        
        Args:
            target_world_pos: Target position in world frame (numpy array)
            drone_odom: Drone odometry message
            
        Returns:
            PointCloud message with target in camera frame
        """
        # Extract drone pose
        drone_pos = np.array([
            drone_odom.pose.pose.position.x,
            drone_odom.pose.pose.position.y,
            drone_odom.pose.pose.position.z
        ])
        
        drone_quat = np.array([
            drone_odom.pose.pose.orientation.x,
            drone_odom.pose.pose.orientation.y,
            drone_odom.pose.pose.orientation.z,
            drone_odom.pose.pose.orientation.w
        ])
        
        # Transform to camera frame
        target_camera = self._world_to_camera_transform(
            target_world_pos,
            drone_pos,
            drone_quat
        )
        
        # Create PointCloud message
        point_cloud = PointCloud()
        point_cloud.header.stamp = rospy.Time.now()
        point_cloud.header.frame_id = "camera_depth_optical_center_link"
        
        point = Point32()
        point.x = float(target_camera[0])
        point.y = float(target_camera[1])
        point.z = float(target_camera[2])
        point_cloud.points = [point]
        
        return point_cloud
        
    def _world_to_camera_transform(self, target_world, drone_pos, drone_quat):
        """
        Transform point from world frame to camera frame
        
        Camera frame convention (FSC):
        - X: right
        - Y: down  
        - Z: forward
        
        Args:
            target_world: Target position in world frame
            drone_pos: Drone position in world frame
            drone_quat: Drone orientation quaternion (x,y,z,w)
            
        Returns:
            Target position in camera frame
        """
        # Drone rotation
        drone_rot = R.from_quat(drone_quat)
        
        # Camera rotation relative to drone (looking forward)
        # FSC camera: Z-forward, X-right, Y-down
        # Drone body: X-forward, Y-left, Z-up
        camera_rot_relative = R.from_euler('xyz', [0, 0, -90], degrees=True) * \
                             R.from_euler('xyz', [90, 0, 0], degrees=True)
        
        # Camera position in world frame
        camera_pos_world = drone_pos + drone_rot.apply(self.camera_offset)
        
        # Camera rotation in world frame
        camera_rot_world = drone_rot * camera_rot_relative
        
        # Transform target to camera frame
        target_relative = target_world - camera_pos_world
        target_camera = camera_rot_world.inv().apply(target_relative)
        
        return target_camera
        
    def run(self):
        """Main loop"""
        rate = rospy.Rate(30)  # 30Hz update rate
        
        while not rospy.is_shutdown():
            rate.sleep()


if __name__ == '__main__':
    try:
        bridge = FSCVisFlyBridge()
        bridge.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("FSC-VisFly Bridge shutting down")