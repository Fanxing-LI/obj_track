#!/usr/bin/env python3
"""
Elastic-Tracker VisFly Bridge Node
Converts between Elastic-Tracker's PositionCommand and VisFly's expected control format
"""

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import PositionCommand, SO3Command
from geometry_msgs.msg import PoseStamped
import torch
from scipy.spatial.transform import Rotation as R
import sys
import os

# Add VisFly path for controller imports
def remove_last_n_folders(path, n=5):
    path = path.rstrip('/\\')
    for _ in range(n):
        path = path[:path.rfind('/')] if '/' in path else ''
    return path

add_path = remove_last_n_folders(os.path.dirname(os.path.abspath(__file__)), 4)
sys.path.append(add_path)

from VisFly.envs.base.controller import PositionController
from VisFly.utils.maths import Quaternion


class ElasticVisFlyBridge:
    """Bridge between Elastic-Tracker and VisFly simulator"""
    
    def __init__(self):
        rospy.init_node('elastic_visfly_bridge', anonymous=False)
        
        # Configuration
        self.num_drones = rospy.get_param('~num_drones', 1)
        self.use_so3_controller = rospy.get_param('~use_so3_controller', False)
        
        # Position controller for converting position commands to bodyrates
        self.position_controller = PositionController()
        
        # State storage
        self.drone_states = [None] * self.num_drones
        self.position_cmds = [None] * self.num_drones
        
        # Setup subscribers and publishers
        self._setup_subscribers()
        self._setup_publishers()
        
        # Control timer (30Hz to match VisFly)
        self.control_timer = rospy.Timer(rospy.Duration(1.0/30.0), self.control_callback)
        
        rospy.loginfo(f"Elastic-VisFly Bridge initialized for {self.num_drones} drones")
        
    def _setup_subscribers(self):
        """Initialize all subscribers"""
        for i in range(self.num_drones):
            # Subscribe to VisFly odometry
            odom_sub = rospy.Subscriber(
                f'visfly/drone_{i}/odom',
                Odometry,
                lambda msg, idx=i: self.visfly_odom_callback(msg, idx),
                queue_size=1
            )
            
            # Subscribe to Elastic-Tracker position commands
            pos_cmd_sub = rospy.Subscriber(
                f'/drone{i}/position_cmd',
                PositionCommand,
                lambda msg, idx=i: self.position_cmd_callback(msg, idx),
                queue_size=1
            )
        
        # Subscribe to VisFly target odometry
        self.target_sub = rospy.Subscriber(
            'visfly/target/odom',
            Odometry,
            self.target_odom_callback,
            queue_size=1
        )
        
    def _setup_publishers(self):
        """Initialize all publishers"""
        self.elastic_odom_pubs = []
        self.elastic_action_pubs = []
        self.so3_cmd_pubs = []
        
        for i in range(self.num_drones):
            # Publish odometry in Elastic format
            odom_pub = rospy.Publisher(
                f'/drone{i}/odom',
                Odometry,
                queue_size=1
            )
            self.elastic_odom_pubs.append(odom_pub)
            
            # Publish converted actions back to VisFly
            action_pub = rospy.Publisher(
                f'/drone{i}/visfly_position_cmd',
                PositionCommand,
                queue_size=1
            )
            self.elastic_action_pubs.append(action_pub)
            
            # Optional SO3Command publisher for lower-level control
            if self.use_so3_controller:
                so3_pub = rospy.Publisher(
                    f'/drone{i}/so3cmd',
                    SO3Command,
                    queue_size=1
                )
                self.so3_cmd_pubs.append(so3_pub)
        
        # Publish target in Elastic format
        self.target_pub = rospy.Publisher(
            '/target/odom',
            Odometry,
            queue_size=1
        )
        
        # Trigger publisher for Elastic planning
        self.trigger_pub = rospy.Publisher(
            '/triger',
            PoseStamped,
            queue_size=1
        )
        
    def visfly_odom_callback(self, msg, drone_idx):
        """
        Forward VisFly odometry to Elastic-Tracker format
        
        Args:
            msg: Odometry from VisFly
            drone_idx: Drone index
        """
        self.drone_states[drone_idx] = msg
        
        # Create Elastic-formatted odometry
        elastic_odom = Odometry()
        elastic_odom.header = msg.header
        elastic_odom.header.frame_id = "world"
        elastic_odom.child_frame_id = f"drone_{drone_idx}"
        
        # Copy pose and twist
        elastic_odom.pose = msg.pose
        elastic_odom.twist = msg.twist
        
        self.elastic_odom_pubs[drone_idx].publish(elastic_odom)
        
    def position_cmd_callback(self, msg, drone_idx):
        """
        Store Elastic-Tracker position commands for processing
        
        Args:
            msg: PositionCommand from Elastic-Tracker
            drone_idx: Drone index
        """
        self.position_cmds[drone_idx] = msg
        
    def target_odom_callback(self, msg):
        """
        Forward target odometry to Elastic-Tracker
        
        Args:
            msg: Target odometry from VisFly
        """
        target_odom = Odometry()
        target_odom.header = msg.header
        target_odom.header.frame_id = "world"
        target_odom.child_frame_id = "target"
        
        # Copy pose and twist
        target_odom.pose = msg.pose
        target_odom.twist = msg.twist
        
        self.target_pub.publish(target_odom)
        
    def control_callback(self, event):
        """
        Main control loop - converts position commands to appropriate format
        This runs at 30Hz to match VisFly's control frequency
        """
        for i in range(self.num_drones):
            if self.position_cmds[i] is None or self.drone_states[i] is None:
                continue
                
            # Get current position command and drone state
            pos_cmd = self.position_cmds[i]
            drone_state = self.drone_states[i]
            
            if self.use_so3_controller:
                # Convert to SO3Command for direct bodyrate control
                so3_cmd = self.position_to_so3(pos_cmd, drone_state)
                if so3_cmd:
                    self.so3_cmd_pubs[i].publish(so3_cmd)
            else:
                # Forward position command directly
                # VisFly's elastic mode handles PositionCommand natively
                self.elastic_action_pubs[i].publish(pos_cmd)
                
    def position_to_so3(self, pos_cmd, drone_state):
        """
        Convert PositionCommand to SO3Command using position controller
        
        Args:
            pos_cmd: PositionCommand message
            drone_state: Current drone odometry
            
        Returns:
            SO3Command message
        """
        # Extract desired state from PositionCommand
        desired_pos = np.array([
            pos_cmd.position.x,
            pos_cmd.position.y,
            pos_cmd.position.z
        ])
        
        desired_vel = np.array([
            pos_cmd.velocity.x,
            pos_cmd.velocity.y,
            pos_cmd.velocity.z
        ])
        
        desired_acc = np.array([
            pos_cmd.acceleration.x,
            pos_cmd.acceleration.y,
            pos_cmd.acceleration.z
        ])
        
        # Extract current state
        current_pos = np.array([
            drone_state.pose.pose.position.x,
            drone_state.pose.pose.position.y,
            drone_state.pose.pose.position.z
        ])
        
        current_vel = np.array([
            drone_state.twist.twist.linear.x,
            drone_state.twist.twist.linear.y,
            drone_state.twist.twist.linear.z
        ])
        
        current_quat = np.array([
            drone_state.pose.pose.orientation.x,
            drone_state.pose.pose.orientation.y,
            drone_state.pose.pose.orientation.z,
            drone_state.pose.pose.orientation.w
        ])
        
        # Use position controller to compute control
        # This is a simplified version - actual implementation would use
        # Elastic's SO3 controller parameters
        pos_error = desired_pos - current_pos
        vel_error = desired_vel - current_vel
        
        # PD control for acceleration command
        kp = np.array([6.0, 6.0, 8.0])
        kd = np.array([4.0, 4.0, 5.0])
        
        acc_cmd = desired_acc + kp * pos_error + kd * vel_error
        
        # Add gravity compensation
        acc_cmd[2] += 9.81
        
        # Compute desired thrust (collective)
        thrust = np.linalg.norm(acc_cmd)
        
        # Compute desired orientation from acceleration
        z_body = acc_cmd / thrust if thrust > 0 else np.array([0, 0, 1])
        yaw = pos_cmd.yaw
        
        x_c = np.array([np.cos(yaw), np.sin(yaw), 0])
        y_body = np.cross(z_body, x_c)
        y_body = y_body / np.linalg.norm(y_body) if np.linalg.norm(y_body) > 0 else np.array([0, 1, 0])
        x_body = np.cross(y_body, z_body)
        
        # Rotation matrix
        R_des = np.column_stack([x_body, y_body, z_body])
        
        # Convert to quaternion
        desired_quat = R.from_matrix(R_des).as_quat()  # [x,y,z,w]
        
        # Create SO3Command
        so3_cmd = SO3Command()
        so3_cmd.header = pos_cmd.header
        
        so3_cmd.force.x = 0.0  # Not used in bodyrate mode
        so3_cmd.force.y = 0.0
        so3_cmd.force.z = thrust
        
        so3_cmd.orientation.x = desired_quat[0]
        so3_cmd.orientation.y = desired_quat[1]
        so3_cmd.orientation.z = desired_quat[2]
        so3_cmd.orientation.w = desired_quat[3]
        
        # Set gains (these would come from Elastic's SO3 controller params)
        so3_cmd.kR = [3.0, 3.0, 1.0]
        so3_cmd.kOm = [0.5, 0.5, 0.25]
        
        so3_cmd.aux.enable_motors = True
        so3_cmd.aux.use_external_yaw = True
        
        return so3_cmd
        
    def trigger_planning(self):
        """Send trigger to start Elastic-Tracker planning"""
        trigger_msg = PoseStamped()
        trigger_msg.header.stamp = rospy.Time.now()
        trigger_msg.header.frame_id = "world"
        trigger_msg.pose.position.x = 0.0
        trigger_msg.pose.position.y = 0.0
        trigger_msg.pose.position.z = 2.0
        trigger_msg.pose.orientation.w = 1.0
        
        self.trigger_pub.publish(trigger_msg)
        rospy.loginfo("Triggered Elastic-Tracker planning")
        
    def run(self):
        """Main loop"""
        # Trigger planning after a short delay
        rospy.Timer(rospy.Duration(2.0), lambda e: self.trigger_planning(), oneshot=True)
        
        rospy.spin()


if __name__ == '__main__':
    try:
        bridge = ElasticVisFlyBridge()
        bridge.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Elastic-VisFly Bridge shutting down")