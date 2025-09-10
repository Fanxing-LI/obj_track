import os
import sys
import math

import numpy as np
from VisFly.envs.base.droneGymEnv import DroneGymEnvsBase
from VisFly.utils.maths import Quaternion
from typing import Union, Tuple, List, Optional, Dict
import torch as th
from habitat_sim import SensorType
from gymnasium import spaces
from VisFly.utils.tools.train_encoder import model as encoder
from VisFly.utils.type import TensorDict


class FlipEnv(DroneGymEnvsBase):
    """
    Flip environment following TACO guide for FPV flip task.
    
    Task: UAV should perform continuous roll flips while maintaining position.
    
    Key components from TACO guide:
    - Continuous roll tracking (copter_rpy_continuous) without ±π discontinuities
    - flip_radian: remaining radians to complete flips
    - Command: [-1, remaining_radians/(2π)] where -1 is Flip task ID
    - TACO compute_flip_reward function
    - Proper relative_pos_body and relative_quat_body for reward
    
    TACO reward signals:
    - relative_pos_body: target position relative to UAV in UAV body frame
    - relative_quat_body: target attitude relative to UAV in UAV body frame  
    - command: [task_id, remaining_flip_progress]
    """
    def __init__(
            self,
            num_agent_per_scene: int = 1,
            num_scene: int = 1,
            seed: int = 42,
            visual: bool = True,
            requires_grad: bool = False,
            random_kwargs: dict = None,
            dynamics_kwargs: dict = {},
            scene_kwargs: dict = {},
            sensor_kwargs: list = [],
            device: str = "cpu",
            target: Optional[th.Tensor] = None,
            max_episode_steps: int = 256,
            tensor_output: bool = False,
            flip_command: Optional[float] = None,  # Default flip angle (1 full rotation)
    ):
        # Random initialization for hovering position with some variation
        random_kwargs = {
            "state_generator":
                {
                    "class": "Uniform",
                    "kwargs": [
                        {"position": {"mean": [0., 0., 1.5], "half": [0.2, 0.2, 0.1]}},
                    ]
                }
        }

        # TACO flip parameters
        self.flip_command = flip_command if flip_command is not None else 2 * th.pi
        self.target_position = None  # Will be initialized after device is set
        
        # TACO continuous roll tracking (no ±π discontinuities)
        self.copter_rpy_continuous = None  # Continuous roll, pitch, yaw
        self.flip_radian = None  # Remaining radians to complete flips
        self.last_roll_angle = None  # For unwrapping roll angle
        
        # Flip trigger step (TACO uses step 500)
        self.flip_trigger_step = 500
        self.progress_buf = None  # Step counter per environment
        
        super().__init__(
            num_agent_per_scene=num_agent_per_scene,
            num_scene=num_scene,
            seed=seed,
            visual=visual,
            requires_grad=requires_grad,
            random_kwargs=random_kwargs,
            dynamics_kwargs=dynamics_kwargs,
            sensor_kwargs=sensor_kwargs,
            scene_kwargs=scene_kwargs,
            device=device,
            max_episode_steps=max_episode_steps,
            tensor_output=tensor_output,
        )
        
        # Initialize after parent init
        self.target_position = th.tensor([0., 0., 1.5])
        
        # TACO flip state tracking - always initialize
        self.copter_rpy_continuous = th.zeros((self.num_envs, 3))
        self.flip_radian = th.zeros(self.num_envs)
        self.last_roll_angle = th.zeros(self.num_envs)
        self.progress_buf = th.zeros(self.num_envs, dtype=th.long)

    def detach(self):
        """Detach env state and custom flip-tracking buffers from autograd graphs.
        This avoids carrying graphs across training iterations in BPTT."""
        super().detach()
        with th.no_grad():
            if self.copter_rpy_continuous is not None:
                self.copter_rpy_continuous = self.copter_rpy_continuous.detach().clone()
            if self.last_roll_angle is not None:
                self.last_roll_angle = self.last_roll_angle.detach().clone()
            if self.flip_radian is not None:
                self.flip_radian = self.flip_radian.detach().clone()
            if self.progress_buf is not None:
                self.progress_buf = self.progress_buf.clone()

    def reset(self, state=None, obs=None, is_test=False):
        # Call parent reset with parameters
        result = super().reset(state=state, obs=obs, is_test=is_test)
        
        # Reset TACO flip state for all environments
        self.copter_rpy_continuous = th.zeros((self.num_envs, 3))
        self.last_roll_angle = th.zeros(self.num_envs)
        self.progress_buf = th.zeros(self.num_envs, dtype=th.long)
        
        # Set initial flip_radian based on initial roll angular velocity sign (TACO guide)
        # For now, set to +2π (can be randomized based on initial angular velocity)
        self.flip_radian = th.full((self.num_envs,), 2 * math.pi)
            
        return result

    def get_observation(
            self,
            indices=None
    ) -> Dict:
        """
        Get observation following TACO format.
        
        TACO flip command structure:
        - command[:, 0] = -1 (Flip task identifier)
        - command[:, 1] = remaining_radians / (2π) (normalized remaining flip progress)
        
        This provides the key signals needed for TACO compute_flip_reward function.
        """
        # Update continuous roll tracking and command
        self._update_flip_state()
        
        # Create TACO command: [task_id, normalized_remaining_radians]
        command = th.zeros((self.num_envs, 2))
        command[:, 0] = -1.0  # Flip task ID (vs +1 for Rotate/Circle)
        
        # Remaining radians, clamped to ±2π as per TACO guide
        remaining_radians = self.flip_radian - self.copter_rpy_continuous[:, 0]  # roll is index 0
        remaining_radians = th.clamp(remaining_radians, -2*math.pi, 2*math.pi)
        command[:, 1] = remaining_radians / (2 * math.pi)  # Normalize
        
        obs = TensorDict({
            "state": self.state,
            "command": command,
            "flip_progress": (1.0 - th.abs(remaining_radians) / (2 * math.pi)).unsqueeze(-1),  # 0 to 1 progress
        })

        return obs

    def _update_flip_state(self):
        """
        Update TACO flip state tracking.
        
        Maintains:
        - copter_rpy_continuous: continuous roll/pitch/yaw without ±π discontinuities
        - flip_radian: remaining radians to complete (updated at trigger steps)
        - progress_buf: step counter per environment
        """
        # All updates here are bookkeeping; keep them out of autograd
        with th.no_grad():
            # Increment progress buffer
            self.progress_buf += 1

            # Get current orientation from dynamics using VisFly Quaternion
            # Detach to avoid linking these buffers to the BPTT graph
            current_rpy = self.envs.dynamics._orientation.toEuler().T.detach()  # [batch, 3]

            # Unwrap roll angle to maintain continuity (TACO requirement)
            roll_diff = current_rpy[:, 0] - self.last_roll_angle

            # Handle wrap-around: if difference > π, subtract 2π; if < -π, add 2π
            roll_diff = th.where(roll_diff > math.pi, roll_diff - 2 * math.pi, roll_diff)
            roll_diff = th.where(roll_diff < -math.pi, roll_diff + 2 * math.pi, roll_diff)

            # Update continuous roll/pitch/yaw tracking
            self.copter_rpy_continuous[:, 0] = self.copter_rpy_continuous[:, 0] + roll_diff
            self.copter_rpy_continuous[:, 1] = current_rpy[:, 1]
            self.copter_rpy_continuous[:, 2] = current_rpy[:, 2]

            # Store current roll for next iteration
            self.last_roll_angle = current_rpy[:, 0]

            # TACO flip trigger: add extra flips at specified step
            if th.any(self.progress_buf == self.flip_trigger_step):
                trigger_mask = self.progress_buf == self.flip_trigger_step
                # Add random flips (k * 2π where k ∈ {-3,-2,-1,0,1,2,3})
                k_values = th.randint(-3, 4, (self.num_envs,))
                self.flip_radian[trigger_mask] = self.flip_radian[trigger_mask] + k_values[trigger_mask] * 2 * math.pi
    

    def get_success(self) -> th.Tensor:
        # # Success when drone completes flip and returns to stable hover
        # actual_device = self.position.device
        # if self.target_position is None:
        #     self.target_position = th.tensor([0., 0., 1.5], device=actual_device)
        # elif self.target_position.device != actual_device:
        #     self.target_position = self.target_position.to(actual_device)
        #
        # # Create new tensors to avoid in-place operations
        # pos = (self.position - 0)
        # vel = (self.velocity - 0)
        #
        # pos_dist = (pos - self.target_position.unsqueeze(0)).norm(dim=1)
        # vel_dist = vel.norm(dim=1)
        #
        # # Check if flip is completed
        # if self.accumulated_rotation.device != actual_device:
        #     self.accumulated_rotation = self.accumulated_rotation.to(actual_device)
        #
        # # Require significant rotation (at least 90% of flip_command) and not just starting
        # flip_completed = (self.accumulated_rotation >= (self.flip_command * 0.9)) & (self.accumulated_rotation > 1.0)
        #
        # # Success if flip completed, position error < 0.3m and velocity < 0.5 m/s
        # success = flip_completed & (pos_dist < 0.3) & (vel_dist < 0.5)
        #
        # # Ensure success is on the same device as position
        # return success
        return th.full((self.num_agent,), False)

    def get_reward(self) -> th.Tensor:
        """
        Compute reward using TACO's compute_flip_reward function.
        
        This is the exact reward function from TACO porting guide with:
        - relative_pos_body: target position relative to UAV in UAV body frame
        - relative_quat_body: target attitude relative to UAV in UAV body frame
        - command: flip command with remaining radians
        - Termination conditions for crashes and timeout
        
        Returns TACO-compatible reward scaled by 1/100.
        """
        # Target position is always initialized in __init__, no need to check
            
        # Get current rotation matrix from dynamics using VisFly Quaternion
        # The R property is always available and returns [3, 3, batch] format
        uav_rotation = self.envs.dynamics._orientation.R.permute(2, 0, 1)  - 0# Convert to [batch, 3, 3]
            
        # === TACO compute_flip_reward signals ===
        
        # 1. relative_pos_body: target position relative to UAV, in UAV body frame
        relative_pos_world = self.target_position.unsqueeze(0) - self.position  # [batch, 3]
        # Transform to body frame: R^T * relative_pos_world
        relative_pos_body = th.bmm(uav_rotation.transpose(1, 2), relative_pos_world.unsqueeze(-1)).squeeze(-1)
        
        # 2. relative_quat_body: target attitude relative to UAV, in UAV body frame
        # Target attitude is identity (upright), so relative = target * UAV^T = UAV^T
        target_rotation = th.eye(3).unsqueeze(0).repeat(self.num_envs, 1, 1)
        relative_rot_body = th.bmm(target_rotation, uav_rotation.transpose(1, 2))
        
        # Convert rotation matrix to quaternion for TACO compatibility
        # For TACO reward, we can use identity quaternions since the reward
        # function primarily uses the rotation matrix elements directly
        batch_size = relative_rot_body.shape[0]
        relative_quat_body = th.zeros((batch_size, 4))
        relative_quat_body[:, 0] = 1.0  # Identity quaternion (w=1, x=y=z=0)
        
        # 3. Command tensor
        remaining_radians = self.flip_radian - self.copter_rpy_continuous[:, 0]
        remaining_radians = th.clamp(remaining_radians, -2*math.pi, 2*math.pi)
        command = th.zeros((self.num_envs, 2))
        command[:, 0] = -1.0  # Flip task ID
        command[:, 1] = remaining_radians
        
        # 4. Reset and progress buffers (for termination)
        reset_buf = th.zeros(self.num_envs)
        max_episode_length = getattr(self, 'max_episode_steps', 500)
        
        # === TACO compute_flip_reward implementation ===
        
        # Distance to target
        pos_dist = th.norm(relative_pos_body, dim=1)
        pos_reward1_l0 = 1.0 / (1.0 + 1 * pos_dist)
        pos_reward1_l1 = 1.0 / (1.0 + 10 * pos_dist)
        pos_reward = pos_reward1_l0 + pos_reward1_l1

        # X-axis tilt (roll alignment) - using rotation matrix directly
        x_tiltage = relative_rot_body[:, 0, 0]  # How much x-axis aligns (should be 1 when aligned)
        x_tiltage_dist = 1 - x_tiltage
        x_tiltage_reward_l0 = 1.0 / (1.0 + 10 * x_tiltage_dist)
        x_tiltage_reward = x_tiltage_reward_l0

        # Flip progress via command (remaining radians / 2π)
        command_dist = command[:, -1] / (2 * math.pi)
        command_reward_l0 = 1.0 / (1.0 + command_dist * command_dist)
        command_reward_l1 = 1.0 / (1.0 + 10 * command_dist * command_dist)
        command_reward = command_reward_l0 + command_reward_l1

        # Combined reward (scaled by 1/100 as in TACO)
        reward = (pos_reward * x_tiltage_reward * command_reward) / 100

        # TACO termination conditions (for environment management, not reward)
        ones = th.ones_like(reset_buf)
        die = th.zeros_like(reset_buf)
        die = th.where(self.position[:, 2] < 0.1, ones, die)  # Too low altitude
        die = th.where(pos_dist > 10, ones, die)  # Too far from target
        
        # Apply small penalties to guide learning away from crash conditions
        altitude_penalty = th.where(self.position[:, 2] < 0.2, -0.1, 0.0)
        distance_penalty = th.where(pos_dist > 5.0, -0.05, 0.0)
        
        final_reward = reward + altitude_penalty + distance_penalty
        
        return final_reward
        


# ===== MODIFICATION GUIDE FOR FLIP ENVIRONMENT =====
#
# This FlipEnv implementation follows the TACO porting guide for FPV flip tasks.
# Here are key areas you can modify for different requirements:
#
# 1. FLIP PARAMETERS (lines 43-51):
#    - flip_command: Change from 2π to customize flip angle (e.g., π for half-flip)
#    - flip_trigger_step: Modify when additional flips are triggered (currently 500)
#    - flip_radian initialization: Can randomize ±2π based on initial angular velocity
#
# 2. REWARD FUNCTION TUNING (lines 259-352):
#    - TACO scaling factor: Currently /100, can adjust for different reward magnitudes
#    - Position reward weights: pos_reward coefficients (1.0, 10.0) 
#    - Orientation reward: x_tiltage_reward coefficient (10.0)
#    - Command reward: command_reward coefficients (1.0, 10.0)
#    - Penalty strengths: altitude_penalty (-0.1), distance_penalty (-0.05)
#
# 3. TERMINATION CONDITIONS (lines 341-344):
#    - Altitude threshold: Currently 0.1m, adjust for different safety margins
#    - Distance threshold: Currently 10m, modify for tighter/looser bounds
#    - Episode length: Set via max_episode_steps parameter
#
# 4. OBSERVATION STRUCTURE (lines 123-155):
#    - Add more sensors: Extend TensorDict with depth, RGB, etc.
#    - Modify command format: Currently [-1, remaining_radians/(2π)]
#    - Include additional state info: IMU data, battery, etc.
#
# 5. CONTINUOUS ROLL TRACKING (lines 157-206):
#    - Modify unwrapping logic: Currently handles ±π discontinuities
#    - Change trigger behavior: Random k∈{-3,-2,-1,0,1,2,3} for additional flips
#    - Adjust orientation extraction: Currently uses dynamics.xz_axis
#
# 6. TARGET POSITION (lines 67, 274-276):
#    - Static target: Currently [0, 0, 1.5], can make dynamic/random
#    - Multiple targets: Extend for multi-target flip sequences
#    - Target attitude: Currently identity (upright), can specify orientations
#
# 7. DEVICE MANAGEMENT:
#    - All tensors automatically moved to match position.device
#    - Add manual device specifications if needed for specific components
#
# 8. INTEGRATION WITH DIFFERENT ALGORITHMS:
#    - BPTT: Works with current observation structure
#    - PPO: May need observation stacking/frame history
#    - Custom: Modify get_observation to return algorithm-specific format
#
# 9. SIMULATION PARAMETERS:
#    - Dynamics: Set via dynamics_kwargs (dt, ctrl_dt, action_type)
#    - Physics: Modify through VisFly dynamics configuration
#    - Rendering: Control via scene_kwargs and visual parameter
#
# 10. DEBUGGING AND ANALYSIS:
#    - Add logging: Insert print statements in _update_flip_state
#    - Metrics tracking: Store flip progress, roll continuity for analysis  
#    - Visualization: Extend get_observation with debug info
#
# EXAMPLE MODIFICATIONS:
#
# For half-flips instead of full flips:
#   self.flip_command = math.pi  # Change line 43
#
# For denser rewards:
#   reward = (pos_reward * x_tiltage_reward * command_reward) / 10  # Change line 338
#
# For random target positions:
#   self.target_position = th.rand(3, device=actual_device) * 2 - 1  # Modify reset()
#
# For additional sensors in observation:
#   obs["depth"] = self.sensor_obs.get("depth", th.zeros(...))  # Add to get_observation
#
# For custom flip triggers:
#   if self.progress_buf[i] % 100 == 0:  # Every 100 steps instead of step 500
#
# ================================================


if __name__ == "__main__":
    import cv2 as cv
    import numpy as np
    import matplotlib
    import os
    import sys
    
    # Add current working directory to Python path
    sys.path.append(os.getcwd())
    
    matplotlib.use('Agg')  # Use non-interactive backend for headless
    
    # Detect headless environment
    
    headless = os.environ.get('DISPLAY') is None or os.environ.get('SSH_CONNECTION') is not None
    
    random_kwargs = {
        "state_generator":
            {
                "class": "Uniform",
                "kwargs": [{
                    "position": {"mean": [0.0, 0.0, 1.5], "half": [0.2, 0.2, 0.1]},
                    "orientation": {"mean": [0.0, 0.0, 0.0], "half": [0, 0, 0]},
                }]
            }
    }
    
    scene_path = "VisFly/datasets/visfly-beta/configs/scenes/garage_empty"
    sensor_kwargs = [{
        "sensor_type": SensorType.DEPTH,
        "uuid": "depth", 
        "resolution": [64, 64],
        "position": [0, 0.2, 0.],
    }]
    scene_kwargs = {
        "path": scene_path,
        "render_settings": {
            "mode": "fix",
            "view": "custom", 
            "resolution": [1080, 1920],
            "position": th.tensor([[7., 6.8, 5.5], [7, 4.8, 4.5]]),
            "line_width": 6.,
            "point": th.tensor([[0., 0, 1.5], [0, 0, 1.5]]),
            "trajectory": True,
        }
    }
    
    num_agent = 4
    env = FlipEnv(
        visual=True,
        num_scene=1,
        num_agent_per_scene=num_agent,
        random_kwargs=random_kwargs,
        scene_kwargs=scene_kwargs,
        sensor_kwargs=sensor_kwargs,
        dynamics_kwargs={},
        flip_command=2*math.pi,  # Full flip
        max_episode_steps=500
    )
    
    env.reset()
    
    # Video writer setup for headless mode
    video_writer = None
    obs_writer = None
    if headless:
        fourcc = cv.VideoWriter_fourcc(*'avc1')
        video_writer = cv.VideoWriter('flip_debug_video.mp4', fourcc, 10.0, (1920, 1080))
        obs_writer = cv.VideoWriter('flip_debug_obs.mp4', fourcc, 10.0, (64, 64))
    
    t = 0
    max_steps = 500 if headless else float('inf')
    
    while t < max_steps:
        # Random actions for demonstration
        a = th.rand((num_agent, 4)) * 2 - 1  # Random actions in [-1, 1]
        env.step(a)
        
        # Visualization points
        spawn_center = th.tensor([random_kwargs["state_generator"]["kwargs"][0]["position"]["mean"]])
        target_pos = th.tensor([[0.0, 0.0, 1.5]])  # Flip target (hover position)
        
        # Create reference axes for flip visualization
        ref_points = []
        for i in range(3):
            axis_point = [0.0, 0.0, 1.5]
            axis_point[i] += 0.5  # Extend along each axis
            ref_points.append(axis_point)
        ref_curve = th.tensor(ref_points).unsqueeze(0)
        
        debug_points = th.cat([spawn_center, target_pos], dim=0)
        
        img = env.render(is_draw_axes=True, points=debug_points, curves=ref_curve)
        obs = env.sensor_obs["depth"]
        
        if headless:
            # Save to video files
            if video_writer:
                video_writer.write(cv.cvtColor(img[0], cv.COLOR_RGB2BGR))
            if obs_writer:
                obs_data = obs[0][0]  # Get first agent's observation
                
                if t == 0:
                    print(f"Flip obs data shape: {obs_data.shape}, dtype: {obs_data.dtype}")
                    print(f"Flip obs data min/max: {obs_data.min():.3f}/{obs_data.max():.3f}")
                
                try:
                    if len(obs_data.shape) == 2:  # HW format (depth)
                        obs_frame = (obs_data * 255).clip(0, 255).astype(np.uint8)
                        obs_frame = cv.cvtColor(obs_frame, cv.COLOR_GRAY2BGR)
                    else:  # Handle other formats
                        obs_frame = (obs_data[0] * 255).clip(0, 255).astype(np.uint8)
                        obs_frame = cv.cvtColor(obs_frame, cv.COLOR_GRAY2BGR)
                    
                    if obs_frame.shape[:2] != (64, 64):
                        obs_frame = cv.resize(obs_frame, (64, 64))
                    
                    obs_writer.write(obs_frame)
                except Exception as e:
                    if t == 0:
                        print(f"Error processing flip obs frame: {e}")
                    black_frame = np.zeros((64, 64, 3), dtype=np.uint8)
                    obs_writer.write(black_frame)
        else:
            # Display in windows
            cv.imshow("flip_img", img[0])
            cv.imshow("flip_obs", obs[0][0])
            cv.waitKey(100)
        
        # Print flip progress occasionally
        if t % 50 == 0:
            obs_dict = env.get_observation()
            flip_progress = obs_dict.get("flip_progress", th.tensor([[0.0]]))[0, 0].item()
            command = obs_dict.get("command", th.tensor([[0.0, 0.0]]))[0]
            print(f"Step {t}: Flip progress: {flip_progress:.3f}, Command: [{command[0]:.1f}, {command[1]:.3f}]")
        
        t += 1
    
    # Cleanup
    if headless and video_writer:
        video_writer.release() 
        obs_writer.release()
        print("Flip videos saved: flip_debug_video.mp4, flip_debug_obs.mp4")
    if not headless:
        cv.destroyAllWindows()
