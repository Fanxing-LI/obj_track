import os
import numpy as np
import torch as th
from habitat_sim.sensor import SensorType
import os
import sys
sys.path.append(os.getcwd())
from VisFly.envs.base.droneGymEnv import DroneGymEnvsBase
from typing import Optional, Dict, Union, Tuple
from gymnasium import spaces
from VisFly.utils.type import TensorDict
from VisFly.utils.maths import Quaternion
import math


class AcrobaticEnv(DroneGymEnvsBase):
    """
    Environment for acrobatic flight maneuvers including circling and flips.
    Based on TACO's Target-and-Command-Oriented Reinforcement Learning approach.
    """
    
    def __init__(
            self,
            num_agent_per_scene: int = 1,
            num_scene: int = 1,
            seed: int = 42,
            visual: bool = True,
            requires_grad: bool = False,
            tensor_output: bool = True,
            random_kwargs: dict = {},
            dynamics_kwargs: dict = {},
            scene_kwargs: dict = {},
            sensor_kwargs: list = [],
            device: str = "cpu",
            max_episode_steps: int = 256,
            task_mode: str = "circle",  # "circle", "flip", or "mix"
    ):
        # Set default dynamics kwargs for acrobatic maneuvers
        dynamics_kwargs = {
            "action_type": "bodyrate",
            "ori_output_type": "quaternion",
            "dt": 0.0025,
            "ctrl_dt": 0.02,
            "integrator": "euler",
            "ctrl_delay": True,
            **dynamics_kwargs
        }
        
        super().__init__(
            num_agent_per_scene=num_agent_per_scene,
            num_scene=num_scene,
            seed=seed,
            visual=visual,
            requires_grad=requires_grad,
            tensor_output=tensor_output,
            random_kwargs=random_kwargs,
            dynamics_kwargs=dynamics_kwargs,
            scene_kwargs=scene_kwargs,
            sensor_kwargs=sensor_kwargs,
            device=device,
            max_episode_steps=max_episode_steps,
        )
        
        self.task_mode = task_mode
        self.device = th.device(device)
        
        # Task parameters
        self.circle_radius = 1.2  # Target circle radius (from TACO)
        self.circle_height = 2.5  # Target circle height
        self.flip_threshold = 0.5  # Success radius for flip positioning
        
        # Command system (from TACO)
        self.command_mode = th.zeros(self.num_envs, dtype=th.long, device=self.device)  # 0: circle, 1: flip
        self.command_param = th.zeros(self.num_envs, device=self.device)  # Circle speed or flip radians
        
        # Initialize commands based on task mode
        if task_mode == "circle":
            self.command_mode[:] = 0
            self.command_param[:] = 5.0  # Default circle speed (m/s)
        elif task_mode == "flip":
            self.command_mode[:] = 1
            self.command_param[:] = 4 * th.pi  # Default 2 flips
        elif task_mode == "mix":
            # Randomly assign tasks
            self.command_mode = th.randint(0, 2, (self.num_envs,), device=self.device)
            self.command_param[self.command_mode == 0] = 5.0  # Circle speed
            self.command_param[self.command_mode == 1] = 4 * th.pi  # Flip radians
        
        # Target position (center for circle, hover point for flip)
        self.target_pos = th.zeros((self.num_envs, 3), device=self.device)
        self.target_pos[:, 2] = self.circle_height
        
        # Flip tracking
        self.flip_start_orientation = None
        self.flip_accumulated_angle = th.zeros(self.num_envs, device=self.device)
        
        # Observation space
        self.observation_space["command"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )
        
        # Update state observation space to include command
        original_state_shape = self.observation_space["state"].shape[-1]
        self.observation_space["state"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(original_state_shape + 2,), dtype=np.float32
        )
        
    def get_observation(self, indices=None) -> TensorDict:
        """Get observation including command information."""
        if indices is None:
            indices = slice(None)
            
        # Get base observation
        base_obs = self.state[indices]
        
        # Add command to observation - ensure same device as base_obs
        command_obs = th.stack([
            self.command_mode[indices].float(),
            self.command_param[indices]
        ], dim=-1).to(base_obs.device)
        
        # Concatenate state and command
        full_state = th.cat([base_obs, command_obs], dim=-1)
        
        obs_dict = TensorDict({"state": full_state, "command": command_obs})
        
        # Add visual observations if enabled
        if self.visual and hasattr(self, 'sensor_obs'):
            for sensor_key in self._visual_sensor_list:
                if sensor_key in self.sensor_obs:
                    obs_dict[sensor_key] = th.from_numpy(self.sensor_obs[sensor_key][indices]).to(self.device)
        
        return obs_dict
    
    def get_success(self) -> th.Tensor:
        """Check success condition based on current task."""
        success = th.zeros(self.num_envs, dtype=th.bool, device=self.device)
        
        # Circle task success: maintain position within threshold
        circle_mask = self.command_mode == 0
        if circle_mask.any():
            circle_dist = th.norm(self.position[circle_mask] - self.target_pos[circle_mask], dim=1)
            success[circle_mask] = circle_dist < 0.3
            
        # Flip task success: completed required flips
        flip_mask = self.command_mode == 1
        if flip_mask.any():
            flip_success = (self.flip_accumulated_angle[flip_mask] >= 
                          (self.command_param[flip_mask] - 0.1))  # Small tolerance
            success[flip_mask] = flip_success
            
        return success
    
    def compute_circle_reward(self, indices):
        """Compute reward for circling task based on TACO's implementation."""
        if not isinstance(indices, th.Tensor):
            indices = th.tensor(indices, device=self.device, dtype=th.long)
            
        # Get relative position to target
        relative_pos = self.position[indices] - self.target_pos[indices]
        
        # Calculate distance to circle
        horizontal_dist = th.norm(relative_pos[:, :2], dim=1) - self.circle_radius
        vertical_dist = th.abs(relative_pos[:, 2])
        pos_dist = th.sqrt(horizontal_dist**2 + vertical_dist**2)
        
        # Position reward
        pos_reward = 1.0 / (1.0 + pos_dist**2) + 1.0 / (1.0 + 10 * pos_dist**2)
        
        # Calculate desired and actual velocity
        circle_speed = self.command_param[indices]
        
        # Create circle coordinate system
        new_x = -relative_pos / (th.norm(relative_pos, dim=1, keepdim=True) + 1e-8)
        new_x[:, 2] = 0
        new_x = new_x / (th.norm(new_x, dim=1, keepdim=True) + 1e-8)
        new_z = th.zeros_like(new_x)
        new_z[:, 2] = 1
        new_y = th.cross(new_z, new_x, dim=1)
        new_y = new_y / (th.norm(new_y, dim=1, keepdim=True) + 1e-8)
        
        # Desired tangential velocity
        desired_vel = th.zeros_like(self.velocity[indices])
        desired_vel[:, 1] = circle_speed
        
        # Project velocity onto circle coordinates
        normal_vel = th.sum(self.velocity[indices] * new_x, dim=1)
        tangential_vel = th.sum(self.velocity[indices] * new_y, dim=1)
        actual_vel = th.stack([normal_vel, tangential_vel, self.velocity[indices, 2]], dim=1)
        
        # Velocity reward
        vel_dist = th.norm(actual_vel - desired_vel, dim=1)
        vel_reward = 1.0 / (1.0 + vel_dist**2) + 1.0 / (1.0 + 10 * vel_dist**2)
        
        # Heading reward (drone should point along tangent)
        heading = self.orientation[indices].to_matrix()[:, :, 0]
        direction_dist = 1 + th.sum(new_x[:, :2] * heading[:, :2], dim=1) / (th.norm(heading[:, :2], dim=1) + 1e-8)
        heading_reward = 1.0 / (1.0 + direction_dist**2) + 1.0 / (1.0 + 10 * direction_dist**2)
        
        # Combined reward
        reward = pos_reward * vel_reward * heading_reward
        
        return reward / 100.0
    
    def compute_flip_reward(self, indices):
        """Compute reward for flip task based on TACO's implementation."""
        if not isinstance(indices, th.Tensor):
            indices = th.tensor(indices, device=self.device, dtype=th.long)
            
        # Position reward (stay near target)
        relative_pos = self.position[indices] - self.target_pos[indices]
        pos_dist = th.norm(relative_pos, dim=1)
        pos_reward = 1.0 / (1.0 + pos_dist) + 1.0 / (1.0 + 10 * pos_dist)
        
        # X-tilt reward (should be tilted for flip)
        x_tilt = self.orientation[indices].to_matrix()[:, 0, 0]
        x_tilt_dist = 1 - x_tilt
        x_tilt_reward = 1.0 / (1.0 + 10 * x_tilt_dist)
        
        # Flip progress reward
        flip_progress = self.flip_accumulated_angle[indices] / (2 * th.pi)  # Number of completed flips
        target_flips = self.command_param[indices] / (2 * th.pi)
        flip_dist = th.abs(flip_progress - target_flips)
        flip_reward = 1.0 / (1.0 + flip_dist**2) + 1.0 / (1.0 + 10 * flip_dist**2)
        
        # Combined reward
        reward = pos_reward * x_tilt_reward * flip_reward
        
        return reward / 100.0
    
    def compute_circle_reward_differentiable(self) -> th.Tensor:
        """Compute differentiable circle reward for all agents."""
        # Position reward - distance to circle center
        dist_to_center = th.norm(self.position[:, :2] - self.target_pos[:, :2], dim=1)
        pos_error = th.abs(dist_to_center - self.circle_radius) 
        pos_reward = th.exp(-2.0 * pos_error)
        
        # Height reward
        height_error = th.abs(self.position[:, 2] - self.circle_height)
        height_reward = th.exp(-2.0 * height_error)
        
        # Velocity reward - encourage movement
        velocity_mag = th.norm(self.velocity[:, :2], dim=1)
        target_velocity = self.command_param  # Circle speed command
        vel_error = th.abs(velocity_mag - target_velocity)
        vel_reward = th.exp(-0.5 * vel_error)
        
        # Orientation reward - stay upright
        z_axis = self.orientation.to_matrix()[:, :, 2]
        upright_reward = z_axis[:, 2] * 0.5 + 0.5
        
        # Combine rewards
        reward = pos_reward * height_reward * vel_reward * upright_reward
        
        return reward
    
    def compute_flip_reward_differentiable(self) -> th.Tensor:
        """Compute differentiable flip reward for all agents."""
        # Position reward - stay near target position
        pos_error = th.norm(self.position - self.target_pos, dim=1)
        pos_reward = th.exp(-2.0 * pos_error)
        
        # Flip progress reward
        flip_progress = self.flip_accumulated_angle / (self.command_param + 1e-6)
        flip_reward = th.tanh(flip_progress * 3.0)
        
        # X-axis tilt reward (encourage flipping around x-axis)
        x_axis = self.orientation.to_matrix()[:, :, 0]
        x_tilt = th.abs(x_axis[:, 2])
        x_tilt_reward = x_tilt * 0.5 + 0.5
        
        # Combine rewards
        reward = pos_reward * x_tilt_reward * flip_reward
        
        return reward / 100.0
    
    def get_reward(self) -> th.Tensor:
        """Compute reward based on current task mode."""
        # When requires_grad is True, compute differentiable rewards
        if self.requires_grad:
            # Use differentiable operations for reward computation
            reward = th.zeros(self.num_envs, device=self.device, requires_grad=False)
            
            # Only compute rewards if orientation is properly initialized
            if hasattr(self, 'orientation') and hasattr(self.orientation, 'to_matrix'):
                # Circle task - use differentiable operations
                circle_mask = (self.command_mode == 0).float()
                if circle_mask.sum() > 0:
                    circle_rewards = self.compute_circle_reward_differentiable()
                    reward = reward + circle_mask * circle_rewards
                    
                # Flip task - use differentiable operations  
                flip_mask = (self.command_mode == 1).float()
                if flip_mask.sum() > 0:
                    flip_rewards = self.compute_flip_reward_differentiable()
                    reward = reward + flip_mask * flip_rewards
                
                # Penalty for low altitude - differentiable
                altitude_penalty = th.sigmoid((0.5 - self.position[:, 2]) * 10.0)
                reward = reward - altitude_penalty
                
            return reward
        else:
            # Non-differentiable reward computation for normal training
            reward = th.zeros(self.num_envs, device=self.device)
            
            # Only compute rewards if orientation is properly initialized
            if hasattr(self, 'orientation') and hasattr(self.orientation, 'to_matrix'):
                # Circle task
                circle_indices = th.where(self.command_mode == 0)[0]
                if len(circle_indices) > 0:
                    reward[circle_indices] = self.compute_circle_reward(circle_indices)
                    
                # Flip task
                flip_indices = th.where(self.command_mode == 1)[0]
                if len(flip_indices) > 0:
                    reward[flip_indices] = self.compute_flip_reward(flip_indices)
                
                # Penalty for low altitude or collision
                reward[self.position[:, 2] < 0.5] -= 1.0
            
            return reward
    
    def step(self, action, is_test=False):
        """Step the environment and update flip tracking."""
        # Store orientation before step for flip tracking
        if self.flip_start_orientation is None:
            self.flip_start_orientation = self.orientation.clone()
        
        # Step the environment
        obs, reward, done, info = super().step(action, is_test)
        
        # Update flip accumulated angle
        flip_mask = self.command_mode == 1
        if flip_mask.any() and self.flip_start_orientation is not None:
            # Calculate angle change from start orientation
            current_orientation = self.orientation[flip_mask]
            start_orientation = self.flip_start_orientation[flip_mask]
            
            # Compute relative quaternion
            rel_quat = current_orientation * start_orientation.inv()
            
            # Extract rotation around x-axis
            rot_matrix = rel_quat.to_matrix()
            x_axis = rot_matrix[:, 0, :]
            
            # Calculate accumulated angle (considering direction)
            angle_change = th.atan2(x_axis[:, 2], x_axis[:, 1])
            self.flip_accumulated_angle[flip_mask] = th.abs(angle_change)
        
        return obs, reward, done, info
    
    def reset_agent_by_id(self, agent_indices=None, state=None, reset_obs=None):
        """Reset agent and flip tracking."""
        # Handle different types of indices
        if agent_indices is None:
            indices_for_reset = None
            indices_for_access = slice(None)
        else:
            indices_for_reset = agent_indices
            indices_for_access = agent_indices
            
        # Reset flip tracking
        self.flip_start_orientation = None
        self.flip_accumulated_angle[indices_for_access] = 0
        
        # For mix mode, randomly reassign tasks
        if self.task_mode == "mix":
            if indices_for_reset is None:
                self.command_mode = th.randint(
                    0, 2, (self.num_envs,), device=self.device
                )
            else:
                self.command_mode[indices_for_access] = th.randint(
                    0, 2, (len(indices_for_access) if hasattr(indices_for_access, '__len__') else 1,), 
                    device=self.device
                )
        
        # Reset target position
        self.target_pos[indices_for_access, 2] = self.circle_height
        
        return super().reset_agent_by_id(indices_for_reset, state, reset_obs)
    
    def reset(self, state=None, obs=None):
        """Reset environment and all tracking."""
        self.flip_start_orientation = None
        self.flip_accumulated_angle.zero_()
        
        # For mix mode, randomly assign tasks
        if self.task_mode == "mix":
            self.command_mode = th.randint(0, 2, (self.num_envs,), device=self.device)
            self.command_param[self.command_mode == 0] = 5.0  # Circle speed
            self.command_param[self.command_mode == 1] = 4 * th.pi  # Flip radians
        
        self.target_pos[:, 2] = self.circle_height
        
        return super().reset(state, obs)