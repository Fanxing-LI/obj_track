import os
import sys
import math
sys.path.append(os.getcwd())
import numpy as np
from VisFly.envs.base.droneGymEnv import DroneGymEnvsBase
from typing import Union, Tuple, List, Optional, Dict
import torch as th
from habitat_sim import SensorType
from gymnasium import spaces
from VisFly.utils.tools.train_encoder import model as encoder
from VisFly.utils.type import TensorDict


class CircleEnv(DroneGymEnvsBase):
    """
    Circle/Rotate environment following TACO guide for FPV circle task.
    
    Task: UAV should fly in a circle around a target with specified radius and tangential speed.
    
    Key components:
    - Desired radius: 1.2m 
    - Tangential speed: configurable (default 1.0 m/s)
    - Command: [1, tangential_speed] where 1 is the Rotate task ID
    - Reward based on position (distance to circle), velocity (tangential alignment), 
      and heading (facing tangent direction)
    
    Reward function directly copied from TACO compute_rotating_reward.
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
            desired_radius: float = 2.2,  # TACO standard radius
            tangential_speed: float = 1.0,  # Default tangential speed in m/s
    ):
        # Random initialization for hovering position with some variation around circle
        # Use provided random_kwargs if given; otherwise, fallback to a reasonable default.
        if random_kwargs is None:
            random_kwargs = {
                "state_generator":
                    {
                        "class": "Uniform",
                        "kwargs": [
                            {"position": {"mean": [5.0, 0.0, 1.5], "half": [0.3, 0.3, 0.1]}},
                        ]
                    }
            }
        # Extract target center from random kwargs (position mean)
        _center = None
        try:
            _center = random_kwargs.get("state_generator", {}).get("kwargs", [])[0].get("position", {}).get("mean", None)
        except Exception:
            _center = None
        if _center is None:
            _center = [0.0, 0.0, 1.5]

        # Circle task parameters
        self.desired_radius = desired_radius
        self.tangential_speed = tangential_speed
        self.target_position = None  # Will be initialized after device is set
        
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
        # Set circle center (target) from random kwargs mean to allow configurable birth position
        self.target_position = th.as_tensor(_center, dtype=th.float32, device=self.device)

        # Expand observation space: base state + [rel_pos(3), rel_linvel(3), tangential_speed(1)]
        try:
            base_dim = int(self.observation_space.spaces["state"].shape[0])
            add_dim = 3 + 3 + 1
            self.observation_space.spaces["state"] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(base_dim + add_dim,),
                dtype=np.float32
            )
        except Exception:
            pass


    def reset(self, state=None, obs=None, is_test=False):
        """Reset environment. Command will be set during get_observation."""
        # Call parent reset with parameters
        result = super().reset(state=state, obs=obs, is_test=is_test)
        
        # Reset is called by parent
            
        return result

    def get_observation(
            self,
            indices=None
    ) -> Dict:
        """
        Observation contains:
        - default UAV state (kept from base)
        - relative position (target - pos) in head frame (3)
        - relative linear velocity (target - vel, target is static) in head frame (3)
        - desired tangential speed as a scalar feature (1)

        Note: command ID removed; speed is embedded as a feature.
        """
        # Base state
        base_state = self.state.to(self.device)

        # Relative quantities in world frame
        relative_pos = (self.target_position.unsqueeze(0) - self.position)
        relative_linvel = (-self.velocity - 0)

        # Transform to head frame
        orientation = self.envs.dynamics._orientation.clone()
        rel_pos_head = orientation.world_to_head(relative_pos.T).T
        rel_linvel_head = orientation.world_to_head(relative_linvel.T).T

        # Optional scaling to keep ranges moderate (similar to HardNavigation)
        rel_pos_head_scaled = rel_pos_head / 10.0
        rel_linvel_head_scaled = rel_linvel_head / 10.0

        # Tangential speed as a per-agent scalar feature
        spd = th.full((self.num_envs, 1), float(self.tangential_speed), device=self.device, dtype=th.float32)

        state = th.hstack([
            base_state,
            rel_pos_head_scaled,
            rel_linvel_head_scaled,
            spd,
        ]).to(self.device)

        return TensorDict({
            "state": state,
        })

    def get_success(self) -> th.Tensor:
        # """
        # Success for the circle task when the agent simultaneously:
        # - stays close to desired radius (± r_tol)
        # - matches desired tangential speed (± v_tol)
        # - keeps altitude near the target (± z_tol)
        #
        # This provides a sparse success signal on top of dense shaping.
        # """
        # relative_pos = self.target_position.unsqueeze(0) - self.position
        # radius_now = th.norm(relative_pos[:, :2], dim=1)
        #
        # # Local frame to get tangential direction
        # new_z = th.zeros_like(relative_pos)
        # new_z[:, 2] = 1
        # new_x = -relative_pos.clone()
        # new_x[:, 2] = 0
        # new_x = new_x / (th.norm(new_x, dim=1, keepdim=True) + 1e-8)
        # new_y = th.cross(new_z, new_x, dim=1)
        # new_y = new_y / (th.norm(new_y, dim=1, keepdim=True) + 1e-8)
        # tangential_speed = th.sum((self.velocity - 0 )* new_y, dim=1)
        #
        # # Tolerances
        # r_tol = 0.25
        # v_tol = max(0.25, 0.25 * float(self.tangential_speed - 0))
        # z_tol = 0.25
        #
        # cond_radius = th.abs(radius_now - self.desired_radius) < r_tol
        # cond_speed = th.abs(tangential_speed - self.tangential_speed) < v_tol
        # cond_alt = th.abs(relative_pos[:, 2]) < z_tol
        #
        # return cond_radius & cond_speed & cond_alt
        return th.full((self.num_agent,), False)

    def get_reward(self):
        """
        Linear shaping for circle (few terms):
        reward = + alive
                 - w_r * |radius - r_des|
                 - w_z * |z - z_des|
                 - w_v * |v_tan - v_des|
                 - w_rr * |v_radial|
                 - w_d * max(0, -v_z)
        """
        # Relative position (target - UAV)
        relative_pos = self.target_position.unsqueeze(0) - self.position
        pos_xy = relative_pos[:, :2]
        radius_now = th.norm(pos_xy, dim=1)

        # Local frame: x radial inward, y tangential (CCW), z up
        new_z = th.zeros_like(relative_pos)
        new_z[:, 2] = 1
        new_x = -relative_pos.clone(); new_x[:, 2] = 0
        # Robust normalization on XY (fallback to +X when nearly zero to avoid undefined tangent)
        _xy_norm = th.norm(new_x, dim=1, keepdim=True)
        _fallback_x = th.zeros_like(new_x); _fallback_x[:, 0] = 1.0
        new_x = th.where(_xy_norm > 1e-6, new_x / (_xy_norm + 1e-8), _fallback_x)
        new_y = th.cross(new_z, new_x, dim=1)
        new_y = new_y / (th.norm(new_y, dim=1, keepdim=True) + 1e-8)

        # Velocity components
        v_world = self.velocity.clone()
        v_radial = th.sum(v_world * new_x, dim=1)
        v_tan = th.sum(v_world * new_y, dim=1)
        v_z = v_world[:, 2]

        # Linear errors
        r_err = th.abs(radius_now - self.desired_radius)
        z_err = th.abs(relative_pos[:, 2])
        v_err = th.abs(v_tan - self.tangential_speed)
        v_radial_mag = th.abs(v_radial)
        # Vertical velocity terms removed per request; manage altitude via z deviation only
        # Direction awareness: penalize motion opposite to desired tangential direction
        dir_sign = 1.0 if float(self.tangential_speed) >= 0.0 else -1.0  # +1: CCW, -1: CW
        wrong_dir = th.clamp(-dir_sign * v_tan, min=0.0)

        # Weights
        w_r = 0.06
        w_z = 8  # strengthen altitude maintenance
        w_v = 0.06
        w_rr = 0.03
        # Removed vertical speed penalties (downward/absolute); rely on altitude deviation only
        w_dir = 0.06

        # Alive bonus to remove incentive for early crash
        alive = 0.1

        reward = alive - (w_r * r_err + w_z * z_err + w_v * v_err + w_rr * v_radial_mag + w_dir * wrong_dir)

        return {
            "reward": reward,
            "r_radius_err": -r_err * w_r,
            "r_altitude_err": -z_err * w_z,
            "r_speed_err": -v_err * w_v,
            "pen_radial_vel": -v_radial_mag * w_rr,
            # vertical speed penalties removed
            "pen_wrong_dir": -wrong_dir * w_dir,
            "alive": th.full_like(reward, alive),
        }

    def get_failure(self) -> th.Tensor:
        """Fail when too low or far outside workable annulus."""
        relative_pos = self.target_position.unsqueeze(0) - self.position
        radius_now = th.norm(relative_pos[:, :2], dim=1)
        too_low = self.position[:, 2] < 0.1
        too_far = th.abs(radius_now - self.desired_radius) > 5.0
        return too_low | too_far | self.is_collision

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
                    "position": {"mean": [5, 0, 1.5], "half": [0.3, 0.3, 0.1]},
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
    env = CircleEnv(
        visual=True,
        num_scene=1,
        num_agent_per_scene=num_agent,
        random_kwargs=random_kwargs,
        scene_kwargs=scene_kwargs,
        sensor_kwargs=sensor_kwargs,
        dynamics_kwargs={},
        desired_radius=1.2,  # TACO standard radius
        tangential_speed=1.0,  # Target speed
        max_episode_steps=500
    )
    
    env.reset()
    
    # Video writer setup for headless mode
    video_writer = None
    obs_writer = None
    if headless:
        fourcc = cv.VideoWriter_fourcc(*'avc1')
        video_writer = cv.VideoWriter('circle_debug_video.mp4', fourcc, 10.0, (1920, 1080))
        obs_writer = cv.VideoWriter('circle_debug_obs.mp4', fourcc, 10.0, (64, 64))
    
    t = 0
    max_steps = 500 if headless else float('inf')
    
    while t < max_steps:
        # Random actions for demonstration
        a = th.rand((num_agent, 4)) * 2 - 1  # Random actions in [-1, 1]
        env.step(a)
        
        # Visualization points
        spawn_center = th.tensor([random_kwargs["state_generator"]["kwargs"][0]["position"]["mean"]])
        target_pos = env.target_position.detach().unsqueeze(0).cpu()  # Circle center from env
        
        # Create circle visualization
        ring_points = []
        ring_radius = env.desired_radius
        for angle in np.linspace(0, 2*np.pi, 32):
            x = target_pos[0, 0] + ring_radius * np.cos(angle)
            y = target_pos[0, 1] + ring_radius * np.sin(angle)
            z = target_pos[0, 2]
            ring_points.append([x, y, z])
        ring_curve = th.tensor(ring_points).unsqueeze(0)
        
        debug_points = th.cat([spawn_center, target_pos], dim=0)
        
        img = env.render(is_draw_axes=True, points=debug_points, curves=ring_curve)
        obs = env.sensor_obs["depth"]
        
        if headless:
            # Save to video files
            if video_writer:
                video_writer.write(cv.cvtColor(img[0], cv.COLOR_RGB2BGR))
            if obs_writer:
                obs_data = obs[0][0]  # Get first agent's observation
                
                if t == 0:
                    print(f"Circle obs data shape: {obs_data.shape}, dtype: {obs_data.dtype}")
                    print(f"Circle obs data min/max: {obs_data.min():.3f}/{obs_data.max():.3f}")
                
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
                        print(f"Error processing circle obs frame: {e}")
                    black_frame = np.zeros((64, 64, 3), dtype=np.uint8)
                    obs_writer.write(black_frame)
        else:
            # Display in windows
            cv.imshow("circle_img", img[0])
            cv.imshow("circle_obs", obs[0][0])
            cv.waitKey(100)
        
        # Print circle progress occasionally
        if t % 50 == 0:
            obs_dict = env.get_observation()
            command = obs_dict.get("command", th.tensor([[0.0, 0.0]]))[0]
            pos = env.position[0]
            dist_to_center = th.norm(pos[:2] - env.target_position[:2])
            print(f"Step {t}: Dist to center: {dist_to_center:.3f}m, Target radius: {env.desired_radius:.1f}m, Command: [{command[0]:.1f}, {command[1]:.3f}]")
        
        t += 1
    
    # Cleanup
    if headless and video_writer:
        video_writer.release()
        obs_writer.release()
        print("Circle videos saved: circle_debug_video.mp4, circle_debug_obs.mp4")
    if not headless:
        cv.destroyAllWindows()
