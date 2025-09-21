import numpy as np
from habitat_sim.sensor import SensorType
import os
import sys
sys.path.append(os.getcwd())
from VisFly.envs.base.droneGymEnv import DroneGymEnvsBase
from typing import Optional, Dict
import torch as th
 

from VisFly.utils.type import TensorDict


class HardNavigationEnv(DroneGymEnvsBase):
    def __init__(
        self,
        num_agent_per_scene: int = 1,
        num_scene: int = 1,
        seed: int = 42,
        visual: bool = True,
        requires_grad: bool = False,
        random_kwargs: dict = {},
        dynamics_kwargs: dict = {},
        scene_kwargs: dict = {},
        sensor_kwargs: list = [],
        device: str = "cpu",
        target: Optional[th.Tensor] = None,
        max_episode_steps: int = 256,
        tensor_output: bool = False,
        collision_config: Optional[Dict] = None,
    ):
        # Default to a 16x16 depth sensor (mirroring EasyNavigationEnv)
        sensor_kwargs = [
            {
                "sensor_type": SensorType.DEPTH,
                "uuid": "depth",
                "resolution": [16, 16],
            }
        ]

        super().__init__(
            num_agent_per_scene=num_agent_per_scene,
            num_scene=num_scene,
            seed=seed,
            visual=visual,
            requires_grad=requires_grad,
            random_kwargs=random_kwargs,
            dynamics_kwargs=dynamics_kwargs,
            scene_kwargs=scene_kwargs,
            sensor_kwargs=[sensor_kwargs] if isinstance(sensor_kwargs, dict) else sensor_kwargs,
            device=device,
            max_episode_steps=max_episode_steps,
            tensor_output=tensor_output,
            collision_config=collision_config,
        )

        self.target = th.ones((self.num_envs, 1)) @ th.as_tensor([15, 0., 1.5] if target is None else target).reshape(1,-1)

    def get_observation(
            self,
            indices=None
    ) -> Dict:
        orientation = self.envs.dynamics._orientation.clone()
        rela = self.target - self.position
        head_target = orientation.world_to_head(rela.T).T
        head_velocity = orientation.world_to_head((self.velocity - 0).T).T
        state = th.hstack([
            head_target / 10,
            self.orientation,
            head_velocity / 10,
            self.angular_velocity / 10,
        ]).to(self.device)
        return TensorDict({
            "state": state,
            "depth": th.as_tensor(self.sensor_obs["depth"]/10).clamp(max=1)
        })

    def get_success(self) -> th.Tensor:
        # success_pos = (self.position - self.target).norm(dim=1) <= self.success_radius
        # success_vel = self.velocity.norm(dim=1) < 0.02
        # return success_pos & success_vel
        return th.full((self.num_agent,), False)

    def get_reward(self) -> th.Tensor:
        # Compute per-agent speed with an extra dimension for safe broadcasting
        speed = (self.velocity - 0).norm(dim=1, keepdim=True)
        unit_velocity = (self.velocity - 0) / (speed + 1e-8)
        align = (self.direction * unit_velocity).sum(dim=1)
        min_vel = 0.2
        max_vel = 0.3
        # Remove the extra dim for subsequent scalar operations
        scale = (speed.squeeze(1) - min_vel) / (max_vel - min_vel)
        scale = scale.clamp(min=0.0, max=1.0)
        # If vel <= min_vel, no aware reward, scale = 0; If vel >= max_vel, scale = 1; In between, linear
        r_percep_aware = (align - 1) * scale * 0.02

        # Restore original shaping coefficients
        pos_factor = -0.05
        threshold = 0.4
        r_pos = (self.position - self.target).norm(dim=1)
        pos_scale = r_pos.clamp_min(threshold).detach()
        r_pos = r_pos / pos_scale * pos_factor

        collision_dis = self.collision_vector.norm(dim=1)
        r_col_dis = (2 - collision_dis).clamp(min=0.0, max=2.0) * -0.6
        collision_direction = self.collision_vector / (collision_dis.unsqueeze(1) + 1e-8)
        r_col_vel = (th.relu((self.velocity - 0) * collision_direction).sum(dim=1) * th.relu(2 - collision_dis)).detach() * -0.4
        # r_col_vel = (((self.velocity - 0) * collision_direction.detach()).sum(dim=1) * th.relu(1 - collision_dis)) * -0.4

        r_vel = (self.velocity - 0).norm(dim=1) * -0.002
        r_ang_vel = (self.angular_velocity - 0).norm(dim=1) * -0.01
        reward = (r_pos  + r_vel + r_ang_vel + r_col_dis + r_col_vel + r_percep_aware)

        if self.tensor_output:
            return {"reward": reward,
                    "r_pos": r_pos.detach(),
                    "r_vel": r_vel.detach(),
                    "r_ang_vel": r_ang_vel.detach(),
                    "r_percep_aware": r_percep_aware.detach(),
                    "r_col_vel": r_col_vel.detach(),
                    "r_col_dis": r_col_dis.detach(),
                    }
        return reward
    

    def get_failure(self) -> th.Tensor:
        return self.is_collision


if __name__ == "__main__":
    import cv2 as cv
    
    import matplotlib

    matplotlib.use('Agg')  # Use non-interactive backend for headless

    # Detect headless environment
    import os

    headless = os.environ.get('DISPLAY') is None or os.environ.get('SSH_CONNECTION') is not None

    random_kwargs = {
        "state_generator":
            {
                "class": "Uniform",
                "kwargs": [
                    {"position": {"mean": [15.0, -5.0, 1], "half": [0.0, 0.0, 0]},
                    "orientation": {"mean": [0, 0, 1.68], "half": [0,0,0]},}
                ]
            }
    }
    target = [15.0, 5.0, 1]
    scene_path = "VisFly/datasets/visfly-beta/configs/scenes/box15_wall_center_tri_pillars"
    sensor_kwargs = [{
                "sensor_type": SensorType.DEPTH,
                "uuid": "depth",
                "resolution": [16, 16],
                "position": [0,0.2,0.],
            }]
    scene_kwargs = {
        "path": scene_path,
        "render_settings": {
            "mode": "fix",
            "view": "custom",
            "resolution": [1080, 1920],
            "position": th.tensor([[15., 0.1, 30.0], [15., 0, 0]]),
            "line_width": 6.,
            "trajectory": True,
        }
    }
    num_agent = 1
    env = HardNavigationEnv(
        visual=True,
        num_scene=1,
        num_agent_per_scene=num_agent,
        random_kwargs=random_kwargs,
        scene_kwargs=scene_kwargs,
        sensor_kwargs=sensor_kwargs,
        dynamics_kwargs={}
    )

    env.reset()

    # Video writer setup for headless mode
    video_writer = None
    obs_writer = None
    if headless:
        fourcc = cv.VideoWriter_fourcc(*'avc1')  # Use avc1 for VSCode/browser compatibility
        video_writer = cv.VideoWriter('debug_video.mp4', fourcc, 10.0, (1920, 1080))
        obs_writer = cv.VideoWriter('debug_obs.mp4', fourcc, 10.0, (128, 128))

    t = 0
    max_steps = 150 if headless else float('inf')  # Limit steps in headless mode

    while t < max_steps:
        a = th.rand((num_agent, 4))
        env.step(a)
        # circile position
        # position = th.tensor([[3., 0, 1]]) + th.tensor([[np.cos(t/10), np.sin(t/10), 0]]) * 2
        # rotation = Quaternion.from_euler(th.tensor(t/10.), th.tensor(t/10.), th.tensor(t/10)).toTensor().unsqueeze(0)
        # env.envs.sceneManager.set_pose(position=position, rotation=rotation)
        # env.envs.update_observation()
        # Create debug points and ring for visualization
        # Sphere for random spawn area (position mean)
        spawn_center = th.tensor([random_kwargs["state_generator"]["kwargs"][0]["position"]["mean"]])
        target_pos = th.tensor([target])

        # Create a ring around target (circle points)
        import numpy as np

        ring_points = []
        ring_radius = 1.0
        for angle in np.linspace(0, 2 * np.pi, 20):
            x = target_pos[0, 0] + ring_radius * np.cos(angle)
            y = target_pos[0, 1] + ring_radius * np.sin(angle)
            z = target_pos[0, 2]
            ring_points.append([x, y, z])
        ring_curve = th.tensor(ring_points).unsqueeze(0)

        debug_points = th.cat([spawn_center, target_pos], dim=0)

        img = env.render(is_draw_axes=True, points=debug_points, curves=ring_curve)
        # print(env.position[0])
        obs = env.sensor_obs["depth"]

        if headless:
            # Save to video files
            if video_writer:
                video_writer.write(cv.cvtColor(img[0], cv.COLOR_RGB2BGR))
            if obs_writer:
                # Handle COLOR sensor observation (RGBA format)
                obs_data = obs[0][0]  # Get first agent's observation
                
                # Debug print to understand the data shape
                if t == 0:
                    print(f"Obs data shape: {obs_data.shape}, dtype: {obs_data.dtype}")
                    print(f"Obs data min/max: {obs_data.min():.3f}/{obs_data.max():.3f}")
                
                try:
                    if len(obs_data.shape) == 3 and obs_data.shape[0] >= 3:  # Multi-channel (RGB/RGBA)
                        # Convert to RGB, then to BGR for OpenCV
                        obs_rgb = obs_data[:3]  # Take first 3 channels (RGB)
                        obs_frame = (obs_rgb * 255).clip(0, 255).astype(np.uint8)
                        obs_frame = np.transpose(obs_frame, (1, 2, 0))  # CHW to HWC
                        obs_frame = cv.cvtColor(obs_frame, cv.COLOR_RGB2BGR)
                    else:  # Single channel or other format
                        if len(obs_data.shape) == 2:  # Already HW format
                            obs_frame = (obs_data * 255).clip(0, 255).astype(np.uint8)
                            obs_frame = cv.cvtColor(obs_frame, cv.COLOR_GRAY2BGR)
                        else:  # CHW format, take first channel
                            obs_frame = (obs_data[0] * 255).clip(0, 255).astype(np.uint8)
                            obs_frame = cv.cvtColor(obs_frame, cv.COLOR_GRAY2BGR)
                    
                    # Ensure correct frame size
                    if obs_frame.shape[:2] != (128, 128):
                        obs_frame = cv.resize(obs_frame, (128, 128))
                    
                    obs_writer.write(obs_frame)
                except Exception as e:
                    if t == 0:
                        print(f"Error processing obs frame: {e}")
                        print(f"Obs data shape: {obs_data.shape}")
                    # Write a black frame as fallback
                    black_frame = np.zeros((128, 128, 3), dtype=np.uint8)
                    obs_writer.write(black_frame)
        else:
            # Display in windows
            cv.imshow("img", img[0])
            # cv.imshow("obs", np.transpose(obs[0], (1, 2, 0)))
            cv.imshow("obs", obs[0][0])
            cv.waitKey(100)

        t += 1

    # Cleanup
    if headless and video_writer:
        video_writer.release()
        obs_writer.release()
        print("Videos saved: debug_video.mp4, debug_obs.mp4")
    if not headless:
        cv.destroyAllWindows()
