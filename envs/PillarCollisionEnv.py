import numpy as np
from habitat_sim.sensor import SensorType
import os
import sys

sys.path.append(os.getcwd())
from VisFly.envs.base.droneGymEnv import DroneGymEnvsBase
from typing import Optional, Dict
import torch as th

from VisFly.utils.type import TensorDict

class PillarCollisionEnv(DroneGymEnvsBase):
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
    ):
        # Ensure depth sensor is always available for NavigationEnv
        if not sensor_kwargs:
            sensor_kwargs = [
                {
                    "sensor_type": SensorType.DEPTH,
                    "uuid": "depth",
                    "resolution": [64, 64],
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
            sensor_kwargs=sensor_kwargs,
            device=device,
            max_episode_steps=max_episode_steps,
            tensor_output=tensor_output,
        )

        self.target = th.ones((self.num_envs, 1)) @ th.as_tensor([15, 0., 1.5] if target is None else target).reshape(1,
                                                                                                                      -1)
        self.success_radius = 0.1

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
            "depth": th.as_tensor(self.sensor_obs["depth"] / 10).clamp(max=1)
        })

    def get_success(self) -> th.Tensor:
        # Use parentheses to ensure correct operator precedence and use logical AND for boolean tensors
        success_pos = (self.position - self.target).norm(dim=1) <= self.success_radius
        success_vel = self.velocity.norm(dim=1) <= 0.1
        return success_pos & success_vel
        # return th.full((self.num_agent,), False)

    # def get_reward(self) -> th.Tensor:
    #     # Position reward with scaling
    #     pos_factor = -0.065
    #     threshold = 0.4
    #     r_pos = (self.position - self.target).norm(dim=1)
    #     pos_scale = r_pos.clamp_min(threshold).detach()
    #     r_pos = r_pos / pos_scale

    #     # Direct collision distance penalty without velocity decomposition
    #     collision_dis = self.collision_vector.norm(dim=1)
    #     collision_direction = self.collision_vector / (collision_dis.unsqueeze(1) + 1e-8)

    #     approaching_velocity = ((self.velocity - 0) * collision_direction).sum(dim=1)
    #     velocity_distance_factor = (2.0 - collision_dis).clamp(min=0.0, max=2.0)

    #     # Apply horizontal weight to reduce ground/ceiling influence
    #     obstacle_spd_r = approaching_velocity.clamp(min=0.0) * velocity_distance_factor * -0.0002

    #     # Linear distance penalty - stronger for harder env
    #     distance_factor = (2 - collision_dis).clamp(min=0.0, max=2.0)
    #     obstacle_dis_r = distance_factor * -0.05

    #     reward = (
    #             r_pos * pos_factor +
    #             (self.velocity - 0).norm(dim=1) * -0.002 +
    #             (self.angular_velocity - 0).norm(dim=1) * -0.01 +
    #             obstacle_spd_r +
    #             obstacle_dis_r
    #     )
    #     if self.tensor_output:
    #         return {
    #             "reward": reward,
    #             "position_r": (r_pos * pos_factor).detach(),
    #             "obstacle_spd_r": obstacle_spd_r.detach(),
    #             "obstacle_dis_r": obstacle_dis_r.detach(),
    #         }
    #     return reward
    
    def get_reward(self):
        base_r = 0.1
        dist = (self.position - self.target).norm(dim=1)
        collision_dis = self.collision_vector.norm(dim=1)
        distance_factor = (2 - collision_dis).clamp(min=0.0, max=2.0)
        r_col_dis = distance_factor * -0.05
        # Smooth distance reward with exponential
        r_pos = dist * -0.1 * 1/9
        
        # # Smooth obstacle field
        # obstacle_field = th.sigmoid((2.0 - collision_dis) * 2)
        # r_obstacle = -obstacle_field * 0.1
        r_vel = (self.velocity - 0).norm(dim=1) * -0.005
        r_ang_vel = (self.angular_velocity - 0).norm(dim=1) * -0.01
        return base_r + r_pos + r_vel + r_ang_vel + r_col_dis

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
                    {"position": {"mean": [15.0, -12.0, 1.0], "half": [ 1,1,0.2]},
                     "orientation": {"mean": [0.0,0.0,1.61], "half": [0,0,0]},}
                ]
            }
    }
    target = [15.0, 10.0, 1]
    scene_path = "VisFly/datasets/visfly-beta/configs/scenes/box15_wall_simple_pillars"
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
            # "position": th.tensor([[6., 6.8, 5.5], [6,4.8,4.5]]),
            "position": th.tensor([[9., 0.1, 30.0], [9., 0, 0]]),
            "line_width": 6.,

            "point": th.tensor([[9., 0, 0], [9, 0, 0]]),
            "trajectory": True,
        }
    }
    num_agent = 4
    env = PillarCollisionEnv(
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
    max_steps = 500 if headless else float('inf')  # Limit steps in headless mode

    while t < max_steps:
        a = th.rand((num_agent, 4))
        env.step(a)
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