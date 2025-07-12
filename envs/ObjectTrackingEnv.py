import os
import sys

import numpy as np
from VisFly.envs.base.droneGymEnv import DroneGymEnvsBase
from typing import Union, Tuple, List, Optional, Dict
import torch as th
from habitat_sim import SensorType
from gymnasium import spaces
from VisFly.utils.tools.train_encoder import model as encoder
from VisFly.utils.type import TensorDict


class ObjectTrackingEnv(DroneGymEnvsBase):
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
    ):
        # random_kwargs = {
        #     "state_generator":
        #         {
        #             # "class": "Uniform",
        #             "class": "TargetUniform",
        #             "kwargs": [
        #                 {"position": {"mean": [10., 0., 1.5], "half": [2.0, 2.0, 0.2]}},
        #             ]
        #         }
        # }

        assert "obj_settings" in scene_kwargs, "scene_kwargs must contain 'obj_settings' for ObjectTrackingEnv"

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
        self.target = th.zeros((self.num_envs, 3), dtype=th.float32, device=self.device)
        self.center = th.as_tensor([5, 0, 1.])
        self.radius_spd = 0.2 * th.pi / 1
        self.height = 0.3
        self.radius = 2
        # self.update_target()

    def update_target(self):
        self.target = self.center
        self.target = th.stack([self.radius * th.cos(self.radius_spd * self.t) + self.center[0],
                                 self.radius * th.sin(self.radius_spd * self.t) + self.center[1],
                                 self.height * th.sin(self.radius_spd * self.t) + self.center[2]
                                 ]).T
        self.target = th.stack([p[0] for p in self.envs.dynamic_object_position])

    def get_observation(
            self,
            indices=None
    ) -> Dict:
        self.update_target()

        rela_tar = self.target - self.position
        orientation = self.envs.dynamics._orientation.clone()
        local_targets = orientation.inv_rotate(rela_tar.T).T
        state = th.hstack([
            local_targets / self.max_sense_radius,
            self.orientation,
            self.velocity / 10,
            self.angular_velocity / 10,
        ]).to(self.device)
        return TensorDict({
            "state": state,
        })

        obs = TensorDict({
            "state": state,
            "depth": th.as_tensor(self.sensor_obs["depth"]).clamp_min(0.2),
        })

        return obs

    def get_success(self) -> th.Tensor:
        return th.full((self.num_agent,), False)

    def get_reward(self) -> th.Tensor:
        base_r = 0.1 * th.ones((self.num_envs,), dtype=th.float32)
        target_vector = self.target - self.position
        normal_target_vector = target_vector / target_vector.norm(dim=1, keepdim=True) - 0
        proj = ((self.direction.clone() - 0) * normal_target_vector - 0).sum(dim=1)
        aware_r = proj * 0.05
        pos_factor = -0.1 * 1 / 9
        pos_r = (self.position - self.target).norm(dim=1) * pos_factor
        keep_pos_r = ((self.position - self.target).norm(dim=1) - 1.0).abs() * pos_factor
        vel_r = (self.velocity - 0).norm(dim=1) * -0.002
        ang_vel_r = (self.angular_velocity - 0).norm(dim=1) * -0.002
        acc_r = (self.envs.acceleration - 0).norm(dim=1) * -0.001
        ang_acc_r = (self.envs.angular_acceleration - 0).norm(dim=1) * -0.001

        diff_r = vel_r + ang_vel_r + aware_r + keep_pos_r  # + acc_r + ang_acc_r
        disc_r = th.zeros_like(diff_r)

        reward = diff_r + disc_r + base_r
        return {"reward": reward, "diff_r": diff_r, "disc_r": th.as_tensor(disc_r)}