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


class FlipEnv(DroneGymEnvsBase):
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
        random_kwargs = {
            "state_generator":
                {
                    "class": "Uniform",
                    "kwargs": [
                        # {"position": {"mean": [1., 0., 1.5], "half": [0.0, 0.0, 0.0]}},
                        {"position": {"mean": [1., 0., 1.5], "half": [1.0, 1.0, 0.5]}},
                    ]
                }
        }

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

    def get_observation(
            self,
            indices=None
    ) -> Dict:
        obs = TensorDict({
            "state": self.state,
        })

        return obs

    def get_success(self) -> th.Tensor:
        return th.full((self.num_agent,), False)

    def get_reward(self) -> th.Tensor:
        base_r = 0.1
        pos_factor = -0.1 * 1 / 9
        xz_axis = self.envs.dynamics.xz_axis
        x_axis = xz_axis[0, :, :].T
        z_axis = xz_axis[1, :, :].T
        omega = th.pi * 2 * 1.5
        target_z_axis = th.stack(
            [th.zeros((self.num_envs,)),
             th.sin(omega*self.t),
             th.cos(omega*self.t)
             ]).T

        x_axis_r = (x_axis - th.tensor([1., 0, 0], )).norm(dim=1)
        z_axis_r = (z_axis - target_z_axis).norm(dim=1)

        reward = (
                base_r +
                x_axis_r * -0.1 +
                z_axis_r * -0.1 +
                (self.velocity - 0).norm(dim=1) * -0.002
        )

        return reward
