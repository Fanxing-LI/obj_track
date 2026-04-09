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


class RacingEnv(DroneGymEnvsBase):
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
            pass_radius: float = 0.5,
            add_vel_obs=False,
            test=False,
            *args, **kwargs
    ):

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
            *args, **kwargs
        )

        self.targets = th.as_tensor([
            [12, 4, 2.],
            [16, 0, 2.],
            [12, -4, 2.5],
            [8, -0, 2.],
        ])
        self.targets_center = self.targets.clone()
        self.targets_radius = 2
        self.targets_v_norm = 1.5 + 0.3*2*(th.rand((len(self.targets),), dtype=th.float32)-0.5)
        # self.targets_v_norm = 1.5 * th.ones((len(self.targets),), dtype=th.float32)
        self.targets_omega = self.targets_v_norm / self.targets_radius
        self.targes_initial_phase = th.rand((len(self.targets),), dtype=th.float32) * 2 * th.pi
        
        # self.targets_ori = th.as_tensor([])
        self._next_target_num = 2
        self._next_target_i = th.zeros((self.num_envs,), dtype=th.int)
        self._past_targets_num = th.zeros((self.num_envs,), dtype=th.int)
        self._is_pass_next = th.zeros((self.num_envs,), dtype=th.bool)
        self.pass_radius = pass_radius
        # if test:
        #     self.targes_initial_phase = th.as_tensor([0,0.25,0.5,0.75], dtype=th.float32) * 2 * th.pi
        #     self.targets_v_norm = 1.5 * th.ones((len(self.targets),), dtype=th.float32) 
        # self.observation_space["gate"] = spaces.Box(
        #     low=0,
        #     high=len(self.targets),
        #     shape=(1,),
        #     dtype=np.int32
        # )
        # state observation includes gates
        self.observation_space["state"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3 * self._next_target_num * (1 if not add_vel_obs else 2) + self.observation_space["state"].shape[0]-3,),
            dtype=np.float32
        )
        self.is_add_vel_obs = add_vel_obs
        
        self._pre_dis = th.zeros((self.num_envs,), dtype=th.float32)

    def collect_info(self, indice, observations):
        _info = super().collect_info(indice, observations)
        _info['episode']["extra"]["past_gate"] = self._past_targets_num[indice].item()
        
        return _info

    @property
    def is_pass_next(self):
        return self._is_pass_next

    def update_target_obs(self):
        self.targets = self.targets_center + th.stack([
            th.cos(self.t[0]* self.targets_omega+self.targes_initial_phase) * self.targets_radius,
            th.sin(self.t[0]* self.targets_omega+self.targes_initial_phase) * self.targets_radius,
            th.zeros_like(self.targets_omega)
        ], dim=1)
        self.targets_v = th.stack(
            [
            -th.sin(self.t[0]* self.targets_omega+self.targes_initial_phase) * self.targets_radius,
            th.cos(self.t[0]* self.targets_omega+self.targes_initial_phase) * self.targets_radius,
            th.zeros_like(self.targets_omega)
        ], dim=1)
        
        next_target_index = (th.arange(self._next_target_num).unsqueeze(0) 
        + self._next_target_i.unsqueeze(1)) % len(self.targets)
        targets = self.targets[next_target_index].to(self.device)
        orientation = self.envs.dynamics._orientation.clone()
        self.rela_targets = targets - self.position.unsqueeze(1)
        self.rela_targets_v = self.targets_v[next_target_index].to(self.device) - self.velocity.unsqueeze(1)

        head_targets = th.stack([orientation.world_to_head(self.rela_targets[:,i,:].T) for i in range(self._next_target_num)], dim=1).permute(2,1,0).reshape(self.num_envs, 3*self._next_target_num)
        self.head_targets = head_targets
        
        self.head_v = orientation.world_to_head(self.velocity.T).to(self.device).T
        
    def get_observation(
            self,
            indices=None,
            predicted_obs=None
    ) -> Dict:
        self.update_target_obs()

        if not self.is_add_vel_obs:
            state = th.hstack([
                self.head_targets / 10,
                # self.rela_targets.reshape(self.num_agent, -1) / 10,
                self.orientation,
                self.head_v / 10,
                # self.velocity / 10,
                self.angular_velocity/ 10,
            ]).to(self.device)
        else:
            state = th.hstack([
                # self.head_targets / 10,
                self.rela_targets.reshape(self.num_agent, -1) / 10,
                self.rela_targets_v.reshape(self.num_agent, -1) / 10,
                self.orientation,
                # self.head_v / 10,
                self.velocity / 10,
                self.angular_velocity/ 10,
            ]).to(self.device)
        

        obs = TensorDict({
            "state": state,
        })

        return obs

    def get_success(self) -> th.Tensor:
        self._is_pass_next = ((self.position - self.targets[self._next_target_i]).norm(dim=1) <= self.pass_radius)
        self._next_target_i = self._next_target_i + self._is_pass_next
        self._next_target_i = self._next_target_i % len(self.targets)
        self._past_targets_num = self._past_targets_num + self._is_pass_next
        if self._is_pass_next.any():
            i = self._is_pass_next.nonzero()[0]
            self._pre_dis[i] = (self.position - self.targets[self._next_target_i]).norm(dim=1)[i]
            
        return th.zeros((self.num_envs,), dtype=th.bool)

    def _reset_attr(self, indices=None, reset_latent=False):
        indices = th.arange(self.num_envs) if indices is None else indices
        self._next_target_i[indices] = th.randint(
            0, 
            len(self.targets),
            (len(indices),), 
            dtype=th.int
            )

        self._past_targets_num[indices] = th.zeros((len(indices),), dtype=th.int)
        self._is_pass_next[indices] = th.zeros((len(indices),), dtype=th.bool)
        self._pre_dis[indices] = (self.position - self.targets[self._next_target_i]).norm(dim=1)[indices]
        obs = super()._reset_attr(indices, reset_latent=reset_latent)
        
        return obs

    def detach(self, *args, **kwargs):
        self._pre_dis = self._pre_dis.detach()
        super().detach(*args, **kwargs)
        
    def reset(self, state=None, obs=None,is_test=False):
        obs = super().reset(state)
        self._next_target_i = th.zeros((self.num_envs,), dtype=th.int)
        self._pre_dis = (self.position - self.targets[self._next_target_i]).norm(dim=1)
        # self._past_targets_num = th.zeros((self.num_envs,), dtype=th.int)
        # self._choose_target()
        return obs

    def get_reward(self,predicted_obs=None) -> th.Tensor:
        # pos_factor = -0.1 * 1 / 9
        self.success_r = 40
        
        # _next_target_i_clamp = self._next_target_i
        # dis = (self.position - self.targets[_next_target_i_clamp]).norm(dim=1)
        # prog_r = self._pre_dis - dis
        # reward = (
        #         0.02 +
        #         prog_r * 1 +
        #         (self.angular_velocity - 0).norm(dim=1) * -0.01 +
        #         self._is_pass_next * self.success_r + 
        #         self.is_collision * -self.success_r
        # )
        # self._pre_dis = dis.clone()
        # return reward
        
        _next_target_i_clamp = self._next_target_i
        dis_vector = (self.targets[_next_target_i_clamp] - self.position)
        dis = (dis_vector-0).norm(dim=1, keepdim=True)
        dis_vector_unit = dis_vector / (dis+1e-6)
        # v_norm = (self.velocity-0).norm(dim=1)
        approaching_v = ((self.velocity-0) * dis_vector_unit).sum(dim=1, keepdim=True).clamp_max(3.)
        approaching_v_vector = dis_vector_unit * approaching_v
        away_v_vector = self.velocity - approaching_v_vector
        away_v = (away_v_vector-0).norm(dim=1) 
        diff_r = (approaching_v.squeeze() * 0.02
                  - away_v * (1/ (6 *dis.squeeze().detach() + 1))* 0.1
                  + (self.angular_velocity - 0).norm(dim=1) * -0.001
                #   + self.is_pass_next * self.success_r
        )
        disc_r = self._is_pass_next * self.success_r + self.is_collision * -10
        reward = diff_r + disc_r
        return {
            "reward": reward,
            "diff_r": diff_r,
            "disc_r": disc_r,
        }