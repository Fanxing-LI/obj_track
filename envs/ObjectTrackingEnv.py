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

cAd = lambda x: x.clone().detach()

class ObjectTrackingEnv(DroneGymEnvsBase):
    semantic_alias = {
        "stone": 5,
        "cup":8,
        "human":7,
        "ball": 2
    }
    def __init__(
            self,
            num_agent_per_scene: int = 1,
            num_scene: int = 1,
            seed: int = 42,
            visual: bool = False,
            requires_grad: bool = False,
            random_kwargs: dict = None,
            dynamics_kwargs: dict = {},
            scene_kwargs: dict = {},
            sensor_kwargs: list = [],
            device: str = "cpu",
            target: Optional[th.Tensor] = None,
            max_episode_steps: int = 256,
            tensor_output: bool = False,
            keep_dis=3.0,
            box_noise=1.0,
            semantic_id =2,
            is_collision_reset=False,
    ):
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
            is_collision_reset=is_collision_reset

        )
        self.target = th.zeros((self.num_envs, 3), dtype=th.float32, device=self.device)
        self.center = th.as_tensor([5, 0, 1.])
        self.radius_spd = 0.2 * th.pi / 1
        self.height = 0.3
        self.radius = 2
        self.box_center = th.ones((self.num_envs, 3), dtype=th.float32, device=self.device) * 0.5
        # self.update_target()
        self.observation_space["state"] = spaces.Box(
            shape=(16,), low=-th.inf, high=th.inf, dtype=np.float32)
        # self.observation_space["state"] = spaces.Box(
        #     shape=(19,), low=-th.inf, high=th.inf, dtype=np.float32)
        test = 1
        # self.update_target()
        self.keep_dis = keep_dis
        self.pre_box_center = th.zeros((self.num_envs, 3), dtype=th.float32, device=self.device)
        self.pre_rebuild_local_targets = th.zeros((self.num_envs, 3), dtype=th.float32, device=self.device)
        # self.pre_box_v = th.zeros((self.num_envs, 3), dtype=th.float32, device=self.device)
        self.pre_dis = th.zeros((self.num_envs,), dtype=th.float32, device=self.device)
        self.smooth_factor = 0.5
        self.box_noise = box_noise
        self.FOV = th.pi/2
        self.semantic_id = semantic_id if isinstance(semantic_id, int) else self.semantic_alias.get(semantic_id, None)

    def reset(self, *args, **kwargs) -> Union[TensorDict, Tuple[TensorDict, Dict]]:
        res = super().reset( *args, **kwargs)

        self.update_target()
        return res

    def _reset_attr(self, indices=None, reset_latent=False):
        super()._reset_attr(indices, reset_latent=reset_latent)

    def update_target(self):
        # update target position and velocity
        if self.envs.visual:
            self.target = th.stack([p[0] for p in self.envs.dynamic_object_position])
            self.target_v = th.stack([v[0] for v in self.envs.dynamic_object_velocity])
        # update target position and velocity in local frame
        rela_tar = self.target - self.position
        rela_v = self.target_v - self.velocity
        orientation = self.envs.dynamics._orientation.clone()
        # self.local_targets = orientation.inv_rotate(rela_tar.T).T
        # add_local_target_v = th.cross(self.angular_velocity-0, self.local_targets-0, dim=1)
        # self.local_targets_v = orientation.inv_rotate(rela_v.T).T - add_local_target_v
        # self.rebuild_local_targets_v = self.rebuild_local_targets_v
        # self.local_v = orientation.inv_rotate(self.velocity.T-0).T-0
        self.head_targets = orientation.world_to_head(rela_tar.T).T
        self.head_targets_v = orientation.world_to_head((rela_v.T-0)).T
        # cali_head_vel = th.cross(self.angular_velocity * th.tensor([[0, 0, 1]]), self.head_targets)
        # self.head_targets_v = self.head_targets_v #- cali_head_vel
        self.head_v = orientation.world_to_head((self.velocity.T-0)).T

    def get_observation(
            self,
            indices=None,
            predicted_obs: Optional[TensorDict] = None,
    ) -> Dict:
        if predicted_obs:
            state = predicted_obs.get("state").to(self.device)
            # self.target = state[:, 0:3] + cAd(self.position)
            # self.target_v = state[:, 3:6] + cAd(self.velocity)
            self.target = state[:, 0:3] + self.position
            self.target_v = state[:, 3:6] + self.velocity
            
        self.update_target()

        state = th.hstack([
            self.head_targets,
            self.head_targets_v,
            self.orientation,
            self.head_v / 10,
            self.angular_velocity / 10,
        ]).to(self.device)
        # return TensorDict({
        #     "state": state,
        # })

        obs = TensorDict({
            "state": state,
        })

        if "color" in self.sensor_obs:
            obs["color"] = th.as_tensor(self.sensor_obs["color"].astype(np.float32))

        return obs
        
    def get_success(self) -> th.Tensor:
        return th.full((self.num_agent,), False)

    def get_reward(self, *args, **kwargs) -> th.Tensor:
        base_r = 0.1 * th.ones((self.num_envs,), dtype=th.float32)
        target_vector = self.target - self.position
        normal_target_vector = target_vector / target_vector.norm(dim=1, keepdim=True) - 0
        proj = ((self.direction.clone() - 0) * normal_target_vector - 0).sum(dim=1)
        aware_r = proj * 0.04
        # aware_r = proj * 0.05
        align_v = self.target_v if hasattr(self, "target_v") else self.velocity
        target_dis = self.keep_dis * (align_v.norm(dim=1)/4).clamp_min(1.0).detach()
        keep_pos_r = ((self.position - self.target).norm(dim=1) - target_dis).abs() * -0.025
        vel_r = (self.velocity - 0).norm(dim=1) * -0.001

        ang_vel_r = (self.angular_velocity - 0).norm(dim=1) * -0.01
        acc_r = (self.envs.acceleration - 0).norm(dim=1) * -0.001
        ang_acc_r = (self.envs.angular_acceleration - 0).norm(dim=1) * -0.002
        act_r = (self._action[:,1:].norm(dim=1).to(keep_pos_r.device) * -0.003
                 + self._action[:,2:3].norm(dim=1).to(keep_pos_r.device) * -0.025)

        act_change_r = (self.envs.dynamics._pre_action[-1].to(self.device)-
                        self.envs.dynamics._pre_action[-2].to(self.device)
                        ).T.norm(dim=-1) * -0.002

        # projection v on backward direction
        unit_v = (self.velocity-0) / ((self.velocity-0).norm(dim=1, keepdim=True)+1e-8)
        inverse_v_proj = (unit_v * (self.direction-0)).sum(dim=1)
        percep_r = inverse_v_proj * 0.02

        disc_r = base_r

        diff_r = (
                
                ang_vel_r + aware_r + keep_pos_r + acc_r+ act_r + act_change_r + vel_r
                + percep_r
                    # + col_dis_r + col_vel_r
        )

        reward = diff_r + disc_r

        return {"reward":reward,
                "keep_pos_r":keep_pos_r.clone().detach(),
                "aware_r":aware_r.clone().detach(),
                "ang_vel_r":ang_vel_r.clone().detach(),
                "ang_acc_r":ang_acc_r.clone().detach(),
                "percp_r":percep_r.clone().detach(),
                # "col_vel_r":col_vel_r.clone().detach(),
                # "col_dis_r":col_dis_r.clone().detach(),
                }
