import torch as th
import numpy as np

def get_reward_debug():
    num_envs = 3
    position = th.tensor([[0., 0., 0.], [10., 0., 0.], [5., 5., 5.]])
    targets = th.tensor([
            [8, 4, 1.],
            [12, 0, 2.],
            [9, -4, 1.],
            [4, -1, 1.],
        ])
    _next_target_i = th.tensor([0, 1, 2])
    velocity = th.tensor([[1., 0., 0.], [0., 1., 0.], [-1., 0., 0.]])
    angular_velocity = th.zeros((3, 3))
    is_pass_next = th.tensor([False, False, True])
    success_r = 10.0

    # Original logic
    _next_target_i_clamp = _next_target_i
    target_pos = targets[_next_target_i_clamp]
    dis_vector = (target_pos - position)
    dis = (dis_vector-0).norm(dim=1, keepdim=True)
    dis_vector_norm = dis_vector / (dis+1e-6)
    
    dis_v_product = ((velocity-0) * dis_vector).sum(dim=1, keepdim=True)
    approaching_v = (dis_v_product / (dis+1e-6)).clamp_max(15.)
    approaching_v_vector = dis_vector_norm * approaching_v
    away_v_vector = velocity - approaching_v_vector
    
    # Check dimensionality issues here
    # away_v_vector: (N, 3)
    # away_v: (away_v_vector-0).norm(dim=1) -> (N,)
    # (1/ (dis.squeeze() + 1)) -> (N,)
    
    away_v = (away_v_vector-0).norm(dim=1) * (1/ (dis.squeeze() + 1))
    
    reward_approaching = approaching_v.squeeze() * 0.02
    reward_away = away_v * 0.02
    reward_pass = is_pass_next * success_r
    reward_ang = (angular_velocity - 0).norm(dim=1) * -0.001
    
    reward = reward_approaching - reward_away + reward_pass + reward_ang
    
    print(f"dis shape: {dis.shape}")
    print(f"approaching_v shape: {approaching_v.shape}")
    print(f"away_v shape: {away_v.shape}")
    print(f"reward shape: {reward.shape}")
    print(f"reward values: {reward}")

get_reward_debug()