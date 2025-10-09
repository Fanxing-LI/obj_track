import os, sys

sys.path.append(os.getcwd())
from VisFly.envs.HoverEnv import HoverEnv
from envs.ObjectTrackingEnv import ObjectTrackingEnv
from envs.VisualHoverEnv import VisualHoverEnv
# from envs.TrackingEnv import AwareTrackEnv
from envs.TrackingEnv import AwareTrackEnv2
from algorithms.BPTT import BPTT
from algorithms.SHAC import SHAC
from VisFly.utils.algorithms.PPO import PPO
import torch as th
import sys
import os
import argparse
from VisFly.utils.common import load_yaml_config
import json
from exps.test.tracking.test import Test as tracking_test
import yaml

# th.autograd.set_detect_anomaly(True)
"""
   Current Best: SHAC_NoCaliHeadV_Pos_Dis3.0_spd3.4_lessNoise_2.zip
   others: SHAC_NoCaliHeadV_Pos_Dis1.5_spd3.4_lessNoise_1.zip for distance 1.5
   SHAC_NoCaliHeadV_Dis4.5_6.zip
   SHAC_deploy_5.zip
   PPO_NoRand_1.zip
"""

def parse_args():
    parser = argparse.ArgumentParser(description='Run experiments', add_help=False)
    parser.add_argument('--comment', '-c', type=str, default="std")
    parser.add_argument("--train", "-t", type=int, default=1)
    parser.add_argument("--algorithm", "-a", type=str, default="SHAC")
    parser.add_argument("--env", "-e", type=str, default="objTracking")
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--weight", "-w", type=str, default="SHAC_deploy_PAction0.025_1.zip", )
    parser.add_argument("--traj", "-tr", type=str, default="bra", )
    parser.add_argument("--velocity", "-v", type=float, default=1.0, )
    parser.add_argument("--distance", "-d", type=float, default=3.0, )
    return parser


env_alias = {
    "objTracking": ObjectTrackingEnv,
}

alg_alias = {
    "BPTT": BPTT,
    "PPO": PPO,
    "SHAC": SHAC,
}



def main(
        traj,
        env="objTracking",
        algorithm="SHAC",
        weight=None,
        ROS_wrapper=False,
        comment=None,
):
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    save_folder = script_dir + f"/saved/{env}/"

    config = load_yaml_config(os.path.dirname(os.path.abspath(__file__)) + f'/alg_cfgs/{env}/{algorithm}.yaml')
    env_config = load_yaml_config(os.path.dirname(os.path.abspath(__file__)) + f'/env_cfgs/{env}.yaml')
    # Auto-expand short trajectory name to full path if user passed a simple label (e.g. 'B')
    # We reuse the vary_v trajectory configs if present.
    traj_path = traj
    if not os.path.isabs(traj_path) and len(traj_path) <= 8:  # heuristic: short label
        candidate = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),  # .../exps/acc_brake
            '../vary_v/configs/obj',
            traj_path
        )
        candidate = os.path.abspath(candidate)
        if os.path.isdir(candidate):
            traj_path = candidate
    # Ensure directory exists; warn if not
    if not os.path.isdir(traj_path):
        print(f"[acc_brake] WARNING: trajectory path '{traj_path}' does not exist; using raw input '{traj}'.")
        traj_path = traj  # fallback
    env_config["eval_env"]["scene_kwargs"]["obj_settings"]["path"] = traj_path
    print(f"[acc_brake] Using trajectory path: {env_config['eval_env']['scene_kwargs']['obj_settings']['path']}")
    comment = comment if comment else algorithm

    eval_env = env_alias[env](
        **env_config["eval_env"]
    )

    if weight:
        model = alg_alias[algorithm].load((save_folder + weight).replace("acc_brake", "std"), env=eval_env)
    else:
        model = alg_alias[algorithm](
            env=eval_env,
            seed=42,
            comment=comment,
            save_path=save_folder,
            **config["algorithm"]
        )
    # from test import Test as tracking_test
    print("--------------------debug:enter the main")
    test_handle = tracking_test(
        model=model,
        save_path=save_folder + "/test",
        name=comment+traj,
        )
    
    if ROS_wrapper:
        config["test"]["is_video"] = False
        # config["test"]["is_video_save"] = False
        config["test"]["is_fig_save"] = False
        config["test"]["is_fig"] = False
        
        return eval_env
    r = test_handle.test(ROS_wrapper=ROS_wrapper, comment=comment, **config["test"])
    if ROS_wrapper:
        # When using ROS_wrapper, the test function runs an infinite loop
        # and never returns, so we should return here
        return r


if __name__ == "__main__":
    args = parse_args().parse_args()
    main(traj=args.traj, algorithm=args.algorithm, env=args.env, weight=args.weight, comment=args.comment)
