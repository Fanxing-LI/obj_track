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
    parser.add_argument("--weight", "-w", type=str, default=None, )
    parser.add_argument("--traj", "-tr", type=str, default="8", )
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


def change_v_in_json(json_file, vel, dis):
    path = os.path.dirname(os.path.abspath(__file__)) + f'/configs/obj/{json_file}/cubic.json'
    env_path = os.path.dirname(os.path.abspath(__file__)) + f'/env_cfgs/objTracking.yaml'
    with open(path, 'r') as file:
        data = json.load(file)
        data["objects"][0]["velocity"]["kwargs"]["mean"] = vel
    with open(path, 'w') as file:
        json.dump(data, file, indent=2)

    with open(env_path, 'r') as file:
        env_data = yaml.safe_load(file)
        env_data["eval_env"]["scene_kwargs"]["obj_settings"]["path"] = path
        env_data["env"]["random_kwargs"]["state_generator"]["kwargs"][0]["min_dis"] = dis-0.001
        env_data["env"]["random_kwargs"]["state_generator"]["kwargs"][0]["max_dis"] = dis+0.001

    with open(env_path, 'w') as file:
        yaml.dump(env_data, file, default_flow_style=False, sort_keys=False)

def main(
        traj,
        velocity,
        distance=3.0,
        env="objTracking",
        algorithm="SHAC",
        weight=None,
        ROS=None,
        comment=None,
        debug=False,
):
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    save_folder = script_dir + f"/saved/{env}/"
    if ROS:
        save_folder = f"/home/lfx-desktop/files/obj_track/exps/vary_v/saved/{env}"
    config = load_yaml_config(os.path.dirname(os.path.abspath(__file__)) + f'/alg_cfgs/{env}/{algorithm}.yaml')
    env_config = load_yaml_config(os.path.dirname(os.path.abspath(__file__)) + f'/env_cfgs/{env}.yaml')
    env_config["eval_env"]["scene_kwargs"]["obj_settings"]["path"] = \
        '/home/lfx-desktop/files/obj_track/exps/vary_v/configs/obj/' + traj

    # Modify action type for elastic mode
    if comment == "elastic":
        env_config["eval_env"]["dynamics_kwargs"] = env_config["eval_env"].get("dynamics_kwargs", {})
        env_config["eval_env"]["dynamics_kwargs"]["action_type"] = "position"
        print(f"[INFO] Modified action_type to 'position' for elastic mode")
    comment = comment if comment else algorithm

    change_v_in_json(traj, velocity, distance)

    eval_env = env_alias[env](
        **env_config["eval_env"]
    )

    if weight:
        model = alg_alias[algorithm].load((save_folder + weight).replace("vary_v", "std"), env=eval_env)
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
        name=f"{weight}_v{velocity}_traj{traj}" if weight else str.join("_", [str(velocity), traj, comment],
        )
    )
    if ROS:
        config["test"]["is_video"] = False
        # config["test"]["is_video_save"] = False
        config["test"]["is_fig_save"] = False
        config["test"]["is_fig"] = False
        return eval_env
    r = test_handle.test(ROS_wrapper=ROS, debug=debug, comment=comment, **config["test"])

    for i in test_handle.obs_all:
        # remove all the image obs
        for key in list(i.keys()):
            if "color" in key or "depth" in key or "semantic" in key:
                del i[key]

    pth_path = save_folder + f"/test/{traj}_{velocity}_{comment}_Dis{distance}.pth"
    th.save({
        "state_all": test_handle.state_all,
        "obs_all": test_handle.obs_all,
        "t": test_handle.t,
        "target_all": test_handle.target_all,
        "collision_all": test_handle.collision_all,
        "reward_all": test_handle.reward_all,
        "action_all": test_handle.action_all,
        "target_dis_all": test_handle.target_dis_all,
        "center_all": test_handle.center_all,
        # "info_all": test_handle.info_all
    },
        pth_path
    )
    print("======================================================================")
    print(f"Test results saved to {pth_path}")
    print("======================================================================")


def auto_get_distance_from_name(weight):
    """
    Extract distance from the weight name.
    Example: SHAC_NoCaliHeadV_Pos_Dis3.0_spd3.4_lessNoise_2.zip -> 3.0
    """
    if "Dis" in weight:
        return float(weight.split("Dis")[1].split("_")[0])
    else:
        return 3.0  # Default distance if not found


if __name__ == "__main__":
    args = parse_args().parse_args()
    distance = auto_get_distance_from_name(args.weight) if args.weight else args.distance
    main(velocity=args.velocity, traj=args.traj, algorithm=args.algorithm, env=args.env, weight=args.weight, distance=distance)
