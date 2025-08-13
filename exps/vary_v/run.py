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


# th.autograd.set_detect_anomaly(True)


def parse_args():
    parser = argparse.ArgumentParser(description='Run experiments', add_help=False)
    parser.add_argument('--comment', '-c', type=str, default="std")
    parser.add_argument("--train", "-t", type=int, default=1)
    parser.add_argument("--algorithm", "-a", type=str, default="SHAC")
    parser.add_argument("--env", "-e", type=str, default="objTracking")
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--weight", "-w", type=str, default=None, )
    parser.add_argument("--traj", "-tr", type=str, default="1", )
    parser.add_argument("--velocity", "-v", type=float, default=3.0, )
    return parser


env_alias = {
    "objTracking": ObjectTrackingEnv,
}

alg_alias = {
    "BPTT": BPTT,
    "PPO": PPO,
    "SHAC": SHAC,
}


def change_v_in_json(json_file, vel):
    path = os.path.dirname(os.path.abspath(__file__)) + f'/configs/obj/{json_file}/cubic.json'
    with open(path, 'r') as file:
        data = json.load(file)
        data["objects"][0]["velocity"]["kwargs"]["mean"] = vel

    with open(path, 'w') as file:
        json.dump(data, file, indent=2)


def get_env(env, traj, v, is_train=False):
    # script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    # save_folder = script_dir + f"/saved/{env}/"

    env_config = load_yaml_config(os.path.dirname(os.path.abspath(__file__)) + f'/env_cfgs/{env}.yaml')
    env_config["eval_env"]["scene_kwargs"]["obj_settings"]["path"] = traj

    change_v_in_json(traj, v)

    env_config["env"]["random_kwargs"]["state_generator"]["kwargs"][0]["position"]["half"] = [1.0, 1.0, 0.1]
    if not is_train:
        env_config["eval_env"]["visual"] = True

    if is_train:
        env = env_alias[env](
            **env_config["env"]
        )
    else:
        env = env_alias[env](
            **env_config["eval_env"]
        )

    return env


def main(
        traj,
        velocity,
        env="objTracking",
        algorithm="BPTT",
        weight=None,
        ROS_wrapper=None,
        comment="",
        debug=False,
):
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    save_folder = script_dir + f"/saved/{env}/"
    if ROS_wrapper:
        save_folder = f"/home/lfx-desktop/files/obj_track/exps/vary_v/saved/{env}"
    config = load_yaml_config(os.path.dirname(os.path.abspath(__file__)) + f'/alg_cfgs/{env}/{algorithm}.yaml')
    env_config = load_yaml_config(os.path.dirname(os.path.abspath(__file__)) + f'/env_cfgs/{env}.yaml')
    env_config["eval_env"]["scene_kwargs"]["obj_settings"]["path"] = \
        '/home/lfx-desktop/files/obj_track/exps/vary_v/configs/obj/' + traj

    change_v_in_json(traj, velocity)

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
    test_handle = tracking_test(
        model=model,
        save_path=save_folder + "/test",
        name=weight if weight else str.join("_", [str(velocity), traj, comment],
        )
    )
    if ROS_wrapper:
        config["test"]["is_video"] = False
        # config["test"]["is_video_save"] = False
        config["test"]["is_fig_save"] = False
        config["test"]["is_fig"] = False
    r = test_handle.test(ROS_wrapper=ROS_wrapper, debug=debug, comment=comment, **config["test"])
    if debug:
        return r
    # save state_all and obs_all together in one file name with velocity
    th.save({
        "state_all": test_handle.state_all,
        "obs_all": test_handle.obs_all,
        "t": test_handle.t,
        "target_all": test_handle.target_all,
        "collision_all": test_handle.collision_all,
        "reward_all": test_handle.reward_all,
        "action_all": test_handle.action_all,
        # "info_all": test_handle.info_all
    },
        save_folder + f"/test/{traj}_{velocity}.pth"
    )
    print("======================================================================")
    print(f"Test results saved to {save_folder}/test/{traj}_{velocity}_{comment}.pth")
    print("======================================================================")


if __name__ == "__main__":
    args = parse_args().parse_args()
    main(velocity=args.velocity, traj=args.traj, algorithm=args.algorithm, env=args.env, weight=args.weight)
