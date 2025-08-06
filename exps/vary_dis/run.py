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


# th.autograd.set_detect_anomaly(True)


def parse_args():
    parser = argparse.ArgumentParser(description='Run experiments', add_help=False)
    parser.add_argument('--comment', '-c', type=str, default="std")
    parser.add_argument("--train", "-t", type=int, default=1)
    parser.add_argument("--algorithm", "-a", type=str, default="SHAC")
    parser.add_argument("--env", "-e", type=str, default="objTracking")
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--weight", "-w", type=str, default=None, )
    parser.add_argument("--dis", "-d", type=float, default=1.5, )
    return parser


env_alias = {
    "objTracking": ObjectTrackingEnv,

}

alg_alias = {
    "BPTT": BPTT,
    "PPO": PPO,
    "SHAC": SHAC,
}

args = parse_args().parse_args()

save_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + f"/saved/{args.env}/"

config = load_yaml_config(os.path.dirname(os.path.abspath(__file__)) + f'/alg_cfgs/{args.env}/{args.algorithm}.yaml')
env_config = load_yaml_config(os.path.dirname(os.path.abspath(__file__)) + f'/env_cfgs/{args.env}.yaml')
env_config["eval_env"]["scene_kwargs"]["obj_settings"]["path"] = "dis_cubic"
env_config["eval_env"]["random_kwargs"]["state_generator"]["kwargs"][0]["max_dis"] = args.dis+0.001
env_config["eval_env"]["random_kwargs"]["state_generator"]["kwargs"][0]["min_dis"] = args.dis-0.001

env_config["env"]["random_kwargs"]["state_generator"]["kwargs"][0]["position"]["half"] = [1.0,1.0,0.1]
if not args.train:
    env_config["eval_env"]["visual"] = True

# if train mode, train the model
if args.train:
    env = env_alias[args.env](
        **env_config["env"]
    )

    model = alg_alias[args.algorithm](
        env=env,
        seed=args.seed,
        comment=args.comment,
        save_path=save_folder,
        **config["algorithm"]
    )

    if args.weight is not None:
        model.load(path=save_folder + args.weight, env=env)

    model.learn(**config["learn"])
    model.save()

else:
    eval_env = env_alias[args.env](
        **env_config["eval_env"]
    )
    model = alg_alias[args.algorithm].load((save_folder + args.weight).replace("vary_dis","std"), env=eval_env)
    from test import Test as tracking_test

    test_handle = tracking_test(
        model=model,
        save_path=save_folder + "/test",
        name=args.weight
    )
    test_handle.test(**config["test"])
    # save state_all and obs_all together in one file name with velocity
    th.save({
        "state_all": test_handle.state_all,
        "obs_all": test_handle.obs_all,
        "t": test_handle.t,
        "target_all": test_handle.target_all,
        "collision_all": test_handle.collision_all,
        "reward_all": test_handle.reward_all,
        "action_all": test_handle.action_all,
        "info_all": test_handle.info_all
    }, save_folder + f"/test/{args.dis}.pth")