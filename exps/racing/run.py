import os, sys

sys.path.append(os.getcwd())
# from VisFly.envs.HoverEnv import HoverEnv2 as HoverEnv
from envs.HoverEnv import HoverEnv
from envs.ObjectTrackingEnv import ObjectTrackingEnv
from envs.DynRacingEnv import RacingEnv
from envs.VisualHoverEnv import VisualHoverEnv
# from envs.TrackingEnv import AwareTrackEnv
from envs.TrackingEnv import AwareTrackEnv2
from algorithms.BPTT_series.BPTT import BPTT
from algorithms.BPTT_series.SHAC import SHAC
from algorithms.BPTT_series.BPTT_KL import BPTT_KL
from VisFly.utils.algorithms.PPO import PPO
from VisFly.utils.algorithms.SAC import SAC
from algorithms.dream_to_fly.algorithms.diff_dreamer3 import DiffDreamer
from algorithms.dream_to_fly.algorithms.dreamer import dreamer
from algorithms.dream_to_fly.algorithms.diff_dreamer_sperate import DiffDreamer_sep
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
    parser.add_argument("--algorithm", "-a", type=str, default="PPO")
    parser.add_argument("--env", "-e", type=str, default="racing")
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--weight", "-w", type=str, default=None, )
    return parser


env_alias = {
    "objTracking": ObjectTrackingEnv,
    "racing": RacingEnv,    
}

alg_alias = {
    "BPTT": BPTT,
    "PPO": PPO,
    "SHAC": SHAC,
    "SAC": SAC,
    "diff_dreamer": DiffDreamer,
    "dreamer": dreamer,
    "diff_dreamer_sep": DiffDreamer_sep,
    "BPTT_KL": BPTT_KL,
}

args = parse_args().parse_args()

save_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + f"/saved/{args.env}/"

config = load_yaml_config(os.path.dirname(os.path.abspath(__file__)) + f'/alg_cfgs/{args.env}/{args.algorithm}.yaml')
env_config = load_yaml_config(os.path.dirname(os.path.abspath(__file__)) + f'/env_cfgs/{args.env}.yaml')

if not args.train:
    env_config["eval_env"]["visual"] = True

# if train mode, train the model
if args.train:
    env = env_alias[args.env](
        **env_config["env"]
    )

    if args.algorithm == "BPTT_sample":
        env_config["env"]["num_agent_per_scene"] = config["algorithm"]["actor_batch_size"]
        env_config["env"]["num_scene"] = 1
        env_config["env"]["visual"] = False
        env_config["env"]["device"] = "cpu"
        train_env = env_alias[args.env](**env_config["env"])
        model = alg_alias[args.algorithm](
            env=env,
            seed=args.seed,
            comment=args.comment,
            save_path=save_folder,
            train_env=train_env,
            **config["algorithm"]
        )
    elif args.algorithm == "diff_dreamer":
        train_env = env_alias[args.env](**env_config["env"])
        model = alg_alias[args.algorithm](
            env=env,
            seed=args.seed,
            comment=args.comment,
            save_path=save_folder,
            # train_env=train_env,
            **config["algorithm"]
        )
    else:
        model = alg_alias[args.algorithm](
            env=env,
            seed=args.seed,
            comment=args.comment,
            save_path=save_folder,
            **config["algorithm"]
        )

    if args.weight is not None:
        # model = model.load(path=save_folder + args.weight, env=env)
        if args.algorithm == "diff_dreamer" or args.algorithm == "dreamer":
            model.load_parameters(save_folder + args.weight, **config["load"])
        else:
            model = model.load(path=save_folder + args.weight, env=env)
        model.create_save_path(args.comment)

    model.learn(**config["learn"])
    model.save()

else:
    eval_env = env_alias[args.env](
        **env_config["eval_env"]
    )
    model = alg_alias[args.algorithm].load(save_folder + args.weight, env=eval_env)
    from test import Test

    test_handle = Test(
        model=model,
        save_path=save_folder + "/test",
        name=args.weight
    )
    test_handle.test(**config["test"])
