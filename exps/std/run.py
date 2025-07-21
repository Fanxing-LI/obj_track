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
    parser.add_argument("--algorithm", "-a", type=str, default="BPTT")
    parser.add_argument("--env", "-e", type=str, default="hovering")
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--weight", "-w", type=str, default=None, )
    return parser


env_alias = {
    "hovering": HoverEnv,
    "hoveringVisual": VisualHoverEnv,
    "objTracking": ObjectTrackingEnv,
    "awareTracking": AwareTrackEnv2

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

train_env = env_alias[args.env](
    **env_config["env"]
)

if not args.train:
    env_config["eval_env"]["visual"] = True

env = env_alias[args.env](
    **env_config["eval_env"]
)

# if train mode, train the model
if args.train:

    if args.algorithm == "PPO":
        model = alg_alias[args.algorithm](
            env=env,
            seed=args.seed,
            comment=args.comment,
            save_path=save_folder,
            **config["algorithm"]
        )
    else:
        model = alg_alias[args.algorithm](
            env=env,
            train_env=train_env,
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

    model = alg_alias[args.algorithm].load(save_folder + args.weight, env=env)
    from exps.test.tracking.test import Test as tracking_test

    if args.weight is not None:
        model.load(path=save_folder + args.weight, env=env)

    test_handle = tracking_test(
        model=model,
        save_path=save_folder + "/test",
        name=args.weight
    )
    test_handle.test(**config["test"])
