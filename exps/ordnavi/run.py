#!/usr/bin/env python3

import sys
import os
import torch as th
import traceback

# Set matplotlib to use non-interactive backend to avoid display issues on remote servers
import matplotlib
matplotlib.use('Agg')

sys.path.append(os.path.join(os.getcwd(), '../../'))
sys.path.append(os.getcwd())
from envs.OrdNavigationEnv import OrdNavigationEnv
from algorithms.BPTT import BPTT
from algorithms.SHAC import SHAC
from VisFly.utils.common import load_yaml_config
from VisFly.utils.policies import extractors  # noqa: F401
import argparse

# GPU manager removed - no longer needed


def parse_args():
    parser = argparse.ArgumentParser(description='Run ordinary navigation experiments', add_help=False)
    parser.add_argument('--comment', '-c', type=str, default="")
    parser.add_argument("--train", "-t", type=int, default=1)
    parser.add_argument("--algorithm", "-a", type=str, default="BPTT",
                        choices=["BPTT", "SHAC"],
                        help="Algorithm to use for training")
    parser.add_argument("--env", "-e", type=str, default="ordinary_navigation",)
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--weight", "-w", type=str, default=None)
    return parser


args = parse_args().parse_args()

save_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + f"/saved/{args.env}/"

# Load configuration files
config = load_yaml_config(os.path.dirname(os.path.abspath(__file__)) + f'/alg_cfgs/{args.env}/{args.algorithm}.yaml')
env_config = load_yaml_config(os.path.dirname(os.path.abspath(__file__)) + f'/env_cfgs/{args.env}.yaml')

# Environment aliases
env_alias = {
    "ordinary_navigation": OrdNavigationEnv,
}

alg_alias = {
    "BPTT": BPTT,
    "SHAC": SHAC,
}

# Create training environment
train_env = env_alias[args.env](
    **env_config["env"]
)

# Create evaluation environment
if not args.train:
    env_config["eval_env"]["visual"] = True

env = env_alias[args.env](
    **env_config["eval_env"]
)

if __name__ == "__main__":
    # Training mode
    if args.train:
        model = alg_alias[args.algorithm](
            env=train_env,
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
        # Evaluation/testing mode
        test_model_path = save_folder + args.weight

        test_env = env_alias[args.env](**env_config["eval_env"])
        test_env.reset()

        print(f"Agent spawn position: {test_env.position[0].cpu().numpy()}")
        print(f"Target position: {test_env.target[0].cpu().numpy()}")
        spawn_to_target_distance = ((test_env.target[0] - test_env.position[0]).norm()).item()
        print(f"Initial distance to target: {spawn_to_target_distance:.2f}m")

        print(f"Loading model from: {test_model_path}")
        model = alg_alias[args.algorithm].load(test_model_path, env=test_env)

        from tst import Test as tracking_test
        output_dir = save_folder + f"/test/{args.env.lower()}_{args.algorithm.lower()}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        import torch as th
        import numpy as np
        spawn_center = th.tensor([env_config["eval_env"]["random_kwargs"]["state_generator"]["kwargs"][0]["position"]["mean"]])
        target_pos = th.tensor([env_config["eval_env"]["target"]])

        ring_points = []
        ring_radius = 1.0
        for angle in np.linspace(0, 2*np.pi, 20):
            x = target_pos[0, 0] + ring_radius * np.cos(angle)
            y = target_pos[0, 1] + ring_radius * np.sin(angle)
            z = target_pos[0, 2]
            ring_points.append([x, y, z])
        ring_curve = th.tensor(ring_points).unsqueeze(0)

        debug_points = th.cat([spawn_center, target_pos], dim=0)
        print(f"Debug visualization: spawn center {spawn_center[0].tolist()}, target {target_pos[0].tolist()}")

        print(f"Initializing test handle with output directory: {output_dir}")
        test_handle = tracking_test(
            env=test_env,
            model=model,
            name=f"{args.env.lower()}_{args.algorithm.lower()}_test",
            save_path=output_dir
        )

        n_eval_episodes = 4
        aggregated_success = 0
        total_agents_eval = test_env.num_envs * n_eval_episodes

        for ep_i in range(n_eval_episodes):
            episode_dir = os.path.join(output_dir, f"episode_{ep_i:03d}")
            if not os.path.exists(episode_dir):
                os.makedirs(episode_dir, exist_ok=True)

            test_handle.obs_all = []
            test_handle.state_all = []
            test_handle.info_all = []
            test_handle.action_all = []
            test_handle.collision_all = []
            test_handle.render_image_all = []
            test_handle.reward_all = []
            test_handle.reward_components = []
            test_handle.t = []
            test_handle.eq_r = []
            test_handle.eq_l = []

            try:
                debug_render_kwargs = config.get('test', {}).get('render_kwargs', {}).copy()
                debug_render_kwargs.update({
                    'is_draw_axes': True,
                    'points': debug_points,
                    'curves': ring_curve
                })

                result = test_handle.test(
                    is_fig=True,
                    is_fig_save=False,
                    is_video=False,
                    is_video_save=True,
                    is_sub_video=False,
                    render_kwargs=debug_render_kwargs,
                )
                if isinstance(result, tuple):
                    figs = result[0] if len(result) > 0 else []
                else:
                    figs = result
                import matplotlib.figure
                if figs is not None and not isinstance(figs, list):
                    if isinstance(figs, matplotlib.figure.Figure):
                        figs = [figs]
                    else:
                        figs = list(figs)

                try:
                    _ = test_handle.draw_debug(save_path=episode_dir)
                    print(f"Episode {ep_i}: debug analysis saved to {os.path.join(episode_dir, 'debug_analysis.png')}")
                except Exception:
                    pass
            except Exception as e:
                print(f"Error during test execution: {e}")
                traceback.print_exc()
                print("Continuing with next episode...")
                continue

            if figs:
                for fig_idx, fig in enumerate(figs):
                    fig_path = os.path.join(episode_dir, f"trajectory_plot_{fig_idx}.png")
                    try:
                        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
                        print(f"Episode {ep_i}: trajectory plot saved to {fig_path}")
                    except Exception:
                        pass

            try:
                test_handle.save_combined_video(episode_dir)
            except Exception as e:
                print(f"Error saving combined video for episode {ep_i}: {e}")
                print("Continuing with next episode...")

            successful_agents = set()
            for timestep_idx, timestep_info in enumerate(test_handle.info_all):
                for agent_idx, agent_info in enumerate(timestep_info):
                    if agent_info and "is_success" in agent_info and agent_info["is_success"]:
                        successful_agents.add(agent_idx)
            episode_successes = len(successful_agents)
            aggregated_success += episode_successes
            print(f"Episode {ep_i}: {episode_successes}/{test_env.num_envs} agents reached target at some point")

        eval_sr = aggregated_success / total_agents_eval
        print(f"\nAggregated evaluation over {n_eval_episodes} episodes → Success Rate: {eval_sr:.3f}\n")
