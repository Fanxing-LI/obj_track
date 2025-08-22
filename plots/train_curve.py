from matplotlib import pyplot as plt
import numpy as np
from VisFly.utils.FigFashion.FigFashion import FigFon
import tensorboard
from tensorboard.backend.event_processing import event_accumulator
import os
import glob

FigFon.set_fashion("IEEE")

fig, axes = FigFon.get_figure_axes(SubFigSize=(1, 1), Column=1)

names = [
    "PPO_for_plot_1",
    "SAC_for_plot_1",
    "SHAC_for_plot_1"
]

def load_tensorboard_file(folder_path):
    # get file paths in the folder


    # Find tensorboard event files in the folder
    event_files = glob.glob(os.path.join(folder_path, "events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No tensorboard event files found in {folder_path}")

    # Use the first (or most recent) event file
    file_path = event_files[0] if len(event_files) == 1 else max(event_files, key=os.path.getctime)

    ea = event_accumulator.EventAccumulator(file_path)
    ea.Reload()
    # return ea.scalars.Items('train/episode_reward')

    # Get the scalar data for rollout/ep_rew_mean
    scalar_events = ea.Scalars('rollout/ep_rew_mean')

    # Extract timestamps and values into separate lists
    timestamps = np.array([event.wall_time for event in scalar_events])
    timesteps = np.array([event.step for event in scalar_events])
    values = np.array([event.value for event in scalar_events])

    # smooth the values keeps the same length as previous values
    if len(values) > 10:
        # Use padding to avoid edge effects
        window_size = 10
        # Pad the array with edge values to avoid boundary effects
        padded_values = np.pad(values, (window_size//2, window_size//2), mode='edge')
        # Apply convolution and then trim to original size
        smoothed = np.convolve(padded_values, np.ones(window_size)/window_size, mode='valid')
        # Ensure the smoothed array has exactly the same length as original
        if len(smoothed) != len(values):
            # Trim or pad to match original length
            smoothed = smoothed[:len(values)]
        values = smoothed

    return values, timesteps

current_folder = os.path.dirname(os.path.abspath(__file__))
current_folder = current_folder.split("obj_track")[0] + "obj_track/exps/std/saved/objTracking/"
for n in names:
    file_path = current_folder + n
    values, timestamps = load_tensorboard_file(file_path)

    axes.plot(timestamps, values, label=n)

import os, sys
# get current running file path and create save path
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
save_path = script_dir + "/train_curve.png"
axes.set_xlabel("Timesteps")
axes.set_ylabel("Episode Reward")
# change x to scientific notation
axes.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
axes.legend([n.split("_for_plot")[0] for n in names], loc="lower right")
fig.savefig(save_path, dpi=300)