from matplotlib import pyplot as plt
import numpy as np
from VisFly.utils.FigFashion.FigFashion import FigFon
import tensorboard
from tensorboard.backend.event_processing import event_accumulator
import os
import glob

FigFon.set_fashion("IEEE")

# Create the main figure using FigFon
fig, axes = FigFon.get_figure_axes(SubFigSize=(1, 1), Column=1, HeightScale=0.6, Border=[0,0,0.8,1])

# Create a narrow right subplot aligned with the main plot area
fig.canvas.draw()  # Ensure the main axes is fully rendered first

# Get the position of the main axes plot area (not including labels/ticks)
main_bbox = axes.get_position()
plot_area = axes.get_window_extent().transformed(fig.transFigure.inverted())

# Calculate position for right subplot - aligned with plot area
right_width = 0.08  # Width of right subplot
gap = 0.01  # Small gap between main plot and right subplot

right_x = plot_area.x1 + gap
right_y = plot_area.y0  # Same bottom as plot area
right_height = plot_area.height  # Same height as plot area

# Create the right subplot with exact alignment to plot area
axes_right = fig.add_axes([right_x, right_y, right_width, right_height])

# Create the main figure using FigFon
fig, axes = FigFon.get_figure_axes(SubFigSize=(1, 1), Column=1, HeightScale=0.6, Border=[0,0,0.8,1])

# Get the position and dimensions of the main axes
axes_position = axes.get_position()
axes_height = axes_position.height
axes_y_pos = axes_position.y0

# Add a narrow right axis using matplotlib's add_axes with same height and y position
right_ax_left = axes_position.x1 - 0.17  # Small gap from main plot
right_ax_width = 0.04  # Very narrow width
axes_right = fig.add_axes([right_ax_left, axes_y_pos+0.19, right_ax_width, axes_height-0.22])

# Configure the right axis - hide ticks and grid
axes_right.set_xticks([])  # Hide x ticks
axes_right.grid(False)  # Hide grid

# Set up the right axis with test results
test_res = {
    "Ours": 281,
    "PPO": 184,
    "SAC": 170
}

# Get line colors for consistent styling (we'll get them later after plotting)
# For now, use default colors
default_colors = ['red', 'blue', 'green']

# Set yticks and labels with corresponding colors
ytick_values = list(test_res.values())
ytick_labels = [f"{name}:{value}" for name, value in test_res.items()]

# Configure right axis to use right y-axis
axes_right.yaxis.set_label_position("right")
axes_right.yaxis.tick_right()

# Set the right axis properties
axes_right.set_ylim(0, 300)  # Same limits as main axis
axes_right.set_yticks(ytick_values)
axes_right.set_yticklabels(ytick_labels, fontsize=6)  # Reduced from 8 to 6
axes_right.set_xlabel("\nTest")

# Remove tick marks (the little spikes)
axes_right.tick_params(axis='y', length=0)

# We'll set colors and draw lines after the main plot is created

names = [
    # "PPO_for_plot_1",
    "PPO_Rand_1",
    "PPO_NoRand_1",
    "SAC_Rand_1",
    "SAC_for_plot_1",
    "SHAC_for_plot_1"
]
names = [
    # "PPO_for_plot_1",
    "SHAC_for_plot_1",
    "PPO_Rand_1",
    # "PPO_NoRand_1",
    "SAC_Rand_1",
    # "SAC_for_plot_1",
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
    original_values = np.array([event.value for event in scalar_events])

    # smooth the values keeps the same length as previous values
    smoothed_values = original_values.copy()
    if len(original_values) > 10:
        # Use padding to avoid edge effects
        window_size = 15  # Increased from 5 to 15 for smoother curves
        # Pad the array with edge values to avoid boundary effects
        padded_values = np.pad(original_values, (window_size//2, window_size//2), mode='edge')
        # Apply convolution and then trim to original size
        smoothed = np.convolve(padded_values, np.ones(window_size)/window_size, mode='valid')
        # Ensure the smoothed array has exactly the same length as original
        if len(smoothed) != len(original_values):
            # Trim or pad to match original length
            smoothed = smoothed[:len(original_values)]
        smoothed_values = smoothed

    return original_values, smoothed_values, timesteps

current_folder = os.path.dirname(os.path.abspath(__file__))
current_folder = current_folder.split("obj_track")[0] + "obj_track/exps/std/saved/objTracking/"

# Store all data for shadow plotting
all_original_data = []
all_smoothed_data = []
all_timestamps = []

for n in names:
    file_path = current_folder + n
    original_values, smoothed_values, timestamps = load_tensorboard_file(file_path)
    
    all_original_data.append(original_values)
    all_smoothed_data.append(smoothed_values) 
    all_timestamps.append(timestamps)
    
    # Plot smoothed line
    axes.plot(timestamps, smoothed_values, label=n)

# Get line colors after plotting
line_colors = [line.get_color() for line in axes.get_lines()]

# Update right axis tick colors to match line colors
axes_right_labels = axes_right.get_yticklabels()
for i, label in enumerate(axes_right_labels):
    if i < len(line_colors):
        label.set_color(line_colors[i])

# Draw horizontal lines on the right axis with matching colors
for i, (pos, color) in enumerate(zip(ytick_values, line_colors[:len(ytick_values)])):
    axes_right.axhline(y=pos, color=color, linestyle='--', alpha=0.7, linewidth=1, zorder=1)

# Add shadow regions showing original data range
for i, (original_vals, smoothed_vals, timestamps) in enumerate(zip(all_original_data, all_smoothed_data, all_timestamps)):
    if i < len(line_colors):
        # Calculate the difference between original and smoothed values
        diff = original_vals - smoothed_vals
        
        # Use smoothed values as the center line, with original data deviation as bounds
        raw_upper_bound = smoothed_vals + np.abs(diff)
        raw_lower_bound = smoothed_vals - np.abs(diff)
        
        # Apply simple smoothing to the error band edges
        smooth_window = 5
        if len(raw_upper_bound) > smooth_window:
            # Pad arrays for edge handling
            padded_upper = np.pad(raw_upper_bound, (smooth_window//2, smooth_window//2), mode='edge')
            padded_lower = np.pad(raw_lower_bound, (smooth_window//2, smooth_window//2), mode='edge')
            
            # Apply convolution smoothing
            smooth_upper = np.convolve(padded_upper, np.ones(smooth_window)/smooth_window, mode='valid')
            smooth_lower = np.convolve(padded_lower, np.ones(smooth_window)/smooth_window, mode='valid')
            
            # Ensure same length as original
            upper_bound = smooth_upper[:len(raw_upper_bound)]
            lower_bound = smooth_lower[:len(raw_lower_bound)]
        else:
            upper_bound = raw_upper_bound
            lower_bound = raw_lower_bound
            
        # Add shadow area showing the smoothed error band
        axes.fill_between(timestamps, lower_bound, upper_bound, 
                         color=line_colors[i], alpha=0.2, zorder=0)

import os, sys
# get current running file path and create save path
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
save_path = script_dir + "/train_curve.png"
axes.set_xlabel("Timesteps")
axes.set_ylabel("Episode Reward")
# change x to scientific notation
axes.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))

# Add alternating gray background every 3M timesteps
x_min, x_max = axes.get_xlim()
x_start = int(x_min // 3e6) * 3e6
x_end = int(x_max // 3e6 + 1) * 3e6
for i in range(int(x_start // 3e6), int(x_end // 3e6)):
    if i % 2 == 0:  # Lighter gray for even intervals
        axes.axvspan(i * 3e6, (i + 1) * 3e6, alpha=0.15, color='gray', zorder=0, linewidth=0)
    else:  # Darker gray for odd intervals
        axes.axvspan(i * 3e6, (i + 1) * 3e6, alpha=0.25, color='gray', zorder=0, linewidth=0)

test_res = {
    "Ours": 281,
    "PPO": 184,
    "SAC": 170
}

# Get the colors of the plotted lines (moved here after plotting and shadow creation)
# line_colors = [line.get_color() for line in axes.get_lines()]
# Make sure legend_names order matches the names list order
legend_names = ["Ours", "PPO", "SAC"]  # This should match the order in names list

# axes.legend([n.split("_")[0] for n in names], loc="lower right")
axes.legend(

    ["Ours",
        "PPO",
        # "PPO w/o rand",
        "SAC",
        # "SAC w/o rand",
        ],loc="lower right")

axes.set_xlim(0,2.5e7)
axes.set_ylim(0,300)

fig.savefig(save_path, dpi=300)