# -*- coding: utf-8 -*-
import os, sys
import json
from VisFly.utils.common import load_yaml_config
import torch as th
from VisFly.utils.FigFashion.FigFashion import FigFon
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from VisFly.utils.FigFashion.colors import colorsets
import numpy as np

colors = colorsets["Modern Scientific"]

FigFon.set_fashion("IEEE")
save_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + "/saved/objTracking/"

def compute_curvature_vector(x, y, t=None):
    """
    Compute curvature vector for a 2D trajectory.
    Args:
        x: 1D array-like, x coordinates
        y: 1D array-like, y coordinates
        t: 1D array-like, time or parameter (optional, will use arc length if None)
    Returns:
        curvature_magnitude: 1D numpy array, curvature magnitude at each point
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if t is None or np.all(t == t[0]):
        # Use cumulative arc length as parameter when t is constant
        dx_raw = np.diff(x, prepend=x[0])
        dy_raw = np.diff(y, prepend=y[0])
        arc_lengths = np.cumsum(np.sqrt(dx_raw ** 2 + dy_raw ** 2))
        t = arc_lengths / 2
    else:
        t = np.asarray(t)

    # First and second derivatives
    dx = np.gradient(x, t)
    dy = np.gradient(y, t)
    ddx = np.gradient(dx, t)
    ddy = np.gradient(dy, t)

    # Curvature magnitude
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = (dx ** 2 + dy ** 2) ** 1.5
    denominator = np.where(denominator < 1e-12, 1e-12, denominator)
    curvature_magnitude = numerator / denominator

    return curvature_magnitude

def add_curvature_background(ax, time_data, curvature_data, alpha=0.15):
    """
    Add curvature-based background shading to a plot
    Args:
        ax: matplotlib axis
        time_data: x-axis data (time)
        curvature_data: curvature values corresponding to time points
        alpha: transparency of the background shading
    """
    # Normalize curvature to [0, 1] for grayscale mapping
    curvature_norm = (curvature_data - np.min(curvature_data)) / (np.max(curvature_data) - np.min(curvature_data) + 1e-12)
    
    # Create segments for background coloring
    for i in range(len(time_data) - 1):
        # Use average curvature between consecutive points
        avg_curvature = (curvature_norm[i] + curvature_norm[i + 1]) / 2
        gray_value = 1 - avg_curvature  # Higher curvature = darker gray
        
        ax.axvspan(time_data[i], time_data[i + 1], 
                   facecolor=(gray_value, gray_value, gray_value), 
                   alpha=alpha, zorder=0)

def parse_args():
    parser = argparse.ArgumentParser(description='Run experiments', add_help=False)
    parser.add_argument('--comment', '-c', type=str, default="std")
    parser.add_argument("--train", "-t", type=int, default=1)
    parser.add_argument("--velocity", "-v", type=str, default="1.0", )
    return parser


def plot_with_missing_data_detection(ax, t, y, label, color=None, threshold=1e-6, min_segment_length=10):
    """
    Plot data with dashed lines for segments where data might be missing (flat horizontal lines)
    Args:
        ax: matplotlib axis
        t: time data
        y: y data
        label: line label
        color: line color (if None, will use default color cycle)
        threshold: threshold for detecting flat segments (derivative close to 0)
        min_segment_length: minimum length of segment to be considered as missing data
    """
    t_np = t.cpu().numpy() if hasattr(t, 'cpu') else t.numpy()
    y_np = y.cpu().numpy() if hasattr(y, 'cpu') else y.numpy()
    
    # Calculate derivatives to detect flat segments
    dy = np.abs(np.gradient(y_np))
    
    # Find segments where derivative is very small (indicating flat lines)
    is_flat = dy < threshold
    
    # Find continuous flat segments
    flat_segments = []
    start_idx = None
    
    for i, flat in enumerate(is_flat):
        if flat and start_idx is None:
            start_idx = i
        elif not flat and start_idx is not None:
            if i - start_idx >= min_segment_length:
                flat_segments.append((start_idx, i))
            start_idx = None
    
    # Check if the last segment extends to the end
    if start_idx is not None and len(is_flat) - start_idx >= min_segment_length:
        flat_segments.append((start_idx, len(is_flat)))
    
    if len(flat_segments) == 0:
        # No missing data detected, plot normally
        ax.plot(t_np, y_np, label=label, color=color, linewidth=1)
    else:
        # Plot segments separately with consistent color
        last_end = 0
        first_segment = True
        
        for start, end in flat_segments:
            # Plot normal segment before flat segment
            if start > last_end:
                ax.plot(t_np[last_end:start+1], y_np[last_end:start+1], 
                       label=label if first_segment else "", color=color, linewidth=1)
                first_segment = False
            
            # Plot flat segment as dashed line with same color
            ax.plot(t_np[start:end], y_np[start:end], 
                   label="" if not first_segment else label, color=color, linewidth=1, 
                   linestyle='--', alpha=0.7)
            first_segment = False
            
            last_end = end
        
        # Plot remaining normal segment after the last flat segment
        if last_end < len(t_np):
            ax.plot(t_np[last_end:], y_np[last_end:], 
                   label="" if not first_segment else label, color=color, linewidth=1)
def plot_with_error_band(ax, x, y, maxy, miny, label=""):
    """
    Plot data with error band (currently simplified)
    """
    # Plot mean line
    ax.plot(x, y, label=label, linewidth=1)
    # Plot error band
    ax.fill_between(x, miny, maxy, alpha=0.3)


args = parse_args().parse_args()

label = args.velocity

# 获取目标位置数据（假设target_all包含位置信息）
algs = [
    "SHAC",
    "PPO",
    "elastic",
]

trajs = ["D", "B", "8"]
vs = ["1.5", "3.0"]

mean_oris = []
std_oris = []
mean_distances = []
std_ditances = []
fig, axes = FigFon.get_figure_axes(SubFigSize=(3, len(vs)*len(trajs)), Column=2, HeightScale=1.1,
                                   sharey=False, Border=[0,0,1,0.95], sharex="col")

file_folder = (os.path.dirname(os.path.abspath(sys.argv[0])).split("obj_track")[0]
               + "/obj_track/exps/vary_v/saved/objTracking/test/")

# Pre-compute curvatures for each trajectory type
trajectory_curvatures = {
    "D": [None, None],
    "B": [None, None],
    "8": [None, None]
}
for traj in trajs:
    for v_i, v in enumerate(vs):

        # Load one file to get the target trajectory (same for all algorithms and velocities)
        sample_data = th.load(file_folder + "{}_{}_SHAC_Dis3.0.pth".format(traj, v))
        target_positions = th.stack([tar for tar in sample_data["target_all"]]).squeeze()[:, 0, :]
        
        # Compute curvature for this trajectory
        curvature = compute_curvature_vector(
            target_positions[:, 0].cpu().numpy(),
            target_positions[:, 1].cpu().numpy()
        )
        trajectory_curvatures[traj][v_i] = curvature

# Add curvature backgrounds first, before the data plotting
for traj_i, traj in enumerate(trajs):
    for v_i, v in enumerate(vs):
        curvature = trajectory_curvatures[traj][v_i].copy()  # Make a copy to avoid modifying original
        # Adjust curvature intensity for different velocities
        if v_i == 1:
            curvature_scale = 1.5  # More controlled scaling factor
            curvature = curvature * curvature_scale
        
        # Load one file to get time data for this trajectory and velocity
        sample_data = th.load(file_folder + "{}_{}_SHAC_Dis3.0.pth".format(traj, v))
        if all(th.stack(sample_data["t"])[:,0] == 0):
            t = th.arange(0, len(sample_data["t"]))*0.03
        else:
            t = th.stack(sample_data["t"])[:,0]
        
        # Normalize time to [0, 1] for consistent curvature background mapping
        t_numpy = t.cpu().numpy() if hasattr(t, 'cpu') else t.numpy()
        # get current x lim
        col_idx = traj_i * 2 + v_i
        add_curvature_background(axes[0, col_idx], t_numpy, curvature)
        add_curvature_background(axes[1, col_idx], t_numpy, curvature)
        add_curvature_background(axes[2, col_idx], t_numpy, curvature)

for v_i , v in enumerate(vs):
    for traj_i, traj in enumerate(trajs):
        for alg_i, alg in enumerate(algs):
            data = th.load(file_folder + "{}_{}_{}_{}.pth".format(traj, v, alg, "Dis3.0"))
            distance = th.stack([tar for tar in data["target_dis_all"]])
            ori = th.stack([cen for cen in data["center_all"]])[:,:,:2]
            x_ori, y_ori = ori[..., 0], ori[..., 1]
            mean_x_ori, std_x_ori = x_ori.mean(dim=1), x_ori.std(dim=1)
            mean_y_ori, std_y_ori = y_ori.mean(dim=1), y_ori.std(dim=1)
            max_x_ori, min_x_ori = x_ori.max(dim=1).values, x_ori.min(dim=1).values
            max_y_ori, min_y_ori = y_ori.max(dim=1).values, y_ori.min(dim=1).values
            # mean_ori, std_ori = ori.mean(dim=1), ori.std(dim=1)
            # max_ori, min_ori = ori.max(dim=1).values, ori.min(dim=1).values
            if all(th.stack(data["t"])[:,0] == 0):
                t = th.arange(0, len(data["t"]))*0.03
            else:
                t = th.stack(data["t"])[:,0]
            states = th.stack(data["state_all"])[:,:,:3]
            # distance = (target_positions-states).norm(dim=-1)
            distance = distance - 3
            mean_distance, std_distance = distance.mean(dim=-1), distance.std(dim=-1)
            max_distance, min_distance = distance.max(dim=-1).values, distance.min(dim=-1).values
            mean_distances.append(mean_distance)
            std_ditances.append(std_distance)

            best_i = 3
            
            # Get color for this algorithm (consistent across all plots)
            alg_colors = colors[:3] # Default matplotlib colors
            if alg_i < len(alg_colors):
                line_color = alg_colors[alg_i]
            else:
                line_color = None
            
            # Use the new plotting function that detects missing data segments
            plot_with_missing_data_detection(axes[0, traj_i * 2 + v_i], t, x_ori[:,best_i], algs[alg_i], color=line_color)
            plot_with_missing_data_detection(axes[1, traj_i * 2 + v_i], t, y_ori[:,best_i], algs[alg_i], color=line_color)
            plot_with_missing_data_detection(axes[2, traj_i * 2 + v_i], t, distance[:,best_i], algs[alg_i], color=line_color)
            
            # Original plotting code (commented out)
            # axes[0, traj_i * 2 + v_i].plot(t, x_ori[:,best_i], label=label)
            # axes[1, traj_i * 2 + v_i].plot(t, y_ori[:,best_i], label=label)
            # axes[2, traj_i * 2 + v_i].plot(t, distance[:,best_i], label=label)
            # plot_with_error_band(axes[0, traj_i*2+v_i], t, mean_x_ori, max_x_ori, min_x_ori)
            # plot_with_error_band(axes[1, traj_i*2+v_i], t, mean_y_ori, max_y_ori, min_y_ori)
            # plot_with_error_band(axes[2, traj_i*2+v_i], t, mean_distance, max_distance,min_distance)
            # axes[2, traj_i * 2 + v_i].set_ylim(-0., 2)

# 为每列的最底行添加x轴标签
for col_idx in range(len(vs) * len(trajs)):
    axes[2, col_idx].set_xlabel("Time (s)")
axes[0,0].set_ylabel("$e_{H}$")
axes[1,0].set_ylabel("$e_{V}$")
axes[2,0].set_ylabel("$e_{d}$")

handles, labels = axes[0,0].get_legend_handles_labels()
algs = ["Ours", "PPO", "Elastic"]
FigFon.set_shared_legend(handles, algs)

# 为每组轨迹添加主标题
fig.text(0.20, 0.97, "Trajectory {}".format(trajs[0]), ha='center', va='center', fontsize=8, weight='bold')
fig.text(0.52, 0.97, "Trajectory {}".format(trajs[1]), ha='center', va='center', fontsize=8, weight='bold')
fig.text(0.85, 0.97, "Trajectory {}".format(trajs[2]), ha='center', va='center', fontsize=8, weight='bold')

# 为每个子图添加速度标题
axes[0,0].set_title("$v={}m/s$".format(vs[0]))
axes[0,1].set_title("$v={}m/s$".format(vs[1]))
axes[0,2].set_title("$v={}m/s$".format(vs[0]))
axes[0,3].set_title("$v={}m/s$".format(vs[1]))
axes[0,4].set_title("$v={}m/s$".format(vs[0]))
axes[0,5].set_title("$v={}m/s$".format(vs[1]))

axes[0,0].set_ylim(-0.15,0.15)
axes[0,1].set_ylim(-0.5,0.5)
axes[0,2].set_ylim(-0.15,0.15)
axes[0,3].set_ylim(-0.5,0.5)
axes[0,4].set_ylim(-0.15,0.15)
axes[0,5].set_ylim(-0.5,0.5)

axes[2,0].set_ylim(-1,1)
axes[2,1].set_ylim(-2,2)
axes[2,2].set_ylim(-1,1)
axes[2,3].set_ylim(-2,2)
axes[2,4].set_ylim(-1,1)
axes[2,5].set_ylim(-2,2)

current_folder = os.path.dirname(os.path.abspath(__file__))
save_folder = current_folder.split("obj_track")[0] + "obj_track/plots/"

fig.savefig("{}vary_traj_all_{}.png".format(save_folder, best_i))

# plt.show()
