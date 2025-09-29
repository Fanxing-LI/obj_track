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

# 设置单一轨迹和速度条件
target_traj = "8"  # 可以修改为需要的轨迹
target_velocity = "1.5"  # 可以修改为需要的速度

# 创建3x3的子图布局：3个指标 x 3个算法
fig, axes = FigFon.get_figure_axes(SubFigSize=(3, 3), Column=1, HeightScale=1.0,
                                   sharey=False, Border=[0,0,1,1], sharex=True)

file_folder = (os.path.dirname(os.path.abspath(sys.argv[0])).split("obj_track")[0]
               + "/obj_track/exps/vary_v/saved/objTracking/test/")

# 计算目标轨迹的曲率
sample_data = th.load(file_folder + "{}_{}_SHAC_Dis3.0.pth".format(target_traj, target_velocity))
target_positions = th.stack([tar for tar in sample_data["target_all"]]).squeeze()[:, 0, :]

# Compute curvature for this trajectory
curvature = compute_curvature_vector(
    target_positions[:, 0].cpu().numpy(),
    target_positions[:, 1].cpu().numpy()
)

# 为每个子图添加曲率背景
if all(th.stack(sample_data["t"])[:,0] == 0):
    t_background = th.arange(0, len(sample_data["t"]))*0.03
else:
    t_background = th.stack(sample_data["t"])[:,0]

t_numpy = t_background.cpu().numpy() if hasattr(t_background, 'cpu') else t_background.numpy()

# Add curvature backgrounds to all subplots
for row in range(3):  # 3 metrics
    for col in range(3):  # 3 algorithms
        add_curvature_background(axes[row, col], t_numpy, curvature)

# 主绘图循环：为每个算法绘制所有飞机的轨迹
for alg_i, alg in enumerate(algs):
    # 加载当前算法的数据
    data = th.load(file_folder + "{}_{}_{}_{}.pth".format(target_traj, target_velocity, alg, "Dis3.0"))
    
    # 提取数据
    distance = th.stack([tar for tar in data["target_dis_all"]])
    ori = th.stack([cen for cen in data["center_all"]])[:,:,:2]
    x_ori, y_ori = ori[..., 0], ori[..., 1]
    
    # 处理时间数据
    if all(th.stack(data["t"])[:,0] == 0):
        t = th.arange(0, len(data["t"]))*0.03
    else:
        t = th.stack(data["t"])[:,0]
    
    # 处理距离数据
    distance = distance - 3
    
    # 获取所有飞机的数量
    num_drones = x_ori.shape[1]
    
    # 为每个飞机绘制轨迹
    for drone_i in range(num_drones):
        # 生成颜色：为每个飞机使用不同的颜色，但同一算法内保持一致的色系
        drone_alpha = 0.3 + 0.7 * (drone_i / max(1, num_drones - 1))  # 透明度从0.3到1.0
        if alg_i == 0:  # SHAC - 蓝色系
            drone_color = plt.cm.Blues(drone_alpha)
        elif alg_i == 1:  # PPO - 橙色系
            drone_color = plt.cm.Oranges(drone_alpha) 
        else:  # Elastic - 绿色系
            drone_color = plt.cm.Greens(drone_alpha)
        drone_color = plt.cm.Blues(drone_alpha)
        
        # 绘制当前飞机的轨迹到对应的子图
        # Row 0: x_ori (水平误差)
        plot_with_missing_data_detection(axes[0, alg_i], t, x_ori[:, drone_i], 
                                       f"Quadrotor {drone_i}" if alg_i == 0 and drone_i < 5 else "", 
                                       color=drone_color)
        
        # Row 1: y_ori (垂直误差)  
        plot_with_missing_data_detection(axes[1, alg_i], t, y_ori[:, drone_i], 
                                       "", color=drone_color)
        
        # Row 2: distance (距离误差)
        plot_with_missing_data_detection(axes[2, alg_i], t, distance[:, drone_i], 
                                       "", color=drone_color)

# 设置轴标签和标题
# 设置行标签（左侧y轴标签）
axes[0,0].set_ylabel("$e_{H}$")
axes[1,0].set_ylabel("$e_{V}$")
axes[2,0].set_ylabel("$e_{d}$")

# 设置列标签（顶部标题）
alg_display_names = ["Ours", "PPO", "Elastic"]
for col_idx, alg_name in enumerate(alg_display_names):
    axes[0, col_idx].set_title(alg_name)

# 设置底部行的x轴标签
for col_idx in range(3):
    axes[2, col_idx].set_xlabel("Time (s)")

# 添加主标题显示轨迹和速度信息
# fig.suptitle("Trajectory {} at v={}m/s - All Drones Performance".format(target_traj, target_velocity), 
#              fontsize=10, y=0.98)

# 设置y轴范围（可根据需要调整）
for col_idx in range(3):
    axes[0, col_idx].set_ylim(-0.2, 0.2)  # 水平误差
    axes[1, col_idx].set_ylim(-0.1, 0.7)  # 垂直误差  
    axes[2, col_idx].set_ylim(-1, 1)       # 距离误差

# 添加图例（只为第一列的第一个子图添加，显示前5个飞机）
handles, labels = axes[0,0].get_legend_handles_labels()
FigFon.set_shared_legend(handles=handles,labels=labels, )

current_folder = os.path.dirname(os.path.abspath(__file__))
save_folder = current_folder.split("obj_track")[0] + "obj_track/plots/"

fig.savefig("{}all_drones_{}_{}.png".format(save_folder, target_traj, target_velocity.replace(".", "_")))

# plt.show()
