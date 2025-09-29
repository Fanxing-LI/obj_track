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

def compute_windowed_variance(data, window_size=10):
    """
    计算滑动窗口方差，返回每个时间点局部方差的平均值
    Args:
        data: shape (T, N) 其中 T 是时间步数，N 是飞机数量
        window_size: 滑动窗口大小
    Returns:
        平均方差值
    """
    T, N = data.shape
    variances = []
    
    for t in range(T):
        # 确定窗口范围
        start_idx = max(0, t - window_size // 2)
        end_idx = min(T, t + window_size // 2 + 1)
        
        # 在窗口内计算每个飞机的方差，然后求平均
        window_data = data[start_idx:end_idx, :]  # shape: (window_len, N)
        if window_data.shape[0] > 1:  # 需要至少2个点才能计算方差
            # 对每个飞机在窗口内的方差求平均
            window_variance = window_data.var(dim=0).mean()  # 先对时间维度求方差，再对飞机维度求平均
            variances.append(window_variance.item())
        else:
            variances.append(0.0)
    
    # 返回所有时间点方差的平均值
    return np.mean(variances)

FigFon.set_fashion("IEEE")
save_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + f"/saved/objTracking/"

def parse_args():
    parser = argparse.ArgumentParser(description='Run experiments', add_help=False)
    parser.add_argument('--comment', '-c', type=str, default="std")
    parser.add_argument("--train", "-t", type=int, default=1)
    parser.add_argument("--velocity", "-v", type=str, default="1.0", )
    return parser

args = parse_args().parse_args()

label = args.velocity

# 获取目标位置数据（算法维度替换距离维度）
algs = [
    "SHAC",
    "PPO",
    "elastic",
]

vs = ["0.5", "1.0","1.5",  "2.0", "2.5", "3.0",]

trajs = ["8", "B", "D"]

mean_oris = []
std_oris = []
mean_distances = []
std_ditances = []
file_folder = (os.path.dirname(os.path.abspath(sys.argv[0])).split("obj_track")[0]
               + f"/obj_track/exps/vary_v/saved/objTracking/test/")

std_metrics = th.zeros((len(trajs), len(vs), len(algs), 3))
metrics = th.zeros((len(trajs), len(vs), len(algs), 3))

for traj_i, traj in enumerate(trajs):
    for v_i, v in enumerate(vs):
        for alg_i, alg in enumerate(algs):
            data = th.load(file_folder+f"{traj}_{v}_{alg}_Dis3.0.pth")

            distance = th.stack([tar for tar in data["target_dis_all"]])
            ori = th.stack([cen for cen in data["center_all"]])[:,:,:2]
            mean_ori, std_ori = ori.abs().mean(dim=1), ori.abs().std(dim=1)
            x_ori, y_ori = ori[...,0], ori[...,1]
            if all(th.stack(data["t"])[:, 0] == 0):
                t = th.arange(0, len(data["t"])) * 0.03
            else:
                t = th.stack(data["t"])[:, 0]
            states = th.stack(data["state_all"])[:,:,:3]
            # distance = (target_positions-states).norm(dim=-1)
            distance = distance - 3
            mean_distance, std_distance = distance.abs().mean(), distance.std(dim=0).mean()

            # 计算平均值和标准差指标
            mean_x_ori, mean_y_ori = x_ori.abs().mean(), y_ori.abs().mean()

            metrics[traj_i, v_i, alg_i, 0] = mean_x_ori
            metrics[traj_i, v_i, alg_i, 1] = mean_y_ori
            metrics[traj_i, v_i, alg_i, 2] = mean_distance

            # 使用滑动窗口方差计算方法
            std_metrics[traj_i, v_i, alg_i, 0] = compute_windowed_variance(x_ori, window_size=15)
            std_metrics[traj_i, v_i, alg_i, 1] = compute_windowed_variance(y_ori, window_size=15)
            std_metrics[traj_i, v_i, alg_i, 2] = compute_windowed_variance(distance, window_size=15)

            test = 1

std_metrics = std_metrics.mean(dim=0)
metrics = metrics.mean(dim=0)

alg_labels = algs  # 使用算法名称作为标签
v_labels = [f"{v} m/s" for v in vs]

fig_heatmap, axes_heatmap = FigFon.get_figure_axes(SubFigSize=(2, 3), Column=1, HeightScale=1.5,
                                                   sharey=True, sharex=True)

metric_names = ["$\\overline{||{e}_H||}$", "$\\overline{||{e}_V||}$", "$\\overline{||{e}_d||}$"]
std_metrics_names = ["$\\delta_{e_H}$", "$\\delta_{e_V}$", "$\\delta_{e_d}$"]
alg_labels = ["Ours", "PPO", "Elastic"]
v_labels = [f"{v}" for v in vs]


# 为每个指标创建一个子图
for metric_idx in range(3):
    # 提取当前指标在所有速度和算法下的数据
    data = metrics[:, :, metric_idx].numpy()  # shape: (len(vs), len(algs))
    std_data = std_metrics[:, :, metric_idx].numpy()  # shape: (len(vs), len(algs))

    # 使用原始数据绘制热力图（不需要手动归一化）
    im = axes_heatmap[0,metric_idx].imshow(data, cmap='viridis', aspect='auto')
    std_im = axes_heatmap[1,metric_idx].imshow(std_data, cmap='viridis', aspect='auto')

    # 设置坐标轴标签
    axes_heatmap[0,metric_idx].set_xticks(range(len(algs)))
    axes_heatmap[0,metric_idx].set_xticklabels(alg_labels)
    axes_heatmap[0,metric_idx].set_yticks(range(len(vs)))
    axes_heatmap[0,metric_idx].set_yticklabels(v_labels)
    axes_heatmap[0,metric_idx].tick_params(axis='both', which='both', length=0)

    axes_heatmap[1, metric_idx].set_xticks(range(len(algs)))
    axes_heatmap[1, metric_idx].set_xticklabels(alg_labels)
    axes_heatmap[1, metric_idx].set_yticks(range(len(vs)))
    axes_heatmap[1, metric_idx].set_yticklabels(v_labels)
    axes_heatmap[1, metric_idx].tick_params(axis='both', which='both', length=0)

    # 添加数值标注
    for i in range(len(vs)):  # 速度维度
        for j in range(len(algs)):  # 算法维度
            text = axes_heatmap[0,metric_idx].text(j, i, f'{data[i, j]:.3f}',
                                                 ha="center", va="center", color="white")
            std_text = axes_heatmap[1,metric_idx].text(j, i, f'{std_data[i, j]:.3f}',
                                                 ha="center", va="center", color="white")
    axes_heatmap[0,metric_idx].set_title(f"{metric_names[metric_idx]}")
    axes_heatmap[1, metric_idx].set_title(f"{std_metrics_names[metric_idx]}")
    axes_heatmap[1,metric_idx].set_xlabel("Algorithm")
    if metric_idx == 0:
        axes_heatmap[0,metric_idx].set_ylabel("$v$ (m/s)")
        axes_heatmap[1, metric_idx].set_ylabel("$v$ (m/s)")
    axes_heatmap[0,metric_idx].grid(False)
    axes_heatmap[1, metric_idx].grid(False)

plt.show()


# FigFon.set_shared_legend(handles, algs)
# for i in range(len(vs)):
#     axes[0,i].set_title(f"$v = {vs[i]}$ m/s")

current_folder = os.path.dirname(os.path.abspath(__file__))
save_folder = current_folder.split("obj_track")[0] + "obj_track/plots/"
fig_heatmap.savefig(f"{save_folder}mesh_vary_v_std_adjacent.png")

plt.show()
