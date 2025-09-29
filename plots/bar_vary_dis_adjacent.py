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
save_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + f"/saved/objTracking/"


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

def parse_args():
    parser = argparse.ArgumentParser(description='Run experiments', add_help=False)
    parser.add_argument('--comment', '-c', type=str, default="std")
    parser.add_argument("--train", "-t", type=int, default=1)
    parser.add_argument("--velocity", "-v", type=str, default="1.0", )
    return parser

args = parse_args().parse_args()

label = args.velocity


vs = ["0.5","1.0", "1.5", "2.0","2.5","3.0"]

dises = ["1.5", "3.0","4.5"]

algs = ["ours", "elastic"]

trajs = ["8", "B", "D"]
# trajs = ["8"]
# vs = ["0.5"]

mean_oris = []
std_oris = []
mean_distances = []
std_ditances = []

file_folder = (os.path.dirname(os.path.abspath(sys.argv[0])).split("obj_track")[0]
               + f"/obj_track/exps/vary_v/saved/objTracking/test/")
# 为每个指标计算全局的min和max值

metrics = th.zeros((len(algs),len(vs),len(trajs), len(dises), 3))
std_metrics = th.zeros((len(algs),len(vs),len(trajs), len(dises), 3))
for alg_i, alg in enumerate(algs):
    for v_i, v in enumerate(vs):
        for dis_i, dis in enumerate(dises):
            for traj_i, traj in enumerate(trajs):
                # if dis_i == 2:
                #     continue
                try:
                    if alg == "ours":
                        data = th.load(file_folder+f"{traj}_{v}_SHAC_Dis{dis}.pth")
                    else:  # elastic
                        data = th.load(file_folder+f"{traj}_{v}_elastic_Dis{dis}.pth")
                except FileNotFoundError:
                    if alg == "ours":
                        path = file_folder+f"{traj}_{v}_SHAC.pth"
                    else:  # elastic
                        path = file_folder+f"{traj}_{v}_elastic.pth"
                    data = th.load(path)
                distance = th.stack([tar for tar in data["target_dis_all"]])
                distance = distance - float(dis)
                ori = th.stack([cen for cen in data["center_all"]])[:,:,:2]
                x_ori, y_ori = ori[...,0], ori[...,1]
                mean_x_ori, mean_y_ori = x_ori.abs().mean(), y_ori.abs().mean()
                mean_distance = distance.abs().mean()

                metrics[alg_i, v_i, traj_i, dis_i, 0] = mean_x_ori
                metrics[alg_i, v_i, traj_i, dis_i, 1] = mean_y_ori
                metrics[alg_i, v_i, traj_i, dis_i, 2] = mean_distance
                
                std_metrics[alg_i, v_i, traj_i, dis_i, 0] = compute_windowed_variance(x_ori, window_size=15)
                std_metrics[alg_i, v_i, traj_i, dis_i, 1] = compute_windowed_variance(y_ori, window_size=15)
                std_metrics[alg_i, v_i, traj_i, dis_i, 2] = compute_windowed_variance(distance, window_size=15)

                test = 1
        # 沿速度和轨迹维度取平均
mean_metrics = metrics.mean(dim=(1,2))  # shape: (len(algs), len(dises), 3)
std_metrics = std_metrics.mean(dim=(1,2))  # shape: (len(algs), len(dises), 3)

# 创建3×2的子图布局 (3个指标 × 2个类型：均值和标准差)
fig_bar, axes_bar = FigFon.get_figure_axes(SubFigSize=(3, 2), Column=1, HeightScale=0.6, share_legend=True)

# 定义标签和颜色
metric_labels = ["$\overline{||e_H||}$", "$\overline{||e_V||}$", "$\overline{||e_d||}$"]
metric_labels_right = ["$\overline{\delta{e_H}}$", "$\overline{\delta{e_V}}$", "$\overline{\delta{e_d}}$"]
alg_labels = ["Ours", "Elastic"]
colors_alg = [colors[0], colors[2]] 

# 设置柱状图参数
x_positions = range(len(dises))  # 3个距离
bar_width = 0.25
offset = 0.15
x1 = [x - offset for x in x_positions]
x2 = [x + offset for x in x_positions]

# 为每个指标创建两列子图（均值和标准差）
for metric_i, metric_label in enumerate(metric_labels):
    # 左图：均值误差
    for alg_i, alg in enumerate(algs):
        mean_values = mean_metrics[alg_i, :, metric_i]  # 当前指标在所有距离下的均值
        if alg_i == 0:
            axes_bar[metric_i, 0].bar(x1, mean_values, bar_width, label=alg_labels[alg_i], 
                                  color=colors_alg[alg_i], alpha=0.8)
        else:
            axes_bar[metric_i, 0].bar(x2, mean_values, bar_width, label=alg_labels[alg_i], 
                                  color=colors_alg[alg_i], alpha=0.8)
    
    # 设置左图属性
    axes_bar[metric_i, 0].set_ylabel(f'{metric_label}')
    axes_bar[metric_i, 1].set_ylabel(f'{metric_labels_right[metric_i]}')

    if metric_i == 0:  # 第一行设置标题
        axes_bar[metric_i, 0].set_title('Mean Absolute Error')
    axes_bar[metric_i, 0].set_xticks(x_positions)
    axes_bar[metric_i, 0].set_xticklabels([f"$d={dis}m$" for dis in dises])
    axes_bar[metric_i, 0].set_xlim(-0.5, len(dises)-0.5)
    axes_bar[metric_i, 0].grid(True, alpha=0.3)
    
    # 右图：标准差
    for alg_i, alg in enumerate(algs):
        std_values = std_metrics[alg_i, :, metric_i]  # 当前指标在所有距离下的标准差
        if alg_i == 0:
            axes_bar[metric_i, 1].bar(x1, std_values, bar_width, label=alg_labels[alg_i], 
                                  color=colors_alg[alg_i], alpha=0.8)
        else:
            axes_bar[metric_i, 1].bar(x2, std_values, bar_width, label=alg_labels[alg_i], 
                                  color=colors_alg[alg_i], alpha=0.8)
    
    # 设置右图属性
    if metric_i == 0:  # 第一行设置标题
        axes_bar[metric_i, 1].set_title('Mean Standard Deviation')
    axes_bar[metric_i, 1].set_xticks(x_positions)
    axes_bar[metric_i, 1].set_xticklabels([f"$d={dis}m$" for dis in dises])
    axes_bar[metric_i, 1].set_xlim(-0.5, len(dises)-0.5)
    axes_bar[metric_i, 1].grid(True, alpha=0.3)

# 设置共享图例
handles, labels = axes_bar[0, 0].get_legend_handles_labels()
handles = [handles[0][0], handles[1][0]]
FigFon.set_shared_legend(handles, labels=["Ours", "Elastic"])
# plt.show()

current_folder = os.path.dirname(os.path.abspath(__file__))
save_folder = current_folder.split("obj_track")[0] + "obj_track/plots/"
fig_bar.savefig(f"{save_folder}bar_vary_dis_adjacent.png")

# plt.show()
