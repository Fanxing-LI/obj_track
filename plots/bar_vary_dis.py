import os, sys
import json
from VisFly.utils.common import load_yaml_config
import torch as th
from VisFly.utils.FigFashion.FigFashion import FigFon
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from VisFly.utils.FigFashion.colors import colorsets

colors = colorsets["Modern Scientific"]

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
                ori = th.stack([cen for cen in data["center_all"]])[:,:,:2]
                x_ori, y_ori = ori[...,0], ori[...,1]
                mean_x_ori, mean_y_ori = x_ori.abs().mean(), y_ori.abs().mean()
                mean_distance = (distance.mean()-float(dis)).abs()

                metrics[alg_i, v_i, traj_i, dis_i, 0] = mean_x_ori
                metrics[alg_i, v_i, traj_i, dis_i, 1] = mean_y_ori
                metrics[alg_i, v_i, traj_i, dis_i, 2] = mean_distance

                std_metrics[alg_i, v_i, traj_i, dis_i, 0] = x_ori.std(dim=0).mean()
                std_metrics[alg_i, v_i, traj_i, dis_i, 1] = y_ori.std(dim=0).mean()
                std_metrics[alg_i, v_i, traj_i, dis_i, 2] = distance.std(dim=0).mean()

        # 沿速度和轨迹维度取平均
mean_metrics = metrics.mean(dim=(1,2))  # shape: (len(algs), len(dises), 3)
std_metrics = std_metrics.mean(dim=(1,2))  # shape: (len(algs), len(dises), 3)

# 创建3×2的子图布局 (3个距离 × 2个指标)
fig_bar, axes_bar = FigFon.get_figure_axes(SubFigSize=(3, 2), Column=1, HeightScale=0.6, share_legend=True)

# 定义标签和颜色
metric_labels = ["$\overline{||e_H||}$", "$\overline{||e_V||}$", "$\overline{||e_d||}$"]
alg_labels = ["Ours", "Elastic"]
colors_alg = [colors[0], colors[2]] 

# 设置柱状图参数
x_positions = range(3)  # H, V, d 三个指标
bar_width = 0.25
offset = 0.15
x1 = [x - offset for x in x_positions]
x2 = [x + offset for x in x_positions]

# 为每个距离和指标组合创建子图
for dis_i, dis in enumerate(dises):
    # 左图：均值误差
    for alg_i, alg in enumerate(algs):
        mean_values = mean_metrics[alg_i, dis_i, :]  # 当前距离的均值
        if alg_i == 0:
            axes_bar[dis_i, 0].bar(x1, mean_values, bar_width, label=alg_labels[alg_i], 
                                  color=colors_alg[alg_i], alpha=0.8)
        else:
            axes_bar[dis_i, 0].bar(x2, mean_values, bar_width, label=alg_labels[alg_i], 
                                  color=colors_alg[alg_i], alpha=0.8)
    
    # 设置左图属性
    axes_bar[dis_i, 0].set_ylabel(f'$d={dis}m$')
    if dis_i == 0:  # 第一行设置标题
        axes_bar[dis_i, 0].set_title('Mean Absolute Error')
    axes_bar[dis_i, 0].set_xticks(x_positions)
    axes_bar[dis_i, 0].set_xticklabels(["$\overline{||e_H||}$", "$\overline{||e_V||}$", "$\overline{||e_d||}$"])
    axes_bar[dis_i, 0].set_xlim(-0.5, 2.5)
    axes_bar[dis_i, 0].grid(True, alpha=0.3)
    
    # 右图：标准差
    for alg_i, alg in enumerate(algs):
        std_values = std_metrics[alg_i, dis_i, :]  # 当前距离的标准差
        if alg_i == 0:
            axes_bar[dis_i, 1].bar(x1, std_values, bar_width, label=alg_labels[alg_i], 
                                  color=colors_alg[alg_i], alpha=0.8)
        else:
            axes_bar[dis_i, 1].bar(x2, std_values, bar_width, label=alg_labels[alg_i], 
                                  color=colors_alg[alg_i], alpha=0.8)
    
    # 设置右图属性
    if dis_i == 0:  # 第一行设置标题
        axes_bar[dis_i, 1].set_title('Standard Deviation')
    axes_bar[dis_i, 1].set_xticks(x_positions)
    axes_bar[dis_i, 1].set_xticklabels(["$\overline{\delta_{e_H}}$", "$\overline{\delta_{e_V}}$", "$\overline{\delta_{e_d}}$"])
    axes_bar[dis_i, 1].set_xlim(-0.5, 2.5)
    axes_bar[dis_i, 1].grid(True, alpha=0.3)

# 设置共享图例
handles, labels = axes_bar[0, 0].get_legend_handles_labels()
handles = [handles[0][0], handles[1][0]]
FigFon.set_shared_legend(handles, labels=["Ours", "Elastic"])
plt.show()

current_folder = os.path.dirname(os.path.abspath(__file__))
save_folder = current_folder.split("obj_track")[0] + "obj_track/plots/"
fig_bar.savefig(f"{save_folder}bar_vary_dis.png")

# plt.show()
