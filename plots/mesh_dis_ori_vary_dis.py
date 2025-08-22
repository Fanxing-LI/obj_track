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
# vs = ["0.5"]

mean_oris = []
std_oris = []
mean_distances = []
std_ditances = []

file_folder = (os.path.dirname(os.path.abspath(sys.argv[0])).split("obj_track")[0]
               + f"/obj_track/exps/vary_v/saved/objTracking/test/")
# 为每个指标计算全局的min和max值

metrics = th.zeros((len(vs), len(dises), 3))
std_metrics = th.zeros((len(vs), len(dises), 3))
for v_i, v in enumerate(vs):
    for dis_i, dis in enumerate(dises):
        # if dis_i == 2:
        #     continue
        try:
            data = th.load(file_folder+f"8_{v}_SHAC_Dis{dis}.pth")
        except FileNotFoundError:
            path = file_folder+f"8_{v}_SHAC.pth"
            data = th.load(path)
        distance = th.stack([tar for tar in data["target_dis_all"]])
        ori = th.stack([cen for cen in data["center_all"]])[:,:,:2]
        x_ori, y_ori = ori[...,0], ori[...,1]
        mean_x_ori, mean_y_ori = x_ori.abs().mean(), y_ori.abs().mean()
        mean_distance = (distance.mean()-float(dis)).abs()

        metrics[v_i, dis_i,0] = mean_x_ori
        metrics[v_i, dis_i,1] = mean_y_ori
        metrics[v_i, dis_i,2] = mean_distance

        std_metrics[v_i, dis_i,0] = x_ori.abs().std()
        std_metrics[v_i, dis_i,1] = y_ori.abs().std()
        std_metrics[v_i, dis_i,2] = distance.std()

        # 创建3D图形

# fig_heatmap, axes_heatmap = FigFon.get_figure_axes(SubFigSize=(2,3), Column=1, HeightScale=1.3,sharey=True, sharex=True)
# axes_heatmap = axes_heatmap.flatten()  # 展平为一维数组便于索引

metric_names = ["Horizontal Error", "Vertical Error", "Distance Error (m)"]
dis_labels = [f"{dis}m" for dis in dises]
v_labels = [f"{v} m/s" for v in vs]

# global_vmin = []
# global_vmax = []
# for metric_idx in range(3):
#     metric_data = metrics[:, :, metric_idx].numpy()  # 所有速度和距离下的该指标数据
#     global_vmin.append(metric_data.min())
#     global_vmax.append(metric_data.max())

fig_heatmap, axes_heatmap = FigFon.get_figure_axes(SubFigSize=(2, 3), Column=1, HeightScale=2.0,
                                                   sharey=True, sharex=True)

metric_names = ["$\\bar{e}_H$", "$\\bar{e}_V$", "$\\bar{e}_d$"]
std_metrics_names = ["$\\delta_{e_H}$","$\\delta_{e_V}$", "$\\delta_{e_d}$"]
dis_labels = [f"{dis}" for dis in dises]
v_labels = [f"{v}" for v in vs]

# 为每个指标创建一个子图
for metric_idx in range(3):
    # 提取当前指标在所有速度和距离下的数据
    data = metrics[:, :, metric_idx].numpy()  # shape: (len(vs), len(dises))
    std_data = std_metrics[:, :, metric_idx].numpy()  # shape: (len(vs), len(dises))

    # 使用原始数据绘制热力图（不需要手动归一化）
    im = axes_heatmap[0,metric_idx].imshow(data, cmap='viridis_r', aspect='auto')
    std_im = axes_heatmap[1,metric_idx].imshow(std_data, cmap='viridis_r', aspect='auto')

    # 设置坐标轴标签
    axes_heatmap[0,metric_idx].set_xticks(range(len(dises)))
    axes_heatmap[0,metric_idx].set_xticklabels(dis_labels)
    axes_heatmap[0,metric_idx].set_yticks(range(len(vs)))
    axes_heatmap[0,metric_idx].set_yticklabels(v_labels)
    axes_heatmap[0,metric_idx].tick_params(axis='both', which='both', length=0)

    axes_heatmap[1, metric_idx].set_xticks(range(len(dises)))
    axes_heatmap[1, metric_idx].set_xticklabels(dis_labels)
    axes_heatmap[1, metric_idx].set_yticks(range(len(vs)))
    axes_heatmap[1, metric_idx].set_yticklabels(v_labels)
    axes_heatmap[1, metric_idx].tick_params(axis='both', which='both', length=0)


    # 添加数值标注
    for i in range(len(vs)):  # 速度维度
        for j in range(len(dises)):  # 距离维度
            text = axes_heatmap[0,metric_idx].text(j, i, f'{data[i, j]:.3f}',
                                                 ha="center", va="center", color="white")
            std_text = axes_heatmap[1,metric_idx].text(j, i, f'{std_data[i, j]:.3f}',
                                                 ha="center", va="center", color="white")
    axes_heatmap[0,metric_idx].set_title(f"{metric_names[metric_idx]}")
    axes_heatmap[1, metric_idx].set_title(f"{std_metrics_names[metric_idx]}")
    axes_heatmap[1,metric_idx].set_xlabel("Distance (m)")
    if metric_idx == 0:
        axes_heatmap[0,metric_idx].set_ylabel("Velocity (m/s)")
        axes_heatmap[1, metric_idx].set_ylabel("Velocity (m/s)")
    axes_heatmap[0,metric_idx].grid(False)
    axes_heatmap[1, metric_idx].grid(False)

plt.tight_layout()
plt.show()




# FigFon.set_shared_legend(handles, algs)
# for i in range(len(vs)):
#     axes[0,i].set_title(f"$v = {vs[i]}$ m/s")


current_folder = os.path.dirname(os.path.abspath(__file__))
save_folder = current_folder.split("obj_track")[0] + "obj_track/plots/"
fig_heatmap.savefig(f"{save_folder}mesh_vary_dis.png")

plt.show()
