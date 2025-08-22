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


def plot_with_error_band(ax, x, y, std):
    """
    x_data: x轴数据
    y_data_list: 包含多次实验结果的列表，每个元素是一次实验的y数据
    """
    # 转换为numpy数组便于计算

    # 计算均值和标准差

    # 绘制均值线
    ax.plot(x, y, label=label, linewidth=1)

    # 绘制误差带（均值 ± 标准差）
    ax.fill_between(x,
                     y - std,
                     y + std,
                     alpha=0.3,
)

args = parse_args().parse_args()

label = args.velocity

# 获取目标位置数据（假设target_all包含位置信息）
algs = [
    "SHAC",
    "PPO",
]

trajs = ["B", "D"]
vs = ["1.0","2.0"]

mean_oris = []
std_oris = []
mean_distances = []
std_ditances = []
fig, axes = FigFon.get_figure_axes(SubFigSize=(3, len(vs)*len(trajs)), Column=1, sharey=True)

file_folder = (os.path.dirname(os.path.abspath(sys.argv[0])).split("obj_track")[0]
               + f"/obj_track/exps/vary_v/saved/objTracking/test/")
for v_i , v in enumerate(vs):
    for traj_i, traj in enumerate(trajs):
        for alg_i, alg in enumerate(algs):
            data = th.load(file_folder+f"{traj}_{v}_{alg}_Dis3.0.pth")
            distance = th.stack([tar for tar in data["target_dis_all"]])
            ori = th.stack([cen for cen in data["center_all"]])[:,:,:2]
            mean_ori, std_ori = ori.mean(dim=1), ori.std(dim=1)
            t = th.stack(data["t"])[:,0]
            states = th.stack(data["state_all"])[:,:,:3]
            # distance = (target_positions-states).norm(dim=-1)
            mean_distance, std_distance = distance.mean(dim=-1), distance.std(dim=-1)
            mean_oris.append(mean_ori)
            std_oris.append(std_ori)
            mean_distances.append(mean_distance)
            std_ditances.append(std_distance)
            # 创建3D图形
            plot_with_error_band(axes[0, traj_i+v_i*2], t, mean_ori[:, 0], std_ori[:, 0])
            plot_with_error_band(axes[1, traj_i+v_i*2], t, mean_ori[:, 1], std_ori[:, 1])
            plot_with_error_band(axes[2, traj_i+v_i*2], t, mean_distance-3, std_distance)

axes[0,0].set_ylabel(f"Horizontal Error")
axes[1,0].set_ylabel(f"Vertical Error")
axes[2,0].set_ylabel(f"Distance Error (m)")

handles, labels = axes[0,0].get_legend_handles_labels()
FigFon.set_shared_legend(handles, algs)

for i in range(len(trajs)):
    axes[0,i].set_title(f"Trajectory {trajs[i]}")


current_folder = os.path.dirname(os.path.abspath(__file__))
save_folder = current_folder.split("obj_track")[0] + "obj_track/plots/"
fig.savefig(f"{save_folder}vary_traj.png")

plt.show()
