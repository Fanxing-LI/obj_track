import os, sys
import json
from VisFly.utils.common import load_yaml_config
import torch as th
from VisFly.utils.FigFashion.FigFashion import FigFon
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from VisFly.utils.FigFashion.colors import colorsets
from plots.plot_dis_ori_vary_traj import plot_with_error_band

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

# 获取目标位置数据（假设target_all包含位置信息）
algs = [
    "SHAC",
    "PPO",
    "elastic",
]
vs = ["0.5","1.0", "1.5"]

vs = ["0.5", "1.0","1.5",  "2.0", "2.5", "3.0",]
# vs = ["0.5"]

mean_oris = []
std_oris = []
mean_distances = []
std_ditances = []
fig, axes = FigFon.get_figure_axes(SubFigSize=(3, len(vs)), Column=2, sharey=True, share_legend=True)
axes = axes.reshape(3, len(vs))
file_folder = (os.path.dirname(os.path.abspath(sys.argv[0])).split("obj_track")[0]
               + f"/obj_track/exps/vary_v/saved/objTracking/test/")

stds = th.zeros(3,len(vs), len(algs))

for v_i, v in enumerate(vs):
    for alg_i, alg in enumerate(algs):
        data = th.load(file_folder+f"8_{v}_{alg}_Dis3.0.pth")

        distance = th.stack([tar for tar in data["target_dis_all"]])
        ori = th.stack([cen for cen in data["center_all"]])[:,:,:2]
        mean_ori, std_ori = ori.abs().mean(dim=1), ori.std(dim=1)
        max_ori, min_ori = ori.abs().max(dim=1).values, ori.abs().min(dim=1).values

        if all(th.stack(data["t"])[:, 0] == 0):
            t = th.arange(0, len(data["t"])) * 0.03
        else:
            t = th.stack(data["t"])[:, 0]
        states = th.stack(data["state_all"])[:,:,:3]
        # distance = (target_positions-states).norm(dim=-1)
        distance = distance - 3
        mean_distance, std_distance = distance.abs().mean(dim=-1), distance.std(dim=-1)
        max_distance, min_distance = distance.abs().max(dim=-1).values, distance.abs().min(dim=-1).values

        mean_oris.append(mean_ori)
        std_oris.append(std_ori)
        mean_distances.append(mean_distance)
        std_ditances.append(std_distance)
        # 创建3D图形
        plot_with_error_band(axes[0, v_i], t, mean_ori[:, 0], max_ori[:, 0], min_ori[:, 0])
        plot_with_error_band(axes[1, v_i], t, mean_ori[:, 1], max_ori[:, 1], min_ori[:, 1])
        plot_with_error_band(axes[2, v_i], t, mean_distance, max_distance, min_distance)
        axes[2, v_i].set_ylim(0,1)

axes[0,0].set_ylabel("$||\\delta_{H}||$")
axes[1,0].set_ylabel("$||\\delta_{V}||$")
axes[2,0].set_ylabel("$||\\delta_{d}||$")

handles, labels = axes[0,0].get_legend_handles_labels()

FigFon.set_shared_legend(handles, algs)
for i in range(len(vs)):
    axes[0,i].set_title(f"$v = {vs[i]}$ m/s")


current_folder = os.path.dirname(os.path.abspath(__file__))
save_folder = current_folder.split("obj_track")[0] + "obj_track/plots/"
fig.savefig(f"{save_folder}vary_v.png")

plt.show()
