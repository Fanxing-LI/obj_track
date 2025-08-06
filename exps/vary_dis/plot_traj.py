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
    ax.plot(x, y, label=label, linewidth=2)

    # 绘制误差带（均值 ± 标准差）
    ax.fill_between(x,
                     y - std,
                     y + std,
                     alpha=0.3,
)

args = parse_args().parse_args()

label = args.velocity

# 获取目标位置数据（假设target_all包含位置信息）
vs = ["1","8", "B"]
# vs = ["1.0"]

all_x_data = []
all_y_data = []


for i, v in enumerate(vs):
    fig = plt.figure(figsize=(12, 6))
    axes = fig.add_subplot(111)
    data = th.load(f"saved/objTracking/test/{v}.pth")
    target_positions = th.stack([th.stack(tar) for tar in data["target_all"]]).squeeze()
    all_x_data.append(target_positions[:, i, 0])
    all_y_data.append(target_positions[:, i, 1])
    axes.plot(target_positions[:, i, 0], target_positions[:, i, 1], linewidth=2)
    # axes.plot(traj[:, 0], traj[:, 2], linewidth=2, alpha=0.5)
    # axes.plot(traj[:, 1], traj[:, 2], linewidth=2, alpha=0.5)
    y_min, y_max = axes.get_ylim()
    axes.set_ylim(y_min - 0.1, y_max + 0.1)
    x_min, x_max = axes.get_xlim()
    axes.set_xlim(x_min - 0.1, x_max + 0.1)
    axes.grid("off")
    # axis equal
    axes.set_aspect('equal')

    # 隐藏所有内容
    axes.set_xticks([])
    axes.set_yticks([])

    # current ylim

    axes.axis('off')
    plt.show()
    fig.savefig(f"traj_{i}.png", dpi=300, bbox_inches='tight')

#get max_and_min_x
# x_max = [max(x) for x in all_x_data]
# x_min = [min(x) for x in all_x_data]
# dif = th.tensor(x_max)-th.tensor(x_min)
# y_max = [max(y) for y in all_y_data]
# y_min = [min(y) for y in all_y_data]
# dif_y = th.tensor(y_max)-th.tensor(y_min)
#
# # global_x_min, global_x_max = min(all_x_data), max(all_x_data)
# # global_y_min, global_y_max = min(all_y_data), max(all_y_data)
# #
#
#
# for i, traj in enumerate(target_positions):
#     traj = traj.cpu().numpy()
#


