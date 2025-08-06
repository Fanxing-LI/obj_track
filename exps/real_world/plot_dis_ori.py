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
vs = ["0.5","1.0", "1.5"]
vs = ["1.0"]
vs = ["0.5", "1.0","1.5",  "2.0"]
# vs = ["0.5"]

mean_oris = []
std_oris = []
mean_distances = []
std_ditances = []
fig, axes = FigFon.get_figure_axes(SubFigSize=(1, 4), Column=2)

for i, v in enumerate(vs):
    data = th.load(f"saved/objTracking/test/{v}.pth")
    target_positions = th.stack([th.stack(tar) for tar in data["target_all"]]).squeeze()
    ori = (th.stack([obs["center"] for obs in data["obs_all"]])[:,:,:2]-0.5).abs()
    mean_ori, std_ori = ori.mean(dim=1), ori.std(dim=1)
    t = th.stack(data["t"])[:,0]
    states = th.stack(data["state_all"])[:,:,:3]
    distance = (target_positions-states).norm(dim=-1)
    mean_distance, std_distance = distance.mean(dim=-1), distance.std(dim=-1)
    mean_oris.append(mean_ori)
    std_oris.append(std_ori)
    mean_distances.append(mean_distance)
    std_ditances.append(std_distance)
    # 创建3D图形
    plot_with_error_band(axes[0], t, mean_ori[:, 0]-0.5, std_ori[:, 0])
    plot_with_error_band(axes[1], t, mean_ori[:, 1]-0.5, std_ori[:, 1])
    plot_with_error_band(axes[2], t, mean_distance-3, std_distance)

axes[0].set_title(f"Horizontal Error")
axes[1].set_title(f"Vertical Error")
axes[2].set_title(f"Distance Error(m)")

axes[3].plot(target_positions[:,0,0], target_positions[:,0,1])
axes[3].axis("equal")

# set ylim a little larger than current ylim
current_ylim = axes[3].get_ylim()
axes[3].set_ylim(current_ylim[0] - 0.1, current_ylim[1] + 0.1)


axes[3].set_title(f"Target Trajectory")
# axes[3].grid(False)
# axes[3].set_xticks([])
# axes[3].set_yticks([])
# axes[3].set_xlabel('')
# axes[3].set_ylabel('')
# axes[3].axis("off")
#
# axes[3].patch.set_visible(False)
# hide grid and axes of axes3

plt.show()
