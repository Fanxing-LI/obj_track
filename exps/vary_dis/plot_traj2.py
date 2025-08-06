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

mean_oris = []
std_oris = []
mean_distances = []
std_ditances = []
fig, axes = FigFon.get_figure_axes(SubFigSize=(1, 3), Column=2)
# 先收集所有轨迹数据，计算全局范围
all_x_data = []
all_y_data = []

for i, v in enumerate(vs):
    data = th.load(f"saved/objTracking/test/{v}.pth")
    target_positions = th.stack([th.stack(tar) for tar in data["target_all"]]).squeeze()
    all_x_data.extend(target_positions[:, i, 0].tolist())
    all_y_data.extend(target_positions[:, i, 1].tolist())

# 计算全局范围
global_x_min, global_x_max = min(all_x_data), max(all_x_data)
global_y_min, global_y_max = min(all_y_data), max(all_y_data)

# 计算最大范围，确保所有图有相同的尺度
x_range = global_x_max - global_x_min
y_range = global_y_max - global_y_min
max_range = max(x_range, y_range)

# 计算中心点
x_center = (global_x_min + global_x_max) / 2
y_center = (global_y_min + global_y_max) / 2

# 添加边距
margin = max_range * 0.1
half_range = max_range / 2 + margin

# 分别绘制和保存每张图
for i, v in enumerate(vs):
    fig, ax = plt.subplots(figsize=(6, 6))  # 正方形画布

    data = th.load(f"saved/objTracking/test/{v}.pth")
    target_positions = th.stack([th.stack(tar) for tar in data["target_all"]]).squeeze()

    ax.plot(target_positions[:, i, 0], target_positions[:, i, 1], color=colors[0], linewidth=2)

    # 设置统一的坐标范围（所有图都使用相同范围）
    ax.set_xlim(x_center - half_range, x_center + half_range)
    ax.set_ylim(y_center - half_range, y_center + half_range)

    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(f'trajectory_{v}.png', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形释放内存

print("所有轨迹图已保存完成")