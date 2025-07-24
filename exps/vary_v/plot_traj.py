import os, sys
import json
from VisFly.utils.common import load_yaml_config
import torch as th
from VisFly.utils.FigFashion.FigFashion import FigFon
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

FigFon.set_fashion("IEEE")
save_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + f"/saved/objTracking/"

label = "0.5"
data = th.load(f"saved/objTracking/test/{label}.pth")

# 获取目标位置数据（假设target_all包含位置信息）
target_positions = th.stack([th.stack(tar) for tar in data["target_all"]]).squeeze()[:,0,:]
t = th.stack(data["t"])[:,0]

# 创建3D图形
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 如果target_positions是3D位置数据
if target_positions.shape[-1] >= 3:
    ax.plot(target_positions[:, 0], target_positions[:, 1], target_positions[:, 2],
            label="Target Trajectory", linewidth=2)

    # 标记起点和终点
    ax.scatter(target_positions[0, 0], target_positions[0, 1], target_positions[0, 2],
               color='green', s=100, label='Start')
    ax.scatter(target_positions[-1, 0], target_positions[-1, 1], target_positions[-1, 2],
               color='red', s=100, label='End')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Target Trajectory')
    ax.legend()
    ax.set_box_aspect([1, 1, 1])

    ax.view_init(elev=80, azim=0)
fig2, axes2 = FigFon.get_figure_axes(SubFigSize=(1, 1), Column=1)
axes2.plot(target_positions[:, 0], target_positions[:, 1], label="Target Trajectory")
axes2.set_xlabel('X (m)')
axes2.set_ylabel('Y (m)')
axes2.set_title('2D Projection of Target Trajectory')

plt.show()