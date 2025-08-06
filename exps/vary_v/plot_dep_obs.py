import os, sys
import json
from VisFly.utils.common import load_yaml_config
import torch as th
from VisFly.utils.FigFashion.FigFashion import FigFon
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

FigFon.set_fashion("IEEE")
save_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + f"/saved/objTracking/"

label = "1.0"
data = th.load(f"saved/objTracking/test/{label}.pth")

# 获取目标位置数据（假设target_all包含位置信息）
target_positions = th.stack([th.stack(tar) for tar in data["target_all"]]).squeeze()
state_obs = th.stack([obs["state"] for obs in data["obs_all"]]).squeeze()
t = th.stack(data["t"])[:,0]

# 创建3D图形
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(121)

# 如果target_positions是3D位置数据
ax.plot(t, state_obs[:,0])
ax = fig.add_subplot(122)
ax.plot(t, state_obs[:,1])


fig2, axes2 = FigFon.get_figure_axes(SubFigSize=(1, 1), Column=1)
axes2.plot(target_positions[:, 0], target_positions[:, 1], label="Target Trajectory")
axes2.set_xlabel('X (m)')
axes2.set_ylabel('Y (m)')
axes2.set_title('2D Projection of Target Trajectory')

plt.show()