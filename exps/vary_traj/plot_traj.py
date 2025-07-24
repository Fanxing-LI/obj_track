import os, sys
import json
from VisFly.utils.common import load_yaml_config
import torch as th
from VisFly.utils.FigFashion.FigFashion import FigFon
import matplotlib.pyplot as plt
FigFon.set_fashion("IEEE")
save_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + f"/saved/objTracking/"

label = "0.5"
data = th.load(f"saved/objTracking/test/{label}.pth")
state = th.tensor(data["state_all"])
target_dis = th.tensor(data["target_all"])
t = th.tensor(data["t"])
fig, axes = FigFon.get_figure_axes(SubFigSize=(1, 1), Column=1)
# axes.plot(t, target_dis)
# axes.set_xlabel("Time (s)")
# axes.set_ylabel("Distance to Target (m)")
# axes.set_ylim(0, 2)
# 3D plot trajectory
axes.plot(state[:, 0], state[:, 1], state[:, 2], label="Drone Trajectory")
plt.show()
pass