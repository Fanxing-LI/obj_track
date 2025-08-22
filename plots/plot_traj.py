import os, sys
import json
from VisFly.utils.common import load_yaml_config
import torch as th
from VisFly.utils.FigFashion.FigFashion import FigFon
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.collections import LineCollection


def compute_curvature_vector(x, y, t=None):
    """
    Compute curvature vector for a 2D trajectory.
    Args:
        x: 1D array-like, x coordinates
        y: 1D array-like, y coordinates
        t: 1D array-like, time or parameter (optional, will use arc length if None)
    Returns:
        curvature_magnitude: 1D numpy array, curvature magnitude at each point
        curvature_vector: 2D numpy array, curvature vector [kx, ky] at each point
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if t is None or np.all(t == t[0]):
        # Use cumulative arc length as parameter when t is constant
        dx_raw = np.diff(x, prepend=x[0])
        dy_raw = np.diff(y, prepend=y[0])
        arc_lengths = np.cumsum(np.sqrt(dx_raw ** 2 + dy_raw ** 2))
        t = arc_lengths/2
    else:
        t = np.asarray(t)

    # First and second derivatives
    dx = np.gradient(x, t)
    dy = np.gradient(y, t)
    ddx = np.gradient(dx, t)
    ddy = np.gradient(dy, t)

    # Curvature magnitude
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = (dx ** 2 + dy ** 2) ** 1.5
    denominator = np.where(denominator < 1e-12, 1e-12, denominator)
    curvature_magnitude = numerator / denominator

    # Unit tangent vector
    speed = np.sqrt(dx ** 2 + dy ** 2)
    speed = np.where(speed < 1e-12, 1e-12, speed)
    tx = dx / speed
    ty = dy / speed

    # Unit normal vector (perpendicular to tangent, pointing toward center of curvature)
    # Rotate tangent 90 degrees counterclockwise, then adjust sign based on curvature
    nx = -ty
    ny = tx

    # Adjust sign based on signed curvature
    signed_curvature = (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5
    nx = np.where(signed_curvature < 0, -nx, nx)
    ny = np.where(signed_curvature < 0, -ny, ny)

    # Curvature vector
    curvature_vector = np.column_stack([curvature_magnitude * nx, curvature_magnitude * ny])

    return curvature_magnitude, curvature_vector

FigFon.set_fashion("IEEE")
save_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + f"/saved/objTracking/"

labels = ["8", "B","D"]
fig, axeses = FigFon.get_figure_axes(SubFigSize=(1, 3), Column=2)

for j, label in enumerate(labels):
    axes = axeses[j]
    data = th.load(f"saved/objTracking/test/{label}_1.0_SHAC.pth")
    
    # 获取目标位置数据（假设target_all包含位置信息）
    target_positions = th.stack([tar for tar in data["target_dis_all"]]).squeeze()[:,0,:]
    t = th.stack(data["t"])[:,0]

    # 计算曲率向量
    curvature_mag, curvature_vec = compute_curvature_vector(
        target_positions[:, 0].cpu().numpy(),
        target_positions[:, 1].cpu().numpy()
    )
    
    # 动态缩放：大曲率用较短的线，小曲率用较长的线
    max_length = 1  # 最大线段长度
    min_length = 0.2  # 最小线段长度
    
    # 归一化曲率并反向映射（高曲率->短线段）
    norm_curvature = (curvature_mag - curvature_mag.min()) / (curvature_mag.max() - curvature_mag.min() + 1e-12)
    adaptive_scale = max_length - (max_length - min_length) * norm_curvature
    
    # 应用自适应缩放
    scaled_curvature_vec = curvature_vec * adaptive_scale.reshape(-1, 1)
    
    # 计算曲率线末端的位置 - 使用自适应缩放后的向量
    trajectory_points = np.column_stack([target_positions[:, 0].cpu().numpy(),
                                       target_positions[:, 1].cpu().numpy()])
    curvature_endpoints = trajectory_points + scaled_curvature_vec
    

    
    # 绘制原始轨迹
    axes.plot(target_positions[:, 0].cpu().numpy(),
              target_positions[:, 1].cpu().numpy(),
              'b-', linewidth=2, label='Target Trajectory')
    
    # 绘制曲率向量线段（每隔几个点画一次，避免过于密集）
    step = 15  # 每5个点画一次
    for i in range(0, len(trajectory_points), step):
        axes.plot([trajectory_points[i, 0], curvature_endpoints[i, 0]],
                  [trajectory_points[i, 1], curvature_endpoints[i, 1]],
                  'r-', linewidth=1, alpha=0.6)
    
    # 绘制曲率末端连成的包络线 - 实线
    axes.plot(curvature_endpoints[:, 0], curvature_endpoints[:, 1],
              'r-', linewidth=1, label='Curvature Envelope', alpha=0.8)
    
    axes.set_xlabel('X (m)')
    axes.set_ylabel('Y (m)')
    # axes.set_title('Target Trajectory with Curvature Vectors and Envelope')
    axes.axis("equal")
    if j == 2:
        axes.set_ylim([-5.5, 3.5])
    else:
        axes.set_ylim([-3.5, 2.5])
    # axes.legend()
    axes.grid(True, alpha=0.3)
    axes.set_title(f"Trajectory {label}")
    
current_folder = os.path.dirname(os.path.abspath(__file__))
save_folder = current_folder.split("obj_track")[0] + "obj_track/plots/"
fig.savefig(f"{save_folder}target_trajectory.png")
plt.show()
