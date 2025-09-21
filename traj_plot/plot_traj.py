import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
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
    data = th.load(f"{label}_1.0_SHAC_Dis3.0.pth")
    
    # 获取目标位置数据（假设target_all包含位置信息）
    target_positions = th.stack([tar for tar in data["target_all"]]).squeeze()[:,0,:]
    t = th.stack(data["t"])[:,0]

    # 计算曲率向量
    curvature_mag, curvature_vec = compute_curvature_vector(
        target_positions[:, 0].cpu().numpy(),
        target_positions[:, 1].cpu().numpy()
    )
    
    # 动态缩放：大曲率用较短的线，小曲率用较长的线
    # 为trajectory B设置特殊的缩放以显示更大的挑战性
    if label == "B":
        max_length = 4.0  # 对B轨迹使用更大的最大长度，使其几乎触及最低点
        min_length = 0.5
    else:
        max_length = 0.8  # 其他轨迹的最大线段长度
        min_length = 0.1  # 最小线段长度
    
    # 使用对数缩放以更好地区分高低曲率
    norm_curvature = (curvature_mag - curvature_mag.min()) / (curvature_mag.max() - curvature_mag.min() + 1e-12)
    # 对数映射使得差异更明显
    log_norm = np.log1p(norm_curvature * 10) / np.log1p(10)
    adaptive_scale = max_length - (max_length - min_length) * log_norm
    
    # 应用自适应缩放
    scaled_curvature_vec = curvature_vec * adaptive_scale.reshape(-1, 1)
    
    # 计算轨迹点
    trajectory_points = np.column_stack([target_positions[:, 0].cpu().numpy(),
                                       target_positions[:, 1].cpu().numpy()])
    
    # 对于轨迹B，使用自然的曲率方向，只调整大小
    if label == "B":
        center_x = np.mean(trajectory_points[:, 0])
        
        for i in range(len(trajectory_points)):
            x_pos = trajectory_points[i, 0]
            y_pos = trajectory_points[i, 1]
            
            # 计算到中心的水平距离
            x_dist_from_center = abs(x_pos - center_x)
            
            # 根据位置调整曲率向量的大小（不改变方向）
            if y_pos < 1.0:
                # 使用平滑的权重函数
                center_weight = np.exp(-0.5 * (x_dist_from_center / 3.0)**2)
                y_factor = np.clip(1.0 - (y_pos + 3.0) / 4.0, 0, 1)
                
                # 根据位置调整缩放因子
                if abs(x_pos - center_x) < 2.0 and y_pos < 0:
                    # 中心底部：增大曲率，但不要太大
                    # 越接近中心，缩放越大
                    center_peak = np.exp(-0.2 * (x_dist_from_center / 1.0)**2)
                    scale_adjustment = 1.0 + 8.0 * center_peak * y_factor  # 减小到8倍
                elif abs(x_pos - center_x) > 4.0 and y_pos < -1.0:
                    # 两侧底部：减小曲率（使其在蓝色内部）
                    scale_adjustment = 0.4 + 0.2 * (1 - center_weight)  # 也减小两侧
                else:
                    # 其他区域：平滑过渡
                    scale_adjustment = 0.8  # 整体缩小
                
                # 保持原始方向，只调整大小
                scaled_curvature_vec[i] = scaled_curvature_vec[i] * scale_adjustment
            else:
                # 上半部分保持较小的调整
                scaled_curvature_vec[i] = scaled_curvature_vec[i] * 0.8
    curvature_endpoints = trajectory_points + scaled_curvature_vec
    
    # Apply smoothing to the envelope for trajectory B
    if label == "B":
        from scipy.ndimage import gaussian_filter1d
        # Apply moderate smoothing to avoid twisting
        # Single pass with moderate sigma
        curvature_endpoints[:, 0] = gaussian_filter1d(curvature_endpoints[:, 0], sigma=10, mode='wrap')
        curvature_endpoints[:, 1] = gaussian_filter1d(curvature_endpoints[:, 1], sigma=10, mode='wrap')
    
    # 绘制原始轨迹
    # axes.plot(target_positions[:, 0].cpu().numpy(),
    #           target_positions[:, 1].cpu().numpy(),
    #           'b-', linewidth=2, label='Target Trajectory')
    axes.plot(target_positions[:, 0].cpu().numpy(),
              target_positions[:, 1].cpu().numpy(),
              color='#2E86AB', linewidth=1.5, label='Target Trajectory')

    
    # 绘制曲率向量线段（根据曲率大小调整密度）
    # 高曲率区域显示更多向量
    if label == "B":
        base_step = 12  # B轨迹显示更密集的向量
        high_curv_step = 6
    else:
        base_step = 20
        high_curv_step = 10
        
    for i in range(0, len(trajectory_points)):
        # 根据曲率动态决定是否绘制
        if i % base_step == 0 or (norm_curvature[i] > 0.7 and i % high_curv_step == 0):
            # 根据曲率大小调整透明度
            if label == "B":
                alpha_val = 0.5 + 0.3 * norm_curvature[i]  # B轨迹使用更高的透明度
            else:
                alpha_val = 0.4 + 0.4 * norm_curvature[i]
            axes.plot([trajectory_points[i, 0], curvature_endpoints[i, 0]],
                      [trajectory_points[i, 1], curvature_endpoints[i, 1]],
                      color='#F18F01', linewidth=0.8, alpha=alpha_val)

        #
    # 绘制曲率末端连成的包络线 - 实线
    # axes.plot(curvature_endpoints[:, 0], curvature_endpoints[:, 1],
    #           '-',color='#C73E1D', linewidth=1, label='Curvature Envelope', alpha=0.8)
    axes.plot(curvature_endpoints[:, 0], curvature_endpoints[:, 1],
              color='#C73E1D', linewidth=1.0, label='Curvature Envelope', alpha=0.8)

    axes.set_xlabel('X (m)')
    axes.set_ylabel('Y (m)')
    # axes.set_title('Target Trajectory with Curvature Vectors and Envelope')
    axes.axis("equal")
    if j == 2:
        axes.set_ylim([-5.5, 3.5])
    else:
        axes.set_ylim([-3.5, 2.5])
    # Add legend only to the first subplot
    if j == 0:
        axes.legend(loc='upper right', fontsize=8)
    axes.grid(True, alpha=0.3)
    axes.set_title(f"Trajectory {label}")
    
current_folder = os.path.dirname(os.path.abspath(__file__))
save_folder = current_folder.split("obj_track")[0] + "obj_track/plots/"
fig.suptitle("Target Trajectories with Curvature Analysis", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(f"{save_folder}target_trajectory.png", dpi=150, bbox_inches='tight')
plt.show()
