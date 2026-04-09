from VisFly.utils.FigFashion.FigFashion import FigFon
import torch as th
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.ndimage import gaussian_filter

FigFon.set_fashion("IEEE")

paths = [
    "dreamer_final_3",
    "SHAC_posObs_VaryDis_H96_changeDisToSelfV_agile_1",
    "DiffDreamer_posObs_VaryDis_H96_changeDisToSelfV_agile_1",
    
]

# short_titles = ["dreamer","SHAC", "DiffDreamer"]
short_titles = [i.split("_")[0] for i in paths]
full_paths = [f"/data0/lee/files/obj_track/exps/std/saved/objTracking/test/{p}/body_targets.pth" for p in paths]


def compute_point_density(points, bins=40):
    """使用 3D 直方图近似每个散点的局部密度。"""
    hist, edges = np.histogramdd(points, bins=bins)
    idx = []
    for i in np.arange(3):
        di = np.digitize(points[:, i], edges[i]) - 1
        di = np.clip(di, 0, bins - 1)
        idx.append(di)
    density = hist[idx[0], idx[1], idx[2]]
    return density


def add_gradient_fov(ax, length, fov_deg_h=90.0, fov_deg_v=90.0, nseg=80, color="deepskyblue"):
    """在 3D 图中绘制从原点出发、沿 +x 的四条 FOV 边界渐变线。"""
    tan_h = np.tan(np.deg2rad(fov_deg_h / 2.0))
    tan_v = np.tan(np.deg2rad(fov_deg_v / 2.0))

    # 四个角方向（右上、右下、左上、左下 对应 y/z 正负）
    dirs = np.array([
        [1.0, +tan_h, +tan_v],
        [1.0, +tan_h, -tan_v],
        [1.0, -tan_h, +tan_v],
        [1.0, -tan_h, -tan_v],
    ], dtype=np.float64)
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)

    t = np.linspace(0.0, length, nseg + 1)
    end_alpha = 0.25
    alphas = np.linspace(0.95, end_alpha, nseg)
    rgb = np.array(mcolors.to_rgb(color), dtype=np.float64)

    for d in dirs:
        pts = t[:, None] * d[None, :]
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        colors = np.zeros((nseg, 4), dtype=np.float64)
        colors[:, :3] = rgb
        colors[:, 3] = alphas
        lc = Line3DCollection(segs, colors=colors, linewidths=2.2)
        ax.add_collection3d(lc)

    # 连接四条射线远端，形成 FOV 横框
    endpoints = length * dirs
    frame_segs = np.array([
        [endpoints[0], endpoints[1]],
        [endpoints[1], endpoints[3]],
        [endpoints[3], endpoints[2]],
        [endpoints[2], endpoints[0]],
    ])
    frame_colors = np.zeros((4, 4), dtype=np.float64)
    frame_colors[:, :3] = rgb
    frame_colors[:, 3] = end_alpha
    frame_lc = Line3DCollection(frame_segs, colors=frame_colors, linewidths=1.8)
    ax.add_collection3d(frame_lc)


def add_gradient_fov_rays_2d(
    ax,
    x_end,
    y_end,
    nseg=80,
    color="red",
    alpha_start=0.95,
    alpha_end=0.0,
    linewidth=1.6,
    fade_power=0.35,
    lw_end_ratio=0.2,
):
    """在 2D 图中绘制从原点出发的两条渐变射线（+y_end 与 -y_end）。"""
    t = np.linspace(0.0, 1.0, nseg + 1)
    fade = np.power(np.linspace(0.0, 1.0, nseg), fade_power)
    alphas = alpha_start + (alpha_end - alpha_start) * fade
    linewidths = linewidth * (1.0 - (1.0 - lw_end_ratio) * fade)
    rgb = np.array(plt.matplotlib.colors.to_rgb(color), dtype=np.float64)

    for sign in (+1.0, -1.0):
        xs = x_end * t
        ys = sign * y_end * t
        for k in range(nseg):
            ax.plot(
                [xs[k], xs[k + 1]],
                [ys[k], ys[k + 1]],
                color=(rgb[0], rgb[1], rgb[2], alphas[k]),
                linewidth=linewidths[k],
            )


def draw_density_heatmap(
    ax,
    hist2d,
    x_edges,
    y_edges,
    cmap_name="Reds",
    sigma=2.0,
    sparse_ratio=0.015,
    min_bin_count=3,
):
    """绘制平滑热力图：压制稀疏轨迹，仅保留主要密集区域。"""
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("white")

    # 先过滤“特别离散”的散点：落在低计数网格中的样本不参与统计
    hist2d_filtered = hist2d.astype(np.float64).copy()
    hist2d_filtered[hist2d_filtered < min_bin_count] = 0.0

    hist2d_smooth = gaussian_filter(hist2d_filtered, sigma=sigma)
    hmax = np.max(hist2d_smooth)
    if hmax > 0:
        # 保留过滤后仍有数据的网格，避免低频离散点重新出现
        keep_mask = (hist2d_smooth >= sparse_ratio * hmax) | (hist2d_filtered > 0)
        hist2d_smooth = np.where(keep_mask, hist2d_smooth, 0.0)

    vmax = float(np.max(hist2d_smooth))
    if vmax <= 1.0:
        vmax = 1.0

    # 将 0 计数区域掩码，确保无数据区域严格为白色
    hist_masked = np.ma.masked_less_equal(hist2d_smooth, 0.0)
    norm = mcolors.PowerNorm(gamma=0.5, vmin=1.0, vmax=vmax)

    # 用 pcolormesh 呈现真正的网格色块，避免插值导致的“线条感”
    ax.pcolormesh(
        x_edges,
        y_edges,
        hist_masked.T,
        cmap=cmap,
        norm=norm,
        shading="flat",
        edgecolors="none",
        linewidth=0.0,
        antialiased=False,
    )


def set_axes_equal_3d(ax):
    xlim = np.array(ax.get_xlim3d())
    ylim = np.array(ax.get_ylim3d())
    zlim = np.array(ax.get_zlim3d())
    center = np.array([xlim.mean(), ylim.mean(), zlim.mean()])
    radius = 0.5 * np.max(np.array([xlim.ptp(), ylim.ptp(), zlim.ptp()]))
    ax.set_xlim3d(center[0] - radius, center[0] + radius)
    ax.set_ylim3d(center[1] - radius, center[1] + radius)
    ax.set_zlim3d(center[2] - radius, center[2] + radius)


fig = plt.figure(figsize=(12, 5))
axs = [
    fig.add_subplot(1, 2, 1, projection="3d"),
    fig.add_subplot(1, 2, 2, projection="3d"),
]

all_points = []
all_density = []

for i in np.arange(np.size(paths)):
    path = full_paths[i]
    body_targets = th.load(path)  # (T, N, 3)
    points = body_targets.reshape(-1, 3).cpu().numpy()
    all_points.append(points)

    density = compute_point_density(points, bins=45)
    density = (density - density.min()) / (density.max() - density.min() + 1e-9)
    all_density.append(density)

all_points_concat = np.concatenate(all_points, axis=0)
fov_len = np.percentile(np.linalg.norm(all_points_concat, axis=1), 98)
fov_color = "deepskyblue"

# 限制 FOV 线条最远只到 x=8
fov_x_cap = 8.0
fov_deg_h = 100.0
fov_deg_v = 100.0
tan_h = np.tan(np.deg2rad(fov_deg_h / 2.0))
tan_v = np.tan(np.deg2rad(fov_deg_v / 2.0))
dir_x = 1.0 / np.sqrt(1.0 + tan_h ** 2 + tan_v ** 2)  # 3D FOV 射线方向的 x 分量
fov_len_3d = np.minimum(fov_len, fov_x_cap / dir_x)

# FOV 在远端的 y/z 投影边界
fov_y = fov_len_3d * np.tan(np.deg2rad(fov_deg_h / 2.0))
fov_z = fov_len_3d * np.tan(np.deg2rad(fov_deg_v / 2.0))

# 2D 投影里把 FOV 射线画短一些，避免线条过长
fov_len_2d = np.minimum(0.72 * fov_len, fov_x_cap)
fov_y_2d = fov_len_2d * np.tan(np.deg2rad(fov_deg_h / 2.0))
fov_z_2d = fov_len_2d * np.tan(np.deg2rad(fov_deg_v / 2.0))

# 按轴设置显示范围：x 轴向左扩展到数据最小值，确保包含 x < -1 的数据
x_data_min = np.min(all_points_concat[:, 0])
x_data_max = np.max(all_points_concat[:, 0])
x_pad = 0.03 * (x_data_max - x_data_min + 1e-9)
x_min = min(-1.0, x_data_min - x_pad)
x_max = np.max(np.array([x_data_max + x_pad, fov_len]))
y_min = np.min(np.array([np.min(all_points_concat[:, 1]), -fov_y]))
y_max = np.max(np.array([np.max(all_points_concat[:, 1]), +fov_y]))
z_min = np.min(np.array([np.min(all_points_concat[:, 2]), -fov_z]))
z_max = np.max(np.array([np.max(all_points_concat[:, 2]), +fov_z]))

# 让 XY 与 XZ 图中原点(0,0)的垂直位置对齐：
# 保持 y/z 各自范围独立，但使 0 在两轴上的相对位置一致。
y_zero_ratio = (0.0 - y_min) / (y_max - y_min + 1e-12)
y_zero_ratio = np.clip(y_zero_ratio, 1e-3, 1.0 - 1e-3)

# 候选1：固定 z_min，扩展 z_max
z_max_candidate = z_min * (y_zero_ratio - 1.0) / y_zero_ratio
span_candidate_1 = (z_max_candidate - z_min) if z_max_candidate >= z_max else np.inf

# 候选2：固定 z_max，扩展 z_min
z_min_candidate = -y_zero_ratio / (1.0 - y_zero_ratio) * z_max
span_candidate_2 = (z_max - z_min_candidate) if z_min_candidate <= z_min else np.inf

if span_candidate_1 <= span_candidate_2:
    z_max = max(z_max, z_max_candidate)
else:
    z_min = min(z_min, z_min_candidate)

x_ticks = np.linspace(x_min, x_max, 5)
y_ticks = np.linspace(y_min, y_max, 5)
z_ticks = np.linspace(z_min, z_max, 5)

# for i in np.arange(np.size(paths)):
#     name = short_titles[i]
#     ax = axs[i]
#     points = all_points[i]
#     density = all_density[i]

#     p = ax.scatter(
#         points[:, 0], points[:, 1], points[:, 2],
#         c=density,
#         cmap="viridis",
#         s=3,
#         alpha=0.8,
#         linewidths=0,
#     )

#     add_gradient_fov(ax, length=fov_len_3d, fov_deg_h=fov_deg_h, fov_deg_v=fov_deg_v, nseg=100, color=fov_color)

#     ax.set_xlim3d(x_min, x_max)
#     ax.set_ylim3d(y_min, y_max)
#     ax.set_zlim3d(z_min, z_max)
#     ax.set_xticks(x_ticks)
#     ax.set_yticks(y_ticks)
#     ax.set_zticks(z_ticks)
#     ax.set_box_aspect((1, 1, 1))

#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     ax.set_zlabel("z")
#     ax.set_title(name)

# fig.suptitle("3D Target Scatter Density + 100°x100° FOV", y=0.98)
# plt.tight_layout()
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, "tar_dist_3d_density_fov.png")
# fig.savefig(save_path,  bbox_inches="tight")
# plt.close(fig)

row_num = np.size(paths)
fig2, ax2 = plt.subplots(row_num, 2, figsize=(8, 3.6 * row_num), squeeze=False)
fig2, ax2 = FigFon.get_figure_axes(SubFigSize=(2,len(paths)), Column=2)
# XY / XZ 统一统计与显示范围
xz_xlim = (-2.0, 18.0)
xz_ylim = (-10.0, 10.0)
# 显示更多离散点：降低最小计数阈值，保留单次落点
heatmap_min_bin_count = 6

for i in np.arange(row_num):
    name = short_titles[i]
    points = all_points[i]

    # 第一行：XY 投影（每列对应一个算法）
    ax_xy = ax2[0, i]
    hist_xy, x_edges_xy, y_edges_xy = np.histogram2d(
        points[:, 0], points[:, 1],
        bins=80,
        range=[[xz_xlim[0], xz_xlim[1]], [xz_ylim[0], xz_ylim[1]]],
    )
    draw_density_heatmap(
        ax_xy,
        hist_xy,
        x_edges_xy,
        y_edges_xy,
        cmap_name="Reds",
        sigma=2.0,
        sparse_ratio=0.015,
        min_bin_count=heatmap_min_bin_count,
    )

    # FOV 的 XY 投影：两条从原点出发的渐变射线（不画右侧横线）
    add_gradient_fov_rays_2d(ax_xy, x_end=fov_len_2d, y_end=fov_y_2d, nseg=100, color=fov_color, linewidth=1.6)

    ax_xy.set_xlim(xz_xlim[0], xz_xlim[1])
    ax_xy.set_ylim(xz_ylim[0], xz_ylim[1])
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")
    ax_xy.set_title(name + " | XY")
    ax_xy.grid(True, alpha=0.25)

    # 第二行：XZ 投影（每列对应一个算法）
    ax_xz = ax2[1, i]
    hist_xz, x_edges_xz, z_edges_xz = np.histogram2d(
        points[:, 0], points[:, 2],
        bins=80,
        range=[[xz_xlim[0], xz_xlim[1]], [xz_ylim[0], xz_ylim[1]]],
    )
    draw_density_heatmap(
        ax_xz,
        hist_xz,
        x_edges_xz,
        z_edges_xz,
        cmap_name="Reds",
        sigma=2.0,
        sparse_ratio=0.015,
        min_bin_count=heatmap_min_bin_count,
    )

    # FOV 的 XZ 投影：两条从原点出发的渐变射线（不画右侧横线）
    add_gradient_fov_rays_2d(ax_xz, x_end=fov_len_2d, y_end=fov_z_2d, nseg=100, color=fov_color,linewidth=1.6)

    ax_xz.set_xlim(xz_xlim[0], xz_xlim[1])
    ax_xz.set_ylim(xz_ylim[0], xz_ylim[1])
    ax_xz.set_aspect("equal", adjustable="datalim")
    ax_xz.set_xlabel("x")
    ax_xz.set_ylabel("z")
    ax_xz.set_title(name + " | XZ")
    ax_xz.grid(True, alpha=0.25)
    
    # fig2.suptitle("2D Projections: each column is one algorithm (XY top, XZ bottom)", y=0.995)
fig2.tight_layout()
# fig2.subplots_adjust(wspace=0.1)

save_path2 = os.path.join(script_dir, "tar_dist_projection_xy_xz.png")
fig2.savefig(save_path2, bbox_inches="tight")
plt.close(fig2)
