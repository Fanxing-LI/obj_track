import json
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np


obj = "box30_high20/track/test_fast"


def _set_axes_equal_3d(ax):
	"""让 3D 坐标轴等比例显示。"""
	x_limits = ax.get_xlim3d()
	y_limits = ax.get_ylim3d()
	z_limits = ax.get_zlim3d()

	x_range = abs(x_limits[1] - x_limits[0])
	y_range = abs(y_limits[1] - y_limits[0])
	z_range = abs(z_limits[1] - z_limits[0])

	x_middle = np.mean(x_limits)
	y_middle = np.mean(y_limits)
	z_middle = np.mean(z_limits)

	radius = 0.5 * max([x_range, y_range, z_range])

	ax.set_xlim3d([x_middle - radius, x_middle + radius])
	ax.set_ylim3d([y_middle - radius, y_middle + radius])
	ax.set_zlim3d([z_middle - radius, z_middle + radius])


def _resolve_obj_json_path(obj_path: str) -> str:
	"""
	支持两种输入：
	1) 目录：例如 box30_high20/track/test_fast
	2) 文件：例如 box30_high20/track/test_fast/cubic_large2.json
	"""
	project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	obj_root = os.path.join(project_root, "VisFly", "configs", "obj")

	abs_candidate = obj_path if os.path.isabs(obj_path) else os.path.join(obj_root, obj_path)

	if os.path.isfile(abs_candidate):
		return abs_candidate

	if not os.path.isdir(abs_candidate):
		raise FileNotFoundError(f"Invalid obj path: {obj_path}")

	json_files = sorted(
		[os.path.join(abs_candidate, f) for f in os.listdir(abs_candidate) if f.endswith(".json")]
	)
	if not json_files:
		raise FileNotFoundError(f"No json file found in: {abs_candidate}")

	return json_files[0]


def load_obj_position_mean(file_or_dir: str) -> np.ndarray:
	"""读取 obj 配置中的 path.points.position.mean，返回 shape=(N,3) 的轨迹点。"""
	json_path = _resolve_obj_json_path(file_or_dir)

	with open(json_path, "r", encoding="utf-8") as f:
		data = json.load(f)

	objects = data.get("objects", [])
	if not objects:
		raise ValueError(f"No objects found in {json_path}")

	# 默认取第一个对象
	obj_cfg = objects[0]
	mean_points = (
		obj_cfg.get("path", {})
		.get("kwargs", {})
		.get("points", {})
		.get("kwargs", {})
		.get("position", {})
		.get("mean", None)
	)

	if mean_points is None:
		raise ValueError(f"No path.kwargs.points.kwargs.position.mean found in {json_path}")

	pts = np.asarray(mean_points, dtype=float)
	if pts.ndim == 1:
		if pts.shape[0] != 3:
			raise ValueError(f"position.mean should be 3D or Nx3, got shape {pts.shape}")
		pts = pts[None, :]
	if pts.ndim != 2 or pts.shape[1] != 3:
		raise ValueError(f"position.mean should be Nx3, got shape {pts.shape}")

	return pts


def plot_obj_position_mean_trajectory(file_or_dir):
	"""绘制输入 obj 路径文件中的 obj position mean 轨迹。"""
	pts = load_obj_position_mean(file_or_dir)

	fig = plt.figure(figsize=(10, 4))
	ax1 = fig.add_subplot(1, 2, 1)
	ax2 = fig.add_subplot(1, 2, 2, projection="3d")

	# XY 轨迹
	ax1.plot(pts[:, 0], pts[:, 1], "o-", linewidth=2, markersize=4)
	ax1.set_xlabel("x")
	ax1.set_ylabel("y")
	ax1.set_title("Obj Position Mean (XY)")
	ax1.axis("equal")
	ax1.grid(True, alpha=0.3)

	# XYZ 轨迹
	ax2.plot(pts[:, 0], pts[:, 1], pts[:, 2], "o-", linewidth=2, markersize=4)
	ax2.set_xlabel("x")
	ax2.set_ylabel("y")
	ax2.set_zlabel("z")
	ax2.set_title("Obj Position Mean (XYZ)")
	_set_axes_equal_3d(ax2)

	plt.tight_layout()
	return fig, [ax1, ax2]


if __name__ == "__main__":
	fig, _ = plot_obj_position_mean_trajectory(obj)

	script_dir = os.path.dirname(os.path.abspath(__file__))
	save_name = obj.replace("/", "_") + "_obj_position_mean.png"
	save_path = os.path.join(script_dir, save_name)
	fig.savefig(save_path, dpi=300)
	plt.show()

