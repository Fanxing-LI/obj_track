import os, sys
import torch as th
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from VisFly.utils.FigFashion.FigFashion import FigFon


def compute_curvature_vector(x, y, t=None):
    """
    Compute curvature vector for a 2D trajectory.
    Args:
        x: 1D array-like, x coordinates
        y: 1D array-like, y coordinates
        t: 1D array-like, time or parameter (optional, will use arc length if None)
    Returns:
        curvature_magnitude: 1D numpy array, curvature magnitude at each point
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if t is None or np.all(t == t[0]):
        # Use cumulative arc length as parameter when t is constant
        dx_raw = np.diff(x, prepend=x[0])
        dy_raw = np.diff(y, prepend=y[0])
        arc_lengths = np.cumsum(np.sqrt(dx_raw ** 2 + dy_raw ** 2))
        t = arc_lengths / 2
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

    return curvature_magnitude


FigFon.set_fashion("IEEE")
save_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + f"/saved/objTracking/"

labels = [ "D","B","8", ]

all_curvatures = []

for j, label in enumerate(labels):
    fig, axeses = FigFon.get_figure_axes(SubFigSize=(1, 1), Column=1, Border=[0, 0, 0.86, 1])
    axes = axeses
    data = th.load(f"exps/vary_v/saved/objTracking/test/{label}_1.0_SHAC_Dis3.0.pth")

    # Get target positions
    target_positions = th.stack([tar for tar in data["target_all"]]).squeeze()[:, 0, :]
    t = th.stack(data["t"])[:, 0]

    # Compute curvature
    curvature_mag = compute_curvature_vector(
        target_positions[:, 0].cpu().numpy(),
        target_positions[:, 1].cpu().numpy()
    )
    all_curvatures.append(curvature_mag)

    # Prepare colored line segments
    points = np.column_stack([target_positions[:, 0].cpu().numpy(),
                              target_positions[:, 1].cpu().numpy()])
    segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)

    # Create color mapping
    norm = plt.Normalize(min(map(np.min, all_curvatures)), max(map(np.max, all_curvatures)))
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(curvature_mag)
    lc.set_linewidth(2)

    # Add to plot
    axes.add_collection(lc)
    axes.autoscale()

    axes.set_xlabel('X (m)')
    axes.set_ylabel('Y (m)')
    axes.axis("equal")
    if j == 0:
        axes.set_ylim([-5.5, 3.5])
    else:
        axes.set_ylim([-3.5, 2.5])
        # expand x lim by 0.2
        current_xlim = axes.get_xlim()
        axes.set_xlim([current_xlim[0]-0.2, current_xlim[1]+0.2])
    axes.grid(True, alpha=0.3)
    axes.set_title(f"Trajectory {label}")


    norm = plt.Normalize(min(map(np.min, all_curvatures)), max(map(np.max, all_curvatures)))
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])

    # Define the position of the colorbar (left, bottom, width, height) in figure coordinates
    cbar_ax = fig.add_axes([0.89, 0.2, 0.01, 0.7])  # Adjust these values as needed
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Curvature Magnitude')

    # Save figure
    current_folder = os.path.dirname(os.path.abspath(__file__))
    save_folder = current_folder.split("obj_track")[0] + "obj_track/plots/"
    fig.savefig(f"{save_folder}target_trajectory_{label}.png")
    plt.show()