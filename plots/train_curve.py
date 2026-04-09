from matplotlib import pyplot as plt
import numpy as np
from VisFly.utils.FigFashion.FigFashion import FigFon
import tensorboard
from tensorboard.backend.event_processing import event_accumulator
from scipy.ndimage import gaussian_filter1d
import os
import glob

FigFon.set_fashion("IEEE")


def load_tensorboard_file(folder_path, tag="rollout/ep_rew_mean", smooth_window=10):
    # Find tensorboard event files in the folder
    event_files = glob.glob(os.path.join(folder_path, "events.out.tfevents.*"))
    if not event_files:
        return np.array([]), np.array([]), np.array([])

    # Use the first (or most recent) event file
    if np.size(event_files) == 1:
        file_path = event_files[0]
    else:
        ctimes = np.array([os.path.getctime(f) for f in event_files], dtype=np.float64)
        file_path = event_files[np.argmax(ctimes)]

    ea = event_accumulator.EventAccumulator(file_path)
    ea.Reload()

    scalar_events = ea.Scalars(tag)

    timesteps = np.array([event.step for event in scalar_events])
    values = np.array([event.value for event in scalar_events])
    raw_values = values.copy()

    if smooth_window and np.size(values) > smooth_window:
        window_size = smooth_window
        padded_values = np.pad(values, (window_size // 2, window_size // 2), mode="edge")
        smoothed = np.convolve(padded_values, np.ones(window_size) / window_size, mode="valid")
        if np.size(smoothed) != np.size(values):
            smoothed = smoothed[: np.size(values)]
        values = smoothed

    return values, timesteps, raw_values


def rolling_bounds(values, window_size):
    """Return rolling lower/upper envelopes for 1D values."""
    if window_size <= 1 or np.size(values) == 0:
        return values.copy(), values.copy()

    half = window_size // 2
    padded = np.pad(values, (half, half), mode="edge")
    lower = np.empty_like(values, dtype=np.float64)
    upper = np.empty_like(values, dtype=np.float64)

    for i in np.arange(np.size(values)):
        seg = padded[i:i + window_size]
        lower[i] = np.min(seg)
        upper[i] = np.max(seg)

    # 对上下界再做一次平滑，避免阴影边界过于锯齿
    sigma = np.maximum(1.0, window_size / 6.0)
    lower = gaussian_filter1d(lower, sigma=sigma, mode="nearest")
    upper = gaussian_filter1d(upper, sigma=sigma, mode="nearest")

    return lower, upper


def plot_train_curves(
    ax,
    run_dirs,
    labels=None,
    tag="rollout/ep_rew_mean",
    smooth_window=10,
    show_presmooth_shadow=True,
    shadow_alpha=0.14,
):
    if labels is None:
        labels = [os.path.basename(p) for p in run_dirs]

    pair_count = np.minimum(np.size(run_dirs), np.size(labels))
    for i in np.arange(pair_count):
        run_dir = run_dirs[i]
        label = labels[i]
        values, timesteps, raw_values = load_tensorboard_file(run_dir, tag=tag, smooth_window=smooth_window)
        if np.size(values) == 0 or np.size(timesteps) == 0:
            continue
        line, = ax.plot(timesteps, values, label=label)

        if show_presmooth_shadow and smooth_window and np.size(raw_values) > 0:
            lower, upper = rolling_bounds(raw_values, np.maximum(1, smooth_window))
            ax.fill_between(
                timesteps,
                lower,
                upper,
                color=line.get_color(),
                alpha=shadow_alpha,
                linewidth=0,
            )

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Reward")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.legend(loc="upper right")


def get_default_run_dirs():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    left_base_dir = os.path.join(project_root, "exps", "std", "saved", "objTracking")
    right_base_dir = os.path.join(project_root, "exps", "racing", "saved","racing")
    left_names = [
        "PPO_final_4",
        "dreamer_final_4",
        "SHAC_posObs_VaryDis_H96_changeDisToSelfV_agile_1",
        "DiffDreamer_posObs_VaryDis_H96_changeDisToSelfV_agile_1",
    ]

    right_names = [
        "PPO_tar1.5_maxv3_1",
        "dreamer_tar1.5_maxv3_1",
        "BPTT_tar1.5_maxv3_1",
        "SepDiffDreamer_tar1.5_maxv3_1",
    ]

    left_dirs = [os.path.join(left_base_dir, n) for n in left_names]
    right_dirs = [os.path.join(right_base_dir, n) for n in right_names]

    left_labels = [n.split("_for_plot")[0] for n in left_names]
    right_labels = [n.split("_for_plot")[0] for n in right_names]

    return (left_dirs, left_labels), (right_dirs, right_labels)


def main(save_path=None):
    left_legends = ["PPO", "Dreamer", "SHAC", "DiffDreamer"]
    right_legends = ["PPO", "Dreamer", "BPTT", "SepDiffDreamer"]
    fig, axes = FigFon.get_figure_axes(SubFigSize=(1, 2), Column=2, HeightScale=0.8)
    axes = np.atleast_1d(axes).ravel()

    (left_dirs, left_labels), (right_dirs, right_labels) = get_default_run_dirs()

    plot_train_curves(
        axes[0],
        left_dirs,
        labels=left_legends if np.size(left_legends) == np.size(left_dirs) else left_labels,
    )
    plot_train_curves(
        axes[1],
        right_dirs,
        labels=right_legends if np.size(right_legends) == np.size(right_dirs) else right_labels,
    )

    if save_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(script_dir, "train_curve.png")

    fig.savefig(save_path, dpi=300)


if __name__ == "__main__":
    main()