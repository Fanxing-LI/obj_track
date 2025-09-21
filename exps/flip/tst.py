import numpy as np
import matplotlib
# Set matplotlib to use non-interactive backend to avoid display issues on remote servers
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from typing import Optional
import os
import cv2

from VisFly.utils.evaluate import TestBase
from VisFly.utils.FigFashion.FigFashion import FigFon  # noqa: F401

class Test(TestBase):
    def __init__(self, env, model, name, save_path: Optional[str] = None):
        super(Test, self).__init__(env=env, model=model, name=name, save_path=save_path)

    def create_combined_video_frame(self, global_frame, agent_sensor_data, timestep_idx):
        """
        Create a combined frame with global view and agent sensor views.
        """
        global_h, global_w = global_frame.shape[:2]

        agent_frames = []
        if "depth" in agent_sensor_data:
            depth_data = agent_sensor_data["depth"]
            if hasattr(depth_data, 'cpu'):
                depth_data = depth_data.cpu().numpy()
            elif hasattr(depth_data, 'numpy'):
                depth_data = depth_data.numpy()
            for agent_idx in range(depth_data.shape[0]):
                depth_frame = depth_data[agent_idx, 0]
                depth_max = depth_frame.max()
                if depth_max > 0:
                    depth_normalized = ((depth_frame / depth_max) * 255).astype(np.uint8)
                else:
                    depth_normalized = np.zeros_like(depth_frame, dtype=np.uint8)
                depth_rgb = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_VIRIDIS)
                agent_frames.append(cv2.resize(depth_rgb, (200, 200)))

        if agent_frames:
            num_agents = len(agent_frames)
            grid_cols = min(4, num_agents)
            grid_rows = (num_agents + grid_cols - 1) // grid_cols
            agent_grid_h = grid_rows * 200
            agent_grid_w = grid_cols * 200
            agent_grid = np.zeros((agent_grid_h, agent_grid_w, 3), dtype=np.uint8)
            for i, frame in enumerate(agent_frames):
                row = i // grid_cols
                col = i % grid_cols
                y_start = row * 200
                x_start = col * 200
                agent_grid[y_start:y_start+200, x_start:x_start+200] = frame
            global_resized = cv2.resize(global_frame, (int(global_w * agent_grid_h / global_h), agent_grid_h))
            combined_frame = np.hstack([global_resized, agent_grid])
        else:
            combined_frame = global_frame

        cv2.putText(combined_frame, f"Timestep: {timestep_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return combined_frame

    def save_combined_video(self, episode_dir):
        if not self.render_image_all:
            print("No render images available for video creation")
            return
        try:
            combined_frames = []
            for i, (render_frame, obs_data) in enumerate(zip(self.render_image_all, self.obs_all)):
                render_bgr = cv2.cvtColor(render_frame, cv2.COLOR_RGB2BGR)
                combined_frame = self.create_combined_video_frame(render_bgr, obs_data, i)
                combined_frames.append(combined_frame)
            if not combined_frames:
                print("No frames to save")
                return
            height, width = combined_frames[0].shape[:2]
            fps = int(1.0 / self.env.envs.dynamics.dt)
            video_path = os.path.join(episode_dir, "combined_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            for frame in combined_frames:
                video_writer.write(frame)
            video_writer.release()
            print(f"Combined video saved: {video_path}")
            self.save_individual_videos(episode_dir)
        except Exception as e:
            print(f"Error creating combined video: {e}")
            print("Continuing without video generation...")

    def save_individual_videos(self, episode_dir):
        if not self.render_image_all:
            return
        try:
            self.save_global_video(episode_dir)
            self.save_agent_depth_videos(episode_dir)
        except Exception as e:
            print(f"Error saving individual videos: {e}")
            print("Continuing without individual video generation...")

    def save_global_video(self, episode_dir):
        try:
            height, width, _ = self.render_image_all[0].shape
            fps = int(1.0 / self.env.envs.dynamics.dt)
            video_path = os.path.join(episode_dir, "global_camera.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            for frame in self.render_image_all:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)
            video_writer.release()
            print(f"Global camera video saved: {video_path}")
        except Exception as e:
            print(f"Error saving global video: {e}")
            print("Continuing without global video...")

    def save_agent_depth_videos(self, episode_dir):
        if not self.obs_all or "depth" not in self.obs_all[0]:
            return
        try:
            num_agents = self.obs_all[0]["depth"].shape[0]
            fps = int(1.0 / self.env.envs.dynamics.dt)
            agent_writers = {}
            for agent_idx in range(num_agents):
                video_path = os.path.join(episode_dir, f"agent_{agent_idx}_depth.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                agent_writers[agent_idx] = cv2.VideoWriter(video_path, fourcc, fps, (200, 200))
            for obs_data in self.obs_all:
                if "depth" in obs_data:
                    depth_data = obs_data["depth"]
                    for agent_idx in range(num_agents):
                        if agent_idx < depth_data.shape[0]:
                            depth_frame = depth_data[agent_idx, 0]
                            if hasattr(depth_frame, 'cpu'):
                                depth_frame = depth_frame.cpu().numpy()
                            elif hasattr(depth_frame, 'numpy'):
                                depth_frame = depth_frame.numpy()
                            depth_normalized = ((depth_frame / (depth_frame.max() + 1e-6)) * 255).astype(np.uint8)
                            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_VIRIDIS)
                            depth_resized = cv2.resize(depth_colored, (200, 200))
                            agent_writers[agent_idx].write(depth_resized)
            for agent_idx, writer in agent_writers.items():
                writer.release()
                print(f"Agent {agent_idx} depth video saved: agent_{agent_idx}_depth.mp4")
        except Exception as e:
            print(f"Error saving agent depth videos: {e}")
            print("Continuing without agent depth videos...")

    def draw(self, names=None):
        import torch as th
        state_data = th.stack(self.state_all).cpu().numpy()

        absolute_positions = state_data[:, :, 0:3]

        # Targets: support FlipEnv which may have 'target_position' instead of 'target'
        targets = getattr(self.env, 'target', None)
        if targets is None:
            targets = getattr(self.env, 'target_position', None)
        if hasattr(targets, 'cpu'):
            targets = targets.cpu().numpy()

        # Collision vectors
        collision_points_list = []
        for rec in self.collision_all:
            col_pt = rec.get("col_pt", None)
            if col_pt is None:
                collision_points_list.append(None)
                continue
            if hasattr(col_pt, 'cpu'):
                col_pt_np = col_pt.cpu().numpy()
            elif hasattr(col_pt, 'numpy'):
                col_pt_np = col_pt.numpy()
            else:
                col_pt_np = np.array(col_pt)
            collision_points_list.append(col_pt_np)
        num_envs_from_state = absolute_positions.shape[1]
        normalized_collision_pts = []
        for pts in collision_points_list:
            if pts is None:
                normalized_collision_pts.append(np.full((num_envs_from_state, 3), np.nan, dtype=float))
            else:
                arr = np.array(pts)
                if arr.ndim == 1 and arr.size == 3:
                    arr = arr.reshape(1, 3)
                normalized_collision_pts.append(arr)
        try:
            collision_points = np.stack(normalized_collision_pts, axis=0)
        except Exception:
            T = absolute_positions.shape[0]
            collision_points = np.full((T, num_envs_from_state, 3), np.nan, dtype=float)
            for t_idx in range(min(T, len(normalized_collision_pts))):
                pts = normalized_collision_pts[t_idx]
                n = min(num_envs_from_state, pts.shape[0]) if pts is not None else 0
                if n > 0:
                    collision_points[t_idx, :n, :] = pts[:n, :]
        collision_vectors = collision_points - absolute_positions

        num_envs = state_data.shape[1]

        success_agents = set()
        for timestep_info in self.info_all:
            for agent_idx, agent_info in enumerate(timestep_info):
                if agent_info and "is_success" in agent_info and agent_info["is_success"]:
                    success_agents.add(agent_idx)

        t_list = []
        for time_tensor in self.t:
            if hasattr(time_tensor, 'cpu'):
                t_list.append(time_tensor.cpu().numpy())
            else:
                t_list.append(np.array(time_tensor))
        t = np.array(t_list)[:, 0]

        fig = plt.figure(figsize=(16, 12))

        def _plot_with_mean(ax, series, labels, title, ylabel, fill_alpha=0.15):
            if series.ndim == 2:
                series = series[:, None, :]
            mean_vals = series.mean(axis=1)
            std_vals = series.std(axis=1)
            for ch in range(series.shape[2]):
                ax.plot(t, mean_vals[:, ch], linewidth=2, label=labels[ch])
                ax.fill_between(
                    t,
                    mean_vals[:, ch] - std_vals[:, ch],
                    mean_vals[:, ch] + std_vals[:, ch],
                    alpha=fill_alpha,
                )
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Time (s)")
            ax.legend()
            ax.grid(True)

        # Position plot
        ax1 = plt.subplot(2, 2, 1)
        _plot_with_mean(ax1, absolute_positions, ["x", "y", "z"], "Position vs Time", "Position (m)")

        # 3D Trajectory
        ax2 = plt.subplot(2, 2, 2, projection='3d')
        for agent_idx in range(num_envs):
            x_pos = absolute_positions[:, agent_idx, 0]
            y_pos = absolute_positions[:, agent_idx, 1]
            z_pos = absolute_positions[:, agent_idx, 2]
            vel = state_data[:, agent_idx, 7:10]
            speed = np.linalg.norm(vel, axis=1)
            ax2.plot(x_pos, y_pos, z_pos, color='lightgrey', linewidth=0.7, alpha=0.6)
            ax2.scatter(x_pos, y_pos, z_pos, c=speed, cmap='viridis', s=8, alpha=0.9)
            ax2.scatter(x_pos[0], y_pos[0], z_pos[0], color='green', s=50, marker='o', edgecolor='black', linewidth=1)
            if agent_idx in success_agents:
                ax2.scatter(x_pos[-1], y_pos[-1], z_pos[-1], color='blue', s=50, marker='s', edgecolor='black', linewidth=1)
            else:
                ax2.scatter(x_pos[-1], y_pos[-1], z_pos[-1], color='red', s=50, marker='x', linewidth=1)
            vec = collision_vectors[:, agent_idx, :]
            valid_mask = ~np.isnan(vec).any(axis=1)
            if np.any(valid_mask):
                X, Y, Z = x_pos[valid_mask], y_pos[valid_mask], z_pos[valid_mask]
                U, V, W = vec[valid_mask, 0], vec[valid_mask, 1], vec[valid_mask, 2]
                mag = np.linalg.norm(vec, axis=1)
                mag_valid = mag[valid_mask]
                near_mask = mag_valid < 1.0
                if np.any(near_mask):
                    ax2.quiver(X[near_mask], Y[near_mask], Z[near_mask], U[near_mask], V[near_mask], W[near_mask],
                               length=1.0, normalize=False, color='red', alpha=0.4, linewidth=0.8)
                mid_mask = (mag_valid >= 1.0) & (mag_valid < 2.0)
                if np.any(mid_mask):
                    ax2.quiver(X[mid_mask], Y[mid_mask], Z[mid_mask], U[mid_mask], V[mid_mask], W[mid_mask],
                               length=1.0, normalize=False, color='orange', alpha=0.3, linewidth=0.6)
                far_mask = mag_valid >= 2.0
                if np.any(far_mask):
                    ax2.quiver(X[far_mask], Y[far_mask], Z[far_mask], U[far_mask], V[far_mask], W[far_mask],
                               length=1.0, normalize=False, color='lightblue', alpha=0.2, linewidth=0.4)

        # Add target marker(s) if available
        if targets is not None:
            if isinstance(targets, np.ndarray) and targets.ndim == 1 and len(targets) >= 3:
                ax2.scatter(targets[0], targets[1], targets[2], color='gold', s=80, marker='*', edgecolor='black', linewidth=1, label='Target')
            elif isinstance(targets, np.ndarray) and targets.ndim == 2:
                for agent_idx in range(min(targets.shape[0], num_envs)):
                    target_pos = targets[agent_idx]
                    if len(target_pos) >= 3:
                        ax2.scatter(target_pos[0], target_pos[1], target_pos[2], color='gold', s=80, marker='*', edgecolor='black', linewidth=1, label='Target' if agent_idx == 0 else None)

        ax2.set_xlabel("X Position (m)")
        ax2.set_ylabel("Y Position (m)")
        ax2.set_zlabel("Z Position (m)")
        ax2.set_title("3D Trajectory")

        # Equal aspect bounds
        all_x = absolute_positions[:, :, 0].flatten()
        all_y = absolute_positions[:, :, 1].flatten()
        all_z = absolute_positions[:, :, 2].flatten()
        if isinstance(targets, np.ndarray):
            if targets.ndim == 1 and len(targets) >= 3:
                all_x = np.append(all_x, targets[0])
                all_y = np.append(all_y, targets[1])
                all_z = np.append(all_z, targets[2])
            elif targets.ndim == 2:
                for i in range(min(targets.shape[0], num_envs)):
                    if len(targets[i]) >= 3:
                        all_x = np.append(all_x, targets[i][0])
                        all_y = np.append(all_y, targets[i][1])
                        all_z = np.append(all_z, targets[i][2])
        max_range = np.array([all_x.max()-all_x.min(), all_y.max()-all_y.min(), all_z.max()-all_z.min()]).max() / 2.0
        mid_x = (all_x.max()+all_x.min()) * 0.5
        mid_y = (all_y.max()+all_y.min()) * 0.5
        mid_z = (all_z.max()+all_z.min()) * 0.5
        ax2.set_xlim(mid_x - max_range, mid_x + max_range)
        ax2.set_ylim(mid_y - max_range, mid_y + max_range)
        ax2.set_zlim(mid_z - max_range, mid_z + max_range)

        # Orientation quaternion
        ax3 = plt.subplot(2, 2, 3)
        _plot_with_mean(ax3, state_data[:, :, 3:7], ["w", "x", "y", "z"], "Orientation Quaternion", "Quaternion")

        # Angular velocity
        ax4 = plt.subplot(2, 2, 4)
        _plot_with_mean(ax4, state_data[:, :, 10:13], ["wx", "wy", "wz"], "Angular Velocity", "Angular Velocity (rad/s)")

        plt.tight_layout()
        try:
            plt.show()
        except Exception:
            pass
        return [fig]

    def draw_debug(self, names=None, save_path: Optional[str] = None):
        import torch as th
        state_data = th.stack(self.state_all).cpu().numpy()

        absolute_positions = state_data[:, :, 0:3]
        velocities = state_data[:, :, 7:10]
        angular_velocities = state_data[:, :, 10:13]

        targets = getattr(self.env, 'target', None)
        if targets is None:
            targets = getattr(self.env, 'target_position', None)
        if hasattr(targets, 'cpu'):
            targets = targets.cpu().numpy()

        num_envs = state_data.shape[1]
        num_timesteps = state_data.shape[0]

        t_list = []
        for time_tensor in self.t:
            if hasattr(time_tensor, 'cpu'):
                t_list.append(time_tensor.cpu().numpy())
            else:
                t_list.append(np.array(time_tensor))
        t = np.array(t_list)[:, 0]

        collision_timesteps = []
        success_timesteps = []
        failure_timesteps = []
        for timestep_idx, timestep_info in enumerate(self.info_all):
            for agent_idx, agent_info in enumerate(timestep_info):
                if agent_info:
                    if "is_collision" in agent_info and agent_info["is_collision"]:
                        collision_timesteps.append((timestep_idx, agent_idx))
                    if "is_success" in agent_info and agent_info["is_success"]:
                        success_timesteps.append((timestep_idx, agent_idx))
                    if "is_failure" in agent_info and agent_info["is_failure"]:
                        failure_timesteps.append((timestep_idx, agent_idx))

        distances_to_target = []
        for timestep_idx in range(num_timesteps):
            timestep_distances = []
            for agent_idx in range(num_envs):
                if isinstance(targets, np.ndarray) and targets.ndim == 1 and len(targets) >= 3:
                    target_pos = targets[:3]
                elif isinstance(targets, np.ndarray) and targets.ndim == 2 and targets.shape[0] > agent_idx:
                    target_pos = targets[agent_idx][:3]
                else:
                    target_pos = np.array([0, 0, 0])
                agent_pos = absolute_positions[timestep_idx, agent_idx, :3]
                distance = np.linalg.norm(agent_pos - target_pos)
                timestep_distances.append(distance)
            distances_to_target.append(timestep_distances)
        distances_to_target = np.array(distances_to_target)

        # Speeds and accelerations
        speeds = np.linalg.norm(velocities, axis=2)
        accelerations = np.zeros_like(velocities)
        for timestep_idx in range(1, num_timesteps):
            dt = t[timestep_idx] - t[timestep_idx-1]
            if dt > 0:
                accelerations[timestep_idx] = (velocities[timestep_idx] - velocities[timestep_idx-1]) / dt
        acceleration_magnitudes = np.linalg.norm(accelerations, axis=2)

        fig = plt.figure(figsize=(18, 14))

        def _plot_with_events(ax, data, title, ylabel, collision_points, success_points):
            mean_vals = data.mean(axis=1)
            std_vals = data.std(axis=1)
            ax.plot(t, mean_vals, linewidth=2, label='Mean')
            ax.fill_between(t, mean_vals - std_vals, mean_vals + std_vals, alpha=0.15)
            for ts, _ in collision_points:
                if ts < len(t):
                    ax.axvline(x=t[ts], color='red', linestyle='--', alpha=0.5)
            for ts, _ in success_points:
                if ts < len(t):
                    ax.axvline(x=t[ts], color='green', linestyle='--', alpha=0.5)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Time (s)")
            ax.grid(True)

        # 1. Speed
        ax1 = plt.subplot(3, 3, 1)
        _plot_with_events(ax1, speeds, "Speed Magnitude", "Speed (m/s)", collision_timesteps, success_timesteps)

        # 2. Acceleration
        ax2 = plt.subplot(3, 3, 2)
        _plot_with_events(ax2, acceleration_magnitudes, "Acceleration Magnitude", "Acceleration (m/s²)", collision_timesteps, success_timesteps)

        # 3. Angular velocity magnitude
        ax3 = plt.subplot(3, 3, 3)
        ang_vel_mag = np.linalg.norm(angular_velocities, axis=2)
        _plot_with_events(ax3, ang_vel_mag, "Angular Velocity Magnitude", "Angular Vel (rad/s)", collision_timesteps, success_timesteps)

        # 4. Distance to target
        ax4 = plt.subplot(3, 3, 4)
        _plot_with_events(ax4, distances_to_target, "Distance to Target", "Distance (m)", collision_timesteps, success_timesteps)

        # 5. Position components
        ax5 = plt.subplot(3, 3, 5)
        _plot_with_events(ax5, np.linalg.norm(absolute_positions, axis=2), "Position Norm", "|pos| (m)", collision_timesteps, success_timesteps)

        # 6. Velocity components
        ax6 = plt.subplot(3, 3, 6)
        _plot_with_events(ax6, np.linalg.norm(velocities, axis=2), "Velocity Norm", "|vel| (m/s)", collision_timesteps, success_timesteps)

        # 7. Collision distance if available
        ax7 = plt.subplot(3, 3, 7)
        if self.collision_all and "col_dis" in self.collision_all[0]:
            col_dis_arr = []
            for rec in self.collision_all:
                if hasattr(rec["col_dis"], 'cpu'):
                    col_dis_arr.append(rec["col_dis"].cpu().numpy())
                elif hasattr(rec["col_dis"], 'numpy'):
                    col_dis_arr.append(rec["col_dis"].numpy())
                else:
                    col_dis_arr.append(np.array(rec["col_dis"]))
            col_dis_arr = np.stack(col_dis_arr, axis=0)
            _plot_with_events(ax7, col_dis_arr, "Distance to Obstacles", "Distance (m)", collision_timesteps, success_timesteps)
        else:
            ax7.text(0.5, 0.5, 'Collision distance data not available',
                     transform=ax7.transAxes, ha='center', va='center')
            ax7.set_title("Distance to Obstacles (Not Available)")

        # 8. Reward analysis if available
        ax8 = plt.subplot(3, 3, 8)
        if hasattr(self.env, '_indiv_rewards') and self.env._indiv_rewards is not None:
            reward_components = self.env._indiv_rewards
            for component_name, component_data in reward_components.items():
                if hasattr(component_data, 'cpu'):
                    component_data = component_data.cpu().numpy()
                elif hasattr(component_data, 'numpy'):
                    component_data = component_data.numpy()
                if component_data.ndim == 1 and len(component_data) == num_timesteps:
                    ax8.plot(t, component_data, label=component_name, alpha=0.7)
            ax8.set_title("Individual Reward Components")
            ax8.set_ylabel("Reward")
            ax8.set_xlabel("Time (s)")
            ax8.legend()
            ax8.grid(True)
        else:
            ax8.text(0.5, 0.5, 'Individual reward data not available',
                     transform=ax8.transAxes, ha='center', va='center')
            ax8.set_title("Individual Reward Components (Not Available)")

        # 9. Motion jerk (smoothness)
        ax9 = plt.subplot(3, 3, 9)
        jerks = np.zeros_like(accelerations)
        for timestep_idx in range(2, num_timesteps):
            dt = t[timestep_idx] - t[timestep_idx-1]
            if dt > 0:
                jerks[timestep_idx] = (accelerations[timestep_idx] - accelerations[timestep_idx-1]) / dt
        jerk_magnitudes = np.linalg.norm(jerks, axis=2)
        _plot_with_events(ax9, jerk_magnitudes, "Motion Jerk (Smoothness)", "Jerk (m/s³)", collision_timesteps, success_timesteps)

        plt.tight_layout()
        target_path = save_path if save_path is not None else self.save_path
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        fig.savefig(f"{target_path}/debug_analysis.png", dpi=150, bbox_inches='tight')

        print("\n=== DEBUG ANALYSIS SUMMARY ===")
        print(f"Total timesteps: {num_timesteps}")
        print(f"Number of agents: {num_envs}")
        if len(success_timesteps) > 0:
            print(f"Success events: {len(success_timesteps)}")
        if len(collision_timesteps) > 0:
            print(f"Collision events: {len(collision_timesteps)}")
        if len(failure_timesteps) > 0:
            print(f"Failure events: {len(failure_timesteps)}")

        try:
            plt.show()
        except Exception:
            pass
        return [fig]

