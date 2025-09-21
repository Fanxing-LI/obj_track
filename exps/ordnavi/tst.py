import numpy as np
import matplotlib
# Set matplotlib to use non-interactive backend to avoid display issues on remote servers
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from typing import Optional
import os, sys
import cv2

from VisFly.utils.evaluate import TestBase
from VisFly.utils.FigFashion.FigFashion import FigFon

class Test(TestBase):
    def __init__(self, env, model, name, save_path: Optional[str] = None):
        super(Test, self).__init__(env=env, model=model, name=name, save_path=save_path)

    def _compute_mean_depth(self, depth_tensor):
        """
        Compute mean depth image across agents for a single timestep.

        Accepts depth in common shapes like (N, 1, H, W) or (N, H, W) where N
        is the number of agents. Returns a 2D array (H, W).
        """
        # Convert to numpy on CPU
        if hasattr(depth_tensor, 'detach') and hasattr(depth_tensor, 'cpu'):
            arr = depth_tensor.detach().cpu().numpy()
        elif hasattr(depth_tensor, 'cpu'):
            arr = depth_tensor.cpu().numpy()
        elif hasattr(depth_tensor, 'numpy'):
            arr = depth_tensor.numpy()
        else:
            arr = np.array(depth_tensor)

        # Handle shapes: (N, 1, H, W) -> (N, H, W); (N, H, W) stays; (H, W) stays
        if arr.ndim == 4 and arr.shape[1] == 1:
            arr = arr[:, 0]
        elif arr.ndim == 4:
            # Unexpected multi-channel, average channels first
            arr = arr.mean(axis=1)
        elif arr.ndim == 3:
            # Already (N, H, W) or (C, H, W). Treat first dim as batch and average over it.
            pass
        elif arr.ndim == 2:
            # Single depth map already
            return arr
        else:
            # Fallback: try to reshape last two dims as HxW if possible
            raise ValueError(f"Unsupported depth tensor shape: {arr.shape}")

        # Mean over agents/batch -> (H, W)
        return arr.mean(axis=0)

    def draw(self, names=None):
        """
        Produce diagnostic plots for navigation runs:
        - Positions over time (mean ± std across agents)
        - 3D trajectories with start/end markers and target(s)
        - Orientation quaternion over time
        - Angular velocity over time
        Returns a list of matplotlib Figure objects.
        """
        # Aggregate state history and move to CPU/NumPy if needed
        state_data = [state for state in self.state_all]
        state_data = [s.cpu() if hasattr(s, 'cpu') else s for s in state_data]
        state_data = np.array(state_data)

        if state_data.size == 0:
            # No data collected; return empty list gracefully
            return []

        # Absolute positions (XYZ) are typically in the first 3 columns
        absolute_positions = state_data[:, :, 0:3]

        # Targets from environment
        targets = getattr(self.env, 'target', None)
        if hasattr(targets, 'cpu'):
            targets = targets.cpu().numpy()

        # Gather collision points over time to visualize vectors from agent to collision point
        collision_points_list = []
        for rec in self.collision_all:
            col_pt = rec.get("col_pt", None) if isinstance(rec, dict) else None
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

        # Normalize to consistent (T, N, 3) shape with NaNs where missing
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

        # Determine which agents succeeded at any point
        success_agents = set()
        for timestep_info in self.info_all:
            for agent_idx, agent_info in enumerate(timestep_info):
                if agent_info and "is_success" in agent_info and agent_info["is_success"]:
                    success_agents.add(agent_idx)

        # Time array
        t_list = []
        for time_tensor in self.t:
            if hasattr(time_tensor, 'cpu'):
                t_list.append(time_tensor.cpu().numpy())
            else:
                t_list.append(np.array(time_tensor))
        t = np.array(t_list)[:, 0] if len(t_list) > 0 else np.arange(absolute_positions.shape[0])

        fig = plt.figure(figsize=(16, 12))

        def _plot_with_mean(ax, series, labels, title, ylabel, fill_alpha=0.15):
            # Ensure shape (T, N, C)
            if series.ndim == 2:  # (T, C)
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

        # 1) Position vs Time
        ax1 = plt.subplot(2, 2, 1)
        _plot_with_mean(ax1, absolute_positions, ["x", "y", "z"], "Position vs Time", "Position (m)")

        # 2) 3D Trajectory Plot with collision awareness vectors
        ax2 = plt.subplot(2, 2, 2, projection='3d')

        # For 3D plots, choose up to 4 agent indexes for visibility
        n_agents_to_plot = min(4, num_envs)

        for agent_idx in range(n_agents_to_plot):
            x, y, z = (
                absolute_positions[:, agent_idx, 0],
                absolute_positions[:, agent_idx, 1],
                absolute_positions[:, agent_idx, 2],
            )
            ax2.plot(x, y, z, color='lightgrey', linewidth=0.7, alpha=0.7)
            ax2.scatter(x, y, z, c=np.arange(len(x)), cmap='viridis', s=6, alpha=0.8)
            ax2.scatter(x[0], y[0], z[0], color='green', s=50, marker='o', edgecolor='black', label=f"Start {agent_idx}")
            ax2.scatter(x[-1], y[-1], z[-1], color='red', s=50, marker='X', edgecolor='black', label=f"End {agent_idx}")

            # Plot collision vectors as quivers when available
            agent_collision_vec = collision_vectors[:, agent_idx, :]
            agent_collision_pts = collision_points[:, agent_idx, :]
            valid_mask = ~np.isnan(agent_collision_vec).any(axis=1)
            if np.any(valid_mask):
                X = absolute_positions[valid_mask, agent_idx, 0]
                Y = absolute_positions[valid_mask, agent_idx, 1]
                Z = absolute_positions[valid_mask, agent_idx, 2]
                U = agent_collision_vec[valid_mask, 0]
                V = agent_collision_vec[valid_mask, 1]
                W = agent_collision_vec[valid_mask, 2]

                # Separate quivers by magnitude for quick visual cues
                magnitudes = np.linalg.norm(np.stack([U, V, W], axis=1), axis=1)
                close_mask = magnitudes < 1.0
                mid_mask = (magnitudes >= 1.0) & (magnitudes < 2.0)
                far_mask = magnitudes >= 2.0

                if np.any(close_mask):
                    ax2.quiver(
                        X[close_mask], Y[close_mask], Z[close_mask],
                        U[close_mask], V[close_mask], W[close_mask],
                        length=1.0, normalize=False, color='red', alpha=0.5, linewidth=0.8,
                    )
                if np.any(mid_mask):
                    ax2.quiver(
                        X[mid_mask], Y[mid_mask], Z[mid_mask],
                        U[mid_mask], V[mid_mask], W[mid_mask],
                        length=1.0, normalize=False, color='orange', alpha=0.35, linewidth=0.6,
                    )
                if np.any(far_mask):
                    ax2.quiver(
                        X[far_mask], Y[far_mask], Z[far_mask],
                        U[far_mask], V[far_mask], W[far_mask],
                        length=1.0, normalize=False, color='lightblue', alpha=0.25, linewidth=0.5,
                    )

                # Optional markers at collision points to highlight contacts
                ax2.scatter(
                    agent_collision_pts[valid_mask, 0],
                    agent_collision_pts[valid_mask, 1],
                    agent_collision_pts[valid_mask, 2],
                    color='magenta', s=30, marker='^', alpha=0.6,
                    label='Collision Point' if agent_idx == 0 else None,
                )

        # Plot target(s)
        if targets is not None:
            if np.ndim(targets) == 2:
                ax2.scatter(targets[:, 0], targets[:, 1], targets[:, 2], c='blue', s=60, marker='*', edgecolor='black', label="Target")
            else:
                ax2.scatter(targets[0], targets[1], targets[2], c='blue', s=60, marker='*', edgecolor='black', label="Target")

        ax2.set_title("3D Trajectories with Collision Vectors")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")
        ax2.legend(loc='upper right', fontsize='small')

        # 3) Orientation over time
        orientation = state_data[:, :, 3:7]
        ax3 = plt.subplot(2, 2, 3)
        _plot_with_mean(ax3, orientation, ["qw", "qx", "qy", "qz"], "Orientation (Quaternion)", "Quaternion Value")

        # 4) Angular velocity over time
        angular_vel = state_data[:, :, 10:13]
        ax4 = plt.subplot(2, 2, 4)
        _plot_with_mean(ax4, angular_vel, ["wx", "wy", "wz"], "Angular Velocity", "Angular Velocity (rad/s)")

        plt.tight_layout()

        return [fig]

    def draw_debug(self, names=None, save_path: Optional[str] = None):
        """Produce comprehensive debug plots for navigation runs."""
        import torch as th

        if len(self.state_all) == 0:
            return []

        state_data = th.stack(self.state_all).cpu().numpy()
        absolute_positions = state_data[:, :, 0:3]
        velocities = state_data[:, :, 7:10]
        angular_velocities = state_data[:, :, 10:13]

        targets = getattr(self.env, 'target', None)
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
                    if agent_info.get("is_collision"):
                        collision_timesteps.append((timestep_idx, agent_idx))
                    if agent_info.get("is_success"):
                        success_timesteps.append((timestep_idx, agent_idx))
                    if agent_info.get("is_failure"):
                        failure_timesteps.append((timestep_idx, agent_idx))

        distances_to_target = []
        for timestep_idx in range(num_timesteps):
            timestep_distances = []
            for agent_idx in range(num_envs):
                if targets is None:
                    target_pos = np.zeros(3, dtype=float)
                elif np.ndim(targets) == 1:
                    target_pos = targets[:3]
                elif np.ndim(targets) == 2 and targets.shape[0] > agent_idx:
                    target_pos = targets[agent_idx][:3]
                else:
                    target_pos = np.zeros(3, dtype=float)
                agent_pos = absolute_positions[timestep_idx, agent_idx, :3]
                timestep_distances.append(np.linalg.norm(agent_pos - target_pos))
            distances_to_target.append(timestep_distances)
        distances_to_target = np.array(distances_to_target)

        speeds = np.linalg.norm(velocities, axis=2)

        accelerations = np.zeros_like(velocities)
        for timestep_idx in range(1, num_timesteps):
            dt = t[timestep_idx] - t[timestep_idx - 1]
            if dt > 0:
                accelerations[timestep_idx] = (velocities[timestep_idx] - velocities[timestep_idx - 1]) / dt
        acceleration_magnitudes = np.linalg.norm(accelerations, axis=2)

        fig = plt.figure(figsize=(20, 16))

        def _plot_with_events(ax, data, title, ylabel):
            if data.ndim == 2:
                data = data[:, None, :]

            mean_vals = data.mean(axis=1)
            std_vals = data.std(axis=1)

            for ch in range(data.shape[2]):
                ax.plot(t, mean_vals[:, ch], linewidth=2, label=f"Channel {ch}")
                ax.fill_between(t, mean_vals[:, ch] - std_vals[:, ch], mean_vals[:, ch] + std_vals[:, ch], alpha=0.15)

            for idx, (timesteps, color, label) in enumerate(
                [
                    (collision_timesteps, 'red', 'Collision'),
                    (success_timesteps, 'green', 'Success'),
                    (failure_timesteps, 'orange', 'Failure'),
                ]
            ):
                if timesteps:
                    for event_idx, (timestep_idx, _) in enumerate(timesteps):
                        if timestep_idx < len(t):
                            ax.axvline(
                                x=t[timestep_idx],
                                color=color,
                                linestyle='--',
                                alpha=0.6,
                                label=label if event_idx == 0 else None,
                            )

            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Time (s)")
            ax.legend()
            ax.grid(True)

        ax1 = plt.subplot(3, 3, 1)
        _plot_with_events(ax1, distances_to_target, "Distance to Target", "Distance (m)")

        ax2 = plt.subplot(3, 3, 2)
        _plot_with_events(ax2, speeds, "Speed Magnitude", "Speed (m/s)")

        ax3 = plt.subplot(3, 3, 3)
        _plot_with_events(ax3, acceleration_magnitudes, "Acceleration Magnitude", "Acceleration (m/s²)")

        ax4 = plt.subplot(3, 3, 4)
        angular_speeds = np.linalg.norm(angular_velocities, axis=2)
        _plot_with_events(ax4, angular_speeds, "Angular Speed", "Angular Speed (rad/s)")

        ax5 = plt.subplot(3, 3, 5)
        _plot_with_events(ax5, velocities, "Velocity Components", "Velocity (m/s)")

        ax6 = plt.subplot(3, 3, 6)
        _plot_with_events(ax6, angular_velocities, "Angular Velocity Components", "Angular Velocity (rad/s)")

        ax7 = plt.subplot(3, 3, 7)
        collision_distances = None
        if hasattr(self.env, 'last_collision_info'):
            collision_info = self.env.last_collision_info
            collision_distances = collision_info.get('distance') if isinstance(collision_info, dict) else None
        if collision_distances is not None:
            collision_distances = np.array(collision_distances)
            time_len = min(len(t), collision_distances.shape[0])
            if time_len > 0:
                time_axis = t[:time_len]
                if collision_distances.ndim == 1:
                    ax7.plot(time_axis, collision_distances[:time_len], label='Collision Distance', alpha=0.7)
                else:
                    for agent_idx in range(min(num_envs, collision_distances.shape[1])):
                        ax7.plot(time_axis, collision_distances[:time_len, agent_idx], label=f"Agent {agent_idx}", alpha=0.7)
                for timestep_idx, agent_idx in collision_timesteps:
                    if timestep_idx < time_len:
                        if collision_distances.ndim == 1:
                            val = collision_distances[timestep_idx]
                        elif agent_idx < collision_distances.shape[1]:
                            val = collision_distances[timestep_idx, agent_idx]
                        else:
                            continue
                        ax7.scatter(t[timestep_idx], val, color='red', s=40, marker='x')
                ax7.set_title("Distance to Obstacles")
                ax7.set_ylabel("Distance (m)")
                ax7.set_xlabel("Time (s)")
                ax7.axhline(y=0.3, color='orange', linestyle='--', alpha=0.6, label='Safety Threshold')
                ax7.legend()
                ax7.grid(True)
            else:
                ax7.text(0.5, 0.5, 'Collision distance data empty', transform=ax7.transAxes, ha='center', va='center')
                ax7.set_axis_off()
        else:
            ax7.text(0.5, 0.5, 'Collision distance data not available', transform=ax7.transAxes, ha='center', va='center')
            ax7.set_axis_off()

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
            ax8.text(0.5, 0.5, 'Individual reward data not available', transform=ax8.transAxes, ha='center', va='center')
            ax8.set_axis_off()

        ax9 = plt.subplot(3, 3, 9)
        jerks = np.zeros_like(accelerations)
        for timestep_idx in range(2, num_timesteps):
            dt = t[timestep_idx] - t[timestep_idx - 1]
            if dt > 0:
                jerks[timestep_idx] = (accelerations[timestep_idx] - accelerations[timestep_idx - 1]) / dt
        jerk_magnitudes = np.linalg.norm(jerks, axis=2)
        _plot_with_events(ax9, jerk_magnitudes, "Motion Jerk (Smoothness)", "Jerk (m/s³)")

        plt.tight_layout()

        target_path = save_path if save_path is not None else self.save_path
        if target_path is not None:
            os.makedirs(target_path, exist_ok=True)
            fig.savefig(os.path.join(target_path, "debug_analysis.png"), dpi=150, bbox_inches='tight')

        return [fig]

    def save_combined_video(self, episode_dir):
        """
        Create and save a combined video that shows:
        - Global camera view (render_image_all)
        - Mean agent depth visualization
        """
        try:
            # Validate inputs
            if not self.render_image_all or not self.obs_all:
                return
            
            # Determine parameters
            height, width, _ = self.render_image_all[0].shape
            fps = int(1.0 / self.env.envs.dynamics.dt)
            
            # Prepare video writer
            video_path = os.path.join(episode_dir, "combined_view.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width + 200, height))
            
            # Iterate over timesteps
            for frame_idx, frame in enumerate(self.render_image_all):
                # Global camera view (left)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Mean agent depth (right)
                depth_panel = np.zeros((height, 200, 3), dtype=np.uint8)
                
                if frame_idx < len(self.obs_all) and "depth" in self.obs_all[frame_idx]:
                    # Arrange depth into a grid and average over agents
                    mean_depth = self._compute_mean_depth(self.obs_all[frame_idx]["depth"])  # (H, W)
                    
                    # Normalize depth to 0-255 and colorize
                    depth_min = float(np.min(mean_depth))
                    depth_max = float(np.max(mean_depth))
                    if depth_max > depth_min:
                        depth_norm = ((mean_depth - depth_min) / (depth_max - depth_min + 1e-6) * 255.0).astype(np.uint8)
                    else:
                        depth_norm = np.zeros_like(mean_depth, dtype=np.uint8)
                    
                    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_VIRIDIS)
                    
                    # Resize to panel size (200x200), then pad vertically
                    depth_color = cv2.resize(depth_color, (200, 200))
                    pad_vert = (height - 200) // 2
                    if pad_vert > 0:
                        depth_panel[pad_vert:pad_vert + 200, :] = depth_color
                    else:
                        # If height < 200, just center by slicing
                        start = (200 - height) // 2
                        depth_panel = depth_color[start:start+height, :]
                
                # Combine horizontally: [global | depth]
                combined = np.concatenate([frame_bgr, depth_panel], axis=1)
                video_writer.write(combined)
            
            video_writer.release()
            print(f"Combined video saved: {video_path}")
            
            # Also save individual videos
            self.save_individual_videos(episode_dir)
            
        except Exception as e:
            print(f"Error creating combined video: {e}")
            print("Continuing without video generation...")

    def save_individual_videos(self, episode_dir):
        """
        Save individual videos for global view and each agent's sensor data.
        """
        if not self.render_image_all:
            return
        
        try:
            # Save global camera video
            self.save_global_video(episode_dir)
            
            # Save agent depth videos
            self.save_agent_depth_videos(episode_dir)
        except Exception as e:
            print(f"Error saving individual videos: {e}")
            print("Continuing without individual video generation...")

    def save_global_video(self, episode_dir):
        """Save global camera view video."""
        try:
            height, width, _ = self.render_image_all[0].shape
            fps = int(1.0 / self.env.envs.dynamics.dt)
            
            video_path = os.path.join(episode_dir, "global_camera.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            
            for frame in self.render_image_all:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)
            
            video_writer.release()
            print(f"Global camera video saved: {video_path}")
        except Exception as e:
            print(f"Error saving global video: {e}")
            print("Continuing without global video...")

    def save_agent_depth_videos(self, episode_dir):
        """Save individual agent depth sensor videos."""
        if not self.obs_all or "depth" not in self.obs_all[0]:
            return
        
        try:
            num_agents = self.obs_all[0]["depth"].shape[0]
            fps = int(1.0 / self.env.envs.dynamics.dt)
            
            # Create videos for each agent
            agent_writers = {}
            
            for agent_idx in range(num_agents):
                video_path = os.path.join(episode_dir, f"agent_{agent_idx}_depth.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                agent_writers[agent_idx] = cv2.VideoWriter(video_path, fourcc, fps, (200, 200))
            
            # Process each timestep
            for obs_data in self.obs_all:
                if "depth" in obs_data:
                    depth_data = obs_data["depth"]
                    for agent_idx in range(num_agents):
                        if agent_idx < depth_data.shape[0]:
                            # Get depth frame for this agent
                            depth_frame = depth_data[agent_idx, 0]  # (H, W)

                            # Convert depth tensor to numpy if needed
                            if hasattr(depth_frame, 'cpu'):
                                depth_frame = depth_frame.cpu().numpy()
                            elif hasattr(depth_frame, 'numpy'):
                                depth_frame = depth_frame.numpy()

                            # Normalize to 0-255
                            depth_max = float(np.max(depth_frame))
                            depth_min = float(np.min(depth_frame))
                            if depth_max > depth_min:
                                depth_normalized = ((depth_frame - depth_min) / (depth_max - depth_min + 1e-6) * 255.0).astype(np.uint8)
                            else:
                                depth_normalized = np.zeros_like(depth_frame, dtype=np.uint8)

                            # Convert to color map for visualization
                            depth_rgb = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_VIRIDIS)

                            # Ensure frame size
                            depth_rgb = cv2.resize(depth_rgb, (200, 200))

                            # Write frame
                            agent_writers[agent_idx].write(depth_rgb)
            
            # Release all writers
            for writer in agent_writers.values():
                writer.release()
                
            print(f"Agent depth videos saved in: {episode_dir}")
        except Exception as e:
            print(f"Error saving agent depth videos: {e}")
            print("Continuing without agent videos...")

    def draw_debug(self, save_path: Optional[str] = None):
        fig = FigFon.drawDebug(self.eq_r, self.eq_l, self.state_all, self.obs_all, self.action_all, self.reward_all, self.info_all)
        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            fig.savefig(os.path.join(save_path, "debug_analysis.png"), dpi=150, bbox_inches='tight')
        return fig
