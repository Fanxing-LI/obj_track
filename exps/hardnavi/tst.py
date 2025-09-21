import numpy as np
import matplotlib
# Set matplotlib to use non-interactive backend to avoid display issues on remote servers
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional
import os, sys
import cv2

from VisFly.utils.evaluate import TestBase
from VisFly.utils.FigFashion.FigFashion import FigFon

class Test(TestBase):
    def __init__(self, env, model, name, save_path: Optional[str] = None):
        super(Test, self).__init__(env=env, model=model, name=name, save_path=save_path)

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

        # 2) 3D Trajectories
        ax2 = plt.subplot(2, 2, 2, projection='3d')
        for agent_idx in range(num_envs):
            x_pos = absolute_positions[:, agent_idx, 0]
            y_pos = absolute_positions[:, agent_idx, 1]
            z_pos = absolute_positions[:, agent_idx, 2]
            vel = state_data[:, agent_idx, 7:10]
            speed = np.linalg.norm(vel, axis=1)
            ax2.plot(x_pos, y_pos, z_pos, color='lightgrey', linewidth=0.7, alpha=0.6)
            ax2.scatter(x_pos, y_pos, z_pos, c=speed, cmap='viridis', s=8, alpha=0.9)

            # Collision vectors (colored by distance)
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
                               length=1.0, normalize=False, color='red', alpha=0.6, linewidth=0.8)
                far_mask = ~near_mask
                if np.any(far_mask):
                    ax2.quiver(X[far_mask], Y[far_mask], Z[far_mask], U[far_mask], V[far_mask], W[far_mask],
                               length=1.0, normalize=False, color='orange', alpha=0.5, linewidth=0.6)

            # Start marker
            ax2.scatter(x_pos[0], y_pos[0], z_pos[0], color='green', s=50, marker='o', edgecolor='black', linewidth=1)
            # End marker (success/failure)
            if agent_idx in success_agents:
                ax2.scatter(x_pos[-1], y_pos[-1], z_pos[-1], color='blue', s=50, marker='s', edgecolor='black', linewidth=1)
            else:
                ax2.scatter(x_pos[-1], y_pos[-1], z_pos[-1], color='red', s=50, marker='x', linewidth=1)

        # Targets
        if targets is not None:
            if isinstance(targets, np.ndarray) and targets.ndim == 1 and targets.size >= 3:
                ax2.scatter(targets[0], targets[1], targets[2], color='gold', s=80, marker='*', edgecolor='black', linewidth=1, label='Target')
            elif isinstance(targets, np.ndarray) and targets.ndim == 2:
                for agent_idx in range(min(targets.shape[0], num_envs)):
                    target_pos = targets[agent_idx]
                    if target_pos.size >= 3:
                        ax2.scatter(target_pos[0], target_pos[1], target_pos[2], color='gold', s=80, marker='*', edgecolor='black', linewidth=1, label='Target' if agent_idx == 0 else None)
        ax2.set_xlabel("X Position (m)")
        ax2.set_ylabel("Y Position (m)")
        ax2.set_zlabel("Z Position (m)")
        ax2.set_title("3D Trajectory")

        # Equalize axes
        all_x = absolute_positions[:, :, 0].flatten()
        all_y = absolute_positions[:, :, 1].flatten()
        all_z = absolute_positions[:, :, 2].flatten()
        if isinstance(targets, np.ndarray):
            if targets.ndim == 1 and targets.size >= 3:
                all_x = np.append(all_x, targets[0])
                all_y = np.append(all_y, targets[1])
                all_z = np.append(all_z, targets[2])
            elif targets.ndim == 2:
                for i in range(min(targets.shape[0], num_envs)):
                    all_x = np.append(all_x, targets[i, 0])
                    all_y = np.append(all_y, targets[i, 1])
                    all_z = np.append(all_z, targets[i, 2])
        max_range = np.array([all_x.max()-all_x.min(), all_y.max()-all_y.min(), all_z.max()-all_z.min()]).max() / 2.0
        mid_x = (all_x.max()+all_x.min()) * 0.5
        mid_y = (all_y.max()+all_y.min()) * 0.5
        mid_z = (all_z.max()+all_z.min()) * 0.5
        ax2.set_xlim(mid_x - max_range, mid_x + max_range)
        ax2.set_ylim(mid_y - max_range, mid_y + max_range)
        ax2.set_zlim(mid_z - max_range, mid_z + max_range)

        # 3) Orientation quaternion (wxyz assumed at indices 3:7)
        ax3 = plt.subplot(2, 2, 3)
        _plot_with_mean(ax3, state_data[:, :, 3:7], ["w", "x", "y", "z"], "Orientation Quaternion", "Quaternion")

        # 4) Angular velocity (indices 10:13)
        ax4 = plt.subplot(2, 2, 4)
        _plot_with_mean(ax4, state_data[:, :, 10:13], ["wx", "wy", "wz"], "Angular Velocity", "Angular Velocity (rad/s)")

        plt.tight_layout()

        # Best-effort show (ignored on headless)
        try:
            plt.show()
        except Exception:
            pass

        return [fig]

    def create_combined_video_frame(self, global_frame, agent_sensor_data, timestep_idx):
        """
        Create a combined frame with global view and agent sensor views.
        
        Args:
            global_frame: Global camera rendered frame
            agent_sensor_data: Dictionary containing sensor data for all agents
            timestep_idx: Current timestep index
        
        Returns:
            Combined frame as numpy array
        """
        # Get global frame dimensions
        global_h, global_w = global_frame.shape[:2]
        
        # Collect agent sensor frames (depth images)
        agent_frames = []
        if "depth" in agent_sensor_data:
            depth_data = agent_sensor_data["depth"]
            
            # Convert to CPU and numpy if it's a CUDA tensor
            if hasattr(depth_data, 'cpu'):
                depth_data = depth_data.cpu().numpy()
            elif hasattr(depth_data, 'numpy'):
                depth_data = depth_data.numpy()
            
            # Normalize and convert depth to RGB for visualization
            for agent_idx in range(depth_data.shape[0]):
                depth_frame = depth_data[agent_idx, 0]  # Remove channel dimension
                # Normalize depth to 0-255 range
                depth_max = depth_frame.max()
                if depth_max > 0:
                    depth_normalized = ((depth_frame / depth_max) * 255).astype(np.uint8)
                else:
                    depth_normalized = np.zeros_like(depth_frame, dtype=np.uint8)
                # Convert to RGB
                depth_rgb = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_VIRIDIS)
                # Resize to reasonable size
                agent_frames.append(cv2.resize(depth_rgb, (200, 200)))
        
        # Create layout: global view on left, agent views on right
        if agent_frames:
            # Calculate grid layout for agent views
            num_agents = len(agent_frames)
            grid_cols = min(4, num_agents)  # Max 4 columns
            grid_rows = (num_agents + grid_cols - 1) // grid_cols
            
            # Create agent grid
            agent_grid_h = grid_rows * 200
            agent_grid_w = grid_cols * 200
            agent_grid = np.zeros((agent_grid_h, agent_grid_w, 3), dtype=np.uint8)
            
            for i, frame in enumerate(agent_frames):
                row = i // grid_cols
                col = i % grid_cols
                y_start = row * 200
                x_start = col * 200
                agent_grid[y_start:y_start+200, x_start:x_start+200] = frame
            
            # Resize global frame to match agent grid height
            global_resized = cv2.resize(global_frame, (int(global_w * agent_grid_h / global_h), agent_grid_h))
            
            # Combine horizontally
            combined_frame = np.hstack([global_resized, agent_grid])
        else:
            combined_frame = global_frame
        
        # Add text overlay with timestep info
        cv2.putText(combined_frame, f"Timestep: {timestep_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return combined_frame

    def save_combined_video(self, episode_dir):
        """
        Save a combined video with global camera view and agent sensor views.
        """
        if not self.render_image_all:
            print("No render images available for video creation")
            return
        
        try:
            # Quiet: avoid verbose start message
            
            # Create combined frames
            combined_frames = []
            for i, (render_frame, obs_data) in enumerate(zip(self.render_image_all, self.obs_all)):
                # Convert render frame from RGB to BGR for OpenCV
                render_bgr = cv2.cvtColor(render_frame, cv2.COLOR_RGB2BGR)
                
                # Create combined frame
                combined_frame = self.create_combined_video_frame(render_bgr, obs_data, i)
                combined_frames.append(combined_frame)
            
            if not combined_frames:
                print("No frames to save")
                return
            
            # Video settings
            height, width = combined_frames[0].shape[:2]
            fps = int(1.0 / self.env.envs.dynamics.dt)  # Use environment timestep
            
            # Save combined video
            video_path = os.path.join(episode_dir, "combined_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            
            for frame in combined_frames:
                video_writer.write(frame)
            
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
