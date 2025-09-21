#!/usr/bin/env python3

import sys
import os
import torch as th
import traceback

# Set offscreen rendering for headless testing
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Set matplotlib to use non-interactive backend to avoid display issues on remote servers
import matplotlib
matplotlib.use('Agg')

sys.path.append(os.path.join(os.getcwd(), '../../'))
sys.path.append(os.getcwd())
from envs.AcrobaticEnv import AcrobaticEnv
from VisFly.utils.algorithms.BPTT import BPTT
from algorithms.SHAC import SHAC
from VisFly.utils.algorithms.PPO import PPO
from VisFly.utils.common import load_yaml_config
from VisFly.utils.policies import extractors  # noqa: F401
from VisFly.utils.evaluate import TestBase
import argparse
import cv2
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def parse_args():
    parser = argparse.ArgumentParser(description='Run acrobatic flight experiments', add_help=True)
    parser.add_argument('--comment', '-c', type=str, default="")
    parser.add_argument("--train", "-t", type=int, default=1)
    parser.add_argument("--algorithm", "-a", type=str, default="BPTT",
                        choices=["BPTT", "SHAC", "PPO"],
                        help="Algorithm to use for training")
    parser.add_argument("--env", "-e", type=str, default="acrobatic")
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--weight", "-w", type=str, default=None)
    parser.add_argument("--task_mode", "-m", type=str, default=None,
                        choices=["circle", "flip", "mix"],
                        help="Override task mode (circle, flip, or mix). If not specified, uses config file setting.")
    return parser

args = parse_args().parse_args()

save_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + f"/saved/{args.env}/"

# Load configuration files
config = load_yaml_config(os.path.dirname(os.path.abspath(__file__)) + f'/alg_cfgs/{args.env}/{args.algorithm}.yaml')
env_config = load_yaml_config(os.path.dirname(os.path.abspath(__file__)) + f'/env_cfgs/{args.env}.yaml')

# Environment aliases
env_alias = {
    "acrobatic": AcrobaticEnv,
}

alg_alias = {
    "BPTT": BPTT,
    "SHAC": SHAC,
    "PPO": PPO,
}

# Import test class for acrobatic flight
class Test(TestBase):
    def __init__(self, env, model, name, save_path=None):
        super(Test, self).__init__(env=env, model=model, name=name, save_path=save_path)
    
    def test(self, is_fig=True, is_video=True, is_sub_video=True, is_fig_save=True, is_video_save=True, render_kwargs={}):
        """Override test method to properly handle is_sub_video parameter."""
        # Call parent test method but handle video saving ourselves
        result = super().test(
            is_fig=is_fig, 
            is_video=False,  # Disable video saving in parent
            is_sub_video=is_sub_video, 
            is_fig_save=is_fig_save, 
            is_video_save=False,  # Handle video saving ourselves
            render_kwargs=render_kwargs
        )
        
        # Handle video saving with correct is_sub_video parameter
        if is_video and is_video_save:
            self.save_video(is_sub_video=is_sub_video)
            
        return result
    
    def save_video(self, is_sub_video=False):
        """Override save_video to use H.264 AVC1 codec for VSCode compatibility."""
        if not self.render_image_all:
            print("No render images available for video creation")
            return
            
        height, width, layers = self.render_image_all[0].shape
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        # Main camera video with H.264 AVC1 codec
        path = f"{self.save_path}/video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec for VSCode compatibility
        fps = int(1.0 / self.env.envs.dynamics.dt)
        video = cv2.VideoWriter(path, fourcc, fps, (width, height))
        
        # Write main video frames
        for image in self.render_image_all:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            video.write(image_bgr)
        
        video.release()
        print(f"video saved in {path}")
        
        # Save sensor videos if requested
        if is_sub_video:
            self.save_sensor_videos()
    
    def save_sensor_videos(self):
        """Save individual sensor videos with H.264 AVC1 codec."""
        if not hasattr(self, 'obs_all') or not self.obs_all:
            print("No observation data available for sensor videos")
            return
        
        # Get sensor names (exclude state and command)
        obs_keys = self.obs_all[0].keys()
        sensor_names = [key for key in obs_keys if key not in ['state', 'command']]
        
        if not sensor_names:
            print("No sensor data found in observations (depth camera may need to be included in training)")
            return
        
        fps = int(1.0 / self.env.envs.dynamics.dt)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec for VSCode compatibility
        
        for sensor_name in sensor_names:
            try:
                if sensor_name == "depth":
                    self.save_depth_video(sensor_name, fourcc, fps)
                else:
                    print(f"Sensor '{sensor_name}' video saving not implemented yet")
            except Exception as e:
                print(f"Error saving {sensor_name} video: {e}")
    
    def save_depth_video(self, sensor_name, fourcc, fps):
        """Save depth sensor video with proper visualization."""
        if sensor_name not in self.obs_all[0]:
            return
            
        # Get depth data shape from first observation
        depth_shape = self.obs_all[0][sensor_name].shape
        if len(depth_shape) != 4:  # Expected: [batch, channels, height, width]
            print(f"Unexpected depth shape: {depth_shape}")
            return
        
        num_agents, channels, height, width = depth_shape
        
        # Create video writers for each agent
        agent_writers = {}
        for agent_idx in range(num_agents):
            video_path = f"{self.save_path}/agent_{agent_idx}_{sensor_name}.mp4"
            # Use reasonable size for depth visualization
            agent_writers[agent_idx] = cv2.VideoWriter(video_path, fourcc, fps, (width*2, height*2))
        
        # Process each timestep
        for obs_data in self.obs_all:
            if sensor_name in obs_data:
                depth_data = obs_data[sensor_name]
                
                # Convert to CPU and numpy if it's a tensor
                if hasattr(depth_data, 'cpu'):
                    depth_data = depth_data.cpu().numpy()
                elif hasattr(depth_data, 'numpy'):
                    depth_data = depth_data.numpy()
                
                # Process each agent's depth data
                for agent_idx in range(num_agents):
                    if agent_idx < depth_data.shape[0]:
                        # Extract single agent depth frame [channels, height, width]
                        agent_depth = depth_data[agent_idx]
                        
                        # Handle single channel depth (most common case)
                        if agent_depth.shape[0] == 1:
                            depth_frame = agent_depth[0]  # Remove channel dimension
                        else:
                            depth_frame = agent_depth[0]  # Take first channel
                        
                        # Normalize depth for visualization (0-10m range)
                        max_depth = 10.0
                        depth_clipped = np.clip(depth_frame, 0, max_depth)
                        depth_normalized = (depth_clipped / max_depth * 255).astype(np.uint8)
                        
                        # Apply colormap for better visualization
                        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_VIRIDIS)
                        
                        # Resize for better visibility
                        depth_resized = cv2.resize(depth_colored, (width*2, height*2))
                        
                        # Write frame
                        agent_writers[agent_idx].write(depth_resized)
        
        # Release all video writers and print paths
        for agent_idx, writer in agent_writers.items():
            writer.release()
            video_path = f"{self.save_path}/agent_{agent_idx}_{sensor_name}.mp4"
            print(f"sensor video saved in {video_path}")
    
    def draw(self, save_path=None):
        """Override the draw method to create trajectory plots."""
        return self.draw_trajectory(save_path)

    def draw_trajectory(self, save_path=None):
        """
        Draw trajectory plots for acrobatic flight.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        figs = []
        
        # Convert lists to numpy arrays for easier manipulation
        position_data = np.array([s[:, :3].cpu().numpy() for s in self.state_all])  # [T, N, 3]
        
        num_agents = position_data.shape[1]
        
        # Create figure for trajectory
        fig = plt.figure(figsize=(15, 10))
        
        # 3D trajectory plot
        ax1 = fig.add_subplot(221, projection='3d')
        
        for agent_idx in range(num_agents):
            trajectory = position_data[:, agent_idx, :]
            ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                    label=f'Agent {agent_idx}', alpha=0.7)
            # Mark start and end
            ax1.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                       marker='o', s=50, c='green')
            ax1.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
                       marker='x', s=50, c='red')
        
        # Draw target circle for circle task
        if hasattr(self.env, 'task_mode') and self.env.task_mode == "circle":
            theta = np.linspace(0, 2*np.pi, 100)
            circle_radius = 1.2  # From AcrobaticEnv
            circle_height = 1.5  # Updated height
            circle_center_x = 6.0  # Updated to new target position
            circle_center_y = 0.0
            circle_x = circle_center_x + circle_radius * np.cos(theta)
            circle_y = circle_center_y + circle_radius * np.sin(theta)
            circle_z = np.ones_like(theta) * circle_height
            ax1.plot(circle_x, circle_y, circle_z, 'b--', linewidth=2, label='Target Circle')
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Trajectory')
        ax1.legend()
        ax1.grid(True)
        
        # XY plane projection
        ax2 = fig.add_subplot(222)
        for agent_idx in range(num_agents):
            trajectory = position_data[:, agent_idx, :]
            ax2.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.7)
            ax2.scatter(trajectory[0, 0], trajectory[0, 1], marker='o', s=30, c='green')
            ax2.scatter(trajectory[-1, 0], trajectory[-1, 1], marker='x', s=30, c='red')
        
        if hasattr(self.env, 'task_mode') and self.env.task_mode == "circle":
            circle = plt.Circle((6.0, 0.0), 1.2, fill=False, linestyle='--', color='blue', linewidth=2)
            ax2.add_patch(circle)
        
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Top-down View (XY Plane)')
        ax2.grid(True)
        ax2.axis('equal')
        
        # Height over time
        ax3 = fig.add_subplot(223)
        time = np.arange(position_data.shape[0]) * self.env.envs.dynamics.ctrl_dt
        for agent_idx in range(num_agents):
            ax3.plot(time, position_data[:, agent_idx, 2], alpha=0.7)
        
        if hasattr(self.env, 'task_mode') and self.env.task_mode == "circle":
            ax3.axhline(y=1.5, color='b', linestyle='--', label='Target Height')
        
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Height (m)')
        ax3.set_title('Height Profile')
        ax3.grid(True)
        ax3.legend()
        
        # Velocity magnitude over time
        ax4 = fig.add_subplot(224)
        velocity_data = np.array([s[:, 7:10].cpu().numpy() for s in self.state_all])
        velocity_mag = np.linalg.norm(velocity_data, axis=2)
        
        for agent_idx in range(num_agents):
            ax4.plot(time, velocity_mag[:, agent_idx], alpha=0.7)
        
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Velocity (m/s)')
        ax4.set_title('Velocity Magnitude')
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(os.path.join(save_path, 'trajectory_plot.png'), dpi=150, bbox_inches='tight')
            print(f"Trajectory plot saved to {os.path.join(save_path, 'trajectory_plot.png')}")
        
        figs.append(fig)
        return figs

# Apply task_mode override if provided
if args.task_mode is not None:
    print(f"Overriding task_mode to: {args.task_mode}")
    env_config["env"]["task_mode"] = args.task_mode
    env_config["eval_env"]["task_mode"] = args.task_mode

# Create training environment
train_env = env_alias[args.env](
    **env_config["env"]
)

# Create evaluation environment
if not args.train:
    env_config["eval_env"]["visual"] = True

env = env_alias[args.env](
    **env_config["eval_env"]
)

def main():
    try:
        # Training mode
        if args.train:
            print(f"Starting training with algorithm: {args.algorithm}")
            
            # Create algorithm instance
            model = alg_alias[args.algorithm](
                env=train_env,
                seed=args.seed,
                comment=args.comment,
                save_path=save_folder,
                **config['algorithm']
            )
            
            # Load existing model if specified
            if args.weight:
                weight_path = save_folder + args.weight
                model.load(path=weight_path, env=env)
                print(f"Loaded weights from: {weight_path}")
            
            # Start training
            model.learn(**config['learn'])
            
            # Save the model with appropriate naming
            if args.comment:
                model_path = f"{save_folder}{args.algorithm}_{args.comment}"
            else:
                # Find the next available index
                import glob
                existing = glob.glob(f"{save_folder}{args.algorithm}__*.pth") + glob.glob(f"{save_folder}{args.algorithm}__*.zip")
                if existing:
                    indices = [int(f.split('__')[-1].split('.')[0]) for f in existing if '__' in f and f.split('__')[-1].split('.')[0].isdigit()]
                    next_index = max(indices) + 1 if indices else 1
                else:
                    next_index = 1
                model_path = f"{save_folder}{args.algorithm}__{next_index}"
            
            model.save(model_path)
            
        # Testing mode
        else:
            if not args.weight:
                print("Error: Weight file required for testing mode (use --weight)")
                sys.exit(1)
            
            print(f"Starting testing with weight: {args.weight}")
            
            # Load model weights
            weight_path = save_folder + args.weight
            
            # Create test environment and reset it
            test_env = env_alias[args.env](**env_config["eval_env"])
            test_env.reset()
            
            # Load the trained model - different algorithms use different loading methods
            print(f"Loading model from: {weight_path}")
            if args.algorithm == "BPTT":
                # BPTT uses instance method load with .pth files
                model = alg_alias[args.algorithm](
                    env=test_env,
                    seed=args.seed,
                    comment=args.comment,
                    save_path=save_folder,
                    **config['algorithm']
                )
                model.load(weight_path)
            elif args.algorithm == "SHAC":
                # SHAC uses class method load with .zip files
                model = alg_alias[args.algorithm].load(weight_path, env=test_env)
            else:  # PPO
                # PPO uses instance method load with .zip files
                model = alg_alias[args.algorithm](
                    env=test_env,
                    seed=args.seed,
                    comment=args.comment,
                    save_path=save_folder,
                    **config['algorithm']
                )
                model.load(weight_path)
            
            # Create test handler
            test_handle = Test(
                env=test_env,
                model=model,
                save_path=os.path.join(save_folder, "test"),
                name=args.weight
            )
            
            # Run test with parameters from config
            test_handle.test(**config['test'])
            
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()