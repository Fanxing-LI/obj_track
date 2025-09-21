"""
Test utilities for Circle environment.

This module provides test and evaluation functionality specifically for the Circle task,
including trajectory analysis, circle tracking metrics, and visualization.

Key features:
- Circle trajectory analysis (radius deviation, speed tracking)
- Tangential velocity alignment metrics
- Heading direction analysis for circle following
- TACO-compatible evaluation framework
"""

import os
import sys
import torch as th
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# Import the base test framework from flip
from exps.flip.tst import Test as BaseTest


class Test(BaseTest):
    """
    Extended test class for Circle environment.
    
    Inherits from the flip test framework but adds circle-specific metrics:
    - Radius tracking error
    - Tangential velocity alignment
    - Heading direction accuracy
    - Circle completion analysis
    """
    
    def __init__(self, env, model, name="circle_test", save_path="./test_results"):
        """
        Initialize circle test framework.
        
        Args:
            env: CircleEnv instance
            model: Trained model for evaluation
            name: Test run identifier
            save_path: Directory to save results
        """
        super().__init__(env, model, name, save_path)
        
        # Circle-specific parameters
        self.desired_radius = getattr(env, 'desired_radius', 1.2)
        self.tangential_speed = getattr(env, 'tangential_speed', 1.0)
        self.target_position = getattr(env, 'target_position', th.tensor([0., 0., 1.5]))
        
        # Circle metrics storage
        self.radius_errors = []
        self.speed_errors = []
        self.heading_errors = []
        self.tangential_velocity_alignment = []
        
    def analyze_circle_metrics(self, positions: th.Tensor, velocities: th.Tensor, 
                             orientations: th.Tensor) -> Dict[str, float]:
        """
        Analyze circle-specific performance metrics.
        
        Args:
            positions: Agent positions over time [T, N, 3]
            velocities: Agent velocities over time [T, N, 3]  
            orientations: Agent orientations over time [T, N, 3x3] (rotation matrices)
            
        Returns:
            Dictionary of circle performance metrics
        """
        # Convert to numpy for easier analysis
        pos = positions.cpu().numpy() if hasattr(positions, 'cpu') else positions
        vel = velocities.cpu().numpy() if hasattr(velocities, 'cpu') else velocities
        target = self.target_position.cpu().numpy() if hasattr(self.target_position, 'cpu') else self.target_position
        
        metrics = {}
        
        # Radius analysis
        relative_pos = pos - target.reshape(1, 1, 3)  # [T, N, 3]
        horizontal_dist = np.linalg.norm(relative_pos[:, :, :2], axis=2)  # [T, N]
        radius_error = np.abs(horizontal_dist - self.desired_radius)
        
        metrics['mean_radius_error'] = np.mean(radius_error)
        metrics['max_radius_error'] = np.max(radius_error)
        metrics['radius_error_std'] = np.std(radius_error)
        
        # Speed analysis
        horizontal_speed = np.linalg.norm(vel[:, :, :2], axis=2)  # [T, N]
        speed_error = np.abs(horizontal_speed - self.tangential_speed)
        
        metrics['mean_speed_error'] = np.mean(speed_error)
        metrics['max_speed_error'] = np.max(speed_error)
        metrics['speed_error_std'] = np.std(speed_error)
        
        # Tangential velocity alignment
        # Compute tangent direction at each point
        for t in range(pos.shape[0]):
            for n in range(pos.shape[1]):
                rel_pos = relative_pos[t, n, :2]  # [2]
                if np.linalg.norm(rel_pos) > 1e-6:
                    # Tangent direction (perpendicular to radius, CCW)
                    tangent_dir = np.array([-rel_pos[1], rel_pos[0]])
                    tangent_dir = tangent_dir / (np.linalg.norm(tangent_dir) + 1e-8)
                    
                    # Actual velocity direction
                    vel_horizontal = vel[t, n, :2]
                    if np.linalg.norm(vel_horizontal) > 1e-6:
                        vel_dir = vel_horizontal / np.linalg.norm(vel_horizontal)
                        # Alignment: dot product of normalized vectors
                        alignment = np.dot(vel_dir, tangent_dir)
                        self.tangential_velocity_alignment.append(alignment)
        
        if self.tangential_velocity_alignment:
            metrics['mean_tangential_alignment'] = np.mean(self.tangential_velocity_alignment)
            metrics['min_tangential_alignment'] = np.min(self.tangential_velocity_alignment)
            
        # Circle completion analysis
        # Check how much of the circle was covered
        angles = []
        for t in range(pos.shape[0]):
            for n in range(pos.shape[1]):
                rel_pos = relative_pos[t, n, :2]
                if np.linalg.norm(rel_pos) > 1e-6:
                    angle = np.arctan2(rel_pos[1], rel_pos[0])
                    angles.append(angle)
        
        if angles:
            angle_range = np.max(angles) - np.min(angles)
            # Handle wrap-around
            if angle_range > np.pi:
                angle_range = 2*np.pi - angle_range
            metrics['circle_coverage'] = angle_range / (2*np.pi)
        else:
            metrics['circle_coverage'] = 0.0
            
        return metrics
    
    def draw_circle_analysis(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive circle task analysis plots.
        
        Args:
            save_path: Directory to save the plot
            
        Returns:
            Matplotlib figure with circle analysis
        """
        if not self.info_all:
            print("No trajectory data available for analysis")
            return None
            
        # Extract trajectory data
        positions = []
        velocities = []
        
        for timestep_obs in self.obs_all:
            if timestep_obs and len(timestep_obs) > 0:
                # Get position from environment state
                obs = timestep_obs[0]  # First agent
                if hasattr(obs, 'position'):
                    positions.append(obs.position.cpu().numpy())
                elif isinstance(obs, dict) and 'state' in obs:
                    # Extract position from state vector (first 3 elements typically)
                    state = obs['state']
                    if hasattr(state, 'cpu'):
                        state = state.cpu().numpy()
                    positions.append(state[:3])
        
        for timestep_info in self.info_all:
            if timestep_info and len(timestep_info) > 0:
                info = timestep_info[0]  # First agent
                if info and 'velocity' in info:
                    velocities.append(info['velocity'])
                elif hasattr(self.env, 'velocity'):
                    velocities.append(self.env.velocity[0].cpu().numpy())
        
        if not positions:
            print("No position data found for circle analysis")
            return None
            
        positions = np.array(positions)  # [T, 3]
        target = self.target_position.cpu().numpy() if hasattr(self.target_position, 'cpu') else self.target_position
        
        # Create analysis figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Circle Task Analysis - Radius: {self.desired_radius}m, Speed: {self.tangential_speed}m/s', fontsize=16)
        
        # 1. 3D trajectory with ideal circle
        ax = axes[0, 0]
        ax.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.7, label='UAV Trajectory')
        ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=8, label='Start')
        ax.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=8, label='End')
        ax.plot(target[0], target[1], 'k*', markersize=12, label='Target Center')
        
        # Draw ideal circle
        angles = np.linspace(0, 2*np.pi, 100)
        circle_x = target[0] + self.desired_radius * np.cos(angles)
        circle_y = target[1] + self.desired_radius * np.sin(angles)
        ax.plot(circle_x, circle_y, 'r--', linewidth=2, label=f'Ideal Circle (r={self.desired_radius}m)')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Top View - Circle Trajectory')
        ax.legend()
        ax.grid(True)
        ax.axis('equal')
        
        # 2. Radius error over time
        ax = axes[0, 1]
        relative_pos = positions - target.reshape(1, 3)
        horizontal_dist = np.linalg.norm(relative_pos[:, :2], axis=1)
        radius_error = np.abs(horizontal_dist - self.desired_radius)
        
        time_steps = np.arange(len(radius_error))
        ax.plot(time_steps, radius_error, 'b-', linewidth=2)
        ax.axhline(y=0.1, color='r', linestyle='--', alpha=0.7, label='±0.1m tolerance')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Radius Error (m)')
        ax.set_title('Radius Tracking Error')
        ax.legend()
        ax.grid(True)
        
        # 3. Height variation
        ax = axes[0, 2]
        height_error = np.abs(positions[:, 2] - target[2])
        ax.plot(time_steps, height_error, 'g-', linewidth=2)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Height Error (m)')
        ax.set_title('Altitude Tracking Error')
        ax.grid(True)
        
        # 4. Speed analysis (if velocity data available)
        ax = axes[1, 0]
        if velocities:
            velocities = np.array(velocities)
            horizontal_speed = np.linalg.norm(velocities[:, :2], axis=1)
            speed_error = np.abs(horizontal_speed - self.tangential_speed)
            
            ax.plot(time_steps[:len(speed_error)], speed_error, 'purple', linewidth=2)
            ax.axhline(y=0.2, color='r', linestyle='--', alpha=0.7, label='±0.2m/s tolerance')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Speed Error (m/s)')
            ax.set_title('Tangential Speed Error')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Velocity data not available', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            ax.set_title('Speed Analysis (No Data)')
        ax.grid(True)
        
        # 5. Distance from target over time
        ax = axes[1, 1]
        target_distance = np.linalg.norm(relative_pos, axis=1)
        ax.plot(time_steps, target_distance, 'orange', linewidth=2)
        ax.axhline(y=self.desired_radius, color='r', linestyle='-', alpha=0.7, 
                  label=f'Desired Distance ({self.desired_radius}m)')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Distance to Target (m)')
        ax.set_title('3D Distance to Target')
        ax.legend()
        ax.grid(True)
        
        # 6. Performance summary statistics
        ax = axes[1, 2]
        stats_text = f"""Circle Performance Summary:
        
Mean Radius Error: {np.mean(radius_error):.3f}m
Max Radius Error: {np.max(radius_error):.3f}m
Radius Error Std: {np.std(radius_error):.3f}m

Mean Height Error: {np.mean(height_error):.3f}m
Max Height Error: {np.max(height_error):.3f}m

Episode Length: {len(positions)} steps
Final Distance: {target_distance[-1]:.3f}m"""
        
        if velocities:
            stats_text += f"""

Mean Speed Error: {np.mean(speed_error):.3f}m/s
Max Speed Error: {np.max(speed_error):.3f}m/s"""
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax.axis('off')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            fig_path = os.path.join(save_path, 'circle_analysis.png')
            fig.savefig(fig_path, dpi=150, bbox_inches='tight')
            print(f"Circle analysis saved: {fig_path}")
            
        return fig
        
    def draw_debug(self, save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """
        Override parent draw_debug to include circle-specific analysis.
        
        Args:
            save_path: Directory to save debug plots
            
        Returns:
            Circle analysis figure
        """
        # Call parent debug function first
        parent_fig = super().draw_debug(save_path)
        
        # Add circle-specific analysis
        circle_fig = self.draw_circle_analysis(save_path)
        
        return circle_fig