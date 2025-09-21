import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch as th
import matplotlib.pyplot as plt
import numpy as np

# Load trajectory B data
data = th.load("B_1.0_SHAC_Dis3.0.pth")
target_positions = th.stack([tar for tar in data["target_all"]]).squeeze()[:,0,:]

# Plot just the trajectory
plt.figure(figsize=(8, 6))
plt.plot(target_positions[:, 0].cpu().numpy(),
         target_positions[:, 1].cpu().numpy(),
         'b-', linewidth=2, label='Trajectory B')
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Trajectory B - Actual Shape')
plt.legend()
plt.savefig('/home/simonwsy/files/obj_track/plots/traj_B_shape.png')
plt.show()

# Print some stats
print(f"Shape: {target_positions.shape}")
print(f"X range: {target_positions[:, 0].min():.2f} to {target_positions[:, 0].max():.2f}")
print(f"Y range: {target_positions[:, 1].min():.2f} to {target_positions[:, 1].max():.2f}")