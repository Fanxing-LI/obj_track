# Hard Navigation Collision Training Improvement Guide

## Current Problem Analysis

The hard navigation environment (`hardnavi`) is struggling with collision avoidance compared to the ordinary navigation environment (`ordnavi`):
- **ordnavi**: Reaches ~500 steps successfully
- **hardnavi**: Only reaches ~200 steps due to collision issues

## Key Differences Between Environments

### Scene Complexity
- **ordnavi**: Uses `box15_wall_slight_pillar` (lighter obstacle density)
- **hardnavi**: Uses `box15_wall_hard_pillar` (denser obstacle placement)

### Current Collision Reward Functions

**ordnavi (working well)**:
```python
# Exponential decay penalty - strong but smooth
r_collision_dist_penalty = -0.01 * th.exp(-1.0 * (collision_dist - 0.8))
```

**hardnavi (current - suboptimal)**:
```python
# Linear penalty - weaker and less smooth
r_collision_dist_penalty = -0.005 * th.relu(0.8 - collision_dist)
```

## Collision Training Improvement Strategies

### 1. **Restore Ordnavi's Exponential Penalty** ⭐ **HIGH PRIORITY**

Replace the current linear penalty with ordnavi's proven exponential penalty:

```python
# In HardNavigationEnv.py, line 130, replace:
# r_collision_dist_penalty = -0.005 * th.relu(0.8 - collision_dist)

# With ordnavi's successful approach:
r_collision_dist_penalty = -0.01 * th.exp(-1.0 * (collision_dist - 0.8))
```

**Why this works better:**
- Exponential decay provides smoother gradients
- Stronger penalty when very close to obstacles
- Avoids cliff-edge behavior of ReLU function

### 2. **Add Velocity-Toward-Obstacle Penalty** ⭐ **HIGH PRIORITY**

Implement the velocity-based collision avoidance from `NavigationEnv.py`:

```python
# Add after line 122 in HardNavigationEnv.py:
collision_unit = self.collision_vector / (collision_dist.unsqueeze(1) + 1e-8)
rel_vel_along_collision = (self.velocity * collision_unit).sum(dim=1)
rel_vel_along_collision = th.relu(rel_vel_along_collision)  # Only penalize moving toward obstacles
r_velocity_toward_obstacle = -0.008 * rel_vel_along_collision * th.relu(0.8 - collision_dist)

# Add to reward calculation (line 141):
reward = (
    (self.position - self.target).norm(dim=1) * pos_factor + 
    r_perception_aware +                
    r_collision_dist_penalty +
    r_velocity_toward_obstacle +         # ADD THIS LINE
    ((self.velocity - 0).norm(dim=1) * -0.002) +   
    ((self.angular_velocity - 0).norm(dim=1) * -0.01) 
)
```

**Benefits:**
- Teaches implicit avoidance behavior
- Penalizes dangerous movements, not just proximity
- More predictive collision avoidance

### 3. **Multi-Scale Distance Penalties** ⭐ **MEDIUM PRIORITY**

Implement a piecewise penalty system for different danger zones:

```python
# Replace single penalty with multi-scale approach:
def get_collision_penalty(collision_dist):
    # Close danger zone (immediate threat)
    close_penalty = -0.02 * th.exp(-3.0 * (collision_dist - 0.3)) * th.relu(0.3 - collision_dist)
    
    # Medium danger zone (preparation for avoidance)
    medium_penalty = -0.01 * th.exp(-1.5 * (collision_dist - 0.6)) * th.relu(0.6 - collision_dist) * th.relu(collision_dist - 0.3)
    
    # Far warning zone (awareness)
    far_penalty = -0.005 * th.exp(-0.8 * (collision_dist - 0.8)) * th.relu(0.8 - collision_dist) * th.relu(collision_dist - 0.6)
    
    return close_penalty + medium_penalty + far_penalty

r_collision_dist_penalty = get_collision_penalty(collision_dist)
```

**Benefits:**
- Different behavior learned for different proximity levels
- Smoother learning curve
- More nuanced collision avoidance

### 4. **Depth-Aware Collision Penalties** ⭐ **MEDIUM PRIORITY**

Use depth sensor information for better spatial awareness:

```python
# Add depth-based collision prediction:
def get_depth_collision_reward(self):
    depth_obs = self.sensor_obs["depth"][0]  # Shape: (num_agents, 1, 64, 64)
    
    # Focus on central region for collision prediction
    center_h, center_w = 32, 32
    region_size = 16
    
    central_depth = depth_obs[:, 0, 
                             center_h-region_size:center_h+region_size,
                             center_w-region_size:center_w+region_size]
    
    # Penalty for very close obstacles in forward direction
    min_forward_depth = central_depth.min(dim=2)[0].min(dim=1)[0]  # (num_agents,)
    r_depth_collision = -0.005 * th.exp(-2.0 * (min_forward_depth - 0.2))
    
    return r_depth_collision

# Add to reward calculation:
r_depth_collision = get_depth_collision_reward(self)
reward = reward + r_depth_collision
```

### 5. **Curriculum Learning for Collision Training** ⭐ **LOW PRIORITY**

Gradually increase collision penalty strength during training:

```python
# Add to __init__ method:
self.training_step = 0
self.collision_curriculum_steps = 1000000  # 1M steps

# In get_reward method:
def get_adaptive_collision_penalty(self, base_penalty):
    # Start with 50% penalty strength, gradually increase to 100%
    curriculum_progress = min(1.0, self.training_step / self.collision_curriculum_steps)
    penalty_scale = 0.5 + 0.5 * curriculum_progress
    self.training_step += 1
    return base_penalty * penalty_scale

r_collision_dist_penalty = get_adaptive_collision_penalty(self, r_collision_dist_penalty)
```

## Implementation Priority

### Phase 1: Quick Wins (1-2 hours)
1. **Restore exponential penalty from ordnavi** ⭐
2. **Add velocity-toward-obstacle penalty** ⭐

### Phase 2: Advanced Improvements (3-5 hours)
3. **Implement multi-scale distance penalties**
4. **Add depth-aware collision prediction**

### Phase 3: Long-term Optimizations (1-2 days)
5. **Implement curriculum learning**
6. **Fine-tune penalty coefficients through hyperparameter search**

## Expected Improvements

With these changes, you should see:
- **Immediate**: 50-100 additional survival steps
- **Medium-term**: Approach ordnavi's ~500 step performance
- **Long-term**: Potentially exceed ordnavi performance with better obstacle prediction

## Hyperparameter Tuning Suggestions

After implementing the core improvements, fine-tune these coefficients:

```python
# Current values → Suggested starting values
collision_distance_penalty: -0.005 → -0.01
velocity_toward_penalty: None → -0.008
collision_threshold: 0.8 → 0.8 (keep)
exponential_decay_rate: None → -1.0 (from ordnavi)
```

## Monitoring and Debugging

Track these metrics during training:
- Average episode length
- Collision rate per episode
- Average minimum collision distance
- Reward component breakdown

Use these debug commands:
```bash
# Monitor training progress
tensorboard --logdir=exps/hardnavi/saved/hard_navigation/

# Test collision behavior
python exps/hardnavi/tst.py
```

## Notes

- Keep all other reward components unchanged as requested
- The exponential penalty from ordnavi is the most critical change
- Test incremental improvements rather than implementing everything at once
- Monitor for reward hacking or unexpected behaviors after each change

---

# Advanced VisFly Collision & Depth Exploitation Strategies

*Based on deep analysis of VisFly's collision detection and depth camera capabilities*

## VisFly System Analysis

### Available Collision Data
VisFly provides rich collision information through:
- `collision_point`: 3D coordinates of nearest collision point
- `collision_vector`: Vector from drone to collision point  
- `collision_dis`: Euclidean distance to collision
- `is_collision`: Boolean collision status
- `get_point_is_collision()`: API for collision testing at arbitrary points

### Depth Sensor Capabilities
The depth camera provides:
- **Resolution**: 64x64 depth map per agent
- **Range**: 0-10m (normalized to 0-1 in observation)
- **Format**: `self.sensor_obs["depth"]` shape `(num_agents, 1, 64, 64)`
- **Frequency**: Updated every environment step

## Novel Advanced Strategies

### 6. **Directional Collision Awareness** ⭐ **HIGH PRIORITY**

Exploit `collision_vector` for spatial collision understanding:

```python
def get_directional_collision_penalty(self):
    collision_dist = self.collision_vector.norm(dim=1)
    
    # Get collision direction in drone's local frame
    orientation = self.envs.dynamics._orientation.clone()
    local_collision_vec = orientation.world_to_head(self.collision_vector.T).T
    
    # Stronger penalty for frontal collisions (negative x in local frame)
    front_collision_factor = th.clamp(-local_collision_vec[:, 0], min=0, max=1)
    
    # Side collision awareness (y-axis)
    side_collision_factor = th.abs(local_collision_vec[:, 1])
    
    # Vertical collision (z-axis) 
    vertical_collision_factor = th.abs(local_collision_vec[:, 2])
    
    # Weighted directional penalty
    r_directional_collision = -0.008 * (
        front_collision_factor * 2.0 +      # Frontal collisions most dangerous
        side_collision_factor * 1.5 +       # Side collisions significant 
        vertical_collision_factor * 1.0     # Vertical collisions moderate
    ) * th.exp(-1.0 * (collision_dist - 0.6))
    
    return r_directional_collision

# Add to reward calculation:
r_directional_collision = get_directional_collision_penalty(self)
reward = reward + r_directional_collision
```

### 7. **Depth Sectoring Strategy** ⭐ **HIGH PRIORITY**

Divide depth image into sectors for intelligent navigation:

```python
def get_depth_sectoring_reward(self):
    depth_obs = self.sensor_obs["depth"]  # (num_agents, 1, 64, 64)
    depth_map = depth_obs[:, 0]  # Remove channel dim: (num_agents, 64, 64)
    
    # Define sectors
    center_y, center_x = 32, 32
    sector_size = 16
    
    # Central forward sector (most critical)
    center_sector = depth_map[:, 
                             center_y-sector_size//2:center_y+sector_size//2,
                             center_x-sector_size//2:center_x+sector_size//2]
    
    # Left and right sectors for steering
    left_sector = depth_map[:, 
                           center_y-sector_size:center_y+sector_size,
                           center_x-sector_size:center_x]
    right_sector = depth_map[:, 
                            center_y-sector_size:center_y+sector_size,
                            center_x:center_x+sector_size]
    
    # Upper sector for altitude awareness
    upper_sector = depth_map[:, 
                            center_y-sector_size:center_y,
                            center_x-sector_size//2:center_x+sector_size//2]
    
    # Calculate minimum depths per sector
    center_min = center_sector.view(center_sector.shape[0], -1).min(dim=1)[0]
    left_min = left_sector.view(left_sector.shape[0], -1).min(dim=1)[0]
    right_min = right_sector.view(right_sector.shape[0], -1).min(dim=1)[0]
    upper_min = upper_sector.view(upper_sector.shape[0], -1).min(dim=1)[0]
    
    # Penalty based on sector depths
    r_center_depth = -0.01 * th.exp(-3.0 * (center_min - 0.3))
    r_side_depth_asymmetry = -0.005 * th.abs(left_min - right_min)  # Encourage balanced flight
    r_altitude_depth = -0.003 * th.exp(-2.0 * (upper_min - 0.4))
    
    return r_center_depth + r_side_depth_asymmetry + r_altitude_depth

# Add to reward calculation:
r_depth_sectoring = get_depth_sectoring_reward(self)
reward = reward + r_depth_sectoring
```

### 8. **Predictive Collision Detection** ⭐ **MEDIUM PRIORITY**

Use `get_point_is_collision()` for look-ahead collision avoidance:

```python
def get_predictive_collision_penalty(self):
    # Sample future positions along velocity vector
    velocity_norm = self.velocity.norm(dim=1, keepdim=True) + 1e-8
    velocity_unit = self.velocity / velocity_norm
    
    # Prediction horizons (in meters)
    prediction_distances = [0.5, 1.0, 1.5, 2.0]
    penalties = []
    
    for pred_dist in prediction_distances:
        future_positions = self.position + velocity_unit * pred_dist
        
        # Check collision at future positions (simplified - would need scene_id handling)
        penalty_weight = 0.01 / pred_dist  # Closer predictions weighted higher
        
        # Approximate penalty based on collision distance
        future_collision_vectors = self.collision_point - future_positions
        future_collision_dist = future_collision_vectors.norm(dim=1)
        
        pred_penalty = -penalty_weight * th.exp(-2.0 * (future_collision_dist - 0.5))
        penalties.append(pred_penalty)
    
    return sum(penalties)

# Add to reward calculation:
r_predictive_collision = get_predictive_collision_penalty(self)
reward = reward + r_predictive_collision
```

### 9. **Temporal Depth Coherence** ⭐ **MEDIUM PRIORITY**

Track depth changes over time to prevent erratic behavior:

```python
def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # Add depth history tracking
    self.depth_history = None
    self.depth_history_length = 5

def get_temporal_depth_reward(self):
    current_depth = self.sensor_obs["depth"]  # (num_agents, 1, 64, 64)
    
    if self.depth_history is None:
        # Initialize depth history
        self.depth_history = current_depth.repeat(1, self.depth_history_length, 1, 1)
        return th.zeros(self.num_agent, device=self.device)
    
    # Update history (shift and add new)
    self.depth_history = th.cat([
        self.depth_history[:, 1:],  # Remove oldest
        current_depth  # Add newest
    ], dim=1)
    
    # Calculate temporal variance in central region
    center_region = self.depth_history[:, :, 24:40, 24:40]  # Central 16x16 region
    center_mean = center_region.mean(dim=(2, 3))  # (num_agents, history_length)
    
    # Penalize high temporal variance (erratic depth changes)
    temporal_variance = center_mean.var(dim=1)
    r_temporal_coherence = -0.003 * temporal_variance
    
    # Reward smooth depth gradients
    depth_gradient = th.abs(center_mean[:, 1:] - center_mean[:, :-1]).mean(dim=1)
    r_smooth_depth = -0.002 * depth_gradient
    
    return r_temporal_coherence + r_smooth_depth

# Add to reward calculation:
r_temporal_depth = get_temporal_depth_reward(self)
reward = reward + r_temporal_depth
```

### 10. **Multi-Resolution Collision System** ⭐ **LOW PRIORITY**

Combine collision distance and depth for layered awareness:

```python
def get_multi_resolution_collision_reward(self):
    collision_dist = self.collision_vector.norm(dim=1)
    depth_obs = self.sensor_obs["depth"][:, 0]  # (num_agents, 64, 64)
    
    # Near-field: Direct collision distance (0-1m)
    near_field_mask = collision_dist < 1.0
    r_near_field = th.where(
        near_field_mask,
        -0.02 * th.exp(-2.0 * (collision_dist - 0.3)),
        th.zeros_like(collision_dist)
    )
    
    # Mid-field: Central depth region (1-3m)
    mid_field_mask = (collision_dist >= 1.0) & (collision_dist < 3.0)
    central_depth = depth_obs[:, 24:40, 24:40].min(dim=2)[0].min(dim=1)[0]
    r_mid_field = th.where(
        mid_field_mask,
        -0.008 * th.exp(-1.0 * (central_depth - 0.5)),
        th.zeros_like(collision_dist)
    )
    
    # Far-field: Full depth awareness (3m+)
    far_field_mask = collision_dist >= 3.0
    min_depth = depth_obs.view(depth_obs.shape[0], -1).min(dim=1)[0]
    r_far_field = th.where(
        far_field_mask,
        -0.003 * th.exp(-0.5 * (min_depth - 0.8)),
        th.zeros_like(collision_dist)
    )
    
    return r_near_field + r_mid_field + r_far_field

# Add to reward calculation:
r_multi_resolution = get_multi_resolution_collision_reward(self)
reward = reward + r_multi_resolution
```

## Updated Implementation Priority

### Phase 1: Enhanced Quick Wins (2-3 hours)
1. **Restore exponential penalty from ordnavi** ⭐
2. **Add velocity-toward-obstacle penalty** ⭐  
3. **Implement directional collision awareness** ⭐ **NEW**
4. **Add depth sectoring strategy** ⭐ **NEW**

### Phase 2: Advanced Collision Intelligence (5-8 hours)
5. **Multi-scale distance penalties**
6. **Predictive collision detection** ⭐ **NEW**
7. **Temporal depth coherence** ⭐ **NEW**

### Phase 3: Sophisticated Systems (1-3 days)
8. **Multi-resolution collision system** ⭐ **NEW**
9. **Curriculum learning**
10. **Hyperparameter optimization with advanced metrics**

## Enhanced Expected Improvements

With the advanced strategies:
- **Phase 1**: 100-200 additional survival steps
- **Phase 2**: Approach and potentially exceed ordnavi's ~500 step performance  
- **Phase 3**: 600+ steps with intelligent obstacle prediction and avoidance

## Advanced Debugging and Monitoring

Additional metrics to track:
- Depth sectoring variance (measure spatial awareness)
- Directional collision frequency (front vs side vs vertical)
- Temporal depth coherence score
- Predictive collision accuracy

```bash
# Enhanced monitoring
tensorboard --logdir=exps/hardnavi/saved/hard_navigation/ --port=6007

# Depth visualization
python -c "
import torch as th
env = HardNavigationEnv(...)
obs = env.reset()
depth = obs['depth'][0, 0].cpu().numpy()
import matplotlib.pyplot as plt
plt.imshow(depth, cmap='viridis')
plt.colorbar()
plt.savefig('depth_analysis.png')
"
```

These advanced strategies exploit VisFly's rich collision and depth information to create more intelligent and robust collision avoidance behaviors.

---

# Depth Camera FOV Optimization

*Based on analysis of VisFly's sensor configuration and Intel D435i specifications*

## Current Depth Camera Configuration

### Intel D435i Real-World Specifications
- **Depth Sensor FOV**: 87° × 58° (horizontal × vertical)
- **RGB Sensor FOV**: 69.4° × 42.5° (horizontal × vertical)  
- **Optimal Range**: 0.3m to 3m (up to 10m max)
- **Current Resolution**: 64×64 pixels

### VisFly Implementation
Current sensor configuration in `/VisFly/utils/SceneManager.py` creates `CameraSensorSpec` objects without explicit FOV settings, using Habitat-Sim defaults.

## FOV Expansion Strategy ⭐ **HIGH PRIORITY**

### 1. **Increase Horizontal FOV to Match D435i**

Modify sensor configuration to use D435i's full 87° horizontal FOV:

```python
# In sensor configuration (navigation env configs or SceneManager.py):
sensor_kwargs = [{
    "sensor_type": "DEPTH",
    "uuid": "depth", 
    "resolution": [64, 64],
    "position": [0, 0.2, 0.0],
    "hfov": 87.0,  # ADD THIS - degrees, matches D435i spec
}]
```

**Alternative approach in SceneManager.py `_load_sensor()` method:**

```python
def _load_sensor(self) -> List[habitat_sim.sensor.SensorSpec]:
    sensor_cfgs_list = []
    for i, sensor_cfg in enumerate(self.sensor_settings):
        sensor_spec = habitat_sim.CameraSensorSpec()
        sensor_spec.uuid = sensor_cfg.get("uuid", "color")
        sensor_spec.resolution = sensor_cfg.get("resolution", [128, 128])
        sensor_spec.orientation = mn.Vector3(sensor_cfg.get("orientation", [0., 0, 0]))
        sensor_spec.position = mn.Vector3(sensor_cfg.get("position", [0, 0, -0.2]))
        
        # ADD FOV configuration
        if sensor_cfg.get("hfov") is not None:
            import magnum as mn
            sensor_spec.hfov = mn.Deg(sensor_cfg.get("hfov", 90.0))  # Default 90° if not specified
        
        # ... rest of sensor configuration
```

### 2. **Enhanced Resolution for Better Peripheral Detection**

Increase resolution to better utilize the expanded FOV:

```python
sensor_kwargs = [{
    "sensor_type": "DEPTH",
    "uuid": "depth",
    "resolution": [128, 128],  # INCREASED from 64×64
    "position": [0, 0.2, 0.0],
    "hfov": 87.0,
    "vfov": 58.0,  # Optional: set vertical FOV to match D435i
}]
```

### 3. **Multi-Camera Setup for 360° Awareness**

Add additional depth sensors for comprehensive coverage:

```python
sensor_kwargs = [
    # Forward camera (main)
    {
        "sensor_type": "DEPTH",
        "uuid": "depth_front", 
        "resolution": [64, 64],
        "position": [0, 0.2, 0.0],
        "orientation": [0, 0, 0],  # Forward facing
        "hfov": 87.0,
    },
    # Left camera
    {
        "sensor_type": "DEPTH", 
        "uuid": "depth_left",
        "resolution": [32, 32],  # Lower res for side cameras
        "position": [0, 0.2, 0.0],
        "orientation": [0, 90, 0],  # 90° left
        "hfov": 60.0,
    },
    # Right camera
    {
        "sensor_type": "DEPTH",
        "uuid": "depth_right", 
        "resolution": [32, 32],
        "position": [0, 0.2, 0.0],
        "orientation": [0, -90, 0],  # 90° right  
        "hfov": 60.0,
    },
]
```

## FOV-Enhanced Reward Strategies

### 11. **Wide-FOV Peripheral Awareness** ⭐ **HIGH PRIORITY**

Exploit the expanded FOV for better obstacle detection:

```python
def get_wide_fov_collision_reward(self):
    depth_obs = self.sensor_obs["depth"][:, 0]  # (num_agents, 64, 64) or (128, 128)
    
    # With 87° HFOV, divide into more sectors
    h, w = depth_obs.shape[1], depth_obs.shape[2]
    
    # Define wider sector coverage
    # Center (forward): 30° cone
    center_start, center_end = int(w * 0.35), int(w * 0.65)
    center_sector = depth_obs[:, :, center_start:center_end]
    
    # Left peripheral: 28.5° (half of remaining FOV)
    left_sector = depth_obs[:, :, :center_start]
    
    # Right peripheral: 28.5°  
    right_sector = depth_obs[:, :, center_end:]
    
    # Calculate minimum depths
    center_min = center_sector.view(center_sector.shape[0], -1).min(dim=1)[0]
    left_min = left_sector.view(left_sector.shape[0], -1).min(dim=1)[0] 
    right_min = right_sector.view(right_sector.shape[0], -1).min(dim=1)[0]
    
    # Enhanced peripheral awareness penalties
    r_center_fov = -0.015 * th.exp(-3.0 * (center_min - 0.3))  # Strong center penalty
    r_peripheral_asymmetry = -0.008 * th.abs(left_min - right_min)  # Balanced flight
    r_peripheral_proximity = -0.006 * (
        th.exp(-2.0 * (left_min - 0.5)) + 
        th.exp(-2.0 * (right_min - 0.5))
    )
    
    return r_center_fov + r_peripheral_asymmetry + r_peripheral_proximity

# Add to reward calculation:
r_wide_fov = get_wide_fov_collision_reward(self)
reward = reward + r_wide_fov
```

### 12. **Multi-Camera Fusion Rewards** ⭐ **MEDIUM PRIORITY**

For multi-camera setups, fuse information from all sensors:

```python
def get_multi_camera_fusion_reward(self):
    # Assuming multi-camera setup
    front_depth = self.sensor_obs.get("depth_front", self.sensor_obs["depth"])[:, 0]
    left_depth = self.sensor_obs.get("depth_left", th.ones_like(front_depth) * 10)[:, 0] 
    right_depth = self.sensor_obs.get("depth_right", th.ones_like(front_depth) * 10)[:, 0]
    
    # Get minimum depth from each camera
    front_min = front_depth.view(front_depth.shape[0], -1).min(dim=1)[0]
    left_min = left_depth.view(left_depth.shape[0], -1).min(dim=1)[0]
    right_min = right_depth.view(right_depth.shape[0], -1).min(dim=1)[0]
    
    # 360° awareness penalty
    min_distance_any_direction = th.min(th.stack([front_min, left_min, right_min], dim=1), dim=1)[0]
    r_360_awareness = -0.01 * th.exp(-2.0 * (min_distance_any_direction - 0.4))
    
    # Encourage escape route identification
    max_distance_any_direction = th.max(th.stack([front_min, left_min, right_min], dim=1), dim=1)[0]
    r_escape_route = 0.003 * th.tanh(max_distance_any_direction - 1.0)  # Positive reward for open paths
    
    return r_360_awareness + r_escape_route

# Add to reward calculation:
r_multi_camera = get_multi_camera_fusion_reward(self)
reward = reward + r_multi_camera
```

## Implementation Steps for FOV Optimization

### Phase 1: Basic FOV Expansion (1-2 hours)
1. **Modify hardnavi environment config** to include `hfov: 87.0` in depth sensor settings
2. **Test current reward functions** with expanded FOV
3. **Implement wide-FOV peripheral awareness rewards**

### Phase 2: Enhanced Resolution (2-3 hours)  
4. **Increase depth resolution** to 128×128 
5. **Update depth sectoring strategies** for higher resolution
6. **Optimize performance** for larger depth maps

### Phase 3: Multi-Camera Setup (1-2 days)
7. **Add side-facing depth cameras** (optional)
8. **Implement multi-camera fusion rewards**
9. **Create 360° collision avoidance behaviors**

## Expected FOV Improvements

With expanded FOV:
- **87° vs default (~60-70°)**: ~25-40% more peripheral vision
- **Better early obstacle detection**: Obstacles visible sooner at trajectory edges
- **Improved navigation**: More intelligent path planning around obstacles
- **Reduced blind spots**: Fewer surprise collisions from periphery

## Configuration Examples

### For hardnavi environment config:
```yaml
# In /exps/hardnavi/env_cfgs/hard_navigation.yaml
env:
  sensor_kwargs:
    - sensor_type: DEPTH
      uuid: depth
      resolution: [128, 128]  # Increased resolution
      position: [0, 0.2, 0.0]
      hfov: 87.0  # D435i horizontal FOV
      vfov: 58.0  # D435i vertical FOV (optional)
```

### For ordnavi (to maintain parity):
```yaml  
# In /exps/ordnavi/env_cfgs/ordinary_navigation.yaml
env:
  sensor_kwargs:
    - sensor_type: DEPTH
      uuid: depth  
      resolution: [128, 128]
      position: [0, 0.2, 0.0]
      hfov: 87.0
      vfov: 58.0
```

## Performance Considerations

- **Memory impact**: 128×128 vs 64×64 = 4× memory per depth frame
- **Computation**: Wider FOV processing requires more sectoring calculations
- **Training time**: More complex depth processing may slow training slightly
- **Benefit**: Significantly better collision avoidance should outweigh costs

The expanded FOV matching real D435i specifications should provide substantial improvements in collision avoidance by giving the agent much better peripheral vision for early obstacle detection and more intelligent navigation decisions.