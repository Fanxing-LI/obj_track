#!/usr/bin/env python3

import os
import argparse
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import yaml
except Exception:
    yaml = None


def load_defaults(cfg_path):
    defaults = {
        'desired_radius': 2.2,
        'tangential_speed': 1.0,
        'center': (0.0, 0.0),
        'title': 'Circle Spec'
    }
    if yaml is None:
        return defaults
    try:
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        env = cfg.get('env', {})
        if 'desired_radius' in env:
            defaults['desired_radius'] = float(env['desired_radius'])
        if 'tangential_speed' in env:
            defaults['tangential_speed'] = float(env['tangential_speed'])
    except Exception:
        pass
    return defaults


def plot_circle_spec(radius, speed, center=(0.0, 0.0), out_path='circle_spec.png', angle_deg=45.0, title=None):
    cx, cy = center
    theta = math.radians(angle_deg)
    px = cx + radius * math.cos(theta)
    py = cy + radius * math.sin(theta)
    # CCW tangent direction at angle theta
    tx = -math.sin(theta)
    ty = math.cos(theta)

    # Arrow length in meters (show actual speed magnitude)
    arrow_len = float(speed)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.4)

    # Draw circle and center
    circle = plt.Circle((cx, cy), radius, fill=False, color='C0', linewidth=2.5)
    ax.add_patch(circle)
    ax.scatter([cx], [cy], c='k', s=30, label='Target center')
    ax.text(cx, cy, '  center', va='center', ha='left', fontsize=9)

    # Draw a sample UAV point on circle and the tangential speed arrow
    ax.scatter([px], [py], c='C1', s=25, label='On circle')
    ax.arrow(px, py, tx * arrow_len, ty * arrow_len,
             width=0.02 * max(0.5, radius/5),
             head_width=0.12 * max(0.5, radius/2),
             head_length=0.18 * max(0.5, radius/2),
             length_includes_head=True,
             color='C3')
    ax.text(px + tx * arrow_len * 1.05,
            py + ty * arrow_len * 1.05,
            f'v_des = {speed:.2f} m/s',
            fontsize=9,
            color='C3',
            va='bottom', ha='left')

    # Annotate radius
    ax.plot([cx, px], [cy, py], color='C2', linestyle=':', linewidth=1.5)
    midx, midy = (cx + px)/2, (cy + py)/2
    ax.text(midx, midy, f'r = {radius:.2f} m', fontsize=9, color='C2', va='bottom')

    # Limits and labels
    margin = max(0.5, 0.25 * radius + arrow_len)
    ax.set_xlim(cx - radius - margin, cx + radius + margin)
    ax.set_ylim(cy - radius - margin, cy + radius + margin)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(title or f'Circle: r={radius:.2f} m, v_des={speed:.2f} m/s')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Plot circle and desired speed spec')
    parser.add_argument('--cfg', type=str, default=os.path.join(os.path.dirname(__file__), 'env_cfgs/circle.yaml'),
                        help='Path to circle env yaml (to read defaults)')
    parser.add_argument('--radius', type=float, default=None, help='Desired radius (m)')
    parser.add_argument('--speed', type=float, default=None, help='Desired tangential speed (m/s)')
    parser.add_argument('--center', type=float, nargs=2, default=None, help='Center x y (m)')
    parser.add_argument('--angle', type=float, default=45.0, help='Point on circle (deg)')
    parser.add_argument('--out', type=str, default=None, help='Output image path')

    args = parser.parse_args()

    defaults = load_defaults(args.cfg)
    r = args.radius if args.radius is not None else defaults['desired_radius']
    v = args.speed if args.speed is not None else defaults['tangential_speed']
    c = tuple(args.center) if args.center is not None else defaults['center']

    out_path = args.out
    if out_path is None:
        # Save next to circle saved folder
        out_path = os.path.join(os.path.dirname(__file__), 'saved/circle/circle_spec.png')

    path = plot_circle_spec(radius=r, speed=v, center=c, out_path=out_path, angle_deg=args.angle,
                            title=defaults.get('title'))
    print(f'Saved: {path}')


if __name__ == '__main__':
    main()

