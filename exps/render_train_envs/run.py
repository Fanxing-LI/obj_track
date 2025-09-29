import os, sys

sys.path.append(os.getcwd())
from VisFly.envs.HoverEnv import HoverEnv
from envs.ObjectTrackingEnv import ObjectTrackingEnv
from envs.VisualHoverEnv import VisualHoverEnv
# from envs.TrackingEnv import AwareTrackEnv
from envs.TrackingEnv import AwareTrackEnv2
from algorithms.BPTT import BPTT
from algorithms.SHAC import SHAC
from VisFly.utils.algorithms.PPO import PPO
import torch as th
import numpy as np
import sys
import os
import argparse
from VisFly.utils.common import load_yaml_config
import json
from exps.test.tracking.test import Test as tracking_test
import yaml
import cv2

# th.autograd.set_detect_anomaly(True)
"""
   Current Best: SHAC_NoCaliHeadV_Pos_Dis3.0_spd3.4_lessNoise_2.zip
   others: SHAC_NoCaliHeadV_Pos_Dis1.5_spd3.4_lessNoise_1.zip for distance 1.5
   SHAC_NoCaliHeadV_Dis4.5_6.zip
   
   PPO_NoRand_1.zip
"""

def parse_args():
    parser = argparse.ArgumentParser(description='Run experiments', add_help=False)
    parser.add_argument('--comment', '-c', type=str, default="std")
    parser.add_argument("--train", "-t", type=int, default=1)
    parser.add_argument("--algorithm", "-a", type=str, default="SHAC")
    parser.add_argument("--env", "-e", type=str, default="objTracking")
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--weight", "-w", type=str, default=None, )
    parser.add_argument("--traj", "-tr", type=str, default="1", )
    parser.add_argument("--velocity", "-v", type=float, default=1.0, )
    parser.add_argument("--distance", "-d", type=float, default=3.0, )
    return parser
args = parse_args().parse_args()

save_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + f"/saved/{args.env}/"

env_config = load_yaml_config(os.path.dirname(os.path.abspath(__file__)) + f'/env_cfgs/{args.env}.yaml')


env_alias = {
    "objTracking": ObjectTrackingEnv,
}

alg_alias = {
    "BPTT": BPTT,
    "PPO": PPO,
    "SHAC": SHAC,
}


eval_env = ObjectTrackingEnv(
    **env_config["eval_env"]
)

test = 1
eval_env.reset()
num_agent = eval_env.num_agent

render_imgs = []
max_t = 180

# Create plots directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

# create a video writer
fourcc = cv2.VideoWriter_fourcc(*"H264")
video_writer = cv2.VideoWriter("plots/train_env_render.mp4", fourcc, 30, (240*8, 240*4))

for i in range(max_t):
    action = th.randn((num_agent, 4)).clamp(-1, 1)
    obs, reward, done, info = eval_env.step(action)
    r_imgs = eval_env.render()

    
    # Convert numpy arrays to tensors before stacking
    if isinstance(r_imgs[0], np.ndarray):
        r_imgs_tensors = [th.from_numpy(img) for img in r_imgs]
    else:
        r_imgs_tensors = r_imgs
    
    stack_r_imgs = th.stack(r_imgs_tensors)
    # reshape into 4*8
    reshape_r_imgs = stack_r_imgs.reshape(4, 8, *stack_r_imgs.shape[1:])
    
    # Convert tensor to numpy and process for video writing
    # Assuming each image is in [H, W, C] format
    frame_data = reshape_r_imgs.cpu().numpy()

    r_imgs = [[r for r in r_imgs[(row_i*8):((row_i+1)*8)]] for row_i in range(4)]
    image = np.vstack([np.hstack(r_imgs_row) for r_imgs_row in r_imgs])
    
    # Ensure image is uint8 and convert color space properly
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Convert RGBA to BGR for OpenCV (remove alpha channel)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Write frame to video
    video_writer.write(image)
    print(f"Processed frame {i+1}/{max_t}")

# Clean up and finalize video
video_writer.release()
print(f"Video saved to plots/train_env_render.mp4")

eval_env.close()
