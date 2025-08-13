from typing import Dict
import torch as th
import numpy as np
from ultralytics import YOLO
import cv2
from sklearn.cluster import KMeans
from VisFly.utils.type import TensorDict
from envs.ObjectTrackingEnv import ObjectTrackingEnv as OriObjectTrackingEnv
from debug.kalman_filter import EKF9D
import mediapipe as mp
from common import extract_nearest_person, frame_img_to_inertial, get_chest_center_pixel_and_depth
import time

DEBUG = True


class intrinsic:
    def __init__(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def to(self, device):
        self.fx = th.tensor(self.fx).to(device)
        self.fy = th.tensor(self.fy).to(device)
        self.cx = th.tensor(self.cx).to(device)
        self.cy = th.tensor(self.cy).to(device)
        return self


class ObjectTrackingEnv(OriObjectTrackingEnv):
    def __init__(self, *args, **kwargs):
        kwargs["semantic_id"] = "human"
        super().__init__(*args, **kwargs)
        # self.intrinsic = intrinsic(fx=322.28, fy=322.28, cx=320.8, cy=178.78).to(self.device)
        # self.intrinsic = intrinsic(fx=320, fy=180, cx=320.8, cy=180).to(self.device)
        self.ekf = EKF9D(dt=self.envs.dynamics.ctrl_dt)

        self.run_count = 0
        self.pre_time = time.time()

    def update_target(self):
        self.run_count += 1
        if self.run_count % 100 == 0:
            print("fps:", 100 / (time.time() - self.pre_time))
            self.pre_time = time.time()
            self.run_count = 0

        super().update_target()
        color = self.sensor_obs["color"][0].transpose(1, 2, 0).clip(max=255).astype(np.uint8)
        depth = self.sensor_obs["depth"][0].transpose(1, 2, 0).clip(max=10)
        result = extract_nearest_person(
            rgb_image=color,
            depth_image=depth,
            conf_threshold=0.5
        )

        cv2.imshow("center", cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        if result:
            color_of_target, depth_of_target, xyxy, center = result
            center2 = center
            center_depth = depth[int(center[1]), int(center[0]), 0]
            debug_center = get_chest_center_pixel_and_depth(color_of_target, depth_of_target, )
            if debug_center:
                center = (xyxy[0] + debug_center[0], xyxy[1] + debug_center[1])
                center_depth = debug_center[2]

            if DEBUG:
                # cv2.imshow("depth", depth / 10)
                cv2.imshow("color", cv2.cvtColor(color_of_target, cv2.COLOR_RGB2BGR))
                cv2.circle(color, (int(center2[0]), int(center2[1])), 6, (0, 255, 0), -1)
                cv2.circle(color, (int(center[0]), int(center[1])), 6, (0, 0, 255), -1)
                cv2.imshow("center", cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

            self.YOLO_local_pos = frame_img_to_inertial(center, self._intrinsic, center_depth)
            orientation = self.envs.dynamics._orientation.clone()
            self.YOLO_head_pos = orientation.local_to_head(self.YOLO_local_pos.T).T

            if not hasattr(self, "pre_YOLO_local_pos"):
                self.pre_YOLO_local_pos = self.YOLO_local_pos.clone()
                self.pre_YOLO_head_pos = self.YOLO_head_pos.clone()

            self.YOLO_estimate_vel = (self.YOLO_local_pos - self.pre_YOLO_local_pos) / self.envs.dynamics.ctrl_dt

            cali_head_vel = th.cross(self.angular_velocity * th.tensor([[0,0,1]]), self.YOLO_head_pos)

            self.YOLO_head_vel = (self.YOLO_head_pos - self.pre_YOLO_head_pos) / self.envs.dynamics.ctrl_dt #+ cali_head_vel
            self.pre_YOLO_head_pos = self.YOLO_head_pos.clone()


        self.ekf.predict()
        self.estimate_pos = self.ekf.x[0:3].T
        self.estimate_vel = self.ekf.x[3:6].T

        # self.ekf.update(z_pos=self.rebuild_local_targets, z_vel=self.rebuild_local_targets_v)
        # self.ekf.update(z_pos=self.YOLO_estimate_pos, z_vel=self.YOLO_estimate_vel)
        self.ekf.update(z_pos=self.YOLO_head_pos, z_vel=self.YOLO_head_vel)
        if not hasattr(self, "pre_estimate_pos"):
            self.pre_estimate_pos = self.estimate_pos
        # self.estimate_vel = (self.estimate_pos-self.pre_estimate_pos) / self.envs.dynamics.ctrl_dt
        self.pre_estimate_pos = self.estimate_pos.clone()
        # self.estimate_pos = self.rebuild_local_targets
        # self.estimate_vel = self.rebuild_local_targets_v
        # 保存结果
        # results[0].save("output.jpg")
        print(
            "estimate_error:", th.norm(self.head_targets - self.estimate_pos),
            "estimate_vel_error:", th.norm(self.head_targets_v - self.estimate_vel)
        )
        test = 1

    def get_observation(
            self,
            indices=None
    ) -> Dict:
        self.update_target()
        self.run_count += 1
        state = th.hstack([
            # self.local_targets,
            # self.local_targets_v,
            # self.rebuild_local_targets,
            # self.rebuild_local_targets_v,
            self.estimate_pos,
            # self.rebuild_local_targets_v,
            self.estimate_vel,
            self.orientation,
            self.head_v / 10,
            self.angular_velocity / 10,
        ]).to(self.device)
        # return TensorDict({
        #     "state": state,
        # })

        obs = TensorDict({
            "state": state,
            "depth": th.as_tensor(self.sensor_obs["depth"]).clamp(0.2, 10),
            "semantic": th.as_tensor(self.sensor_obs["semantic"].astype(np.float32)),
            "color": th.as_tensor(self.sensor_obs["color"].astype(np.float32)),
        })

        return obs
