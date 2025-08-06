from typing import Dict
import torch as th
import numpy as np
from ultralytics import YOLO
import cv2
from sklearn.cluster import KMeans
from VisFly.utils.type import TensorDict
from envs.ObjectTrackingEnv import ObjectTrackingEnv as OriObjectTrackingEnv
model = YOLO("yolov8n.pt")
from debug.kalman_filter import EKF3DSnap
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
        self.intrinsic = intrinsic(fx=322.28, fy=322.28, cx=320.8, cy=178.78).to(self.device)
        self.intrinsic = intrinsic(fx=320, fy=180, cx=320.8, cy=180).to(self.device)
        self.ekf = EKF3DSnap(dt=self.envs.dynamics.ctrl_dt)

    def update_target(self):
        super().update_target()
        img = cv2.cvtColor(self.sensor_obs["color"].squeeze().transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
        results = model.predict(img, verbose=False)  # Use the color image for detection
        # results[0].show()
        depth = self.sensor_obs["depth"][0].transpose(1, 2, 0).clip(max=10)
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes

            # Filter for human detections (class ID 0 is 'person' in COCO dataset)
            person_mask = boxes.cls == 0

            if person_mask.any():
                # Get only human bounding boxes
                human_boxes = boxes[person_mask]
                best_human_idx = human_boxes.conf.argmax()
                x1, y1, x2, y2 = human_boxes.xyxy[best_human_idx].cpu().numpy()
                img_shape = img.shape[:2]
                center = [(x1 + x2) / 2,(y1 + y2) / 2]
                # confidence = human_boxes.conf[best_human_idx].cpu().numpy()
                # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                depth_of_target =depth[int(y1):int(y2),int(x1):int(x2),:]
                # target_depths_flat = depth_of_target.flatten()
                # kmeans = KMeans(n_clusters=1)  # 聚类成一个簇
                # kmeans.fit(target_depths_flat.reshape(-1, 1))
                # target_depth = kmeans.cluster_centers_[0]
                cv2.imshow("box", depth_of_target/10)
                target_depth = self.sensor_obs["depth"][0,0, int(center[1]), int(center[0])]
                estimate_pos = target_depth * th.tensor([[
                    1,
                    -(center[0]-self.intrinsic.cx)/self.intrinsic.fx,
                    -(center[1]-self.intrinsic.cy)/self.intrinsic.fy]
                ])

                if not hasattr(self, "pre_estimated_target_pos"):
                    self.pre_estimate_pos = estimate_pos
                else:
                    self.pre_estimate_pos = self.estimate_pos.clone()
                self.estimate_pos = self.ekf.x[0:3].T
                self.estimate_vel = self.ekf.x[3:6].T
                # self.estimate_vel = (self.estimate_pos - self.pre_estimate_pos) / self.envs.dynamics.ctrl_dt
        cv2.imshow("depth",depth/10)
        cv2.waitKey(1)

        self.ekf.predict()
        self.ekf.update(self.rebuild_local_targets)
        self.estimate_pos = self.ekf.x[0:3].T
        self.estimate_vel = self.ekf.x[3:6].T
        # 保存结果
        # results[0].save("output.jpg")
        test = 1

    def get_observation(
            self,
            indices=None
    ) -> Dict:
        self.update_target()

        state = th.hstack([
            # self.local_targets,
            # self.local_targets_v,
            # self.rebuild_local_targets,
            # self.rebuild_local_targets_v,
            self.estimate_pos,
            self.rebuild_local_targets_v,
            # self.estimate_vel,
            self.orientation,
            self.local_v / 10,
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
