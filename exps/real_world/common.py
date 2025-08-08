import numpy as np
import mediapipe as mp
import numpy as np
import cv2
from ultralytics import YOLO
import torch as th

# 初始化 YOLOv8n 模型（只需执行一次）
yolo_model = YOLO('yolov8n.pt')  # 自动下载模型
mp_pose = mp.solutions.pose

DEBUG = True


def get_chest_center_pixel_and_depth(rgb_image, depth_image, min_visibility=0.5):
    H, W = rgb_image.shape[:2]
    with mp_pose.Pose(static_image_mode=True) as pose:
        result = pose.process(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        if not result.pose_landmarks:
            return None

        lm = result.pose_landmarks.landmark

        # 获取四个关键点信息
        pts = {}
        for name, key in {
            "l_shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
            "r_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
            "l_hip": mp_pose.PoseLandmark.LEFT_HIP,
            "r_hip": mp_pose.PoseLandmark.RIGHT_HIP
        }.items():
            landmark = lm[key]
            if landmark.visibility >= min_visibility:
                pts[name] = (landmark.x * W, landmark.y * H)

        # 根据可用关键点个数选择策略
        if len(pts) >= 4:
            # 四点都在，用肩膀中心与胯中心做中点
            u_s = (pts["l_shoulder"][0] + pts["r_shoulder"][0]) / 2
            v_s = (pts["l_shoulder"][1] + pts["r_shoulder"][1]) / 2
            u_h = (pts["l_hip"][0] + pts["r_hip"][0]) / 2
            v_h = (pts["l_hip"][1] + pts["r_hip"][1]) / 2
            u_c = int((u_s + u_h) / 2)
            v_c = int((v_s + v_h) / 2)
        elif len(pts) == 3:
            # 三个点，找最长边对应两个点
            keys = list(pts.keys())
            coords = [pts[k] for k in keys]
            dists = [np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))
                     for i in range(3) for j in range(i+1, 3)]
            i, j = divmod(np.argmax(dists), 2)
            u_c = int((coords[i][0] + coords[j][0]) / 2)
            v_c = int((coords[i][1] + coords[j][1]) / 2)
        elif len(pts) == 2:
            # 两个点，直接中点
            coords = list(pts.values())
            u_c = int((coords[0][0] + coords[1][0]) / 2)
            v_c = int((coords[0][1] + coords[1][1]) / 2)
        else:
            # 可用关键点太少
            return None

        if not (0 <= u_c < W and 0 <= v_c < H):
            return None

        depth = float(depth_image[v_c, u_c])
        if depth == 0 or np.isnan(depth):
            return None

        return (u_c, v_c, depth)

def extract_nearest_person(rgb_image, depth_image, conf_threshold=0.0):
    """
    从图像中提取离图像中心最近的人（基于 bounding box 中心点）

    Args:
        rgb_image: HxWx3 RGB 图像 (np.uint8)
        depth_image: HxW 深度图 (单位可为米或毫米)
        conf_threshold: YOLO 检测置信度阈值

    Returns:
        rgb_crop: 裁剪出的 RGB 子图
        depth_crop: 裁剪出的深度图
        bbox: [x1, y1, x2, y2] 原图坐标
        None: 如果没有检测到人
    """
    H, W = rgb_image.shape[:2]
    image_center = np.array([W / 2, H / 2])

    results = yolo_model(rgb_image, verbose=False)
    people = []

    for det in results[0].boxes:
        cls_id = int(det.cls)
        conf = float(det.conf)
        if cls_id == 0 and conf >= conf_threshold:  # class 0 = person
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            dist = np.linalg.norm(np.array([cx, cy]) - image_center)
            people.append(((x1, y1, x2, y2), dist))

    if not people:
        return None

    # 选择距离图像中心最近的人
    people.sort(key=lambda x: x[1])
    x1, y1, x2, y2 = people[0][0]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W - 1, x2), min(H - 1, y2)

    rgb_crop = rgb_image[y1:y2, x1:x2]
    depth_crop = depth_image[y1:y2, x1:x2]

    mean_x = (x1 + x2) / 2
    mean_y = (y1 + y2) / 2

    return rgb_crop, depth_crop, (mean_x, mean_y)


def frame_img_to_inertial(center, intrinsic, depth):
    pos = depth * th.tensor([[
        1.0,
        -(center[0] - intrinsic.cx) / intrinsic.fx * 2,
        -(center[1] - intrinsic.cy) / intrinsic.fy * 2,
    ]])
    return pos
