import numpy as np
import mediapipe as mp
import numpy as np
import cv2
from ultralytics import YOLO
import torch as th

# 初始化 YOLOv8n 模型（只需执行一次）
yolo_model = YOLO('yolov10n.pt')  # 自动下载模型
yolo_model.overrides = {
    'conf': 0.25,      # 提高置信度阈值
    'iou': 0.7,        # 提高NMS阈值
    'max_det': 5,      # 限制最大检测数量（人体检测通常不需要很多）
    'half': True,      # FP16推理
    'device': 'cuda',  # GPU推理
    'verbose': False   # 关闭日志
}
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=0,  # 使用最轻量级模型 (0/1/2)
    enable_segmentation=False,  # 关闭分割
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

DEBUG = False


def get_chest_center_pixel_and_depth(rgb_image, depth_image, min_visibility=0.5):
    H, W = rgb_image.shape[:2]
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    annotated_image = rgb_image.copy()

    # 直接使用全局的pose_detector，不再重新初始化
    result = pose_detector.process(rgb_image)
    if not result.pose_landmarks:
        return None

    lm = result.pose_landmarks.landmark
    pts = {}
    indices = {
        "l_shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
        "r_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
        "l_hip": mp_pose.PoseLandmark.LEFT_HIP,
        "r_hip": mp_pose.PoseLandmark.RIGHT_HIP
    }

    for name, idx in indices.items():
        p = lm[idx]
        if p.visibility >= min_visibility:
            u = p.x * W  # 保持float精度
            v = p.y * H  # 保持float精度
            pts[name] = (u, v)
            if DEBUG:
                cv2.circle(annotated_image, (int(u), int(v)), 4, (0, 255, 0), -1)
                cv2.putText(annotated_image, name, (int(u) + 5, int(v) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    if len(pts) >= 4:
        u_c = (pts["l_shoulder"][0] + pts["r_shoulder"][0] + pts["l_hip"][0] + pts["r_hip"][0]) / 4
        v_c = (pts["l_shoulder"][1] + pts["r_shoulder"][1] + pts["l_hip"][1] + pts["r_hip"][1]) / 4
    elif len(pts) == 3:
        coords = list(pts.values())
        dists = [np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))
                 for i in range(3) for j in range(i + 1, 3)]
        i, j = divmod(np.argmax(dists), 2)
        u_c = (coords[i][0] + coords[j][0]) / 2
        v_c = (coords[i][1] + coords[j][1]) / 2
    elif len(pts) == 2:
        coords = list(pts.values())
        u_c = (coords[0][0] + coords[1][0]) / 2
        v_c = (coords[0][1] + coords[1][1]) / 2
    elif len(pts) == 1:
        pass
    else:
        return None

    # 在这里才转换为int用于索引
    u_c_int = int(u_c)
    v_c_int = int(v_c)

    if not (0 <= u_c_int < W and 0 <= v_c_int < H):
        return None

    depth = float(depth_image[v_c_int, u_c_int])
    if depth == 0 or np.isnan(depth):
        return None

    if DEBUG:
        cv2.circle(annotated_image, (u_c_int, v_c_int), 6, (0, 0, 255), -1)
        cv2.putText(annotated_image, f"Chest: ({u_c_int},{v_c_int}) d={depth:.2f}",
                    (u_c_int + 10, v_c_int), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow("Chest Detection", annotated_image)
        cv2.waitKey(1)

    return (u_c, v_c, depth)  # 返回float精度的坐标


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
    mean_y = y1 + (y2-y1) * 1/3 #(y1 + y2) / 2

    return rgb_crop, depth_crop, people[0][0], (mean_x, mean_y)


def frame_img_to_inertial(center, intrinsic, depth):
    pos = depth * th.tensor([[
        1.0,
        -(center[0] - intrinsic.cx) / intrinsic.fx * 2,
        -(center[1] - intrinsic.cy) / intrinsic.fy * 2,
    ]])
    return pos
