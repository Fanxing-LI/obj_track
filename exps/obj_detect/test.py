from ultralytics import YOLO

# 加载模型（自动下载）
model = YOLO("yolo11n.pt")  # yolov8n.pt, yolov8s.pt 等

# 使用内置测试图片推理
results = model.predict("ball4.png")

# 显示结果（自动弹出窗口）
results[0].show()

# 保存结果
results[0].save("output.jpg")


