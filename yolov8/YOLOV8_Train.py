import os

from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')

model.train(data='directory/config.yaml', epochs = 100, imgsz = 320, batch = -1, device=0, plots = False)
