import os

from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')

model.train(data='/content/gdrive/My Drive/PUNO_NG_PAGMAMAHAL/TASK_FILES/YOLOV8_Colab/config.yaml', epochs = 100, imgsz = 320, batch = -1, device=0, plots = False)
