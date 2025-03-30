import os
os.environ["ULTRALYTICS_NO_CHECKS"] = "1"  # Disable online checks
import cv2 as cv
from ultralytics import YOLO

# Load a pretrained YOLOv8 model (e.g., YOLOv8n for a smaller and faster model)
# model = YOLO('yolov8n.pt') # replace 'n' with 's','m','l', or 'x' for different sizes
model = YOLO('yolov8x.pt') # replace 'n' with 's','m','l', or 'x' for different sizes

results = model.predict('./images/knife.jpg')
img = results[0].plot()

cv.imshow('YOLOv8 Object Detection', img)
cv.waitKey(0)  # Wait for a key press
cv.destroyAllWindows()  # Close window