from ultralytics import YOLO
import cv2
import numpy as np

image=cv2.imread('blank_screen.jpg',cv2.COLOR_BGR2RGB)
model=YOLO('yolov8n.pt')
results=model.predict([image])
for result in results:
    boxes=result.boxes
    print(boxes)
    masks=result.masks
    keypoints = result.keypoints
    probs = result.probs
    truth_array=boxes.cls == 0.
    truth_array=np.array(truth_array)
    print(truth_array[0])
