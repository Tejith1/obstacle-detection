"""
Simple Class Mapper - Terminal Based

This script helps you identify classes by showing what's detected
and letting you label them via terminal input.

USAGE:
1. Run: python simple_class_mapper.py
2. Point camera at object
3. Press 'l' in the video window when you see a detection
4. Type the object name in the terminal
5. Press 'q' to quit and save

Much simpler than the interactive version!
"""

import sys
import cv2
import torch
import numpy as np
import os
import pathlib
import json

# Fix for Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

REPO = "yolov5"
sys.path.insert(0, REPO)

from models.common import DetectMultiBackend
from utils.general import (non_max_suppression, check_img_size)
from utils.augmentations import letterbox
from utils.torch_utils import select_device

WEIGHTS = "best.pt"
IMG_SIZE = 640
CONF_THRESH = 0.25
IOU_THRESH = 0.45
DEVICE = ''
CAMERA_INDEX = 1
MAPPING_FILE = "class_mapping.json"

class_mappings = {}

def load_mappings():
    global class_mappings
    if os.path.exists(MAPPING_FILE):
        with open(MAPPING_FILE, 'r') as f:
            class_mappings = json.load(f)
        print(f"✅ Loaded {len(class_mappings)} existing mappings\n")

def save_mappings():
    with open(MAPPING_FILE, 'w') as f:
        json.dump(class_mappings, f, indent=2)
    
    with open("class_mapping.txt", 'w') as f:
        f.write("CLASS MAPPING FOR best.pt\n")
        f.write("=" * 50 + "\n\n")
        for class_id in sorted(class_mappings.keys(), key=lambda x: int(x.replace('class', ''))):
            f.write(f"{class_id:8} → {class_mappings[class_id]}\n")
    
    print(f"\n✅ Saved to {MAPPING_FILE} and class_mapping.txt")

def scale_boxes(boxes, ratio, dwdh, shape):
    if boxes is None or len(boxes) == 0:
        return np.array([])
    boxes = np.array(boxes)
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, 4) if len(boxes) == 4 else np.array([])
    if len(boxes) == 0 or boxes.shape[1] != 4:
        return np.array([])
    
    dw, dh = dwdh
    h0, w0 = shape
    ratio = float(ratio[0] if isinstance(ratio, (list, tuple, np.ndarray)) else ratio)
    
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / ratio
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w0 - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h0 - 1)
    
    return boxes

print("\n" + "=" * 60)
print("SIMPLE CLASS MAPPER")
print("=" * 60)
print("\nHow to use:")
print("1. Point camera at a known object")
print("2. Wait for detection (you'll see bounding box)")
print("3. Press 'L' in the video window")
print("4. Type object name in THIS TERMINAL and press Enter")
print("5. Repeat for all classes")
print("6. Press 'Q' to quit and save\n")
print("=" * 60 + "\n")

load_mappings()

device = select_device(DEVICE)
model = DetectMultiBackend(WEIGHTS, device=device, dnn=False, fp16=False)
stride = int(model.stride)
names = model.names
imgsz = check_img_size(IMG_SIZE, s=stride)

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"❌ Cannot open webcam {CAMERA_INDEX}")
    exit(1)

print("✅ Ready! Point camera at objects...\n")

last_detections = {}

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img0 = frame.copy()
        h0, w0 = img0.shape[:2]

        img, ratio, (dw, dh) = letterbox(img0, imgsz, stride=stride)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        im = torch.from_numpy(img).to(device).float() / 255.0
        if im.ndim == 3:
            im = im.unsqueeze(0)

        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, CONF_THRESH, IOU_THRESH, None, False, max_det=1000)

        current_detections = {}

        if len(pred) and pred[0] is not None and len(pred[0]):
            det = pred[0].clone().detach().cpu().numpy()
            boxes = det[:, :4].copy()
            boxes = scale_boxes(boxes, ratio, (dw, dh), (h0, w0))

            if len(boxes) > 0:
                det[:, :4] = boxes
                
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)
                    cls = int(cls)
                    class_name = names[cls] if cls < len(names) else f"class{cls}"
                    
                    current_detections[class_name] = conf
                    
                    # Color: green if mapped, orange if not
                    color = (0, 255, 0) if class_name in class_mappings else (0, 165, 255)
                    display = class_mappings.get(class_name, class_name)
                    
                    cv2.rectangle(img0, (x1, y1), (x2, y2), color, 2)
                    label = f"{display} {conf:.2f}"
                    cv2.putText(img0, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        last_detections = current_detections

        # Show stats
        y = 30
        cv2.putText(img0, f"Mapped: {len(class_mappings)}/22", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        y += 40
        
        if current_detections:
            cv2.putText(img0, f"Detecting {len(current_detections)} classes:", 
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            y += 30
            for cn in sorted(current_detections.keys()):
                status = "✓" if cn in class_mappings else "?"
                color = (0,255,0) if cn in class_mappings else (0,165,255)
                cv2.putText(img0, f"{status} {cn}", (10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y += 25

        cv2.putText(img0, "Press L to label | Q to quit", 
                   (10, img0.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        cv2.imshow("Point at objects and press L to label", img0)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('l') and last_detections:
            # Find first unmapped class
            unmapped = [cn for cn in last_detections.keys() if cn not in class_mappings]
            if unmapped:
                class_to_map = unmapped[0]
                print(f"\n🎯 Class '{class_to_map}' detected (conf: {last_detections[class_to_map]:.2f})")
                print(f"What object is this? Type name: ", end='', flush=True)
                
                object_name = input().strip()
                if object_name:
                    class_mappings[class_to_map] = object_name
                    print(f"✅ Mapped '{class_to_map}' → '{object_name}'")
                    print(f"Progress: {len(class_mappings)}/22 classes mapped\n")
                else:
                    print("⏭️  Skipped\n")
            else:
                print("✅ All visible classes are already mapped!\n")

except KeyboardInterrupt:
    print("\n\n⚠️  Stopped by user")
finally:
    cap.release()
    cv2.destroyAllWindows()
    
    if class_mappings:
        save_mappings()
        print(f"\n✅ Mapped {len(class_mappings)}/22 classes:")
        for k, v in sorted(class_mappings.items(), key=lambda x: int(x[0].replace('class',''))):
            print(f"  {k:8} → {v}")
    
    print("\n💡 Use these mappings in webcam_detect.py by adding:")
    print("   CLASS_NAME_MAP = { ... } (see class_mapping.txt)\n")
