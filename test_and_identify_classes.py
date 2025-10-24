"""
Interactive Class Identifier for Your Obstacle Detection Model

This script helps you identify what each class (class0, class1, ..., class21) 
actually detects by testing with real objects.

HOW TO USE:
1. Run this script: python test_and_identify_classes.py
2. Point camera at a known object (e.g., your phone, a bottle, yourself)
3. When you see a detection, press a key to label it:
   - Type the object name and press Enter
   - Press 's' to skip
   - Press 'q' to quit and save
4. Build your class mapping automatically!

The script will save a 'class_mapping.txt' file with your findings.
"""

import sys
import time
import cv2
import torch
import numpy as np
import os
import pathlib
from collections import defaultdict
import json

# Fix for Windows when loading Linux-trained YOLO weights
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# ensure yolov5 repo is importable
REPO = "yolov5"
sys.path.insert(0, REPO)

from models.common import DetectMultiBackend
from utils.general import (non_max_suppression, check_img_size)
from utils.augmentations import letterbox
from utils.torch_utils import select_device

# Config
WEIGHTS = "best.pt"
IMG_SIZE = 640
CONF_THRESH = 0.25
IOU_THRESH = 0.45
DEVICE = ''
CAMERA_INDEX = 1

# Store mappings
class_mappings = {}
MAPPING_FILE = "class_mapping.json"

def load_existing_mappings():
    """Load any existing class mappings from file."""
    global class_mappings
    if os.path.exists(MAPPING_FILE):
        try:
            with open(MAPPING_FILE, 'r') as f:
                class_mappings = json.load(f)
            print(f"✅ Loaded existing mappings from {MAPPING_FILE}")
            print(f"   Found {len(class_mappings)} mapped classes")
        except:
            pass

def save_mappings():
    """Save class mappings to file."""
    with open(MAPPING_FILE, 'w') as f:
        json.dump(class_mappings, f, indent=2)
    
    # Also save as readable text
    with open("class_mapping.txt", 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("CLASS MAPPING FOR best.pt MODEL\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total classes mapped: {len(class_mappings)} / 22\n\n")
        
        for class_id in sorted(class_mappings.keys(), key=lambda x: int(x.replace('class', ''))):
            f.write(f"{class_id:8} → {class_mappings[class_id]}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("To use these in webcam_detect.py, add this mapping:\n")
        f.write("=" * 60 + "\n")
        f.write("CLASS_NAME_MAP = {\n")
        for class_id in sorted(class_mappings.keys(), key=lambda x: int(x.replace('class', ''))):
            f.write(f'    "{class_id}": "{class_mappings[class_id]}",\n')
        f.write("}\n")
    
    print(f"\n✅ Saved mappings to {MAPPING_FILE} and class_mapping.txt")

def scale_boxes_from_letterbox_np(boxes, ratio, dwdh, original_shape):
    """Rescale boxes from letterboxed image size back to original image size."""
    if boxes is None or len(boxes) == 0:
        return np.array([])

    boxes = np.array(boxes)
    if boxes.ndim == 1:
        if len(boxes) == 4:
            boxes = boxes.reshape(1, 4)
        else:
            return np.array([])

    if boxes.shape[1] != 4:
        return np.array([])

    dw, dh = dwdh
    h0, w0 = original_shape

    if isinstance(ratio, (list, tuple, np.ndarray)):
        ratio = ratio[0] if len(ratio) > 0 else 1.0
    
    ratio = float(ratio)

    boxes[:, 0] = (boxes[:, 0] - dw) / ratio
    boxes[:, 1] = (boxes[:, 1] - dh) / ratio
    boxes[:, 2] = (boxes[:, 2] - dw) / ratio
    boxes[:, 3] = (boxes[:, 3] - dh) / ratio

    boxes[:, 0] = np.clip(boxes[:, 0], 0, w0 - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h0 - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w0 - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h0 - 1)

    return boxes

def main():
    print("\n" + "=" * 60)
    print("INTERACTIVE CLASS IDENTIFIER")
    print("=" * 60)
    print("\nInstructions:")
    print("1. Point camera at a KNOWN object")
    print("2. When detected, the class name will be highlighted")
    print("3. Press 'l' to LABEL that class with the object name")
    print("4. Press 's' to SKIP")
    print("5. Press 'q' to QUIT and save")
    print("\nMapped classes will be saved to 'class_mapping.json'")
    print("=" * 60 + "\n")
    
    load_existing_mappings()
    
    # device
    device = select_device(DEVICE)
    print("Using device:", device)

    # Load model
    model = DetectMultiBackend(WEIGHTS, device=device, dnn=False, fp16=False)
    stride = int(model.stride)
    names = model.names
    imgsz = check_img_size(IMG_SIZE, s=stride)
    print("Model loaded successfully!\n")

    # open webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam (index {CAMERA_INDEX})")

    current_detections = {}
    waiting_for_label = False
    selected_class = None
    
    print("🎥 Webcam opened! Starting detection...\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img0 = frame.copy()
            h0, w0 = img0.shape[:2]

            # Preprocess
            img, ratio, (dw, dh) = letterbox(img0, imgsz, stride=stride)
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            im = torch.from_numpy(img).to(device)
            im = im.float() / 255.0
            if im.ndim == 3:
                im = im.unsqueeze(0)

            # Inference
            pred = model(im, augment=False, visualize=False)
            pred = non_max_suppression(pred, CONF_THRESH, IOU_THRESH, None, False, max_det=1000)

            current_detections = {}
            
            # Process detections
            if len(pred) and pred[0] is not None and len(pred[0]):
                det = pred[0].clone().detach().cpu().numpy()
                boxes = det[:, :4].copy()

                if boxes is not None and len(boxes) > 0:
                    boxes = scale_boxes_from_letterbox_np(boxes, ratio, (dw, dh), (h0, w0))
                else:
                    boxes = []

                if len(boxes) > 0:
                    det[:, :4] = boxes
                    
                    for *xyxy, conf, cls in det:
                        x1, y1, x2, y2 = map(int, xyxy)
                        cls = int(cls)
                        class_name = names[cls] if cls < len(names) else f"class{cls}"
                        
                        # Track detections
                        if class_name not in current_detections:
                            current_detections[class_name] = []
                        current_detections[class_name].append((xyxy, conf))
                        
                        # Color based on mapping status
                        if class_name in class_mappings:
                            color = (0, 255, 0)  # Green = mapped
                            display_name = class_mappings[class_name]
                        else:
                            color = (0, 165, 255)  # Orange = unmapped
                            display_name = class_name
                        
                        # Highlight if selected
                        if waiting_for_label and class_name == selected_class:
                            color = (255, 0, 255)  # Magenta = selected
                        
                        # Draw box
                        cv2.rectangle(img0, (x1, y1), (x2, y2), color, 2)
                        label = f"{display_name} {conf:.2f}"
                        
                        # Label background
                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(img0, (x1, y1 - h - 6), (x1 + w + 6, y1), color, -1)
                        cv2.putText(img0, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # Display status
            y_offset = 30
            cv2.putText(img0, f"Mapped: {len(class_mappings)}/22 classes", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            y_offset += 35
            
            if current_detections:
                cv2.putText(img0, f"Detected: {len(current_detections)} different classes", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                y_offset += 30
                
                for class_name in sorted(current_detections.keys()):
                    status = "✓ MAPPED" if class_name in class_mappings else "? UNKNOWN"
                    text = f"{class_name}: {status}"
                    color = (0,255,0) if class_name in class_mappings else (0,165,255)
                    cv2.putText(img0, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    y_offset += 25
            
            # Instructions
            y_bottom = img0.shape[0] - 10
            cv2.putText(img0, "Press 'L' to label | 'S' to skip | 'Q' to quit", 
                       (10, y_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            cv2.imshow("Class Identifier (Point at known objects!)", img0)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('l') and current_detections:
                # Select first unmapped class to label
                for class_name in sorted(current_detections.keys()):
                    if class_name not in class_mappings:
                        waiting_for_label = True
                        selected_class = class_name
                        print(f"\n🎯 Selected: {selected_class}")
                        print(f"What object is this? (Type name and press Enter): ", end='', flush=True)
                        break
                else:
                    print("\n✅ All detected classes are already mapped!")
            elif key == ord('s'):
                waiting_for_label = False
                selected_class = None
                print("\n⏭️  Skipped")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        if class_mappings:
            save_mappings()
            print(f"\n✅ Done! Mapped {len(class_mappings)}/22 classes")
            print(f"\nYour mappings:")
            for k, v in sorted(class_mappings.items()):
                print(f"  {k} → {v}")
        else:
            print("\n⚠️  No classes were mapped")

if __name__ == "__main__":
    if not os.path.isdir(REPO):
        print(f"Error: {REPO} folder not found.")
        sys.exit(1)
    if not os.path.exists(WEIGHTS):
        print(f"Error: {WEIGHTS} not found.")
        sys.exit(1)
    
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
