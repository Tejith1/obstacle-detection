"""
Attendance System using YOLOv5 Face Detection
Counts faces (class 9) for student attendance tracking
"""

import sys
import time
import cv2
import torch
import numpy as np
import os
import pathlib
from collections import Counter, defaultdict
from datetime import datetime
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

# Configuration
WEIGHTS = "best.pt"
IMG_SIZE = 640
CONF_THRESH = 0.25
IOU_THRESH = 0.45
DEVICE = ''
CAMERA_INDEX = 1
FACE_CLASS_ID = 9  # Assuming class 9 is faces

class AttendanceTracker:
    def __init__(self):
        self.attendance_log = []
        self.current_count = 0
        self.max_count_today = 0
        self.session_start = datetime.now()
        self.face_history = []  # Track faces over time for stability
        
    def update_count(self, face_detections):
        """Update attendance count based on face detections"""
        current_faces = len(face_detections)
        
        # Add to history for smoothing
        self.face_history.append(current_faces)
        if len(self.face_history) > 10:  # Keep last 10 frames
            self.face_history.pop(0)
        
        # Use average of recent frames for stability
        stable_count = int(np.mean(self.face_history))
        
        self.current_count = stable_count
        if stable_count > self.max_count_today:
            self.max_count_today = stable_count
            
        # Log significant changes
        if len(self.attendance_log) == 0 or abs(stable_count - self.attendance_log[-1]['count']) >= 2:
            self.attendance_log.append({
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'count': stable_count,
                'raw_detections': current_faces
            })
    
    def get_attendance_summary(self):
        """Get current attendance statistics"""
        return {
            'current_present': self.current_count,
            'max_today': self.max_count_today,
            'session_duration': str(datetime.now() - self.session_start).split('.')[0],
            'total_logs': len(self.attendance_log)
        }
    
    def save_attendance_report(self, filename=None):
        """Save attendance report to file"""
        if filename is None:
            filename = f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'session_info': {
                'start_time': self.session_start.strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'duration': str(datetime.now() - self.session_start).split('.')[0]
            },
            'statistics': {
                'max_attendance': self.max_count_today,
                'final_count': self.current_count,
                'total_changes': len(self.attendance_log)
            },
            'detailed_log': self.attendance_log
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        return filename

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

def draw_attendance_overlay(frame, tracker, face_detections):
    """Draw attendance information on frame"""
    h, w = frame.shape[:2]
    
    # Background for attendance info
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Title
    cv2.putText(frame, "ATTENDANCE SYSTEM", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Current count
    stats = tracker.get_attendance_summary()
    cv2.putText(frame, f"PRESENT: {stats['current_present']}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Max count today
    cv2.putText(frame, f"MAX TODAY: {stats['max_today']}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Session duration
    cv2.putText(frame, f"SESSION: {stats['session_duration']}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Detection count
    cv2.putText(frame, f"DETECTIONS: {len(face_detections)}", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Instructions
    cv2.putText(frame, "Press 'S' to save report | 'Q' to quit", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

def main():
    print("\n" + "=" * 60)
    print("ATTENDANCE SYSTEM - YOLOv5 Face Detection")
    print("=" * 60)
    print(f"Looking for faces as class {FACE_CLASS_ID}")
    print("Press 'S' to save attendance report")
    print("Press 'Q' to quit")
    print("=" * 60 + "\n")
    
    # Initialize
    device = select_device(DEVICE)
    model = DetectMultiBackend(WEIGHTS, device=device, dnn=False, fp16=False)
    stride = int(model.stride)
    names = model.names
    imgsz = check_img_size(IMG_SIZE, s=stride)
    
    print(f"Model loaded. Looking for class: {names.get(FACE_CLASS_ID, f'class{FACE_CLASS_ID}')}")
    
    # Initialize tracker
    tracker = AttendanceTracker()
    
    # Open webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam (index {CAMERA_INDEX})")
    
    prev_time = 0.0
    
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

            face_detections = []
            
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
                        cls = int(cls)
                        
                        # Only process face detections (class 9)
                        if cls == FACE_CLASS_ID:
                            x1, y1, x2, y2 = map(int, xyxy)
                            face_detections.append([x1, y1, x2, y2, conf, cls])
                            
                            # Draw face detection
                            cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"Face {conf:.2f}"
                            cv2.putText(img0, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Update attendance
            tracker.update_count(face_detections)
            
            # Draw attendance overlay
            draw_attendance_overlay(img0, tracker, face_detections)

            # FPS
            cur_time = time.time()
            fps = 1.0 / (cur_time - prev_time) if prev_time else 0.0
            prev_time = cur_time
            cv2.putText(img0, f"FPS: {fps:.1f}", (w0-100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            # Show
            cv2.imshow("Attendance System - Face Detection", img0)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = tracker.save_attendance_report()
                print(f"\n✅ Attendance report saved: {filename}")
                stats = tracker.get_attendance_summary()
                print(f"📊 Session Summary:")
                print(f"   Current Present: {stats['current_present']}")
                print(f"   Max Today: {stats['max_today']}")
                print(f"   Duration: {stats['session_duration']}\n")

    except KeyboardInterrupt:
        print("\n⚠️ Stopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Final report
        final_stats = tracker.get_attendance_summary()
        print(f"\n📋 Final Attendance Summary:")
        print(f"   Students Present: {final_stats['current_present']}")
        print(f"   Peak Attendance: {final_stats['max_today']}")
        print(f"   Session Duration: {final_stats['session_duration']}")
        
        # Auto-save final report
        filename = tracker.save_attendance_report()
        print(f"📄 Final report saved: {filename}")

if __name__ == "__main__":
    if not os.path.isdir(REPO):
        print(f"Error: {REPO} folder not found.")
        sys.exit(1)
    if not os.path.exists(WEIGHTS):
        print(f"Error: {WEIGHTS} not found.")
        sys.exit(1)
    
    main()