"""
Professional Web Interface for Drone Obstacle Detection System
Flask-based dashboard with real-time video streaming and controls
"""

from flask import Flask, render_template, Response, jsonify, request
import cv2
import torch
import numpy as np
import json
import threading
import time
from datetime import datetime
import base64
import os
import pathlib

# Fix for Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

import sys
sys.path.insert(0, "yolov5")

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, check_img_size
from utils.augmentations import letterbox
from utils.torch_utils import select_device
from drone_navigation import DroneNavigator, get_navigation_command

app = Flask(__name__)

class WebDetectionSystem:
    def __init__(self):
        self.device = select_device('')
        self.model = DetectMultiBackend("best.pt", device=self.device, dnn=False, fp16=False)
        self.stride = int(self.model.stride)
        self.names = self.model.names
        self.imgsz = check_img_size(640, s=self.stride)
        
        self.cap = None
        self.navigator = DroneNavigator()
        self.is_running = False
        self.current_frame = None
        self.detection_stats = {
            'total_obstacles': 0,
            'class_breakdown': {},
            'navigation': {},
            'fps': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Settings
        self.settings = {
            'conf_threshold': 0.25,
            'iou_threshold': 0.45,
            'camera_index': 1,
            'show_navigation': True,
            'show_zones': True,
            'detection_mode': 'obstacles'  # 'obstacles' or 'attendance'
        }
    
    def start_camera(self):
        """Start camera capture"""
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.settings['camera_index'])
            if self.cap.isOpened():
                # Initialize navigator with frame size
                ret, frame = self.cap.read()
                if ret:
                    h, w = frame.shape[:2]
                    self.navigator = DroneNavigator(w, h)
                return True
        return False
    
    def stop_camera(self):
        """Stop camera capture"""
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def scale_boxes(self, boxes, ratio, dwdh, shape):
        """Scale boxes from letterbox to original image size"""
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
    
    def process_frame(self):
        """Process single frame for detection"""
        if not self.cap or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        start_time = time.time()
        img0 = frame.copy()
        h0, w0 = img0.shape[:2]
        
        # Preprocess
        img, ratio, (dw, dh) = letterbox(img0, self.imgsz, stride=self.stride)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        im = torch.from_numpy(img).to(self.device).float() / 255.0
        if im.ndim == 3:
            im = im.unsqueeze(0)
        
        # Inference
        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, self.settings['conf_threshold'], 
                                 self.settings['iou_threshold'], None, False, max_det=1000)
        
        detections = []
        class_counts = {}
        
        # Process detections
        if len(pred) and pred[0] is not None and len(pred[0]):
            det = pred[0].clone().detach().cpu().numpy()
            boxes = det[:, :4].copy()
            boxes = self.scale_boxes(boxes, ratio, (dw, dh), (h0, w0))
            
            if len(boxes) > 0:
                det[:, :4] = boxes
                
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)
                    cls = int(cls)
                    class_name = self.names[cls] if cls < len(self.names) else f"class{cls}"
                    
                    detections.append([x1, y1, x2, y2, conf, cls])
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    
                    # Draw detection
                    color = (0, 255, 0)
                    if self.settings['detection_mode'] == 'attendance' and cls == 9:
                        color = (255, 0, 0)  # Red for faces
                    
                    cv2.rectangle(img0, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(img0, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Navigation analysis
        nav_result = self.navigator.analyze_obstacles(detections)
        
        # Draw overlays
        if self.settings['show_navigation']:
            img0 = self.navigator.draw_navigation_overlay(img0, nav_result, detections)
        
        # Update stats
        fps = 1.0 / (time.time() - start_time)
        self.detection_stats = {
            'total_obstacles': len(detections),
            'class_breakdown': class_counts,
            'navigation': nav_result,
            'fps': round(fps, 1),
            'timestamp': datetime.now().isoformat()
        }
        
        return img0

detection_system = WebDetectionSystem()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    def generate():
        detection_system.start_camera()
        while True:
            frame = detection_system.process_frame()
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)  # ~30 FPS
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def get_stats():
    """Get current detection statistics"""
    return jsonify(detection_system.detection_stats)

@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    """Get or update system settings"""
    if request.method == 'POST':
        data = request.json
        detection_system.settings.update(data)
        return jsonify({'status': 'success', 'settings': detection_system.settings})
    else:
        return jsonify(detection_system.settings)

@app.route('/api/navigation_command')
def get_navigation_command():
    """Get drone navigation command"""
    nav_result = detection_system.detection_stats.get('navigation', {})
    if nav_result:
        command = get_navigation_command(nav_result)
        return jsonify(command)
    return jsonify({'error': 'No navigation data available'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)