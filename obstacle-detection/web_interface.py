"""
Professional Web Interface for Drone Obstacle Detection System
Flask-based dashboard with real-time video streaming and controls
OPTIMIZED for better FPS and performance
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
from queue import Queue

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
        # Reduced resolution for better FPS
        self.imgsz = check_img_size(416, s=self.stride)  # Changed from 640 to 416
        
        self.cap = None
        self.navigator = DroneNavigator()
        self.is_running = False
        self.current_frame = None
        self.frame_queue = Queue(maxsize=2)  # Frame buffer for smoother streaming
        self.lock = threading.Lock()
        self.last_process_time = time.time()
        
        self.detection_stats = {
            'total_obstacles': 0,
            'class_breakdown': {},
            'navigation': {},
            'fps': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Settings - optimized for better performance
        self.settings = {
            'conf_threshold': 0.25,
            'iou_threshold': 0.45,
            'camera_index': 0,
            'show_navigation': True,
            'show_zones': True,
            'frame_skip': 1,  # Process every N frames (1 = no skip, 2 = skip every other frame)
            'jpeg_quality': 80  # JPEG compression quality (lower = faster, smaller)
        }
        
        # Performance tracking
        self.frame_count = 0
        self.fps_start_time = time.time()
    
    def start_camera(self):
        """Start camera capture with optimized settings"""
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.settings['camera_index'])
            if self.cap.isOpened():
                # Optimize camera settings for better FPS
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
                self.cap.set(cv2.CAP_PROP_FPS, 30)  # Request 30 FPS
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
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
        """Process single frame for detection - OPTIMIZED"""
        if not self.cap or not self.cap.isOpened():
            return None
        
        # Frame skipping for better performance
        self.frame_count += 1
        if self.frame_count % self.settings['frame_skip'] != 0:
            ret, frame = self.cap.read()
            return frame if ret else None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        start_time = time.time()
        
        # Resize frame for faster processing
        scale_factor = 0.75  # Process at 75% of original size
        frame_resized = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
        img0 = frame_resized.copy()
        h0, w0 = img0.shape[:2]
        
        # Preprocess
        img, ratio, (dw, dh) = letterbox(img0, self.imgsz, stride=self.stride, auto=False)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        im = torch.from_numpy(img).to(self.device).float() / 255.0
        if im.ndim == 3:
            im = im.unsqueeze(0)
        
        # Inference with torch.no_grad() for better performance
        with torch.no_grad():
            pred = self.model(im, augment=False, visualize=False)
            pred = non_max_suppression(pred, self.settings['conf_threshold'], 
                                     self.settings['iou_threshold'], None, False, max_det=100)
        
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
                    
                    # Scale coordinates back to original frame size
                    x1, y1, x2, y2 = int(x1/scale_factor), int(y1/scale_factor), int(x2/scale_factor), int(y2/scale_factor)
                    
                    detections.append([x1, y1, x2, y2, conf, cls])
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Draw on original sized frame for better quality
        output_frame = frame.copy()
        
        # Draw detections with optimized rendering
        for x1, y1, x2, y2, conf, cls in detections:
            color = (0, 255, 0)
            
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
            class_name = self.names[cls] if cls < len(self.names) else f"class{cls}"
            label = f"{class_name} {conf:.2f}"
            cv2.putText(output_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Navigation analysis with full overlay
        nav_result = {}
        if self.settings['show_navigation']:
            nav_result = self.navigator.analyze_obstacles(detections)
            # Use the full navigation overlay for better visualization
            output_frame = self.navigator.draw_navigation_overlay(output_frame, nav_result, detections)
            
            # Add obstacle count display (positioned to not overlap)
            if len(detections) > 0:
                count_text = f"OBSTACLES: {len(detections)}"
                cv2.rectangle(output_frame, (8, 145), (250, 185), (0, 0, 0), -1)
                cv2.rectangle(output_frame, (8, 145), (250, 185), (0, 255, 0), 2)
                cv2.putText(output_frame, count_text, (15, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add FPS indicator (top-left, below obstacle count)
            h, w = output_frame.shape[:2]
            fps_text = f"FPS: {self.detection_stats.get('fps', 0)}"
            cv2.putText(output_frame, fps_text, (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Calculate FPS
        process_time = time.time() - start_time
        fps = 1.0 / process_time if process_time > 0 else 0
        
        # Update stats with thread safety
        with self.lock:
            self.detection_stats = {
                'total_obstacles': len(detections),
                'class_breakdown': class_counts,
                'navigation': nav_result,
                'fps': round(fps, 1),
                'process_time': round(process_time * 1000, 1),  # in ms
                'timestamp': datetime.now().isoformat()
            }
        
        return output_frame

detection_system = WebDetectionSystem()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route - OPTIMIZED"""
    def generate():
        detection_system.start_camera()
        while True:
            frame = detection_system.process_frame()
            if frame is not None:
                # Dynamic JPEG quality based on settings
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), detection_system.settings['jpeg_quality']]
                ret, buffer = cv2.imencode('.jpg', frame, encode_param)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                time.sleep(0.01)  # Small delay if no frame
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def get_stats():
    """Get current detection statistics with thread safety"""
    with detection_system.lock:
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
    print("\n" + "=" * 70)
    print("🚁 DRONE OBSTACLE DETECTION - WEB INTERFACE")
    print("=" * 70)
    print(f"\n✅ Server starting...")
    print(f"📡 Access dashboard at: http://localhost:5000")
    print(f"🌐 Network access at: http://0.0.0.0:5000")
    print(f"\n⚡ Performance Optimizations Enabled:")
    print(f"   - Reduced inference resolution (416x416)")
    print(f"   - Frame skipping support")
    print(f"   - Optimized JPEG compression")
    print(f"   - Thread-safe operations")
    print(f"\n💡 Tips for better FPS:")
    print(f"   - Lower confidence threshold")
    print(f"   - Enable frame skipping (skip 1-2 frames)")
    print(f"   - Reduce JPEG quality to 60-70")
    print(f"   - Close other camera applications")
    print("\n" + "=" * 70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)