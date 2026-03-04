"""
Multi-Model Ensemble Detection API Backend
Serves 4 object detection models via REST API with live webcam streaming
"""

from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import cv2
import torch
import numpy as np
import json
import threading
import time
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
import base64
from io import BytesIO
from PIL import Image
import platform
from queue import Queue
import platform

app = Flask(__name__)
CORS(app)

# Get backend directory for relative paths
BACKEND_DIR = Path(__file__).resolve().parent
MODELS_DIR = BACKEND_DIR.parent / "models"

# Model configurations
MODELS_CONFIG = {
    "model_1": {
        "name": "YOLOv5 (Obstacle Detection)",
        "path": MODELS_DIR / "obstacle_detection_yolov5_best.pt",
        "type": "yolov5",
        "description": "Real-time obstacle detection from locally trained YOLOv5"
    },
    "model_2": {
        "name": "YOLO11n (VisDrone v1-1)",
        "path": MODELS_DIR / "visdrone_yolo11n_best.pt",
        "type": "yolo_ultralytics",
        "description": "Nano model trained on VisDrone dataset (30 epochs)"
    },
    "model_3": {
        "name": "YOLO11m (VisDrone v1-2)",
        "path": MODELS_DIR / "visdrone_yolo11m_best.pt",
        "type": "yolo_ultralytics",
        "description": "Medium model trained on VisDrone dataset (50 epochs)"
    },
    "model_4": {
        "name": "Results Model (YOLO)",
        "path": MODELS_DIR / "results_model_best.pt",
        "type": "yolo_ultralytics",
        "description": "Model from results.zip training artifacts"
    }
}

class ModelManager:
    """Manages all 4 models and webcam streams"""
    
    def __init__(self):
        self.models = {}
        self.webcams = {}
        self.frames = {}
        self.locks = {}
        self.frame_queues = {}
        self.frame_counts = {}
        self.settings = {
            "conf_threshold": 0.25,
            "iou_threshold": 0.45,
            "imgsz": 320,  # Reduced from 416 for even faster inference (4x reduction in pixels)
            "frame_skip": 4  # Process every 4th frame (skip 3) to significantly reduce lag
        }
        self.load_models()
    
    def load_models(self):
        """Load all models"""
        for model_id, config in MODELS_CONFIG.items():
            try:
                if config["path"].exists():
                    if config["type"] == "yolo_ultralytics":
                        self.models[model_id] = YOLO(str(config["path"]))
                    elif config["type"] == "yolov5":
                        # For YOLOv5, we'll use Ultralytics wrapper which supports v5 models
                        self.models[model_id] = YOLO(str(config["path"]))
                    
                    print(f"✓ Loaded {model_id}: {config['name']}")
                else:
                    print(f"✗ Model not found: {config['path']}")
            except Exception as e:
                print(f"✗ Failed to load {model_id}: {str(e)}")
                self.models[model_id] = None
    
    def get_model(self, model_id):
        """Get model by ID"""
        return self.models.get(model_id)
    
    def start_webcam(self, model_id, camera_index=0):
        """Start webcam for a model"""
        if model_id not in self.webcams:
            self.webcams[model_id] = cv2.VideoCapture(camera_index)
            # Reduce resolution for faster capture (320x240)
            self.webcams[model_id].set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            self.webcams[model_id].set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            self.webcams[model_id].set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Prevent frame buffering
            self.frames[model_id] = None
            self.locks[model_id] = threading.Lock()
            self.frame_queues[model_id] = Queue(maxsize=1)  # Keep only latest frame
            self.frame_counts[model_id] = 0
            
            # Start frame capture thread
            thread = threading.Thread(target=self._capture_frames, args=(model_id,), daemon=True)
            thread.start()
            return True
        return False
    
    def _capture_frames(self, model_id):
        """Capture frames from webcam (only store latest, skip frames for performance)"""
        cap = self.webcams[model_id]
        frame_skip = self.settings.get("frame_skip", 2)
        frame_counter = 0
        
        while cap.isOpened() and model_id in self.webcams:
            ret, frame = cap.read()
            if ret:
                frame_counter += 1
                # Only process every N frames to reduce lag
                if frame_counter % frame_skip == 0:
                    with self.locks[model_id]:
                        self.frames[model_id] = frame
                    frame_counter = 0
            else:
                time.sleep(0.01)
    
    def stop_webcam(self, model_id):
        """Stop webcam"""
        if model_id in self.webcams:
            self.webcams[model_id].release()
            del self.webcams[model_id]
            if model_id in self.frames:
                del self.frames[model_id]
            if model_id in self.locks:
                del self.locks[model_id]
            if model_id in self.frame_queues:
                del self.frame_queues[model_id]
            if model_id in self.frame_counts:
                del self.frame_counts[model_id]
            return True
        return False
    
    def get_frame(self, model_id):
        """Get latest frame"""
        if model_id in self.frames and self.locks[model_id]:
            with self.locks[model_id]:
                return self.frames[model_id]
        return None
    
    def run_inference(self, model_id, frame):
        """Run inference on frame"""
        model = self.get_model(model_id)
        if model is None or frame is None:
            return None
        
        try:
            results = model.predict(
                source=frame,
                conf=self.settings["conf_threshold"],
                iou=self.settings["iou_threshold"],
                imgsz=self.settings["imgsz"],
                verbose=False
            )
            return results[0]
        except Exception as e:
            print(f"Inference error for {model_id}: {str(e)}")
            return None
    
    def draw_annotations(self, frame, result):
        """Draw bounding boxes on frame"""
        if result is None:
            return frame
        
        annotated = result.plot()
        return annotated[:, :, ::-1] if len(annotated.shape) == 3 else annotated
    
    def get_model_info(self, model_id):
        """Get model information"""
        if model_id not in MODELS_CONFIG:
            return None
        
        config = MODELS_CONFIG[model_id]
        model = self.get_model(model_id)
        
        return {
            "id": model_id,
            "name": config["name"],
            "description": config["description"],
            "type": config["type"],
            "loaded": model is not None,
            "classes": list(model.names.values()) if model else []
        }

# Initialize manager
manager = ModelManager()


# ==================== UTILITY FUNCTIONS ====================

def detect_available_cameras(max_cameras=10):
    """Detect available cameras on the system"""
    cameras = []
    
    for i in range(max_cameras):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Check if camera actually works
                ret, _ = cap.read()
                if ret:
                    # Get camera name/properties
                    cameras.append({
                        "index": i,
                        "name": f"Camera {i}",
                        "available": True
                    })
                cap.release()
        except:
            break
    
    # If no cameras found, suggest index 0 as default
    if not cameras:
        cameras = [{"index": 0, "name": "Default Camera", "available": True}]
    
    return cameras


# ==================== API ENDPOINTS ====================

@app.route("/api/health", methods=["GET"])
def health():
    """Health check"""
    return jsonify({"status": "ok", "service": "ensemble-models-api"}), 200


@app.route("/api/cameras", methods=["GET"])
def get_cameras():
    """Get available cameras"""
    cameras = detect_available_cameras()
    return jsonify({
        "cameras": cameras,
        "available_count": len(cameras)
    }), 200


@app.route("/api/models", methods=["GET"])
def list_models():
    """Get all models info"""
    models_info = {}
    for model_id in MODELS_CONFIG.keys():
        models_info[model_id] = manager.get_model_info(model_id)
    
    return jsonify(models_info), 200


@app.route("/api/models/<model_id>", methods=["GET"])
def get_model_info(model_id):
    """Get specific model info"""
    info = manager.get_model_info(model_id)
    if info is None:
        return jsonify({"error": "Model not found"}), 404
    return jsonify(info), 200


@app.route("/api/webcam/<model_id>/start", methods=["POST"])
def start_webcam(model_id):
    """Start webcam for model"""
    if model_id not in MODELS_CONFIG:
        return jsonify({"error": "Model not found"}), 404
    
    if manager.get_model(model_id) is None:
        return jsonify({"error": "Model not loaded"}), 400
    
    camera_index = request.json.get("camera_index", 0) if request.json else 0
    success = manager.start_webcam(model_id, camera_index)
    
    return jsonify({
        "success": success,
        "model_id": model_id,
        "message": "Webcam started" if success else "Webcam already running"
    }), 200


@app.route("/api/webcam/<model_id>/stop", methods=["POST"])
def stop_webcam(model_id):
    """Stop webcam for model"""
    success = manager.stop_webcam(model_id)
    return jsonify({
        "success": success,
        "model_id": model_id
    }), 200


@app.route("/api/webcam/<model_id>/frame", methods=["GET"])
def get_frame_with_inference(model_id):
    """Get current frame with inference"""
    if model_id not in MODELS_CONFIG:
        return jsonify({"error": "Model not found"}), 404
    
    frame = manager.get_frame(model_id)
    if frame is None:
        return jsonify({"error": "No frame available"}), 400
    
    # Run inference
    result = manager.run_inference(model_id, frame)
    
    # Draw annotations
    annotated_frame = manager.draw_annotations(frame, result)
    
    # Encode to JPEG with lower quality for faster transmission (optimized for 4 models)
    _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
    frame_bytes = buffer.tobytes()
    
    # Return as base64
    frame_b64 = base64.b64encode(frame_bytes).decode('utf-8')
    
    # Extract detections info
    detections = []
    if result is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = result.names[class_id]
            detections.append({
                "class": class_name,
                "confidence": round(confidence, 3)
            })
    
    # Count by class
    class_counts = {}
    for det in detections:
        class_counts[det["class"]] = class_counts.get(det["class"], 0) + 1
    
    return jsonify({
        "model_id": model_id,
        "frame": frame_b64,
        "detections": detections,
        "detection_count": len(detections),
        "class_counts": class_counts,
        "timestamp": time.time()
    }), 200


@app.route("/api/inference", methods=["POST"])
def run_inference():
    """Run inference on uploaded image"""
    if model_id not in MODELS_CONFIG:
        return jsonify({"error": "Model not found"}), 404
    
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    try:
        file = request.files["image"]
        model_id = request.form.get("model_id")
        
        # Read image
        image_data = file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run inference
        result = manager.run_inference(model_id, frame)
        
        # Draw annotations
        annotated_frame = manager.draw_annotations(frame, result)
        
        # Encode to JPEG with optimized quality
        _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        frame_bytes = buffer.tobytes()
        
        # Return as base64
        frame_b64 = base64.b64encode(frame_bytes).decode('utf-8')
        
        # Extract detections
        detections = []
        if result is not None:
            for box in result.boxes:
                detections.append({
                    "class": result.names[int(box.cls[0])],
                    "confidence": float(box.conf[0])
                })
        
        return jsonify({
            "model_id": model_id,
            "frame": frame_b64,
            "detections": detections,
            "detection_count": len(detections)
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/settings", methods=["GET", "POST"])
def settings():
    """Get/update settings"""
    if request.method == "GET":
        return jsonify(manager.settings), 200
    
    if request.method == "POST":
        data = request.json
        manager.settings.update(data)
        return jsonify({"success": True, "settings": manager.settings}), 200


# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    print("\n" + "="*60)
    print("ENSEMBLE MODELS API SERVER")
    print("="*60)
    print("\nLoading models...")
    print(f"Models loaded: {sum(1 for m in manager.models.values() if m is not None)}/{len(MODELS_CONFIG)}")
    print("\nStarting Flask server on http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
