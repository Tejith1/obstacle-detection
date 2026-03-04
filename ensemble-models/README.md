# 🤖 Ensemble Models - Multi-Model Detection System

Complete multi-model object detection system with **4 different YOLO models** running simultaneously through a modern React frontend and Flask backend.

## 📦 Project Structure

```
ensemble-models/
├── models/                          # All 4 trained models
│   ├── obstacle_detection_yolov5_best.pt      # Model 1: YOLOv5 (Obstacle Detection)
│   ├── visdrone_yolo11n_best.pt               # Model 2: YOLO11n (VisDrone v1-1)
│   ├── visdrone_yolo11m_best.pt               # Model 3: YOLO11m (VisDrone v1-2)
│   ├── results_model_best.pt                  # Model 4: Custom YOLO (Results)
│   └── visdrone.yaml                         # Dataset configuration
├── backend/                         # Python Flask API
│   ├── app.py                      # Main API server
│   ├── requirements.txt            # Python dependencies
│   └── .gitignore
├── frontend/                        # React + Vite application
│   ├── src/
│   │   ├── main.jsx               # Entry point
│   │   ├── App.jsx                # Main app component
│   │   ├── App.css                # App styles
│   │   ├── index.css              # Global styles
│   │   └── components/
│   │       ├── ModelBox.jsx       # Individual model UI component
│   │       └── ModelBox.css       # Component styles
│   ├── public/
│   │   └── index.html             # HTML template
│   ├── package.json               # Node dependencies
│   ├── vite.config.js             # Vite configuration
│   ├── .gitignore
│   └── requirements.txt           # (legacy)
└── README.md                        # This file
```

## 🎯 Models Overview

| Model | Framework | Source | Training | Purpose |
|-------|-----------|--------|----------|---------|
| **Model 1** | YOLOv5 | obstacle-detection | Pre-trained | Real-time obstacle detection |
| **Model 2** | YOLO11n | VisDrone v1-1 | 30 epochs, imgsz 640 | Smaller, faster detection |
| **Model 3** | YOLO11m | VisDrone v1-2 | 50 epochs, imgsz 640, SGD | Medium-sized, balanced |
| **Model 4** | YOLO | results.zip | Custom training | Experimental/alternative model |

All models are trained on the **VisDrone dataset** (except Model 1) with 10 classes: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ (for backend)
- Node.js 16+ (for frontend)
- Webcam (or video source)

### 1. Backend Setup

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install CPU PyTorch first (recommended for Windows)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
python -m pip install -r requirements.txt
```

### 2. Frontend Setup

```powershell
cd frontend
npm install
```

### 3. Run the System

**Terminal 1 - Start Backend (Port 5000):**
```powershell
cd backend
.\venv\Scripts\Activate.ps1
python app.py
```

**Terminal 2 - Start Frontend (Port 3000):**
```powershell
cd frontend
npm start
```

Then open your browser and navigate to:
```
http://localhost:3000
```

## 🎨 Frontend Features

- **4 Model Output Boxes**: Each model in a dedicated card with live inference
- **Real-time Webcam Streaming**: Independent webcam streams for each model
- **Detection Visualization**: Bounding boxes and labels on annotated frames
- **Live Statistics**: 
  - FPS counter for each model
  - Detection count and class breakdown
  - Confidence scores
- **Model Information**: Classes, descriptions, and status for each model
- **Start/Stop Controls**: Individual control for each model's webcam

## 🔌 Backend API

### Health & Info
- `GET /api/health` - Server health check
- `GET /api/models` - List all loaded models
- `GET /api/models/<model_id>` - Get specific model info

### Webcam Control
- `POST /api/webcam/<model_id>/start` - Start webcam for model
- `POST /api/webcam/<model_id>/stop` - Stop webcam
- `GET /api/webcam/<model_id>/frame` - Get current annotated frame with detections

### Inference
- `POST /api/inference` - Run inference on uploaded image

### Settings
- `GET /api/settings` - Get current settings
- `POST /api/settings` - Update settings (conf_threshold, iou_threshold, imgsz)

## ⚙️ Configuration

Edit settings in the React app or via API:

```python
{
  "conf_threshold": 0.25,   # Confidence threshold (0-1)
  "iou_threshold": 0.45,    # IOU threshold for NMS
  "imgsz": 640              # Inference image size
}
```

## 🎓 Training Models Used

### Model 1: YOLOv5
- Source: `obstacle-detection/best.pt`
- Custom trained for obstacle detection
- Uses local YOLOv5 repo

### Model 2: YOLO11n (Nano)
- Source: `v1-1/best.pt`
- **30 epochs**, imgsz 640, optimizer: auto
- Lightweight, good for real-time inference

### Model 3: YOLO11m (Medium)
- Source: `v1-2/best.pt`
- **50 epochs**, imgsz 640, optimizer: SGD, batch: 16
- Better accuracy, slower than nano

### Model 4: Custom Results
- Source: `results.zip` extracted model
- Alternative training experiment
- Available for A/B testing

## 🖥️ System Architecture

```
┌─────────────────────────────────┐
│   React Frontend (Port 3000)    │
│   - 4 Model Boxes               │
│   - Webcam Controls             │
│   - Real-time Display           │
└────────────┬────────────────────┘
             │ HTTP/REST API
             ▼
┌─────────────────────────────────┐
│   Flask Backend (Port 5000)     │
│   - Model Manager               │
│   - Webcam Streams              │
│   - Inference Engine            │
└────────────┬────────────────────┘
             │ OpenCV/PyTorch
             ▼
┌─────────────────────────────────┐
│   4 YOLO Models                 │
│   - YOLOv5                      │
│   - YOLO11n, YOLO11m            │
│   - Custom YOLO                 │
└─────────────────────────────────┘
```

## 🛠️ Troubleshooting

### Models not loading?
- Check that all `.pt` files exist in `models/` folder
- Verify Python can import torch and ultralytics
- Check console output for specific error messages

### Webcam not starting?
- Ensure only one application is using the webcam
- Try changing `camera_index` in API calls (0, 1, 2, etc.)
- On Windows, grant camera permissions to Python

### Slow performance?
- Reduce `imgsz` in settings (e.g., 416 or 320)
- Increase `conf_threshold` to reduce detections
- Close other applications using GPU/CPU

### CORS errors in browser?
- Backend runs on port 5000, frontend on port 3000
- Flask-CORS is configured automatically
- Ensure both servers are running

## 📊 Performance Tips

1. **For CPU-only systems**: Use Model 2 (YOLO11n) - smallest and fastest
2. **For GPU systems**: Use Model 3 (YOLO11m) - best accuracy
3. **Parallel inference**: All 4 models run independently, can start any/all
4. **Batch processing**: API supports image upload for non-webcam use

## 📝 Notes

- All models are in YOLO format (`.pt` files from Ultralytics)
- Models can run simultaneously without interference
- Each model has independent settings (can be enhanced in future)
- Frame capture runs in background threads for smooth streaming
- Inference results are cached per-model for optimal performance

## 📚 References

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- [Ultralytics YOLOv8/11](https://github.com/ultralytics/ultralytics)
- [VisDrone Dataset](http://aiskyeye.com/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [React Documentation](https://react.dev/)
