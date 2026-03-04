# 🎯 Getting Started - Step by Step

## What You Have

✅ **4 Production-Ready YOLO Models**
✅ **Professional React Frontend**
✅ **Flask REST API Backend**
✅ **Complete Documentation**
✅ **Everything in One Folder**

All models are copied into `ensemble-models/models/` with clear, descriptive names:
- `obstacle_detection_yolov5_best.pt` ← From obstacle-detection folder
- `visdrone_yolo11n_best.pt` ← From v1-1 (30 epochs, nano)
- `visdrone_yolo11m_best.pt` ← From v1-2 (50 epochs, medium)
- `results_model_best.pt` ← From results.zip training

---

## Step 1: One-Time Setup (5 minutes)

### Option A: PowerShell (Windows 10+)
```powershell
cd ensemble-models
.\setup.ps1
```

### Option B: Command Prompt (Batch)
```cmd
cd ensemble-models
setup.bat
```

### Option C: Manual Setup
```powershell
# Backend setup
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt

# Frontend setup
cd ..\frontend
npm install
```

---

## Step 2: Start the System

### Terminal 1 - Backend Server
```powershell
cd ensemble-models\backend
.\venv\Scripts\Activate.ps1
python app.py
```

**Expected Output:**
```
============================================================
  ENSEMBLE MODELS API SERVER
============================================================

Loading models...
Models loaded: 4/4
✓ Loaded model_1: YOLOv5 (Obstacle Detection)
✓ Loaded model_2: YOLO11n (VisDrone v1-1)
✓ Loaded model_3: YOLO11m (VisDrone v1-2)
✓ Loaded model_4: Results Model (YOLO)

Starting Flask server on http://localhost:5000
============================================================
```

### Terminal 2 - Frontend Server
```powershell
cd ensemble-models\frontend
npm start
```

**Expected Output:**
```
> ensemble-models-frontend@1.0.0 start
> vite

Local:   http://localhost:3000
```

---

## Step 3: Open Your Browser

1. Open: **http://localhost:3000**
2. Wait for page to load (will show 4 model boxes)
3. Click **"Start Webcam"** on any model box
4. Point camera at objects
5. Watch detections appear in real-time!

---

## What You'll See

### Dashboard Layout

```
┌─────────────────────────────────────────────────────┐
│  🤖 Ensemble Models Detection Dashboard              │
│  Real-time Multi-Model Object Detection System       │
└─────────────────────────────────────────────────────┘
┌─────────────┬──────────────┬──────────────┬────────┐
│   Model 1   │   Model 2    │   Model 3    │ Model 4│
│ (YOLOv5)    │ (YOLO11n)    │ (YOLO11m)    │(Custom)│
│  📷         │  📷          │  📷          │ 📷     │
│  [Start]    │  [Start]     │  [Start]     │[Start] │
│  🎯 0 dets  │  🎯 5 dets   │  🎯 7 dets   │🎯 4det │
└─────────────┴──────────────┴──────────────┴────────┘
```

Each model box has:
- **Live Webcam Feed** - Real-time video
- **Bounding Boxes** - Green boxes around detected objects
- **Labels** - Class name + confidence score
- **FPS Counter** - Frames per second
- **Detection Stats** - Count and breakdown by class
- **Control Buttons** - Start/Stop webcam
- **Class List** - Shows what this model can detect

---

## Common Tasks

### Compare All Models
```
1. Click "Start Webcam" on all 4 boxes
2. Point camera at objects
3. Compare which models detect what
4. Observe FPS and accuracy differences
```

### Focus on One Model
```
1. Click "Start Webcam" on Model 3 (Most Balanced)
2. Adjust confidence slider if needed
3. Test different objects/distances
```

### Stop Everything
```
1. Click "Stop Webcam" on each running model
2. Or just close the browser tab
```

### Check Backend API
```
curl http://localhost:5000/api/models
curl http://localhost:5000/api/health
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **"Python not found"** | Install Python 3.8+ from python.org |
| **"Node not found"** | Install Node.js 16+ from nodejs.org |
| **Port 5000 already in use** | Change port in `backend/app.py` (line ~450) |
| **Port 3000 already in use** | Change port in `frontend/vite.config.js` (line 6) |
| **Webcam not showing** | Close other apps using camera, try different camera_index |
| **Slow inference** | Use Model 2 (YOLO11n), reduce imgsz in settings |
| **Models not loaded** | Check console output, verify `.pt` files exist in models/ |
| **CORS errors** | Ensure both backend (5000) and frontend (3000) running |
| **"Module not found"** | Run setup script again: `.\setup.ps1` |

---

## Documentation Files

- **README.md** - Full project documentation
- **QUICKSTART.md** - 30-second quick start guide
- **MODELS.md** - Detailed info about each model
- **COMPLETE.md** - System overview and features

---

## Model Performance Guide

### Best for Speed (Real-time)
→ Use **Model 2 (YOLO11n)**
- Nano variant, fastest inference
- Good accuracy for real-time

### Best for Accuracy
→ Use **Model 3 (YOLO11m)**
- Medium variant, trained for 50 epochs
- Best accuracy-speed balance
- Recommended for most uses

### For Comparison Testing
→ Use **Models 1 & 4**
- Model 1: Obstacle-specific training
- Model 4: Alternative training experiment

### For Learning
→ Start with **All 4 Together**
- See how different models compare
- Understand pros/cons of each
- Learn what affects detection

---

## System Architecture (Overview)

```
Browser (Port 3000)
    ↓
    ↓ HTTP Requests
    ↓
Flask API (Port 5000)
    ↓
    ├─→ Webcam 1 → Inference → Model 1
    ├─→ Webcam 2 → Inference → Model 2
    ├─→ Webcam 3 → Inference → Model 3
    └─→ Webcam 4 → Inference → Model 4
    ↓
    ↓ JSON Response (with Base64 Frame)
    ↓
React Frontend Displays Results
```

---

## File Organization

Everything is **self-contained in ensemble-models/**:

```
ensemble-models/                    ← You are here
├── models/                         ← 4 trained models
├── backend/                        ← Flask API server
├── frontend/                       ← React app
├── README.md                       ← Full docs
├── QUICKSTART.md                   ← Quick start
├── MODELS.md                       ← Model details
├── COMPLETE.md                     ← System overview
└── setup.ps1 / setup.bat          ← Setup helpers
```

**No files outside this folder were modified.**

---

## Next Steps After First Run

1. ✅ Verify all models load successfully
2. ✅ Test each model independently
3. ✅ Run all models together
4. ✅ Adjust confidence thresholds
5. ✅ Test with different camera angles
6. ✅ Document which model performs best
7. ✅ Explore the React code
8. ✅ Check the Flask API endpoints

---

## Need Help?

1. **Check README.md** for complete documentation
2. **Check QUICKSTART.md** for common issues
3. **Check MODELS.md** for model-specific info
4. **Look at backend console** for error messages
5. **Check browser DevTools** for frontend errors

---

## You're All Set! 🚀

Just run the 3 commands:
```powershell
cd ensemble-models\backend
.\venv\Scripts\Activate.ps1
python app.py
```

Then in another terminal:
```powershell
cd ensemble-models\frontend
npm start
```

Then open: **http://localhost:3000**

### Happy detecting! 🎉
