# 🚀 Quick Start Guide - Ensemble Models

## 30-Second Setup

### One-time Setup
```powershell
cd ensemble-models
.\setup.ps1        # Run PowerShell setup script
# OR
setup.bat          # Run batch setup script
```

## Running the System

### Step-by-Step

**1. Start Backend (Terminal 1):**
```powershell
cd ensemble-models\backend
.\venv\Scripts\Activate.ps1
python app.py
```

You should see:
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

**2. Start Frontend (Terminal 2):**
```powershell
cd ensemble-models\frontend
npm start
```

You should see:
```
> ensemble-models-frontend@1.0.0 start
> vite

Local:   http://localhost:3000
```

**3. Open Browser:**
Navigate to: `http://localhost:3000`

## What You'll See

A professional dashboard with:
- **📦 4 Model Boxes** - One for each YOLO model
- **📷 Live Webcam Feeds** - Independent stream for each model
- **🎯 Detection Results** - Real-time bounding boxes and labels
- **📊 Statistics** - FPS, detection count, class breakdown
- **🎮 Controls** - Start/Stop buttons for each model's webcam

## Testing the System

### Scenario 1: Compare All Models Side-by-Side
1. Click "Start Webcam" on all 4 models
2. Point camera at objects
3. See which models detect what
4. Compare FPS and detection accuracy

### Scenario 2: Test Single Model
1. Start just one model's webcam
2. Fine-tune confidence threshold in settings
3. Observe detection quality

### Scenario 3: Upload Image for Inference
Use the API with `curl` or Python:
```bash
curl -X POST http://localhost:5000/api/inference \
  -F "image=@path/to/image.jpg" \
  -F "model_id=model_1"
```

## Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| "Module not found" | Run `.\setup.ps1` again to ensure all deps installed |
| Webcam not starting | Close other apps using camera, try different camera_index |
| Port already in use | Change port in `backend/app.py` or `frontend/vite.config.js` |
| Slow inference | Reduce `imgsz` in frontend or select smaller model (Model 2) |
| CORS errors | Ensure backend (5000) and frontend (3000) both running |

## API Endpoints (for reference)

```
GET  http://localhost:5000/api/health
GET  http://localhost:5000/api/models
GET  http://localhost:5000/api/models/model_1
POST http://localhost:5000/api/webcam/model_1/start
POST http://localhost:5000/api/webcam/model_1/stop
GET  http://localhost:5000/api/webcam/model_1/frame
POST http://localhost:5000/api/inference
```

## Performance Notes

### Model Performance (Approximate)
| Model | Speed | Accuracy | Best For |
|-------|-------|----------|----------|
| Model 1 (YOLOv5) | Very Fast | Good | Real-time, low-spec hardware |
| Model 2 (YOLO11n) | Very Fast | Good | Mobile/edge devices |
| Model 3 (YOLO11m) | Fast | Best | Balance of speed & accuracy |
| Model 4 (Custom) | Fast | Good | Experimental/comparison |

### Tips for Better Performance
- Close Chrome DevTools (saves 10-15% GPU)
- Run only needed models (not all 4 at once)
- Use CPU with smaller models for good results
- Reduce resolution if needed

## Next Steps

After successful setup, explore:
1. **Threshold Tuning** - Adjust confidence to filter false positives
2. **Different Cameras** - Test with phone cameras, USB cameras
3. **Image Upload** - Test models on static images
4. **Model Comparison** - Document which performs best for your use case

## Need Help?

Check the main [README.md](README.md) for:
- Detailed architecture
- Full API documentation
- Model training information
- Troubleshooting guide
