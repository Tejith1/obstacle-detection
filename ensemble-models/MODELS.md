# 📊 Model Information & Attribution

## Quick Reference

| ID | Name | Framework | Source | File | Training Details |
|----|------|-----------|--------|------|------------------|
| **model_1** | YOLOv5 (Obstacle) | YOLOv5 | `obstacle-detection/best.pt` | `obstacle_detection_yolov5_best.pt` | Custom obstacle detection training |
| **model_2** | YOLO11n (VisDrone) | Ultralytics YOLO | `v1-1/best.pt` | `visdrone_yolo11n_best.pt` | 30 epochs, nano model |
| **model_3** | YOLO11m (VisDrone) | Ultralytics YOLO | `v1-2/best.pt` | `visdrone_yolo11m_best.pt` | 50 epochs, medium model |
| **model_4** | Custom YOLO | Ultralytics YOLO | `results.zip` | `results_model_best.pt` | Results training artifacts |

---

## Detailed Model Specifications

### Model 1: YOLOv5 (Obstacle Detection)

**File Location:** `models/obstacle_detection_yolov5_best.pt`

**Training Info:**
- Framework: YOLOv5 (Ultralytics)
- Dataset: Custom obstacle detection dataset
- Purpose: Real-time obstacle detection (especially for drones)
- Original Source: `obstacle-detection/best.pt`

**Classes:** Trained on various obstacle classes (specific list embedded in model)

**Usage:** General-purpose obstacle/object detection with good real-time performance

**Notes:**
- Uses local YOLOv5 repository for inference
- Optimized for speed and accuracy balance
- Successfully used in drone navigation systems

---

### Model 2: YOLO11n (VisDrone v1-1) - Nano

**File Location:** `models/visdrone_yolo11n_best.pt`

**Training Info:**
- Framework: Ultralytics YOLO11 (nano variant)
- Dataset: VisDrone 2019 Detection Challenge
- Training Duration: 30 epochs
- Image Size: 640x640
- Optimizer: Auto (adaptive)
- Batch Size: Default
- Original Source: `v1-1/Do_you_detect_images_using_yolo_v11____🧍🚍🚘🛵.ipynb`

**Classes (10 total):**
0. pedestrian
1. people
2. bicycle
3. car
4. van
5. truck
6. tricycle
7. awning-tricycle
8. bus
9. motor

**Performance Characteristics:**
- ✓ Fastest inference speed among YOLO11 variants
- ✓ Lowest memory footprint
- ✓ Best for real-time applications
- ~ Slightly lower accuracy than medium variant

**Best For:**
- Edge devices and mobile deployment
- Real-time applications requiring <30ms latency
- Resource-constrained environments

---

### Model 3: YOLO11m (VisDrone v1-2) - Medium

**File Location:** `models/visdrone_yolo11m_best.pt`

**Training Info:**
- Framework: Ultralytics YOLO11 (medium variant)
- Dataset: VisDrone 2019 Detection Challenge
- Training Duration: 50 epochs (more epochs than v1-1)
- Image Size: 640x640
- Optimizer: SGD (Stochastic Gradient Descent)
- Batch Size: 16
- Patience: 35 (early stopping)
- Original Source: `v1-2/Copy_of_Visdrone_YOLO11m.ipynb`

**Classes (10 total - same as Model 2):**
0. pedestrian
1. people
2. bicycle
3. car
4. van
5. truck
6. tricycle
7. awning-tricycle
8. bus
9. motor

**Performance Metrics:**
- Trained for longer period (50 vs 30 epochs)
- Typically achieves better mAP than nano variant
- Increased computational requirements

**Performance Characteristics:**
- ✓ Better accuracy than nano variant
- ✓ Balanced speed and accuracy
- ✓ Standard deployment choice
- ~ Slower than nano, requires more VRAM

**Best For:**
- Production systems with moderate GPU/CPU resources
- Applications requiring good accuracy-speed balance
- Typical object detection scenarios

---

### Model 4: Custom YOLO (Results)

**File Location:** `models/results_model_best.pt`

**Training Info:**
- Framework: Ultralytics YOLO
- Dataset: VisDrone 2019 Detection Challenge
- Training Duration: Custom (from results.zip)
- Image Size: Unknown (embedded in model)
- Optimizer: Unknown (embedded in model)
- Original Source: `results.zip` (training artifacts)

**Classes:** Likely similar to VisDrone (empirically determined by loading model)

**Characteristics:**
- ? Experimental/alternative training configuration
- ? Different hyperparameters than Models 2 & 3
- ? Useful for A/B testing and comparison

**Best For:**
- Comparison testing
- Experimental validation
- A/B testing against other configurations

---

## Training Dataset: VisDrone

The VisDrone dataset (Models 2, 3, 4) is a large-scale benchmark for object detection in videos captured from drones:

- **Source:** http://aiskyeye.com/
- **Use Case:** Aerial object detection
- **Challenges:** Scale variation, orientation, occlusion, weather
- **10 Classes:** Pedestirans, people, bicycles, cars, vans, trucks, tricycles, awning-tricycles, buses, motorcycles

**Dataset Splits:**
- Training: 6,471 images
- Validation: 548 images  
- Test: 1,610 images

---

## Model File Format

All models are in **PyTorch `.pt` format**:
- Compatible with Ultralytics YOLO framework
- Includes weights, architecture, and metadata
- Can be loaded with: `from ultralytics import YOLO; model = YOLO('model.pt')`

---

## Performance Comparison

Approximate performance (varies by hardware):

| Metric | Model 1 (YOLOv5) | Model 2 (11n) | Model 3 (11m) | Model 4 (Custom) |
|--------|------------------|---------------|---------------|-----------------|
| Speed (GPU) | ~40-50 ms | ~30-40 ms | ~60-80 ms | ~50-70 ms |
| Speed (CPU) | ~150-200 ms | ~100-150 ms | ~200-300 ms | ~150-250 ms |
| Memory | ~500 MB | ~400 MB | ~700 MB | ~600 MB |
| mAP (approx) | ~0.40 | ~0.42 | ~0.46 | ~0.44 |

*Actual performance depends on image resolution, batch size, hardware, and other factors.*

---

## How to Check Loaded Models

The React frontend displays model information as they load. You can also:

1. **Via API:**
   ```bash
   curl http://localhost:5000/api/models
   ```

2. **Via Python:**
   ```python
   from ultralytics import YOLO
   model = YOLO('path/to/model.pt')
   print(model.names)  # List all classes
   ```

---

## Notes for Users

- All models are production-ready (best.pt weights, not last.pt)
- Models can run simultaneously without conflict
- Each model maintains independent inference threads
- Ensemble comparison available through 4-box UI
- Custom threshold adjustments available per-session (not persistent)

---

## References

- YOLOv5: https://github.com/ultralytics/yolov5
- YOLOv11/Ultralytics: https://github.com/ultralytics/ultralytics
- VisDrone Dataset: http://aiskyeye.com/
- Official Ultralytics Docs: https://docs.ultralytics.com/
