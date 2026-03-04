# Technology Stack

## Core Framework
- **YOLOv5**: Object detection framework (cloned in `yolov5/` directory)
- **PyTorch**: Deep learning backend for model inference
- **OpenCV**: Computer vision library for webcam handling and image processing

## Key Dependencies
- `torch>=1.8.0` + `torchvision>=0.9.0`: PyTorch ecosystem
- `opencv-python>=4.1.1`: Video capture and image processing
- `ultralytics>=8.2.64`: YOLO utilities and model handling
- `numpy>=1.23.5`: Numerical computations
- `matplotlib>=3.3`: Plotting and visualization
- `pillow>=10.3.0`: Image processing
- `PyYAML>=5.3.1`: Configuration file handling

## Development Environment
- **Python 3.x** required
- **Virtual environment**: `venv/` directory (recommended)
- **Windows compatibility**: Includes pathlib fixes for cross-platform model loading

## Common Commands

### Setup
```cmd
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Applications
```cmd
# Main obstacle detection
python webcam_detect.py

# Class identification tools
python simple_class_mapper.py
python test_and_identify_classes.py

# Check model classes
python check_model_classes.py
```

### Model Management
- Model weights: `best.pt` (custom trained, 22 classes)
- Default camera index: `1` (configurable in scripts)
- Inference size: `640x640` pixels
- Confidence threshold: `0.25`
- IoU threshold: `0.45`

## Architecture Patterns
- **Modular detection pipeline**: Preprocessing → Inference → Post-processing → Visualization
- **Real-time processing**: Frame-by-frame webcam analysis
- **Cross-platform compatibility**: Windows path handling for YOLOv5 integration
- **Configuration-driven**: Adjustable thresholds and camera settings