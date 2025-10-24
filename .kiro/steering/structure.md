# Project Structure

## Root Directory Layout
```
├── best.pt                          # Custom trained YOLOv5 model (22 classes)
├── requirements.txt                 # Project dependencies
├── webcam_detect.py                 # Main obstacle detection application
├── simple_class_mapper.py           # Terminal-based class identification
├── test_and_identify_classes.py     # Interactive class mapping tool
├── check_model_classes.py           # Model class inspection utility
├── venv/                           # Python virtual environment
├── yolov5/                         # YOLOv5 framework (git submodule)
└── .kiro/                          # Kiro IDE configuration
    └── steering/                   # AI assistant guidance rules
```

## Documentation Files
- `CLASS_EXPLANATION.md`: Detailed explanation of YOLO class system
- `HOW_TO_IDENTIFY_CLASSES.md`: Guide for mapping custom classes
- `IMPLEMENTATION_SUMMARY.md`: Feature implementation details

## YOLOv5 Submodule Structure
```
yolov5/
├── detect.py                       # YOLOv5 detection script
├── train.py                        # Model training script
├── models/                         # Model architectures
├── utils/                          # Utility functions
├── data/                          # Dataset configurations
└── requirements.txt               # YOLOv5 dependencies
```

## Code Organization Patterns

### Main Application (`webcam_detect.py`)
- **Configuration section**: Model paths, thresholds, camera settings
- **Utility functions**: Box drawing, coordinate scaling
- **Main loop**: Webcam capture → Inference → Visualization
- **Real-time counting**: Obstacle statistics with visual feedback

### Class Mapping Tools
- **Simple mapper**: Terminal-based, minimal interaction
- **Interactive mapper**: GUI-based with visual feedback
- **Output format**: JSON + human-readable text files

### Import Structure
```python
# Windows compatibility fix
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# YOLOv5 imports
sys.path.insert(0, "yolov5")
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
```

## File Naming Conventions
- **Main scripts**: Descriptive names (`webcam_detect.py`, `simple_class_mapper.py`)
- **Model files**: `best.pt` for custom weights
- **Output files**: `class_mapping.json`, `class_mapping.txt`
- **Documentation**: ALL_CAPS with underscores (`.md` extension)

## Configuration Management
- **Hardcoded constants**: At top of each script for easy modification
- **Camera index**: Configurable per script (default: `1`)
- **Model parameters**: Confidence/IoU thresholds, image size
- **Paths**: Relative to project root, cross-platform compatible