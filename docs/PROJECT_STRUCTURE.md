# 📁 Project Structure

## Root Directory
```
drone-obstacle-detection/
├── 📄 README.md                    # Main project documentation
├── 📄 requirements.txt             # Python dependencies
├── 📄 .gitignore                   # Git ignore rules
├── 🎯 webcam_detect.py             # Main obstacle detection with navigation
├── 🌐 web_interface.py             # Professional web dashboard
├── 🧭 drone_navigation.py          # Navigation logic and zone analysis
├── 🔧 check_model_classes.py       # Model inspection utility
├── 🔧 simple_class_mapper.py       # Terminal-based class identification
├── 🔧 test_and_identify_classes.py # Interactive class mapping tool
├── 📁 docs/                        # Documentation files
├── 📁 templates/                   # Web interface templates
├── 📁 yolov5/                      # YOLOv5 framework (submodule)
└── 📁 .kiro/                       # IDE configuration
```

## Core Applications

### 🎯 Main Detection System
- **`webcam_detect.py`**: Primary application with real-time obstacle detection and navigation guidance
- **`drone_navigation.py`**: Navigation logic, zone analysis, and directional recommendations

### 🌐 Web Interface
- **`web_interface.py`**: Flask-based web server for professional dashboard
- **`templates/dashboard.html`**: Modern web interface with real-time controls

## Utility Tools

### 🔧 Model Management
- **`check_model_classes.py`**: Inspect model classes and capabilities
- **`simple_class_mapper.py`**: Terminal-based class identification
- **`test_and_identify_classes.py`**: Interactive GUI-based class mapping

## Documentation

### 📚 User Guides
- **`README.md`**: Complete project overview and usage instructions
- **`docs/INSTALLATION.md`**: Detailed installation guide
- **`docs/USAGE.md`**: Application usage examples
- **`docs/API.md`**: Web interface API documentation

### 📚 Technical Documentation
- **`CLASS_EXPLANATION.md`**: YOLO class system explanation
- **`HOW_TO_IDENTIFY_CLASSES.md`**: Class mapping tutorial
- **`IMPLEMENTATION_SUMMARY.md`**: Technical implementation details
- **`model_improvement_guide.md`**: Performance optimization guide

## Dependencies

### 📦 Core Framework
- **YOLOv5**: Object detection framework (in `yolov5/` directory)
- **PyTorch**: Deep learning backend
- **OpenCV**: Computer vision and webcam handling

### 📦 Web Interface
- **Flask**: Web framework for dashboard
- **HTML/CSS/JavaScript**: Frontend technologies

### 📦 Python Libraries
See `requirements.txt` for complete dependency list.

## File Naming Conventions

### 🎯 Applications
- **Main scripts**: Descriptive names (`webcam_detect.py`, `drone_navigation.py`)
- **Utility scripts**: Purpose-based names (`check_model_classes.py`)

### 📚 Documentation
- **Markdown files**: ALL_CAPS with underscores (`.md` extension)
- **User guides**: Descriptive names (`INSTALLATION.md`, `USAGE.md`)

### 📁 Directories
- **`docs/`**: All documentation files
- **`templates/`**: Web interface templates
- **`yolov5/`**: YOLOv5 framework (git submodule)

## Configuration Files

### ⚙️ Project Configuration
- **`requirements.txt`**: Python package dependencies
- **`.gitignore`**: Git version control exclusions
- **`.kiro/steering/`**: AI assistant guidance rules

### ⚙️ Model Files
- **`best.pt`**: Custom trained YOLOv5 model (22 classes)
- **Model weights**: Not included in repository (add your own)

## Output Files

### 📊 Generated Reports
- **`class_mapping.json`**: Machine-readable class mappings
- **`class_mapping.txt`**: Human-readable class reference
- **Detection logs**: Runtime detection statistics

### 📊 Temporary Files
- **Performance metrics**: FPS and accuracy measurements