# 🛠️ Installation Guide

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.14+, or Ubuntu 18.04+
- **Python**: 3.7 or higher (3.8+ recommended)
- **RAM**: 4GB minimum (8GB+ recommended)
- **Storage**: 5GB free space
- **Camera**: USB webcam or built-in camera

### Recommended Requirements
- **CPU**: Quad-core processor or better
- **GPU**: NVIDIA GTX 1060+ (optional, for faster inference)
- **RAM**: 8GB or more
- **Python**: 3.8 or 3.9

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/drone-obstacle-detection.git
cd drone-obstacle-detection
```

### 2. Set Up Python Virtual Environment

#### Windows
```cmd
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt
```

### 4. Install Additional Dependencies for Web Interface

```bash
pip install flask
```

### 5. Verify Installation

```bash
# Check if YOLOv5 is properly installed
python check_model_classes.py
```

If you see model class information, the installation is successful!

## Model Setup

### Option 1: Use Pre-trained Model
If you have a `best.pt` file:
1. Place it in the project root directory
2. Run `python check_model_classes.py` to verify

### Option 2: Train Your Own Model
1. Prepare your dataset in YOLO format
2. Use YOLOv5 training scripts in the `yolov5/` directory
3. Follow YOLOv5 documentation for training

## Troubleshooting

### Common Issues

#### ❌ "No module named 'torch'"
```bash
# Install PyTorch manually
pip install torch torchvision
```

#### ❌ "Could not open webcam"
- Check camera permissions
- Try different camera indices (0, 1, 2) in the scripts
- Ensure no other applications are using the camera

#### ❌ "YOLOv5 import errors"
```bash
# Ensure you're in the project root directory
cd drone-obstacle-detection
python webcam_detect.py
```

#### ❌ "best.pt not found"
- Ensure the model file is in the project root
- Check file permissions
- Verify the file isn't corrupted

### GPU Setup (Optional)

#### NVIDIA GPU Support
```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Performance Optimization

#### For Better Speed
```python
# In webcam_detect.py, adjust these settings:
IMG_SIZE = 416          # Smaller size = faster
CONF_THRESH = 0.4       # Higher threshold = fewer detections
```

#### For Better Accuracy
```python
# In webcam_detect.py, adjust these settings:
IMG_SIZE = 832          # Larger size = more accurate
CONF_THRESH = 0.15      # Lower threshold = more detections
```

## Verification Steps

### 1. Test Main Detection System
```bash
python webcam_detect.py
```
Expected: Video window with obstacle detection and navigation overlays

### 2. Test Web Interface
```bash
python web_interface.py
```
Then open: `http://localhost:5000`
Expected: Professional web dashboard with live video feed

### 3. Test Attendance System
```bash
python attendance_system.py
```
Expected: Face detection with attendance counting

### 4. Test Class Mapping Tools
```bash
python simple_class_mapper.py
```
Expected: Interactive class identification interface

## Development Setup

### For Contributors

```bash
# Install development dependencies
pip install -r requirements.txt
pip install flask pytest black flake8

# Run code formatting
black *.py

# Run linting
flake8 *.py
```

### IDE Setup
- **VS Code**: Install Python extension
- **PyCharm**: Configure Python interpreter to use virtual environment
- **Kiro IDE**: Project includes `.kiro/` configuration

## Docker Setup (Optional)

### Build Docker Image
```bash
docker build -t drone-detection .
```

### Run Container
```bash
docker run -p 5000:5000 --device=/dev/video0 drone-detection
```

## Next Steps

After successful installation:
1. Read the [Usage Guide](USAGE.md)
2. Check the [API Documentation](API.md)
3. Explore the [Project Structure](PROJECT_STRUCTURE.md)
4. Review the main [README.md](../README.md)

## Support

If you encounter issues:
1. Check this troubleshooting section
2. Review the [GitHub Issues](https://github.com/yourusername/drone-obstacle-detection/issues)
3. Create a new issue with detailed error information