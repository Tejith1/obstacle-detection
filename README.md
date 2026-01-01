# 🚁 Drone Obstacle Detection & Navigation System

A real-time obstacle detection and navigation system for drones using YOLOv5 deep learning model. This system processes live camera feeds to detect obstacles and provides intelligent navigation recommendations for autonomous drone flight.

## 🌟 Features

- **Real-time Obstacle Detection**: Uses YOLOv5 for fast and accurate obstacle detection
- **Intelligent Navigation**: Analyzes obstacle positions and suggests optimal flight directions
- **Web Dashboard**: Modern, responsive web interface for monitoring and control
- **Performance Optimized**: Multiple settings for balancing speed and accuracy
- **Zone-based Analysis**: Divides the view into zones for strategic navigation decisions
- **Live Statistics**: Real-time FPS, detection counts, and navigation status

## 🏗️ System Architecture

```
┌─────────────────┐
│  Camera Feed    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  YOLOv5 Model   │ ← Detection
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Drone Navigator │ ← Analysis
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Web Interface  │ ← Visualization
└─────────────────┘
```

## 📋 Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for better performance)
- Webcam or camera device
- Pre-trained YOLOv5 model (`best.pt`)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify model file**
   - Ensure `best.pt` (trained YOLOv5 model) is in the project root directory

## 💻 Usage

### Starting the Web Interface

```bash
python web_interface.py
```

Then open your browser and navigate to:
- Local access: `http://localhost:5000`
- Network access: `http://0.0.0.0:5000`

### Command Line Detection

For basic webcam detection without the web interface:

```bash
python webcam_detect.py
```

### Drone Navigation Module

The navigation system can also be used standalone:

```python
from drone_navigation import DroneNavigator, get_navigation_command

navigator = DroneNavigator(frame_width=640, frame_height=480)
nav_result = navigator.analyze_obstacles(detections)
command = get_navigation_command(nav_result)
```

## 🎮 Web Interface Controls

### Detection Statistics
- **Total Obstacles**: Real-time count of detected objects
- **FPS**: Frames per second performance indicator
- **Class Breakdown**: Distribution of detected object types

### Navigation Status
- **CLEAR_PATH**: Safe to proceed forward
- **CAUTION**: Obstacles present, alternative route suggested
- **DANGER_STOP**: Critical obstacles, immediate stop required

### Performance Settings
- **Confidence Threshold** (0.1-0.9): Minimum detection confidence
- **IoU Threshold** (0.1-0.9): Intersection over Union for NMS
- **Frame Skip** (1-3): Process every Nth frame for better FPS
- **JPEG Quality** (50-95): Video stream compression quality

### Quick Actions
- **Max FPS**: Optimize for maximum performance
- **Max Quality**: Optimize for best detection accuracy
- **Balanced**: Balanced performance and quality
- **Reset**: Restore default settings

## 📊 Navigation System

The navigation system divides the camera view into 5 zones:

```
┌─────────┬─────────┬─────────┐
│         │   TOP   │         │
│  LEFT   ├─────────┤  RIGHT  │
│         │ CENTER  │         │
├─────────┼─────────┼─────────┤
│         │ BOTTOM  │         │
└─────────┴─────────┴─────────┘
```

### Navigation Logic
1. **Obstacle Detection**: Detects all objects in frame
2. **Zone Assignment**: Maps each obstacle to one or more zones
3. **Danger Scoring**: Calculates danger level per zone based on:
   - Number of obstacles
   - Size of obstacles
   - Proximity to center
4. **Direction Recommendation**: Suggests safest navigation direction:
   - FORWARD (if center clear)
   - LEFT/RIGHT (if center blocked)
   - UP/DOWN (if sides blocked)
   - STOP (if all paths dangerous)

## 📁 Project Structure

```
detection/
├── web_interface.py          # Flask web server and main interface
├── drone_navigation.py       # Navigation logic and zone analysis
├── webcam_detect.py          # Standalone webcam detection
├── best.pt                   # YOLOv5 trained model
├── requirements.txt          # Python dependencies
├── templates/
│   └── dashboard.html        # Web dashboard UI
├── yolov5/                   # YOLOv5 framework
└── docs/                     # Additional documentation
    ├── API.md
    ├── INSTALLATION.md
    ├── PROJECT_STRUCTURE.md
    └── USAGE.md
```

## 🔧 Configuration

### Model Settings
Edit the configuration in `web_interface.py`:

```python
WEIGHTS = "best.pt"           # Model file
IMG_SIZE = 416                # Input image size (lower = faster)
CONF_THRESH = 0.25           # Confidence threshold
IOU_THRESH = 0.45            # NMS IoU threshold
```

### Camera Settings
```python
CAMERA_INDEX = 0             # Camera device index
FRAME_WIDTH = 640            # Camera resolution width
FRAME_HEIGHT = 480           # Camera resolution height
```

## 🎯 Performance Optimization Tips

1. **For Maximum FPS** (30+ FPS):
   - Set confidence threshold to 0.35+
   - Enable frame skipping (2-3 frames)
   - Reduce JPEG quality to 60-70
   - Use lower resolution (416x416)

2. **For Maximum Accuracy**:
   - Set confidence threshold to 0.20
   - Disable frame skipping
   - Increase JPEG quality to 90+
   - Close other camera applications

3. **Balanced Performance** (20-30 FPS):
   - Confidence threshold: 0.25
   - Frame skip: 1
   - JPEG quality: 80
   - Resolution: 640x480

## 🐛 Troubleshooting

### Camera Not Found
```bash
# List available cameras
python -c "import cv2; print([i for i in range(5) if cv2.VideoCapture(i).isOpened()])"
```

### Low FPS
- Close other applications using the camera
- Reduce detection resolution
- Enable frame skipping
- Lower JPEG quality
- Ensure GPU is being used (check CUDA availability)

### Model Loading Error
- Verify `best.pt` exists in the project directory
- Check PyTorch and CUDA versions are compatible
- Try re-downloading the model file

## 📈 Future Enhancements

- [ ] Multi-drone coordination
- [ ] 3D obstacle mapping
- [ ] Path planning algorithms
- [ ] Autonomous flight integration
- [ ] Cloud-based processing
- [ ] Mobile app support
- [ ] Recording and replay functionality

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **YOLOv5** by Ultralytics for the object detection framework
- **Flask** for the web framework
- **OpenCV** for computer vision capabilities
- **PyTorch** for deep learning infrastructure

## 📞 Support

For issues, questions, or contributions, please:
- Open an issue on GitHub
- Check existing documentation in the `docs/` folder
- Review troubleshooting section above

---

**Built with ❤️ for autonomous drone navigation**
