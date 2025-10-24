# 🎮 Usage Guide

## Quick Start

### 1. Basic Obstacle Detection
```bash
python webcam_detect.py
```
- **Purpose**: Real-time obstacle detection with navigation guidance
- **Controls**: Press 'Q' to quit
- **Output**: Live video with bounding boxes, navigation arrows, and statistics

### 2. Professional Web Interface
```bash
python web_interface.py
```
Then open: `http://localhost:5000`
- **Purpose**: Professional dashboard with web controls
- **Features**: Live streaming, statistics, settings panel
- **Access**: Any web browser on your network

### 3. Attendance System
```bash
python attendance_system.py
```
- **Purpose**: Face detection for student attendance
- **Controls**: Press 'S' to save report, 'Q' to quit
- **Output**: Real-time face counting with session reports

## Detailed Application Guide

### 🎯 Main Detection System (`webcam_detect.py`)

#### Features
- **Real-time Detection**: Identifies obstacles using 22-class model
- **Navigation Guidance**: Shows directional arrows (LEFT/RIGHT/UP/DOWN/STOP)
- **Zone Analysis**: Divides view into 5 zones for spatial awareness
- **Visual Feedback**: Color-coded status and confidence indicators

#### On-Screen Display
```
┌─────────────────────────────────────────────┐
│ STATUS: REDIRECT SAFE                       │ ← Navigation status
│ CONFIDENCE: 85%                             │ ← Safety confidence
│                                             │
│     [Navigation Arrow: →]                   │ ← Direction guidance
│     [Green box: car 0.89]                   │ ← Object detections
│     [Green box: person 0.76]                │
│                                             │
│ Proceed right with caution                  │ ← Recommendation
│ FPS: 24.5                                   │ ← Performance
└─────────────────────────────────────────────┘
```

#### Navigation Status Types
- **CLEAR_PATH** (Green): Safe to proceed forward
- **REDIRECT_SAFE** (Yellow): Obstacle in center, safe alternative available
- **REDIRECT_CAUTION** (Orange): Multiple obstacles, proceed carefully
- **DANGER_STOP** (Red): Obstacles in all directions, stop immediately

#### Configuration
Edit these values in `webcam_detect.py`:
```python
CONF_THRESH = 0.25      # Detection confidence (0.1-0.9)
IOU_THRESH = 0.45       # Overlap threshold (0.1-0.9)
IMG_SIZE = 640          # Input size (416/640/832)
CAMERA_INDEX = 1        # Camera selection (0/1/2)
```

### 🌐 Web Interface (`web_interface.py`)

#### Dashboard Features
- **Live Video Stream**: Real-time detection feed
- **Statistics Panel**: Obstacle counts, FPS, navigation status
- **Settings Control**: Adjust parameters without restarting
- **Mode Switching**: Toggle between obstacle detection and attendance

#### API Endpoints
- `GET /`: Main dashboard page
- `GET /video_feed`: Live video stream
- `GET /api/stats`: Current detection statistics
- `GET/POST /api/settings`: System configuration
- `GET /api/navigation_command`: Drone movement commands

#### Usage Example
```javascript
// Get current statistics
fetch('/api/stats')
  .then(response => response.json())
  .then(data => {
    console.log('Obstacles:', data.total_obstacles);
    console.log('Navigation:', data.navigation.direction);
  });

// Update settings
fetch('/api/settings', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    conf_threshold: 0.3,
    detection_mode: 'obstacles'
  })
});
```

### 👥 Attendance System (`attendance_system.py`)

#### Features
- **Face Detection**: Uses class 9 for face recognition
- **Stability Filtering**: Averages counts over multiple frames
- **Session Tracking**: Records attendance changes over time
- **Report Generation**: Saves detailed JSON reports

#### Display Information
```
┌─────────────────────────────────────────────┐
│ ATTENDANCE SYSTEM                           │
│ PRESENT: 15                                 │ ← Current count
│ MAX TODAY: 18                               │ ← Peak attendance
│ SESSION: 01:23:45                           │ ← Duration
│ DETECTIONS: 15                              │ ← Raw detections
│                                             │
│     [Red box: Face 0.92]                    │ ← Face detections
│     [Red box: Face 0.87]                    │
│                                             │
│ Press 'S' to save | 'Q' to quit            │
└─────────────────────────────────────────────┘
```

#### Report Format
```json
{
  "session_info": {
    "start_time": "2024-01-15 09:00:00",
    "end_time": "2024-01-15 10:30:00",
    "duration": "1:30:00"
  },
  "statistics": {
    "max_attendance": 18,
    "final_count": 15,
    "total_changes": 12
  },
  "detailed_log": [
    {"timestamp": "09:05:23", "count": 5, "raw_detections": 5},
    {"timestamp": "09:12:45", "count": 12, "raw_detections": 11}
  ]
}
```

## Utility Tools

### 🔧 Model Class Inspector (`check_model_classes.py`)

```bash
python check_model_classes.py
```

**Output Example:**
```
Class ID | Class Name
--------------------------
   0     | person
   1     | bicycle
   2     | car
   9     | face
  ...
```

### 🔧 Class Mapping Tools

#### Simple Terminal Mapper
```bash
python simple_class_mapper.py
```
- Point camera at known objects
- Press 'L' when detection appears
- Type object name in terminal
- Builds mapping automatically

#### Interactive Visual Mapper
```bash
python test_and_identify_classes.py
```
- GUI-based identification
- Visual feedback with progress tracking
- Real-time mapping updates

## Performance Optimization

### Speed Optimization
```python
# For faster performance (lower accuracy)
IMG_SIZE = 416
CONF_THRESH = 0.4
IOU_THRESH = 0.5
```

### Accuracy Optimization
```python
# For better accuracy (slower performance)
IMG_SIZE = 832
CONF_THRESH = 0.15
IOU_THRESH = 0.3
```

### Hardware-Specific Settings

#### CPU-Only Systems
```python
DEVICE = 'cpu'
IMG_SIZE = 416          # Smaller for CPU
```

#### GPU Systems
```python
DEVICE = '0'            # Use first GPU
IMG_SIZE = 640          # Standard size
```

#### Multiple GPUs
```python
DEVICE = '0,1'          # Use multiple GPUs
```

## Integration Examples

### Drone Command Integration
```python
from drone_navigation import get_navigation_command

# Get navigation recommendation
nav_result = navigator.analyze_obstacles(detections)
command = get_navigation_command(nav_result)

# Example output:
# {
#   'movement': {'x': -0.3, 'y': 0, 'z': 0},
#   'action': 'LEFT',
#   'confidence': 0.85,
#   'should_stop': False
# }
```

### CCTV Integration
```python
# Replace webcam with IP camera
cap = cv2.VideoCapture('rtsp://camera_ip:554/stream')

# Or use video file
cap = cv2.VideoCapture('path/to/video.mp4')
```

### Custom Model Integration
```python
# Use different model
WEIGHTS = "path/to/your_model.pt"

# Check classes
python check_model_classes.py
```

## Troubleshooting

### Common Issues

#### Low Detection Accuracy
1. **Lower confidence threshold**: `CONF_THRESH = 0.15`
2. **Increase image size**: `IMG_SIZE = 832`
3. **Check lighting conditions**
4. **Verify camera focus**

#### Poor Performance
1. **Reduce image size**: `IMG_SIZE = 416`
2. **Increase confidence threshold**: `CONF_THRESH = 0.4`
3. **Close other applications**
4. **Use GPU if available**

#### Navigation Issues
1. **Check zone boundaries** in navigation overlay
2. **Verify obstacle detection** is working
3. **Adjust confidence thresholds**
4. **Test with clear obstacles**

### Debug Mode
Add debug prints to any script:
```python
print(f"Detections: {len(detections)}")
print(f"Navigation: {nav_result}")
print(f"FPS: {fps}")
```

## Best Practices

### Camera Setup
- **Good lighting**: Avoid backlighting and shadows
- **Stable mounting**: Minimize camera shake
- **Clear lens**: Keep camera clean
- **Appropriate height**: Match intended use case

### Model Usage
- **Consistent environment**: Train and test in similar conditions
- **Regular calibration**: Update thresholds as needed
- **Performance monitoring**: Track FPS and accuracy
- **Backup configurations**: Save working parameter sets

### Development
- **Version control**: Use git for code changes
- **Testing**: Verify changes with multiple scenarios
- **Documentation**: Update guides when modifying features
- **Performance profiling**: Monitor resource usage