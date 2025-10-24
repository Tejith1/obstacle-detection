# 🌐 Web Interface API Documentation

## Overview

The web interface (`web_interface.py`) provides a RESTful API for controlling and monitoring the drone obstacle detection system. The API enables real-time parameter adjustment, statistics retrieval, and system control.

## Base URL
```
http://localhost:5000
```

## Endpoints

### 🏠 Main Dashboard

#### `GET /`
Returns the main dashboard HTML page.

**Response**: HTML page with embedded JavaScript for real-time updates

**Usage**:
```bash
curl http://localhost:5000/
```

### 📹 Video Streaming

#### `GET /video_feed`
Provides real-time video stream with detection overlays.

**Response**: Multipart HTTP stream (MJPEG format)

**Content-Type**: `multipart/x-mixed-replace; boundary=frame`

**Usage**:
```html
<img src="http://localhost:5000/video_feed" alt="Live Feed">
```

### 📊 Statistics API

#### `GET /api/stats`
Returns current detection and navigation statistics.

**Response Format**:
```json
{
  "total_obstacles": 4,
  "class_breakdown": {
    "car": 2,
    "person": 1,
    "bicycle": 1
  },
  "navigation": {
    "status": "REDIRECT_SAFE",
    "direction": "LEFT",
    "confidence": 0.85,
    "obstacles_by_zone": {
      "center": 2,
      "left": 0,
      "right": 1,
      "top": 0,
      "bottom": 1
    },
    "danger_scores": {
      "center": 0.8,
      "left": 0.0,
      "right": 0.3,
      "top": 0.0,
      "bottom": 0.2
    },
    "recommendation": "Obstacle in center - safe path available left"
  },
  "fps": 24.5,
  "timestamp": "2024-01-15T10:30:45.123Z"
}
```

**Usage**:
```javascript
fetch('/api/stats')
  .then(response => response.json())
  .then(data => {
    console.log('Total obstacles:', data.total_obstacles);
    console.log('Navigation direction:', data.navigation.direction);
    console.log('Confidence:', data.navigation.confidence);
  });
```

### ⚙️ Settings Management

#### `GET /api/settings`
Returns current system settings.

**Response Format**:
```json
{
  "conf_threshold": 0.25,
  "iou_threshold": 0.45,
  "camera_index": 1,
  "show_navigation": true,
  "show_zones": true,
  "detection_mode": "obstacles"
}
```

#### `POST /api/settings`
Updates system settings.

**Request Format**:
```json
{
  "conf_threshold": 0.3,
  "iou_threshold": 0.4,
  "camera_index": 0,
  "show_navigation": true,
  "show_zones": false,
  "detection_mode": "attendance"
}
```

**Response Format**:
```json
{
  "status": "success",
  "settings": {
    "conf_threshold": 0.3,
    "iou_threshold": 0.4,
    "camera_index": 0,
    "show_navigation": true,
    "show_zones": false,
    "detection_mode": "attendance"
  }
}
```

**Usage**:
```javascript
// Update detection sensitivity
fetch('/api/settings', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    conf_threshold: 0.15,
    detection_mode: 'obstacles'
  })
})
.then(response => response.json())
.then(data => console.log('Settings updated:', data));
```

### 🧭 Navigation Commands

#### `GET /api/navigation_command`
Returns drone-ready navigation commands based on current obstacle analysis.

**Response Format**:
```json
{
  "movement": {
    "x": -0.425,
    "y": 0.0,
    "z": 0.0
  },
  "action": "LEFT",
  "confidence": 0.85,
  "should_stop": false
}
```

**Movement Coordinates**:
- `x`: Left (-) / Right (+) movement
- `y`: Backward (-) / Forward (+) movement  
- `z`: Down (-) / Up (+) movement
- Values range from -0.5 to +0.5 (scaled by confidence)

**Actions**:
- `FORWARD`: Safe to proceed straight
- `LEFT`: Move left to avoid obstacles
- `RIGHT`: Move right to avoid obstacles
- `UP`: Move up to avoid obstacles
- `DOWN`: Move down to avoid obstacles
- `STOP`: Stop immediately, obstacles in all directions

**Usage**:
```javascript
fetch('/api/navigation_command')
  .then(response => response.json())
  .then(command => {
    if (command.should_stop) {
      drone.stop();
    } else {
      drone.move(command.movement.x, command.movement.y, command.movement.z);
    }
  });
```

## Parameter Reference

### Detection Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `conf_threshold` | float | 0.1-0.9 | 0.25 | Minimum confidence for detections |
| `iou_threshold` | float | 0.1-0.9 | 0.45 | IoU threshold for NMS |
| `camera_index` | int | 0-10 | 1 | Camera device index |

### Display Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `show_navigation` | boolean | true | Show navigation overlays |
| `show_zones` | boolean | true | Show zone boundaries |
| `detection_mode` | string | "obstacles" | "obstacles" or "attendance" |

### Navigation Status Values

| Status | Description | Color Code |
|--------|-------------|------------|
| `CLEAR_PATH` | No obstacles, safe to proceed | Green |
| `REDIRECT_SAFE` | Obstacle in center, safe alternative | Yellow |
| `REDIRECT_CAUTION` | Multiple obstacles, proceed carefully | Orange |
| `DANGER_STOP` | Obstacles everywhere, stop immediately | Red |

## Real-Time Updates

### JavaScript Integration

```javascript
// Auto-update statistics every second
setInterval(() => {
  fetch('/api/stats')
    .then(response => response.json())
    .then(updateDashboard);
}, 1000);

function updateDashboard(data) {
  // Update obstacle count
  document.getElementById('obstacleCount').textContent = data.total_obstacles;
  
  // Update navigation status
  const statusElement = document.getElementById('navStatus');
  statusElement.textContent = data.navigation.status;
  statusElement.className = `status-${data.navigation.status.toLowerCase()}`;
  
  // Update FPS
  document.getElementById('fps').textContent = data.fps;
}
```

### WebSocket Alternative (Future Enhancement)

For real-time updates without polling:
```javascript
const socket = new WebSocket('ws://localhost:5000/ws');
socket.onmessage = function(event) {
  const data = JSON.parse(event.data);
  updateDashboard(data);
};
```

## Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request (invalid parameters) |
| 404 | Endpoint not found |
| 500 | Internal server error |

### Error Response Format

```json
{
  "error": "Invalid parameter value",
  "details": "conf_threshold must be between 0.1 and 0.9",
  "code": 400
}
```

### JavaScript Error Handling

```javascript
fetch('/api/settings', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({conf_threshold: 1.5}) // Invalid value
})
.then(response => {
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }
  return response.json();
})
.catch(error => {
  console.error('Settings update failed:', error);
  showErrorMessage('Failed to update settings');
});
```

## Integration Examples

### Drone Control Integration

```python
import requests
import time

def get_navigation_command():
    response = requests.get('http://localhost:5000/api/navigation_command')
    return response.json()

def control_drone():
    while True:
        command = get_navigation_command()
        
        if command['should_stop']:
            drone.stop()
            print("STOPPING: Obstacles detected")
        else:
            x, y, z = command['movement'].values()
            drone.move(x, y, z)
            print(f"Moving {command['action']}: confidence {command['confidence']:.1%}")
        
        time.sleep(0.1)  # 10Hz control loop
```

### Monitoring Dashboard

```python
import requests
import json

def monitor_system():
    response = requests.get('http://localhost:5000/api/stats')
    stats = response.json()
    
    print(f"Obstacles: {stats['total_obstacles']}")
    print(f"Navigation: {stats['navigation']['direction']}")
    print(f"Confidence: {stats['navigation']['confidence']:.1%}")
    print(f"FPS: {stats['fps']}")
    
    return stats

# Log statistics to file
with open('detection_log.json', 'a') as f:
    stats = monitor_system()
    json.dump(stats, f)
    f.write('\n')
```

## Security Considerations

### Local Network Only
- Default binding to `localhost` (127.0.0.1)
- Change to `0.0.0.0` only in trusted networks
- Consider authentication for production use

### Input Validation
- All parameters are validated server-side
- Invalid values return appropriate error messages
- No direct file system access through API

### Rate Limiting
- Consider implementing rate limiting for production
- Monitor resource usage with high-frequency requests