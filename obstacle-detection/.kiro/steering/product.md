# Product Overview

This is a **real-time obstacle detection system** built on YOLOv5 for computer vision applications. The system uses a webcam to detect and count obstacles in real-time, displaying both total counts and breakdowns by object type.

## Key Features

- **Real-time detection**: Live webcam feed with instant obstacle identification
- **Obstacle counting**: Displays total obstacle count and breakdown by class type
- **Custom model support**: Uses a trained `best.pt` model with 22 custom classes (class0-class21)
- **Class mapping tools**: Interactive utilities to identify what each class detects
- **Visual feedback**: Color-coded bounding boxes and on-screen statistics

## Primary Use Cases

- Navigation systems for autonomous vehicles/robots
- Safety monitoring in industrial environments  
- Traffic analysis and pedestrian counting
- Warehouse automation and path planning
- General computer vision research and development

## Core Components

- `webcam_detect.py`: Main detection application with real-time counting
- `simple_class_mapper.py`: Terminal-based class identification tool
- `test_and_identify_classes.py`: Interactive class mapping utility
- `best.pt`: Custom trained YOLOv5 model weights