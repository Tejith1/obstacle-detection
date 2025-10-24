# Obstacle Detection (YOLOv5)

A simple obstacle detection project using YOLOv5 for real-time object detection from webcam input.

This repository contains a local copy of YOLOv5 (in the `yolov5/` directory), a `webcam_detect.py` script to run webcam detection, and a trained weights file `best.pt`.

## Contents

- `webcam_detect.py` — main script to run webcam object detection
- `best.pt` — trained YOLOv5 weights (model)
- `yolov5/` — local copy of the Ultralytics YOLOv5 repository (integrated into this project)

## Requirements

- Windows (development tested on Windows)
- Python 3.12 (virtual environment recommended)
- A webcam (or use a video file by modifying the script)

Recommended: use the bundled `.venv` virtual environment or create a new one.

## Setup (recommended)

1. Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Upgrade pip and install required packages:

```powershell
python -m pip install --upgrade pip
python -m pip install -r yolov5/requirements.txt
# or install packages used by this project manually:
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install opencv-python matplotlib numpy
```

Note: The Yolov5 `requirements.txt` may include extra packages. If you have a GPU and compatible CUDA, install the appropriate `torch` build instead of the CPU wheel above.

## Run webcam detection

From the project root (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
python webcam_detect.py
```

This will open your default webcam (index 0). To change the camera index or run on a video file, edit `webcam_detect.py`.

## Common issues and troubleshooting

1. ModuleNotFoundError: No module named 'matplotlib.backends.registry'
   - This indicates a corrupted matplotlib installation. Fix by reinstalling matplotlib in your virtualenv:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --force-reinstall matplotlib
```

2. numpy/opencv version incompatibility
   - If you get an error when installing numpy (e.g., `opencv-python requires numpy<2.3.0,>=2`), install compatible versions:

```powershell
python -m pip install numpy==2.2.6 opencv-python==4.12.0.88
python -m pip install --force-reinstall matplotlib
```

3. Embedded `yolov5` repository warning when adding to git
   - If you see a warning about an embedded git repo when running `git add .`, remove the `yolov5/.git` folder (if you don't want it as a submodule):

```powershell
Remove-Item -Recurse -Force "yolov5\.git"
git add .
```

## Notes

- This project uses YOLOv5 (Ultralytics) for detection. See `yolov5/README.md` for more details about the model and training.
- If you want to track `yolov5` separately, use `git submodule add <url> yolov5` instead of embedding the repo.

## License and Acknowledgements

- YOLOv5 and Ultralytics: https://github.com/ultralytics/yolov5
- This repository bundles YOLOv5 code and is intended for educational and development use.

