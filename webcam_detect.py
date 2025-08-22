# webcam_detect_local_draw.py
import sys
import time
import cv2
import torch
import numpy as np
import os
import pathlib
import sys

# Fix for Windows when loading Linux-trained YOLO weights
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
# ensure yolov5 repo is importable
REPO = "yolov5"   # relative path to your cloned yolov5 folder
sys.path.insert(0, REPO)

# local imports from yolov5
from models.common import DetectMultiBackend
from utils.general import (non_max_suppression, check_img_size)
from utils.augmentations import letterbox
from utils.torch_utils import select_device

# ---------- Config ----------
WEIGHTS = "best.pt"       # path to your weights (relative to project root)
IMG_SIZE = 640            # inference size (square). Increase for accuracy, slower speed
CONF_THRESH = 0.25
IOU_THRESH = 0.45
DEVICE = ''               # '' = auto, or 'cpu' or '0' or '0,1' etc.
SHOW_FPS = True
CAMERA_INDEX = 0
# Map class name -> custom sentence (edit to fit your classes)
MSG_MAP = {
    # "person": "Person detected — be careful!",
    # "dog": "Dog detected — say hello!",
    # add your own names from model.names
}
# --------------------------------

def draw_box(img, xyxy, label=None, color=(0,255,0), thickness=2):
    """
    Draw bounding box with label on image (cv2).
    xyxy: [x1,y1,x2,y2]
    """
    x1, y1, x2, y2 = map(int, xyxy)
    # rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        # text background
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        t = 1
        (w, h), _ = cv2.getTextSize(label, font, scale, t)
        # make filled rectangle for label
        cv2.rectangle(img, (x1, y1 - h - 6), (x1 + w + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 4), font, scale, (255,255,255), t, cv2.LINE_AA)


def scale_boxes_from_letterbox_np(boxes, ratio, dwdh, original_shape):
    """
    Rescale boxes from letterboxed image size back to original image size.
    Handles empty or malformed detections safely.
    """
    if boxes is None or len(boxes) == 0:
        return np.array([])

    boxes = np.array(boxes)

    # Ensure boxes have correct shape (N, 4)
    if boxes.ndim == 1:
        if len(boxes) == 4:
            boxes = boxes.reshape(1, 4)  # single box
        else:
            return np.array([])  # invalid shape, skip

    if boxes.shape[1] != 4:
        return np.array([])  # skip malformed results

    dw, dh = dwdh
    h0, w0 = original_shape

    # scale boxes
    boxes[:, 0] = (boxes[:, 0] - dw) / ratio
    boxes[:, 1] = (boxes[:, 1] - dh) / ratio
    boxes[:, 2] = (boxes[:, 2] - dw) / ratio
    boxes[:, 3] = (boxes[:, 3] - dh) / ratio

    # clip boxes to image size
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w0 - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h0 - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w0 - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h0 - 1)

    return boxes


def main():
    # device
    device = select_device(DEVICE)
    print("Using device:", device)

    # Load model
    model = DetectMultiBackend(WEIGHTS, device=device, dnn=False, fp16=False)
    stride = int(model.stride)
    names = model.names
    pt = model.pt  # bool: is torchscript/pt model
    imgsz = check_img_size(IMG_SIZE, s=stride)  # ensure size is multiple of stride
    print("Model loaded. stride:", stride, "pt:", pt)
    print("Class names (first 10):", list(names)[:10])

    # open webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)  # change index if you have multiple cameras
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam (index {CAMERA_INDEX}). Try changing the index.")

    prev_time = 0.0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from webcam")
                break

            img0 = frame.copy()                      # original shape (H0, W0, C)
            h0, w0 = img0.shape[:2]

            # Preprocess: letterbox returns (img, ratio, (dw, dh))
            img, ratio, (dw, dh) = letterbox(img0, imgsz, stride=stride)
            # convert to CHW, RGB, float32
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            im = torch.from_numpy(img).to(device)
            im = im.float() / 255.0
            if im.ndim == 3:
                im = im.unsqueeze(0)

            # Inference
            pred = model(im, augment=False, visualize=False)
            pred = non_max_suppression(pred, CONF_THRESH, IOU_THRESH, None, False, max_det=1000)

            # process detections
            if len(pred) and pred[0] is not None and len(pred[0]):
                det = pred[0].clone().detach().cpu().numpy()  # Nx6: x1,y1,x2,y2,conf,cls
                boxes = det[:, :4].copy()

                # ✅ safe check before scaling
                if boxes is not None and len(boxes) > 0:
                    boxes = scale_boxes_from_letterbox_np(boxes, ratio, (dw, dh), (h0, w0))
                else:
                    boxes = []

                if len(boxes) > 0:
                    det[:, :4] = boxes
                    for *xyxy, conf, cls in det:
                        x1, y1, x2, y2 = map(int, xyxy)
                        cls = int(cls)
                        label_name = names[cls] if cls < len(names) else f"class{cls}"
                        label = f"{label_name} {conf:.2f}"
                        draw_color = (0, 255, 0)
                        draw_box(img0, (x1, y1, x2, y2), label=label, color=draw_color, thickness=2)

                        # custom sentence mapping
                        sentence = MSG_MAP.get(label_name, f"Command: {label_name} detected")
                        cv2.putText(img0, sentence, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            # FPS
            if SHOW_FPS:
                cur_time = time.time()
                fps = 1.0 / (cur_time - prev_time) if prev_time else 0.0
                prev_time = cur_time
                cv2.putText(img0, f"FPS: {fps:.1f}", (10, img0.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            # show
            cv2.imshow("YOLOv5 Webcam (local)", img0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    if not os.path.isdir(REPO):
        print(f"Error: {REPO} folder not found. Be sure you run this from the project root that contains the cloned yolov5 repo.")
        sys.exit(1)
    if not os.path.exists(WEIGHTS):
        print(f"Error: weights file {WEIGHTS} not found in project root. Put best.pt next to yolov5 folder or adjust WEIGHTS path.")
        sys.exit(1)
    main()
