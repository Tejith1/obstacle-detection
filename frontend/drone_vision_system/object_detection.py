"""
object_detection.py — YOLO11-based object detection wrapper.

Upgrade: Uses YOLO11s (small) for better accuracy. Class-colored
bounding boxes with rounded corners and glow effects.
"""

import cv2
from ultralytics import YOLO
from utils import (
    TARGET_CLASSES, YOLO_CONF_THRESHOLD,
    COLOR_WHITE, COLOR_HUD_BG, COLOR_DIM_WHITE,
    get_class_color, draw_rounded_rect, draw_glow_circle,
)


class ObjectDetector:
    """YOLO11-based real-time object detector with polished rendering."""

    def __init__(self, model_path: str = "yolo11s.pt",
                 conf: float = YOLO_CONF_THRESHOLD):
        print(f"[INFO] Loading model: {model_path} ...")
        self.model = YOLO(model_path)
        self.conf = conf
        self.class_names = self.model.names
        print(f"[INFO] Model loaded. Classes: {len(self.class_names)}")

    # ── detection ─────────────────────────────────────────────────────
    def detect(self, frame):
        """Run inference and return list of detection dicts."""
        results = self.model.predict(frame, conf=self.conf, verbose=False)
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = self.class_names.get(cls_id, "unknown")
                if cls_name not in TARGET_CLASSES:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                detections.append({
                    "class_name": cls_name,
                    "confidence": conf,
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "center": (cx, cy),
                    "bbox_height": y2 - y1,
                })
        return detections

    # ── polished drawing ──────────────────────────────────────────────
    @staticmethod
    def draw_detections(frame, detections):
        """Annotate frame with class-colored rounded bounding boxes."""
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_name = det["class_name"]
            conf = det["confidence"]
            color = get_class_color(cls_name)

            # Outer glow (soft)
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2),
                          color, 2)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

            # Main bounding box with rounded corners
            draw_rounded_rect(frame, (x1, y1), (x2, y2), color,
                              thickness=2, radius=6)

            # Corner accents (small L-shaped brackets)
            corner_len = min(15, (x2 - x1) // 4, (y2 - y1) // 4)
            # Top-left
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 2)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 2)
            # Top-right
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 2)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 2)
            # Bottom-left
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 2)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 2)
            # Bottom-right
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 2)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 2)

            # Label with confidence bar
            label = f"{cls_name} {conf:.0%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)

            label_x = x1
            label_y = y1 - 8
            pad = 4

            # Label background
            overlay = frame.copy()
            cv2.rectangle(overlay,
                          (label_x - pad, label_y - th - pad),
                          (label_x + tw + pad + 50, label_y + pad),
                          COLOR_HUD_BG, cv2.FILLED)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            # Label text
            cv2.putText(frame, label, (label_x, label_y),
                        font, font_scale, color, 1, cv2.LINE_AA)

            # Confidence bar
            bar_x = label_x + tw + 8
            bar_y = label_y - th + 2
            bar_w = 40
            bar_h = th
            cv2.rectangle(frame,
                          (bar_x, bar_y),
                          (bar_x + bar_w, bar_y + bar_h),
                          COLOR_DIM_WHITE, 1)
            fill_w = int(bar_w * conf)
            cv2.rectangle(frame,
                          (bar_x, bar_y),
                          (bar_x + fill_w, bar_y + bar_h),
                          color, cv2.FILLED)

            # Center dot with glow
            cx, cy = int(det["center"][0]), int(det["center"][1])
            draw_glow_circle(frame, (cx, cy), 3, color, intensity=0.3)

        return frame
