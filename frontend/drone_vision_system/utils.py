"""
utils.py — Constants, helper functions, and coordinate transforms
for the Drone Vision System v2.

Upgrades: cyberpunk color palette, glow effects, rounded drawing helpers,
          class-based color coding, gradient utilities.
"""

import cv2
import math
import numpy as np

# ─── Display / Layout ────────────────────────────────────────────────
DISPLAY_WIDTH = 1024
DISPLAY_HEIGHT = 680
MAP_HEIGHT = DISPLAY_HEIGHT // 2       # 340
CAM_HEIGHT = DISPLAY_HEIGHT // 2       # 340
MAP_WIDTH = DISPLAY_WIDTH
CAM_WIDTH = DISPLAY_WIDTH

# ─── Processing resolution ───────────────────────────────────────────
PROC_WIDTH = 640
PROC_HEIGHT = 420

# ─── FPS / Detection ─────────────────────────────────────────────────
TARGET_FPS = 30
YOLO_CONF_THRESHOLD = 0.40

# ─── Cyberpunk Color Palette (BGR) ───────────────────────────────────
COLOR_BG_DARK      = (15, 12, 10)
COLOR_BG_GRAD_TOP  = (30, 18, 12)
COLOR_BG_GRAD_BOT  = (12, 8, 6)
COLOR_GRID_LINE    = (40, 35, 30)
COLOR_GRID_MAJOR   = (55, 48, 40)

# Neon accent colors
COLOR_CYAN         = (240, 220, 0)       # neon cyan
COLOR_MAGENTA      = (200, 50, 255)      # neon magenta/pink
COLOR_GREEN        = (80, 255, 120)      # neon green
COLOR_AMBER        = (50, 180, 255)      # warm amber/orange
COLOR_RED          = (60, 40, 255)       # bright red
COLOR_BLUE         = (255, 160, 40)      # electric blue
COLOR_PURPLE       = (220, 100, 180)     # soft purple
COLOR_TEAL         = (200, 200, 0)       # teal

# UI colors
COLOR_HUD_BG       = (25, 20, 18)
COLOR_HUD_BORDER   = (60, 55, 50)
COLOR_HUD_TEXT     = (200, 195, 185)
COLOR_WHITE        = (255, 255, 255)
COLOR_DIM_WHITE    = (140, 135, 130)
COLOR_FPS          = COLOR_GREEN

# Detection class colors
CLASS_COLORS = {
    "person":        (80, 255, 120),    # green
    "car":           (240, 220, 0),     # cyan
    "bicycle":       (50, 180, 255),    # amber
    "motorcycle":    (200, 50, 255),    # magenta
    "truck":         (255, 160, 40),    # blue
    "bus":           (220, 100, 180),   # purple
    "cat":           (100, 255, 255),   # yellow
    "dog":           (150, 200, 255),   # light salmon
    "bird":          (200, 200, 0),     # teal
    "traffic light": (0, 255, 200),     # lime
    "stop sign":     (60, 40, 255),     # red
}
DEFAULT_CLASS_COLOR = (200, 195, 185)

# Avoidance grid colors
COLOR_GRID_CLEAR   = (40, 60, 25)
COLOR_GRID_DANGER  = (30, 20, 160)
COLOR_GRID_AMBER   = (20, 100, 200)
COLOR_GRID_CENTER  = (50, 30, 220)

# Drone colors
COLOR_DRONE_BODY   = COLOR_GREEN
COLOR_DRONE_GLOW   = (60, 200, 100)
COLOR_DRONE_PATH   = (50, 180, 80)
COLOR_DRONE_PATH2  = (30, 100, 50)

# Warning
COLOR_WARNING      = COLOR_RED
COLOR_WARNING_GLOW = (30, 20, 120)

# ─── Target COCO classes ─────────────────────────────────────────────
TARGET_CLASSES = {
    "person", "car", "bicycle", "motorcycle",
    "truck", "cat", "dog", "bird",
    "bus", "traffic light", "stop sign",
}

# ─── 3×3 Grid Labels ─────────────────────────────────────────────────
GRID_LABELS = {
    (0, 0): "TL", (0, 1): "TC", (0, 2): "TR",
    (1, 0): "ML", (1, 1): "CC", (1, 2): "MR",
    (2, 0): "BL", (2, 1): "BC", (2, 2): "BR",
}

# ─── Drone simulation defaults ───────────────────────────────────────
DRONE_SPEED = 1.2
DRONE_SPEED_MIN = 0.3
DRONE_SPEED_MAX = 4.0
WORLD_SCALE = 4.0
OBSTACLE_WARN_DIST = 80
MANUAL_TURN_RATE = 0.06


# ═══════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════

def bbox_to_world(cx, cy, frame_w, frame_h, drone_x, drone_y, scale=WORLD_SCALE):
    """Convert bounding-box center (pixel) to world coordinates."""
    x_world = (cx - frame_w / 2) / scale + drone_x
    y_world = (frame_h - cy) / scale + drone_y
    return x_world, y_world


def get_grid_cell(cx, cy, frame_w, frame_h):
    """Return (row, col) in a 3×3 grid for the given pixel centre."""
    col = min(int(cx / (frame_w / 3)), 2)
    row = min(int(cy / (frame_h / 3)), 2)
    return row, col


def get_class_color(class_name):
    """Return the BGR color for a detection class."""
    return CLASS_COLORS.get(class_name, DEFAULT_CLASS_COLOR)


def create_gradient_bg(height, width, top_color, bot_color):
    """Create a vertical gradient background image."""
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        ratio = y / max(height - 1, 1)
        color = tuple(int(top_color[i] + (bot_color[i] - top_color[i]) * ratio)
                      for i in range(3))
        bg[y, :] = color
    return bg


def draw_text_with_bg(img, text, org, font_scale=0.55, color=COLOR_WHITE,
                      bg_color=COLOR_HUD_BG, thickness=1, padding=5,
                      border_color=None):
    """Draw text with a semi-transparent rounded-rect background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    x1 = x - padding
    y1 = y - th - padding
    x2 = x + tw + padding
    y2 = y + baseline + padding

    # Semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, cv2.FILLED)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    if border_color:
        cv2.rectangle(img, (x1, y1), (x2, y2), border_color, 1)

    cv2.putText(img, text, (x, y), font, font_scale, color, thickness,
                cv2.LINE_AA)


def draw_rounded_rect(img, pt1, pt2, color, thickness=1, radius=8,
                      fill=False, alpha=1.0):
    """Draw a rounded rectangle. If fill=True, fills with optional alpha."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, abs(x2 - x1) // 2, abs(y2 - y1) // 2)

    if fill:
        overlay = img.copy()
        # Fill the main rectangle areas
        cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, cv2.FILLED)
        cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, cv2.FILLED)
        # Corner circles
        cv2.circle(overlay, (x1 + r, y1 + r), r, color, cv2.FILLED)
        cv2.circle(overlay, (x2 - r, y1 + r), r, color, cv2.FILLED)
        cv2.circle(overlay, (x1 + r, y2 - r), r, color, cv2.FILLED)
        cv2.circle(overlay, (x2 - r, y2 - r), r, color, cv2.FILLED)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    else:
        # Draw border lines
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
        # Corner arcs
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def draw_glow_circle(img, center, radius, color, intensity=0.4):
    """Draw a circle with a soft glow effect."""
    overlay = img.copy()
    # Outer glow (larger, dimmer)
    cv2.circle(overlay, center, radius + 6, color, cv2.FILLED)
    cv2.addWeighted(overlay, intensity * 0.3, img, 1 - intensity * 0.3, 0, img)
    overlay = img.copy()
    cv2.circle(overlay, center, radius + 3, color, cv2.FILLED)
    cv2.addWeighted(overlay, intensity * 0.5, img, 1 - intensity * 0.5, 0, img)
    # Core circle
    cv2.circle(img, center, radius, color, cv2.FILLED)


def draw_scanlines(img, spacing=4, alpha=0.06):
    """Add subtle CRT-style scanlines."""
    overlay = img.copy()
    h, w = img.shape[:2]
    for y in range(0, h, spacing):
        cv2.line(overlay, (0, y), (w, y), (0, 0, 0), 1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_grid_hud(img, grid_status, steer_cmd="NONE",
                  x_offset=10, y_offset=10, cell_size=32):
    """Draw a polished 3×3 grid HUD with steer command."""
    total_w = 3 * cell_size + 12
    total_h = 3 * cell_size + 32

    # Background panel
    draw_rounded_rect(img,
                      (x_offset - 4, y_offset - 4),
                      (x_offset + total_w, y_offset + total_h),
                      COLOR_HUD_BG, fill=True, alpha=0.8, radius=6)
    draw_rounded_rect(img,
                      (x_offset - 4, y_offset - 4),
                      (x_offset + total_w, y_offset + total_h),
                      COLOR_HUD_BORDER, thickness=1, radius=6)

    for r in range(3):
        for c in range(3):
            x1 = x_offset + c * cell_size
            y1 = y_offset + r * cell_size
            x2 = x1 + cell_size - 2
            y2 = y1 + cell_size - 2

            if grid_status[r][c]:
                if (r, c) == (1, 1):
                    fill_color = COLOR_GRID_CENTER
                else:
                    fill_color = COLOR_GRID_AMBER
            else:
                fill_color = COLOR_GRID_CLEAR

            # Fill cell
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), fill_color, cv2.FILLED)
            cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
            cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_HUD_BORDER, 1)

            # Cell label
            label = GRID_LABELS.get((r, c), "")
            cv2.putText(img, label,
                        (x1 + 4, y1 + cell_size // 2 + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                        COLOR_DIM_WHITE, 1, cv2.LINE_AA)

    # Steer command below grid
    cmd_y = y_offset + 3 * cell_size + 14
    cmd_color = COLOR_RED if steer_cmd not in ("NONE", "CLEAR") else COLOR_GREEN
    cv2.putText(img, steer_cmd,
                (x_offset, cmd_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, cmd_color, 1, cv2.LINE_AA)


def draw_hud_panel(img, x, y, w, h, title="", border_color=COLOR_HUD_BORDER):
    """Draw a semi-transparent HUD panel with title bar."""
    draw_rounded_rect(img, (x, y), (x + w, y + h),
                      COLOR_HUD_BG, fill=True, alpha=0.8, radius=6)
    draw_rounded_rect(img, (x, y), (x + w, y + h),
                      border_color, thickness=1, radius=6)
    if title:
        # Title bar accent line
        cv2.line(img, (x + 8, y + 20), (x + w - 8, y + 20),
                 border_color, 1)
        cv2.putText(img, title, (x + 8, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    COLOR_CYAN, 1, cv2.LINE_AA)


def draw_controls_help(img, x, y):
    """Draw keyboard controls reference on the frame."""
    controls = [
        ("Q", "Quit"),
        ("P", "Pause"),
        ("R", "Reset"),
        ("G", "Grid"),
        ("A", "Auto/Man"),
        ("D", "Debug"),
        ("/\\", "Spd+"),
        ("\\/", "Spd-"),
        ("<>", "Steer"),
    ]
    draw_hud_panel(img, x, y, 120, len(controls) * 18 + 28,
                   title="CONTROLS", border_color=COLOR_CYAN)
    for i, (key, desc) in enumerate(controls):
        cy = y + 30 + i * 18
        cv2.putText(img, f"[{key}]", (x + 6, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                    COLOR_AMBER, 1, cv2.LINE_AA)
        cv2.putText(img, desc, (x + 46, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                    COLOR_DIM_WHITE, 1, cv2.LINE_AA)
