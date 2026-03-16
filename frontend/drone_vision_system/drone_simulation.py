"""
drone_simulation.py — 2D drone map simulator with 3×3 grid-based
obstacle avoidance, polished cyberpunk rendering, and glow effects.
"""

import cv2
import math
import time
import numpy as np
from utils import (
    MAP_WIDTH, MAP_HEIGHT, DRONE_SPEED, WORLD_SCALE,
    OBSTACLE_WARN_DIST, MANUAL_TURN_RATE,
    DRONE_SPEED_MIN, DRONE_SPEED_MAX,
    COLOR_BG_DARK, COLOR_GRID_LINE, COLOR_GRID_MAJOR,
    COLOR_DRONE_BODY, COLOR_DRONE_GLOW, COLOR_DRONE_PATH, COLOR_DRONE_PATH2,
    COLOR_WHITE, COLOR_DIM_WHITE, COLOR_HUD_TEXT,
    COLOR_CYAN, COLOR_AMBER, COLOR_GREEN, COLOR_RED, COLOR_MAGENTA,
    COLOR_WARNING, COLOR_WARNING_GLOW,
    COLOR_HUD_BG, COLOR_HUD_BORDER,
    COLOR_GRID_CLEAR, COLOR_GRID_DANGER, COLOR_GRID_AMBER, COLOR_GRID_CENTER,
    GRID_LABELS,
    bbox_to_world, get_grid_cell, get_class_color,
    draw_text_with_bg, draw_glow_circle, draw_rounded_rect,
    draw_hud_panel, draw_scanlines, create_gradient_bg,
)


# Cache the gradient background (created once)
_gradient_bg = None


def _get_gradient_bg():
    global _gradient_bg
    if _gradient_bg is None:
        _gradient_bg = create_gradient_bg(MAP_HEIGHT, MAP_WIDTH,
                                          (30, 18, 12), (12, 8, 6))
    return _gradient_bg.copy()


# ═══════════════════════════════════════════════════════════════════════
# GridAvoidance
# ═══════════════════════════════════════════════════════════════════════

class GridAvoidance:
    """3×3 grid obstacle analysis & steering commands."""

    def analyze(self, detections, frame_w, frame_h):
        grid = [[False] * 3 for _ in range(3)]
        for det in detections:
            cx, cy = det["center"]
            r, c = get_grid_cell(cx, cy, frame_w, frame_h)
            grid[r][c] = True
        cmd, val = self._decide(grid)
        return grid, cmd, val

    @staticmethod
    def _decide(g):
        if g[1][1]:
            left_free = not g[1][0]
            right_free = not g[1][2]
            if left_free and right_free:
                return "EVADE LEFT", -0.8
            if left_free:
                return "EVADE LEFT", -0.9
            if right_free:
                return "EVADE RIGHT", 0.9
            return "CLIMB", 0.0

        if g[0][1]:
            return "DESCEND", 0.0
        if g[2][1]:
            return "CLIMB", 0.0

        left_occ = g[0][0] or g[1][0] or g[2][0]
        right_occ = g[0][2] or g[1][2] or g[2][2]

        if left_occ and right_occ:
            return "HOLD", 0.0
        if left_occ:
            return "STEER RIGHT", 0.6
        if right_occ:
            return "STEER LEFT", -0.6

        return "CLEAR", 0.0


# ═══════════════════════════════════════════════════════════════════════
# DroneSimulator
# ═══════════════════════════════════════════════════════════════════════

class DroneSimulator:
    """Drone movement, state, and full map rendering."""

    def __init__(self):
        self.x = MAP_WIDTH / 2
        self.y = MAP_HEIGHT / 2
        self.heading = -math.pi / 2
        self.speed = DRONE_SPEED
        self.path = [(self.x, self.y)]
        self.steer_cmd = "CLEAR"
        self.auto_mode = True
        self.altitude = 50.0
        self.frame_count = 0

    # ── state update ──────────────────────────────────────────────────
    def update(self, avoidance_steer: float = 0.0,
               manual_steer: float = 0.0):
        """Advance drone one tick. In auto mode, use avoidance_steer.
        In manual mode, use manual_steer."""
        if self.auto_mode:
            self.heading += avoidance_steer * 0.04
        else:
            self.heading += manual_steer * MANUAL_TURN_RATE

        self.heading += np.random.normal(0, 0.005)

        dx = math.cos(self.heading) * self.speed
        dy = math.sin(self.heading) * self.speed
        self.x += dx
        self.y += dy

        self.x %= MAP_WIDTH
        self.y %= MAP_HEIGHT

        self.path.append((self.x, self.y))
        if len(self.path) > 1000:
            self.path = self.path[-1000:]

        self.frame_count += 1

    def reset(self):
        self.x = MAP_WIDTH / 2
        self.y = MAP_HEIGHT / 2
        self.heading = -math.pi / 2
        self.speed = DRONE_SPEED
        self.path = [(self.x, self.y)]
        self.steer_cmd = "CLEAR"
        self.altitude = 50.0

    def change_speed(self, delta):
        self.speed = max(DRONE_SPEED_MIN,
                         min(DRONE_SPEED_MAX, self.speed + delta))

    def toggle_auto(self):
        self.auto_mode = not self.auto_mode

    @staticmethod
    def estimate_distance(bbox_height, real_height=1.7, focal_length=800):
        if bbox_height <= 0:
            return float("inf")
        return focal_length * real_height / bbox_height

    # ── rendering ─────────────────────────────────────────────────────
    def render(self, detections, frame_w, frame_h, grid_status=None,
               steer_cmd="CLEAR"):
        self.steer_cmd = steer_cmd

        # Gradient background
        canvas = _get_gradient_bg()

        self._draw_bg_grid(canvas)

        if grid_status is not None:
            self._draw_avoidance_grid(canvas, grid_status)

        self._draw_objects(canvas, detections, frame_w, frame_h)
        self._draw_path(canvas)
        self._draw_drone(canvas)
        self._draw_hud(canvas, detections, grid_status)

        # Subtle scanlines for cyberpunk feel
        draw_scanlines(canvas, spacing=3, alpha=0.04)

        return canvas

    # ·· background grid ···············································
    def _draw_bg_grid(self, canvas):
        # Minor grid
        for x in range(0, MAP_WIDTH, 40):
            cv2.line(canvas, (x, 0), (x, MAP_HEIGHT), COLOR_GRID_LINE, 1)
        for y in range(0, MAP_HEIGHT, 40):
            cv2.line(canvas, (0, y), (MAP_WIDTH, y), COLOR_GRID_LINE, 1)
        # Major grid
        for x in range(0, MAP_WIDTH, 160):
            cv2.line(canvas, (x, 0), (x, MAP_HEIGHT), COLOR_GRID_MAJOR, 1)
        for y in range(0, MAP_HEIGHT, 160):
            cv2.line(canvas, (0, y), (MAP_WIDTH, y), COLOR_GRID_MAJOR, 1)

    # ·· 3×3 avoidance grid ············································
    def _draw_avoidance_grid(self, canvas, grid_status):
        cell_w = MAP_WIDTH // 3
        cell_h = MAP_HEIGHT // 3
        overlay = canvas.copy()
        for r in range(3):
            for c in range(3):
                x1 = c * cell_w
                y1 = r * cell_h
                x2 = x1 + cell_w
                y2 = y1 + cell_h
                if grid_status[r][c]:
                    if (r, c) == (1, 1):
                        color = COLOR_GRID_CENTER
                    else:
                        color = COLOR_GRID_AMBER
                else:
                    color = COLOR_GRID_CLEAR
                cv2.rectangle(overlay, (x1, y1), (x2, y2),
                              color, cv2.FILLED)
                # Border
                cv2.rectangle(canvas, (x1, y1), (x2, y2),
                              (50, 45, 40), 1)
                # Label
                label = GRID_LABELS.get((r, c), "")
                cv2.putText(canvas, label, (x1 + 6, y1 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                            COLOR_DIM_WHITE, 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0, canvas)

    # ·· detected objects ··············································
    def _draw_objects(self, canvas, detections, frame_w, frame_h):
        for det in detections:
            cx, cy = det["center"]
            wx, wy = bbox_to_world(cx, cy, frame_w, frame_h,
                                   self.x, self.y, WORLD_SCALE)
            mx = int(wx) % MAP_WIDTH
            my = int(wy) % MAP_HEIGHT

            dist = self.estimate_distance(det["bbox_height"])
            color = get_class_color(det["class_name"])

            # Glow marker
            draw_glow_circle(canvas, (mx, my), 8, color, intensity=0.35)
            cv2.circle(canvas, (mx, my), 10, COLOR_WHITE, 1, cv2.LINE_AA)

            # Label
            obj_label = f'{det["class_name"]} {dist:.0f}m'
            cv2.putText(canvas, obj_label, (mx + 14, my + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        color, 1, cv2.LINE_AA)

            # Danger line for close objects
            ddx = mx - self.x
            ddy = my - self.y
            d = math.sqrt(ddx * ddx + ddy * ddy)
            if d < OBSTACLE_WARN_DIST:
                # Pulsing dashed line
                pulse = 0.5 + 0.5 * math.sin(self.frame_count * 0.15)
                line_color = (int(COLOR_WARNING[0] * pulse),
                              int(COLOR_WARNING[1] * pulse),
                              int(min(255, COLOR_WARNING[2] * (0.5 + pulse * 0.5))))
                cv2.line(canvas, (int(self.x), int(self.y)),
                         (mx, my), line_color, 2, cv2.LINE_AA)

    # ·· drone path (gradient fade) ····································
    def _draw_path(self, canvas):
        if len(self.path) < 2:
            return
        n = len(self.path)
        for i in range(1, n):
            alpha = i / n
            color = (int(COLOR_DRONE_PATH2[0] + (COLOR_DRONE_PATH[0] - COLOR_DRONE_PATH2[0]) * alpha),
                     int(COLOR_DRONE_PATH2[1] + (COLOR_DRONE_PATH[1] - COLOR_DRONE_PATH2[1]) * alpha),
                     int(COLOR_DRONE_PATH2[2] + (COLOR_DRONE_PATH[2] - COLOR_DRONE_PATH2[2]) * alpha))
            pt1 = (int(self.path[i-1][0]), int(self.path[i-1][1]))
            pt2 = (int(self.path[i][0]), int(self.path[i][1]))
            # Skip if points are far apart (wrap-around)
            if abs(pt1[0] - pt2[0]) < MAP_WIDTH // 2 and abs(pt1[1] - pt2[1]) < MAP_HEIGHT // 2:
                thickness = max(1, int(alpha * 2))
                cv2.line(canvas, pt1, pt2, color, thickness, cv2.LINE_AA)

    # ·· drone icon ····················································
    def _draw_drone(self, canvas):
        # Outer glow (pulsing)
        pulse = 0.6 + 0.4 * math.sin(self.frame_count * 0.08)
        glow_r = int(22 + 4 * pulse)
        draw_glow_circle(canvas, (int(self.x), int(self.y)),
                         glow_r, COLOR_DRONE_GLOW, intensity=0.15 * pulse)

        # Drone body (triangle with arms)
        size = 16
        angle = self.heading
        # Main triangle
        pts = []
        for i, a in enumerate([0, 2.3, -2.3]):
            r = size if i == 0 else size * 0.55
            px = int(self.x + math.cos(angle + a) * r)
            py = int(self.y + math.sin(angle + a) * r)
            pts.append([px, py])
        pts_np = np.array(pts, dtype=np.int32)
        cv2.fillPoly(canvas, [pts_np], COLOR_DRONE_BODY)
        cv2.polylines(canvas, [pts_np], True, COLOR_WHITE, 1, cv2.LINE_AA)

        # Center dot
        cv2.circle(canvas, (int(self.x), int(self.y)), 2,
                   COLOR_WHITE, cv2.FILLED)

        # Heading indicator line
        hx = int(self.x + math.cos(self.heading) * (size + 8))
        hy = int(self.y + math.sin(self.heading) * (size + 8))
        cv2.line(canvas, (int(self.x), int(self.y)), (hx, hy),
                 COLOR_CYAN, 1, cv2.LINE_AA)

    # ·· HUD overlay ··················································
    def _draw_hud(self, canvas, detections, grid_status):
        # ── Left panel: status ──
        draw_hud_panel(canvas, 8, 8, 240, 150,
                       title="DRONE STATUS", border_color=COLOR_CYAN)

        info_lines = [
            (f"POS   X:{self.x:.0f}  Y:{self.y:.0f}", COLOR_HUD_TEXT),
            (f"HDG   {math.degrees(self.heading):.0f} deg", COLOR_HUD_TEXT),
            (f"SPD   {self.speed:.1f} u/t", COLOR_AMBER),
            (f"ALT   {self.altitude:.0f} m", COLOR_HUD_TEXT),
            (f"MODE  {'AUTO' if self.auto_mode else 'MANUAL'}",
             COLOR_GREEN if self.auto_mode else COLOR_MAGENTA),
        ]
        for i, (text, color) in enumerate(info_lines):
            cv2.putText(canvas, text, (18, 40 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        color, 1, cv2.LINE_AA)

        # ── Right panel: detections ──
        det_count = len(detections)
        draw_hud_panel(canvas, MAP_WIDTH - 210, 8, 200, 30,
                       title="", border_color=COLOR_AMBER)
        cv2.putText(canvas,
                    f"DETECTIONS: {det_count}",
                    (MAP_WIDTH - 200, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    COLOR_AMBER, 1, cv2.LINE_AA)

        # ── Navigation command (center-top) ──
        cmd = self.steer_cmd
        if cmd not in ("CLEAR", "NONE"):
            cmd_color = COLOR_RED
        else:
            cmd_color = COLOR_GREEN
            cmd = "ALL CLEAR"

        cmd_text = f"NAV: {cmd}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, _), _ = cv2.getTextSize(cmd_text, font, 0.55, 1)
        cx = (MAP_WIDTH - tw) // 2
        draw_text_with_bg(canvas, cmd_text, (cx, 28),
                          font_scale=0.55, color=cmd_color,
                          border_color=cmd_color)

        # ── Obstacle warning (center-bottom, pulsing) ──
        if grid_status is not None and grid_status[1][1]:
            pulse = 0.5 + 0.5 * math.sin(self.frame_count * 0.2)
            warn_color = (int(COLOR_WARNING[0] * pulse),
                          int(COLOR_WARNING[1] * pulse),
                          int(min(255, COLOR_WARNING[2] * (0.5 + pulse * 0.5))))

            warn_text = "!! OBSTACLE AHEAD !!"
            (tw, th), _ = cv2.getTextSize(warn_text, font, 0.85, 2)
            wx = (MAP_WIDTH - tw) // 2
            wy = MAP_HEIGHT - 25

            # Glow background
            overlay = canvas.copy()
            cv2.rectangle(overlay,
                          (wx - 15, wy - th - 12),
                          (wx + tw + 15, wy + 12),
                          COLOR_WARNING_GLOW, cv2.FILLED)
            cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)

            cv2.putText(canvas, warn_text, (wx, wy),
                        font, 0.85, warn_color, 2, cv2.LINE_AA)

        # ── Bottom-left: Title ──
        cv2.putText(canvas, "DRONE NAVIGATION SYSTEM v2",
                    (10, MAP_HEIGHT - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    COLOR_DIM_WHITE, 1, cv2.LINE_AA)
