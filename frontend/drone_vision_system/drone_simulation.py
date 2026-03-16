"""
drone_simulation.py — 2D drone map simulator with A* path planning,
source-to-destination navigation, and obstacle-based rerouting.
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
    COLOR_PATH_PLANNED, COLOR_PATH_REROUTED,
    COLOR_SOURCE, COLOR_DESTINATION, COLOR_OBSTACLE_ZONE, COLOR_PATH_ARRIVED,
    GRID_LABELS,
    bbox_to_world, get_grid_cell, get_class_color,
    draw_text_with_bg, draw_glow_circle, draw_rounded_rect,
    draw_hud_panel, draw_scanlines, create_gradient_bg,
)
from path_planner import PathPlanner


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
    """Drone with A* path-following from source to destination."""

    # Default source (bottom-left) and destination (top-right)
    DEFAULT_SRC = (80, MAP_HEIGHT - 60)
    DEFAULT_DST = (MAP_WIDTH - 80, 60)

    def __init__(self):
        self.source = self.DEFAULT_SRC
        self.destination = self.DEFAULT_DST

        self.x = float(self.source[0])
        self.y = float(self.source[1])
        # Point initial heading toward destination
        self.heading = math.atan2(
            self.destination[1] - self.source[1],
            self.destination[0] - self.source[0]
        )
        self.speed = DRONE_SPEED
        self.trail = [(self.x, self.y)]   # actual movement trail
        self.steer_cmd = "CLEAR"
        self.auto_mode = True
        self.altitude = 50.0
        self.frame_count = 0

        # Path planner
        self.planner = PathPlanner(MAP_WIDTH, MAP_HEIGHT)
        success = self.planner.plan(source_px=self.source, dest_px=self.destination)
        self.nav_status = "NAVIGATING"   # NAVIGATING | REROUTING | ARRIVED | NO_PATH
        self._reroute_flash = 0          # countdown for reroute visual

        # Keep a FULL copy of the planned path for display (won't be consumed)
        self.original_path = list(self.planner.path)
        # Also keep current full planned path for rendering
        self.full_planned_path = list(self.planner.path)

        print(f"[INFO] Path planned: {len(self.planner.path)} waypoints, success={success}")
        print(f"[INFO] Source: {self.source}  Destination: {self.destination}")

    # ── state update ──────────────────────────────────────────────────
    def update(self, avoidance_steer: float = 0.0,
               manual_steer: float = 0.0,
               detections=None, frame_w=0, frame_h=0):
        """Advance drone one tick — follows A* waypoints."""

        # Update obstacle grid from detections
        if detections and frame_w > 0:
            self.planner.update_obstacles_from_detections(
                detections, frame_w, frame_h,
                self.x, self.y, WORLD_SCALE, bbox_to_world
            )

            # Check if current path is blocked → reroute
            if self.planner.path and self.planner.is_path_blocked():
                drone_pos = (int(self.x), int(self.y))
                success = self.planner.replan(drone_pos)
                if success:
                    self.full_planned_path = list(self.planner.path)
                    self.nav_status = "REROUTING"
                    self._reroute_flash = 60   # show alert for ~2s
                else:
                    self.nav_status = "NO_PATH"

        # Check arrival
        if self.planner.has_arrived((self.x, self.y)):
            self.nav_status = "ARRIVED"
            self.frame_count += 1
            return

        # Get next waypoint and steer toward it
        wp = self.planner.get_next_waypoint((self.x, self.y))
        if wp is None:
            if self.nav_status != "ARRIVED":
                self.nav_status = "ARRIVED"
            self.frame_count += 1
            return

        # Calculate desired heading toward waypoint
        target_heading = math.atan2(wp[1] - self.y, wp[0] - self.x)

        # Smoothly rotate toward target heading
        angle_diff = target_heading - self.heading
        # Normalize to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        turn_rate = 0.20
        if abs(angle_diff) < turn_rate:
            self.heading = target_heading
        else:
            self.heading += turn_rate if angle_diff > 0 else -turn_rate

        # Move forward
        dx = math.cos(self.heading) * self.speed
        dy = math.sin(self.heading) * self.speed
        self.x += dx
        self.y += dy

        # Clamp to map bounds
        self.x = max(0, min(MAP_WIDTH - 1, self.x))
        self.y = max(0, min(MAP_HEIGHT - 1, self.y))

        self.trail.append((self.x, self.y))
        if len(self.trail) > 2000:
            self.trail = self.trail[-2000:]

        if self._reroute_flash > 0:
            self._reroute_flash -= 1
            if self._reroute_flash == 0 and self.nav_status == "REROUTING":
                self.nav_status = "NAVIGATING"

        self.frame_count += 1

    def reset(self):
        self.x = float(self.source[0])
        self.y = float(self.source[1])
        self.heading = math.atan2(
            self.destination[1] - self.source[1],
            self.destination[0] - self.source[0]
        )
        self.speed = DRONE_SPEED
        self.trail = [(self.x, self.y)]
        self.steer_cmd = "CLEAR"
        self.altitude = 50.0
        self.nav_status = "NAVIGATING"
        self._reroute_flash = 0
        self.planner.clear_obstacles()
        self.planner.is_rerouted = False
        self.planner.plan(source_px=self.source, dest_px=self.destination)
        self.original_path = list(self.planner.path)
        self.full_planned_path = list(self.planner.path)
        print(f"[INFO] Reset: {len(self.planner.path)} waypoints")

    def set_source(self, px, py):
        self.source = (int(px), int(py))
        self.x = float(self.source[0])
        self.y = float(self.source[1])
        self.heading = math.atan2(
            self.destination[1] - self.source[1],
            self.destination[0] - self.source[0]
        )
        self.trail = [(self.x, self.y)]
        self.nav_status = "NAVIGATING"
        self.planner.is_rerouted = False
        self.planner.plan(source_px=self.source, dest_px=self.destination)
        self.original_path = list(self.planner.path)
        self.full_planned_path = list(self.planner.path)

    def set_destination(self, px, py):
        self.destination = (int(px), int(py))
        self.nav_status = "NAVIGATING"
        self.planner.is_rerouted = False
        drone_pos = (int(self.x), int(self.y))
        self.planner.plan(source_px=drone_pos, dest_px=self.destination)
        self.original_path = list(self.planner.path)
        self.full_planned_path = list(self.planner.path)

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

        canvas = _get_gradient_bg()
        self._draw_bg_grid(canvas)

        if grid_status is not None:
            self._draw_avoidance_grid(canvas, grid_status)

        # Draw obstacle zones
        self._draw_obstacle_zones(canvas)

        # Draw planned path(s)
        self._draw_planned_path(canvas)

        # Draw detected objects
        self._draw_objects(canvas, detections, frame_w, frame_h)

        # Draw drone trail
        self._draw_trail(canvas)

        # Draw source and destination markers
        self._draw_source_dest(canvas)

        # Draw drone
        self._draw_drone(canvas)

        # HUD
        self._draw_hud(canvas, detections, grid_status)

        # Reroute alert
        if self._reroute_flash > 0:
            self._draw_reroute_alert(canvas)

        # Arrival celebration
        if self.nav_status == "ARRIVED":
            self._draw_arrived_alert(canvas)

        draw_scanlines(canvas, spacing=3, alpha=0.04)
        return canvas

    # ·· background grid ···············································
    def _draw_bg_grid(self, canvas):
        for x in range(0, MAP_WIDTH, 40):
            cv2.line(canvas, (x, 0), (x, MAP_HEIGHT), COLOR_GRID_LINE, 1)
        for y in range(0, MAP_HEIGHT, 40):
            cv2.line(canvas, (0, y), (MAP_WIDTH, y), COLOR_GRID_LINE, 1)
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
                cv2.rectangle(canvas, (x1, y1), (x2, y2),
                              (50, 45, 40), 1)
                label = GRID_LABELS.get((r, c), "")
                cv2.putText(canvas, label, (x1 + 6, y1 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                            COLOR_DIM_WHITE, 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0, canvas)

    # ·· obstacle zones (blocked cells) ·································
    def _draw_obstacle_zones(self, canvas):
        overlay = canvas.copy()
        for px, py, cs in self.planner.get_blocked_cells_px():
            cv2.rectangle(overlay, (px, py), (px + cs, py + cs),
                          COLOR_OBSTACLE_ZONE, cv2.FILLED)
        cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

    # ·· planned path ···················································
    def _draw_planned_path(self, canvas):
        # Draw original path (dimmed) if rerouted
        if self.planner.is_rerouted and len(self.original_path) > 1:
            for i in range(1, len(self.original_path)):
                pt1 = (int(self.original_path[i-1][0]),
                        int(self.original_path[i-1][1]))
                pt2 = (int(self.original_path[i][0]),
                        int(self.original_path[i][1]))
                if (abs(pt1[0] - pt2[0]) < MAP_WIDTH // 2 and
                        abs(pt1[1] - pt2[1]) < MAP_HEIGHT // 2):
                    # Dashed effect — draw every other segment
                    if i % 3 != 0:
                        cv2.line(canvas, pt1, pt2, (60, 50, 40), 1,
                                 cv2.LINE_AA)

        # Draw the FULL planned path (not the consumed remaining path)
        path = self.full_planned_path
        if len(path) < 2:
            return

        path_color = (COLOR_PATH_REROUTED if self.planner.is_rerouted
                       else COLOR_PATH_PLANNED)

        # Animated dashes — creates a flowing animation effect
        dash_offset = self.frame_count % 12

        for i in range(1, len(path)):
            pt1 = (int(path[i-1][0]), int(path[i-1][1]))
            pt2 = (int(path[i][0]), int(path[i][1]))
            if (abs(pt1[0] - pt2[0]) < MAP_WIDTH // 2 and
                    abs(pt1[1] - pt2[1]) < MAP_HEIGHT // 2):
                # Animated dash pattern
                if (i + dash_offset) % 4 != 0:
                    cv2.line(canvas, pt1, pt2, path_color, 2, cv2.LINE_AA)

        # Glow effect on path (subtle)
        if len(path) > 2:
            overlay = canvas.copy()
            for i in range(1, len(path)):
                pt1 = (int(path[i-1][0]), int(path[i-1][1]))
                pt2 = (int(path[i][0]), int(path[i][1]))
                if (abs(pt1[0] - pt2[0]) < MAP_WIDTH // 2 and
                        abs(pt1[1] - pt2[1]) < MAP_HEIGHT // 2):
                    cv2.line(overlay, pt1, pt2, path_color, 5, cv2.LINE_AA)
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

            draw_glow_circle(canvas, (mx, my), 8, color, intensity=0.35)
            cv2.circle(canvas, (mx, my), 10, COLOR_WHITE, 1, cv2.LINE_AA)

            obj_label = f'{det["class_name"]} {dist:.0f}m'
            cv2.putText(canvas, obj_label, (mx + 14, my + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        color, 1, cv2.LINE_AA)

            # Danger line for close objects
            ddx = mx - self.x
            ddy = my - self.y
            d = math.sqrt(ddx * ddx + ddy * ddy)
            if d < OBSTACLE_WARN_DIST:
                pulse = 0.5 + 0.5 * math.sin(self.frame_count * 0.15)
                line_color = (int(COLOR_WARNING[0] * pulse),
                              int(COLOR_WARNING[1] * pulse),
                              int(min(255, COLOR_WARNING[2] * (0.5 + pulse * 0.5))))
                cv2.line(canvas, (int(self.x), int(self.y)),
                         (mx, my), line_color, 2, cv2.LINE_AA)

    # ·· drone trail (actual movement path) ·····························
    def _draw_trail(self, canvas):
        if len(self.trail) < 2:
            return
        n = len(self.trail)
        for i in range(1, n):
            alpha = i / n
            color = (int(COLOR_DRONE_PATH2[0] + (COLOR_DRONE_PATH[0] - COLOR_DRONE_PATH2[0]) * alpha),
                     int(COLOR_DRONE_PATH2[1] + (COLOR_DRONE_PATH[1] - COLOR_DRONE_PATH2[1]) * alpha),
                     int(COLOR_DRONE_PATH2[2] + (COLOR_DRONE_PATH[2] - COLOR_DRONE_PATH2[2]) * alpha))
            pt1 = (int(self.trail[i-1][0]), int(self.trail[i-1][1]))
            pt2 = (int(self.trail[i][0]), int(self.trail[i][1]))
            if abs(pt1[0] - pt2[0]) < MAP_WIDTH // 2 and abs(pt1[1] - pt2[1]) < MAP_HEIGHT // 2:
                thickness = max(1, int(alpha * 2))
                cv2.line(canvas, pt1, pt2, color, thickness, cv2.LINE_AA)

    # ·· source & destination markers ···································
    def _draw_source_dest(self, canvas):
        pulse = 0.5 + 0.5 * math.sin(self.frame_count * 0.1)

        # Source marker — pulsing green diamond
        sx, sy = int(self.source[0]), int(self.source[1])
        r = int(12 + 3 * pulse)
        draw_glow_circle(canvas, (sx, sy), r, COLOR_SOURCE, intensity=0.3 * pulse)
        # Diamond shape
        src_pts = np.array([
            [sx, sy - r], [sx + r, sy], [sx, sy + r], [sx - r, sy]
        ], dtype=np.int32)
        cv2.polylines(canvas, [src_pts], True, COLOR_SOURCE, 2, cv2.LINE_AA)
        cv2.putText(canvas, "SRC", (sx + r + 4, sy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_SOURCE, 1, cv2.LINE_AA)

        # Destination marker — pulsing magenta star/circle
        dx, dy = int(self.destination[0]), int(self.destination[1])
        r2 = int(14 + 4 * pulse)
        draw_glow_circle(canvas, (dx, dy), r2, COLOR_DESTINATION,
                         intensity=0.35 * pulse)
        # Crosshair
        cv2.line(canvas, (dx - r2, dy), (dx + r2, dy),
                 COLOR_DESTINATION, 1, cv2.LINE_AA)
        cv2.line(canvas, (dx, dy - r2), (dx, dy + r2),
                 COLOR_DESTINATION, 1, cv2.LINE_AA)
        cv2.circle(canvas, (dx, dy), r2, COLOR_DESTINATION, 2, cv2.LINE_AA)
        cv2.circle(canvas, (dx, dy), r2 // 2, COLOR_DESTINATION, 1, cv2.LINE_AA)
        cv2.putText(canvas, "DST", (dx + r2 + 4, dy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_DESTINATION, 1,
                    cv2.LINE_AA)

    # ·· drone icon ····················································
    def _draw_drone(self, canvas):
        pulse = 0.6 + 0.4 * math.sin(self.frame_count * 0.08)
        glow_r = int(22 + 4 * pulse)
        draw_glow_circle(canvas, (int(self.x), int(self.y)),
                         glow_r, COLOR_DRONE_GLOW, intensity=0.15 * pulse)

        size = 16
        angle = self.heading
        pts = []
        for i, a in enumerate([0, 2.3, -2.3]):
            r = size if i == 0 else size * 0.55
            px = int(self.x + math.cos(angle + a) * r)
            py = int(self.y + math.sin(angle + a) * r)
            pts.append([px, py])
        pts_np = np.array(pts, dtype=np.int32)
        cv2.fillPoly(canvas, [pts_np], COLOR_DRONE_BODY)
        cv2.polylines(canvas, [pts_np], True, COLOR_WHITE, 1, cv2.LINE_AA)

        cv2.circle(canvas, (int(self.x), int(self.y)), 2,
                   COLOR_WHITE, cv2.FILLED)

        hx = int(self.x + math.cos(self.heading) * (size + 8))
        hy = int(self.y + math.sin(self.heading) * (size + 8))
        cv2.line(canvas, (int(self.x), int(self.y)), (hx, hy),
                 COLOR_CYAN, 1, cv2.LINE_AA)

    # ·· HUD overlay ··················································
    def _draw_hud(self, canvas, detections, grid_status):
        # ── Left panel: status ──
        draw_hud_panel(canvas, 8, 8, 240, 175,
                       title="DRONE STATUS", border_color=COLOR_CYAN)

        info_lines = [
            (f"POS   X:{self.x:.0f}  Y:{self.y:.0f}", COLOR_HUD_TEXT),
            (f"HDG   {math.degrees(self.heading):.0f} deg", COLOR_HUD_TEXT),
            (f"SPD   {self.speed:.1f} u/t", COLOR_AMBER),
            (f"ALT   {self.altitude:.0f} m", COLOR_HUD_TEXT),
            (f"MODE  {'AUTO' if self.auto_mode else 'MANUAL'}",
             COLOR_GREEN if self.auto_mode else COLOR_MAGENTA),
            (f"SRC   ({self.source[0]},{self.source[1]})", COLOR_SOURCE),
            (f"DST   ({self.destination[0]},{self.destination[1]})",
             COLOR_DESTINATION),
        ]
        for i, (text, color) in enumerate(info_lines):
            cv2.putText(canvas, text, (18, 40 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                        color, 1, cv2.LINE_AA)

        # ── Right panel: detections & reroute count ──
        det_count = len(detections)
        draw_hud_panel(canvas, MAP_WIDTH - 210, 8, 200, 50,
                       title="", border_color=COLOR_AMBER)
        cv2.putText(canvas,
                    f"DETECTIONS: {det_count}",
                    (MAP_WIDTH - 200, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    COLOR_AMBER, 1, cv2.LINE_AA)
        cv2.putText(canvas,
                    f"REROUTES: {self.planner.reroute_count}",
                    (MAP_WIDTH - 200, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    COLOR_PATH_REROUTED if self.planner.reroute_count > 0
                    else COLOR_DIM_WHITE, 1, cv2.LINE_AA)

        # ── Navigation status (center-top) ──
        status_colors = {
            "NAVIGATING": COLOR_GREEN,
            "REROUTING": COLOR_PATH_REROUTED,
            "ARRIVED": COLOR_PATH_ARRIVED,
            "NO_PATH": COLOR_RED,
        }
        status_color = status_colors.get(self.nav_status, COLOR_GREEN)
        status_text = f"NAV: {self.nav_status}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, _), _ = cv2.getTextSize(status_text, font, 0.55, 1)
        cx = (MAP_WIDTH - tw) // 2
        draw_text_with_bg(canvas, status_text, (cx, 28),
                          font_scale=0.55, color=status_color,
                          border_color=status_color)

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

            overlay = canvas.copy()
            cv2.rectangle(overlay,
                          (wx - 15, wy - th - 12),
                          (wx + tw + 15, wy + 12),
                          COLOR_WARNING_GLOW, cv2.FILLED)
            cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)
            cv2.putText(canvas, warn_text, (wx, wy),
                        font, 0.85, warn_color, 2, cv2.LINE_AA)

        # ── Bottom-left: Title ──
        cv2.putText(canvas, "DRONE NAVIGATION SYSTEM v3 - PATH PLANNING",
                    (10, MAP_HEIGHT - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    COLOR_DIM_WHITE, 1, cv2.LINE_AA)

    # ·· reroute flash alert ···········································
    def _draw_reroute_alert(self, canvas):
        pulse = 0.5 + 0.5 * math.sin(self.frame_count * 0.25)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = ">> PATH REROUTED <<"
        (tw, th), _ = cv2.getTextSize(text, font, 0.7, 2)
        x = (MAP_WIDTH - tw) // 2
        y = MAP_HEIGHT // 2

        # Glowing background
        overlay = canvas.copy()
        cv2.rectangle(overlay,
                      (x - 20, y - th - 15),
                      (x + tw + 20, y + 15),
                      (20, 60, 120), cv2.FILLED)
        cv2.addWeighted(overlay, 0.5 * pulse, canvas,
                        1 - 0.5 * pulse, 0, canvas)

        color = (int(30 * (1 - pulse) + 30),
                 int(140 * pulse),
                 int(255 * pulse))
        cv2.putText(canvas, text, (x, y),
                    font, 0.7, color, 2, cv2.LINE_AA)

    # ·· arrival alert ·················································
    def _draw_arrived_alert(self, canvas):
        pulse = 0.5 + 0.5 * math.sin(self.frame_count * 0.12)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "DESTINATION REACHED"
        (tw, th), _ = cv2.getTextSize(text, font, 0.65, 2)
        x = (MAP_WIDTH - tw) // 2
        y = MAP_HEIGHT // 2 - 20

        overlay = canvas.copy()
        cv2.rectangle(overlay,
                      (x - 20, y - th - 15),
                      (x + tw + 20, y + 15),
                      (40, 80, 30), cv2.FILLED)
        cv2.addWeighted(overlay, 0.4 * pulse, canvas,
                        1 - 0.4 * pulse, 0, canvas)

        cv2.putText(canvas, text, (x, y),
                    font, 0.65, COLOR_PATH_ARRIVED, 2, cv2.LINE_AA)

        # Sub-text
        sub = f"Reroutes: {self.planner.reroute_count}"
        (tw2, _), _ = cv2.getTextSize(sub, font, 0.45, 1)
        cv2.putText(canvas, sub, ((MAP_WIDTH - tw2) // 2, y + 25),
                    font, 0.45, COLOR_DIM_WHITE, 1, cv2.LINE_AA)
