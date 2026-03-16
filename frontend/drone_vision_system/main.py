"""
main.py — Entry point for the Drone Vision System v3.

Combines live YOLO11 camera detection (bottom half) with a 2D
drone navigation map (top half) in a single OpenCV window.

The drone navigates from a source to a destination using A* path
planning. When obstacles are detected in the planned path, the
route dynamically reroutes around them.

Controls:
    Q        — Quit
    P        — Pause / resume
    R        — Reset drone to source & replan
    G        — Toggle grid overlay
    D        — Toggle debug info
    UP/DOWN  — Increase/decrease speed
    Left-click on map  — Set new SOURCE
    Right-click on map — Set new DESTINATION
"""

import time
import cv2
import numpy as np

from utils import (
    DISPLAY_WIDTH, DISPLAY_HEIGHT, MAP_HEIGHT, CAM_HEIGHT, CAM_WIDTH,
    PROC_WIDTH, PROC_HEIGHT,
    COLOR_GREEN, COLOR_RED, COLOR_CYAN, COLOR_AMBER,
    COLOR_MAGENTA, COLOR_DIM_WHITE, COLOR_HUD_TEXT, COLOR_WHITE,
    COLOR_HUD_BG, COLOR_HUD_BORDER,
    COLOR_SOURCE, COLOR_DESTINATION,
    draw_text_with_bg, draw_grid_hud, draw_scanlines,
    draw_hud_panel, draw_controls_help,
)
from object_detection import ObjectDetector
from drone_simulation import DroneSimulator, GridAvoidance


# ── Mouse state ──────────────────────────────────────────────────────
_mouse_event = None    # will be set by callback


def _mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks on the map portion (top half)."""
    global _mouse_event
    if y > MAP_HEIGHT:
        return  # clicked on camera half, ignore

    if event == cv2.EVENT_LBUTTONDOWN:
        _mouse_event = ("source", x, y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        _mouse_event = ("dest", x, y)


def main():
    global _mouse_event

    # ── Camera ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check your webcam connection.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, PROC_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PROC_HEIGHT)

    # ── Modules ───────────────────────────────────────────────────────
    detector = ObjectDetector()
    drone = DroneSimulator()
    grid_avoidance = GridAvoidance()

    paused = False
    show_grid = True
    show_debug = False
    prev_time = time.time()
    frame_count = 0
    fps_smooth = 30.0

    print("[INFO] Drone Vision System v3 started. Press Q to quit.")
    print("[INFO] Controls: Q=Quit P=Pause R=Reset G=Grid D=Debug")
    print("[INFO]           Left-click=Set Source  Right-click=Set Destination")
    print("[INFO]           Arrow Keys=Speed +/-")

    # Create window and attach mouse callback
    win_name = "Drone Vision System v3"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, DISPLAY_WIDTH, DISPLAY_HEIGHT)
    cv2.setMouseCallback(win_name, _mouse_callback)

    # ── Main loop ─────────────────────────────────────────────────────
    while True:
        ret, raw_frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame.")
            break

        frame = cv2.resize(raw_frame, (PROC_WIDTH, PROC_HEIGHT))
        frame_count += 1

        # ── Object detection ──
        detections = detector.detect(frame)

        # ── Grid avoidance ──
        grid_status, steer_cmd, steer_val = grid_avoidance.analyze(
            detections, PROC_WIDTH, PROC_HEIGHT
        )

        # ── Draw detections on camera frame ──
        ObjectDetector.draw_detections(frame, detections)

        # ── Grid HUD on camera feed (top-right) ──
        if show_grid:
            hud_x = PROC_WIDTH - 115
            draw_grid_hud(frame, grid_status, steer_cmd,
                          x_offset=hud_x, y_offset=8, cell_size=30)

        # ── Controls help (bottom-left of camera) ──
        draw_controls_help(frame, 8, PROC_HEIGHT - 200)

        # ── Handle mouse events ──
        if _mouse_event is not None:
            action, mx, my = _mouse_event
            # The map is displayed at DISPLAY_WIDTH × MAP_HEIGHT but the
            # drone map coordinates are MAP_WIDTH × MAP_HEIGHT.
            # Since MAP_WIDTH == DISPLAY_WIDTH (1024), coordinates match.
            if action == "source":
                drone.set_source(mx, my)
                print(f"[INFO] Source set to ({mx}, {my})")
            elif action == "dest":
                drone.set_destination(mx, my)
                print(f"[INFO] Destination set to ({mx}, {my})")
            _mouse_event = None

        # ── Update drone (pass detections for obstacle rerouting) ──
        if not paused:
            drone.update(
                avoidance_steer=steer_val,
                detections=detections,
                frame_w=PROC_WIDTH,
                frame_h=PROC_HEIGHT,
            )

        # ── Render map ──
        map_frame = drone.render(
            detections, PROC_WIDTH, PROC_HEIGHT,
            grid_status if show_grid else None,
            steer_cmd
        )

        # ── Resize for display ──
        cam_display = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))
        map_display = cv2.resize(map_frame, (DISPLAY_WIDTH, MAP_HEIGHT))

        # ── Separator line between halves ──
        cv2.line(map_display,
                 (0, MAP_HEIGHT - 1), (DISPLAY_WIDTH, MAP_HEIGHT - 1),
                 COLOR_CYAN, 2)

        # ── FPS counter ──
        now = time.time()
        dt = max(now - prev_time, 1e-6)
        fps = 1.0 / dt
        fps_smooth = fps_smooth * 0.9 + fps * 0.1
        prev_time = now

        fps_color = (COLOR_GREEN if fps_smooth >= 20
                     else COLOR_AMBER if fps_smooth >= 10 else COLOR_RED)
        draw_text_with_bg(cam_display, f"FPS: {fps_smooth:.0f}",
                          (10, 28), font_scale=0.50, color=fps_color,
                          border_color=fps_color)

        # Navigation status indicator on camera
        nav_color = (COLOR_GREEN if drone.nav_status == "NAVIGATING"
                     else COLOR_RED if drone.nav_status == "NO_PATH"
                     else COLOR_AMBER if drone.nav_status == "REROUTING"
                     else COLOR_CYAN)
        draw_text_with_bg(cam_display, drone.nav_status,
                          (CAM_WIDTH // 2 - 50, 28),
                          font_scale=0.50, color=nav_color,
                          border_color=nav_color)

        # Speed indicator
        draw_text_with_bg(cam_display,
                          f"SPD: {drone.speed:.1f}",
                          (CAM_WIDTH // 2 + 60, 28),
                          font_scale=0.45, color=COLOR_AMBER)

        # Pause indicator
        if paused:
            pause_text = "|| PAUSED ||"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, _), _ = cv2.getTextSize(pause_text, font, 0.7, 2)
            px = (CAM_WIDTH - tw) // 2
            draw_text_with_bg(cam_display, pause_text,
                              (px, CAM_HEIGHT // 2),
                              font_scale=0.7, color=COLOR_RED,
                              border_color=COLOR_RED, thickness=2)

        # Debug info
        if show_debug:
            debug_lines = [
                f"Frame: {frame_count}",
                f"Detections: {len(detections)}",
                f"Drone: ({drone.x:.0f}, {drone.y:.0f})",
                f"Heading: {drone.heading:.2f} rad",
                f"Grid cmd: {steer_cmd}",
                f"Nav: {drone.nav_status}",
                f"Path pts: {len(drone.planner.path)}",
                f"Reroutes: {drone.planner.reroute_count}",
            ]
            for i, line in enumerate(debug_lines):
                cv2.putText(cam_display, line,
                            (CAM_WIDTH - 250, 30 + i * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            COLOR_DIM_WHITE, 1, cv2.LINE_AA)

        # Light scanlines on camera feed
        draw_scanlines(cam_display, spacing=4, alpha=0.03)

        # ── Combine halves ──
        combined = np.vstack([map_display, cam_display])

        cv2.imshow(win_name, combined)

        # ── Keyboard ──
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == ord("Q"):
            break
        elif key == ord("p") or key == ord("P"):
            paused = not paused
        elif key == ord("r") or key == ord("R"):
            drone.reset()
            print("[INFO] Drone reset to source. Path replanned.")
        elif key == ord("g") or key == ord("G"):
            show_grid = not show_grid
        elif key == ord("d") or key == ord("D"):
            show_debug = not show_debug
        # Arrow keys
        elif key == 0:
            pass
        elif key == 82 or key == 72:  # UP arrow
            drone.change_speed(0.2)
        elif key == 84 or key == 80:  # DOWN arrow
            drone.change_speed(-0.2)
        elif key == ord("+") or key == ord("="):
            drone.change_speed(0.2)
        elif key == ord("-") or key == ord("_"):
            drone.change_speed(-0.2)

    # ── Cleanup ───────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Drone Vision System stopped.")


if __name__ == "__main__":
    main()
