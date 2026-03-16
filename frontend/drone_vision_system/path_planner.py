"""
path_planner.py — A* grid-based path planner with obstacle avoidance.

Maintains an occupancy grid over the 2D map. When obstacles from YOLO
detections are projected onto the map, cells are marked as blocked.
The planner computes an optimal path using A* and supports dynamic
replanning when obstacles block the current route.
"""

import math
import heapq
import numpy as np

# ─── Grid config (imported by others) ────────────────────────────────
PATH_CELL_SIZE = 20          # pixels per grid cell
OBSTACLE_INFLATE = 2         # extra cells around each obstacle


class PathPlanner:
    """A* path planner on a 2D occupancy grid."""

    def __init__(self, map_w, map_h, cell_size=PATH_CELL_SIZE):
        self.map_w = map_w
        self.map_h = map_h
        self.cell_size = cell_size
        self.cols = map_w // cell_size
        self.rows = map_h // cell_size

        # occupancy: 0 = free, 1 = blocked
        self.grid = np.zeros((self.rows, self.cols), dtype=np.uint8)

        # Current planned path as list of (px, py) map-pixel coords
        self.path = []
        self.path_grid = []          # path as grid (row, col) list
        self.source = None           # (px, py)
        self.destination = None      # (px, py)
        self.is_rerouted = False     # True after an obstacle-triggered replan
        self.reroute_count = 0

    # ── coordinate helpers ────────────────────────────────────────────
    def _px_to_cell(self, px, py):
        """Convert pixel coords to grid (row, col)."""
        c = max(0, min(self.cols - 1, int(px) // self.cell_size))
        r = max(0, min(self.rows - 1, int(py) // self.cell_size))
        return r, c

    def _cell_to_px(self, r, c):
        """Convert grid (row, col) to pixel center."""
        px = c * self.cell_size + self.cell_size // 2
        py = r * self.cell_size + self.cell_size // 2
        return px, py

    # ── obstacle management ───────────────────────────────────────────
    def clear_obstacles(self):
        """Reset the occupancy grid."""
        self.grid.fill(0)

    def mark_obstacle(self, px, py, inflate=OBSTACLE_INFLATE):
        """Mark a map-pixel location (and inflated neighbours) as blocked."""
        r0, c0 = self._px_to_cell(px, py)
        for dr in range(-inflate, inflate + 1):
            for dc in range(-inflate, inflate + 1):
                nr = r0 + dr
                nc = c0 + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    self.grid[nr][nc] = 1

    def update_obstacles_from_detections(self, detections, frame_w, frame_h,
                                          drone_x, drone_y, world_scale,
                                          bbox_to_world_fn):
        """Clear and rebuild obstacle grid from current YOLO detections."""
        self.clear_obstacles()
        for det in detections:
            cx, cy = det["center"]
            wx, wy = bbox_to_world_fn(cx, cy, frame_w, frame_h,
                                       drone_x, drone_y, world_scale)
            mx = int(wx) % self.map_w
            my = int(wy) % self.map_h
            self.mark_obstacle(mx, my)

    # ── path checking ─────────────────────────────────────────────────
    def is_path_blocked(self):
        """Check if any cell on the current planned path is now blocked."""
        for r, c in self.path_grid:
            if 0 <= r < self.rows and 0 <= c < self.cols:
                if self.grid[r][c] == 1:
                    return True
        return False

    # ── A* planner ────────────────────────────────────────────────────
    def plan(self, source_px=None, dest_px=None):
        """
        Run A* from source to destination.
        Updates self.path (pixel waypoints) and self.path_grid.
        Returns True if a valid path was found.
        """
        if source_px is not None:
            self.source = source_px
        if dest_px is not None:
            self.destination = dest_px

        if self.source is None or self.destination is None:
            self.path = []
            self.path_grid = []
            return False

        sr, sc = self._px_to_cell(*self.source)
        dr, dc = self._px_to_cell(*self.destination)

        # If source or dest is on a blocked cell, still try (we'll find
        # the nearest reachable cell)
        path_cells = self._astar(sr, sc, dr, dc)
        if path_cells is None:
            # No valid path — keep old path or empty
            self.path = []
            self.path_grid = []
            return False

        self.path_grid = path_cells
        # Convert grid cells to pixel waypoints
        self.path = [self._cell_to_px(r, c) for r, c in path_cells]
        return True

    def replan(self, drone_px):
        """Replan from the drone's current position to the destination."""
        self.is_rerouted = True
        self.reroute_count += 1
        return self.plan(source_px=drone_px)

    def _astar(self, sr, sc, dr, dc):
        """A* search returning list of (row, col) or None."""
        if sr == dr and sc == dc:
            return [(sr, sc)]

        # 8-directional movement
        DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)]
        COST_STRAIGHT = 1.0
        COST_DIAG = 1.414

        open_set = []
        heapq.heappush(open_set, (0.0, sr, sc))
        came_from = {}
        g_score = {(sr, sc): 0.0}

        def heuristic(r, c):
            return math.sqrt((r - dr) ** 2 + (c - dc) ** 2)

        visited = set()

        while open_set:
            _, cr, cc = heapq.heappop(open_set)

            if (cr, cc) in visited:
                continue
            visited.add((cr, cc))

            if cr == dr and cc == dc:
                # Reconstruct path
                path = [(cr, cc)]
                while (cr, cc) in came_from:
                    cr, cc = came_from[(cr, cc)]
                    path.append((cr, cc))
                path.reverse()
                return path

            for ddr, ddc in DIRS:
                nr, nc = cr + ddr, cc + ddc
                if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
                    continue
                if (nr, nc) in visited:
                    continue
                if self.grid[nr][nc] == 1:
                    continue

                move_cost = COST_DIAG if (ddr != 0 and ddc != 0) else COST_STRAIGHT
                tentative_g = g_score[(cr, cc)] + move_cost

                if tentative_g < g_score.get((nr, nc), float('inf')):
                    g_score[(nr, nc)] = tentative_g
                    f = tentative_g + heuristic(nr, nc)
                    came_from[(nr, nc)] = (cr, cc)
                    heapq.heappush(open_set, (f, nr, nc))

        return None  # no path found

    # ── query helpers ─────────────────────────────────────────────────
    def get_next_waypoint(self, drone_px, advance_dist=15):
        """
        Return the next waypoint the drone should steer toward.
        Automatically advances past waypoints that are close enough.
        Returns (px, py) or None if path is empty / arrived.
        """
        if not self.path:
            return None

        dx, dy = drone_px
        while self.path:
            wx, wy = self.path[0]
            dist = math.sqrt((wx - dx) ** 2 + (wy - dy) ** 2)
            if dist < advance_dist:
                self.path.pop(0)
                if self.path_grid:
                    self.path_grid.pop(0)
            else:
                return self.path[0]

        return None  # arrived

    def has_arrived(self, drone_px, threshold=25):
        """Check if drone is close enough to the destination."""
        if self.destination is None:
            return False
        dx = drone_px[0] - self.destination[0]
        dy = drone_px[1] - self.destination[1]
        return math.sqrt(dx * dx + dy * dy) < threshold

    def get_blocked_cells_px(self):
        """Yield (px, py, cell_size) for every blocked grid cell."""
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == 1:
                    px = c * self.cell_size
                    py = r * self.cell_size
                    yield px, py, self.cell_size
