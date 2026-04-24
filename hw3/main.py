import math
import random
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np

from map_processor import (
    load_and_filter_map,
    select_start,
    get_goal_pixels,
    pixel_to_world,
)
from navigator import init_sim, execute_waypoint_path


POINT_CLOUD_DATA = "semantic_3d_pointcloud/point.npy"
COLOR_DATA = "semantic_3d_pointcloud/color0255.npy"

# Semantic colour and index dictionaries for five required target categories.
# Colours come from the 101-category colour map; indices from info_semantic.json.
SEMANTIC_DICTS = {
    "colors": {
        "rack": [[0, 255, 133]],
        "cooktop": [[7, 255, 224]],
        "sofa": [[10, 0, 255]],
        "cushion": [[255, 5, 153]],
        "stair": [[255, 31, 0]],
    },
    "indices": {
        "rack": 8,
        "cooktop": 280,
        "sofa": 196,
        "cushion": 268,
        "stair": 30,
    },
}


# =====================================================================
# RRT Planner
# =====================================================================

class RRTNode:
    """A single node in the RRT tree."""
    __slots__ = ("x", "y", "parent")

    def __init__(self, x: int, y: int, parent: Optional["RRTNode"] = None):
        self.x = x          # pixel column
        self.y = y          # pixel row
        self.parent = parent


def _is_free(occupancy_map: np.ndarray, x: int, y: int) -> bool:
    """Return True if pixel (col=x, row=y) is free (not an obstacle)."""
    h, w = occupancy_map.shape[:2]
    if x < 0 or x >= w or y < 0 or y >= h:
        return False
    return occupancy_map[y, x] == 0


def _line_collision_free(occupancy_map: np.ndarray,
                         x0: int, y0: int, x1: int, y1: int) -> bool:
    """Check if the straight line between two pixels is entirely in free space.

    Uses Bresenham-style sampling along the line.
    """
    dist = max(abs(x1 - x0), abs(y1 - y0))
    if dist == 0:
        return _is_free(occupancy_map, x0, y0)
    for i in range(dist + 1):
        t = i / dist
        cx = int(round(x0 + t * (x1 - x0)))
        cy = int(round(y0 + t * (y1 - y0)))
        if not _is_free(occupancy_map, cx, cy):
            return False
    return True


def plan_path(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    occupancy_map: np.ndarray,
    max_iter: int = 8000,
    step_size: int = 15,
    goal_bias: float = 0.15,
    goal_tolerance: int = 15,
) -> Optional[List[Tuple[int, int]]]:
    """Plan a collision-free path using the RRT algorithm.

    Parameters
    ----------
    start : (x, y)  pixel coordinate (col, row)
    goal  : (x, y)  pixel coordinate (col, row)
    occupancy_map : 2-D uint8 array, 0=free, 255=obstacle
    max_iter : maximum RRT iterations
    step_size : how far (pixels) each extension grows
    goal_bias : probability of sampling the goal instead of a random point
    goal_tolerance : distance (pixels) at which we consider the goal reached

    Returns
    -------
    path : list of (x, y) pixel waypoints from start to goal, or None
    """
    h, w = occupancy_map.shape[:2]
    sx, sy = start
    gx, gy = goal

    if not _is_free(occupancy_map, sx, sy):
        print("[RRT] WARNING: start pixel is inside an obstacle – "
              "trying to proceed anyway.")
    if not _is_free(occupancy_map, gx, gy):
        print("[RRT] WARNING: goal pixel is inside an obstacle – "
              "trying to proceed anyway.")

    root = RRTNode(sx, sy)
    nodes: List[RRTNode] = [root]

    for iteration in range(max_iter):
        # ----- Sample -----
        if random.random() < goal_bias:
            rand_x, rand_y = gx, gy
        else:
            rand_x = random.randint(0, w - 1)
            rand_y = random.randint(0, h - 1)

        # ----- Nearest neighbour (brute-force, fine for map sizes <1k) -----
        best_node = None
        best_dist = float("inf")
        for node in nodes:
            d = math.hypot(node.x - rand_x, node.y - rand_y)
            if d < best_dist:
                best_dist = d
                best_node = node

        # ----- Steer towards the sample -----
        dx = rand_x - best_node.x
        dy = rand_y - best_node.y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            continue
        if dist > step_size:
            dx = dx / dist * step_size
            dy = dy / dist * step_size
        new_x = int(round(best_node.x + dx))
        new_y = int(round(best_node.y + dy))

        # ----- Collision check -----
        if not _line_collision_free(occupancy_map, best_node.x, best_node.y, new_x, new_y):
            continue

        new_node = RRTNode(new_x, new_y, parent=best_node)
        nodes.append(new_node)

        # ----- Goal reached? -----
        if math.hypot(new_x - gx, new_y - gy) <= goal_tolerance:
            # Connect to exact goal if collision-free
            if _line_collision_free(occupancy_map, new_x, new_y, gx, gy):
                goal_node = RRTNode(gx, gy, parent=new_node)
                nodes.append(goal_node)
            else:
                goal_node = new_node

            # Back-trace the path
            path: List[Tuple[int, int]] = []
            cur: Optional[RRTNode] = goal_node
            while cur is not None:
                path.append((cur.x, cur.y))
                cur = cur.parent
            path.reverse()
            print(f"[RRT] Path found after {iteration + 1} iterations, "
                  f"{len(path)} waypoints.")
            return path

    print("[RRT] Failed to find a path within the iteration budget.")
    return None


# =====================================================================
# Path smoothing (optional improvement / bonus)
# =====================================================================

def smooth_path(path: List[Tuple[int, int]],
                occupancy_map: np.ndarray) -> List[Tuple[int, int]]:
    """Greedily remove unnecessary waypoints when a short-cut is collision-free."""
    if len(path) <= 2:
        return path
    smoothed = [path[0]]
    i = 0
    while i < len(path) - 1:
        # Try to skip as far ahead as possible
        furthest = i + 1
        for j in range(len(path) - 1, i, -1):
            if _line_collision_free(occupancy_map,
                                    path[i][0], path[i][1],
                                    path[j][0], path[j][1]):
                furthest = j
                break
        smoothed.append(path[furthest])
        i = furthest
    print(f"[Smooth] Reduced path from {len(path)} to {len(smoothed)} waypoints.")
    return smoothed


# =====================================================================
# Visualisation
# =====================================================================

def visualize_path(map_img: np.ndarray,
                   path: List[Tuple[int, int]],
                   start: Tuple[int, int],
                   goal: Tuple[int, int],
                   window_name: str = "RRT Path") -> np.ndarray:
    """Draw the RRT path, start, and goal on a copy of the map."""
    vis = (map_img.copy() * 255).astype(np.uint8)
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    # Draw path segments
    for i in range(len(path) - 1):
        cv2.line(vis, path[i], path[i + 1], (0, 255, 0), 2)

    # Waypoint dots
    for pt in path:
        cv2.circle(vis, pt, 3, (255, 255, 0), -1)

    # Start & goal markers
    cv2.circle(vis, start, 6, (0, 255, 0), -1)      # green = start
    cv2.putText(vis, "Start", (start[0] + 8, start[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.circle(vis, goal, 6, (0, 0, 255), -1)        # red = goal
    cv2.putText(vis, "Goal", (goal[0] + 8, goal[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow(window_name, vis)
    print("Press any key on the visualisation window to continue...")
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)
    return vis


# =====================================================================
# Goal selection helpers
# =====================================================================

def pick_goal(map_img: np.ndarray) -> Tuple[str, Tuple[int, int]]:
    prompt = "Enter semantic destination (ex: 'rack', 'cushion', 'sofa', 'stair', 'cooktop'): "
    goal_prompt = input(prompt).strip().lower()
    if goal_prompt not in SEMANTIC_DICTS["colors"]:
        print(f"Goal '{goal_prompt}' is not valid.")
        sys.exit(1)

    goal_pixels = get_goal_pixels(map_img, SEMANTIC_DICTS["colors"], goal_prompt)
    goal = random.choice(goal_pixels)
    return goal_prompt, goal


def run_in_sim(start_world: Tuple[float, float],
               world_path: List[Tuple[float, float]],
               goal_prompt: str):
    start_x, start_z = start_world
    print(f"Spawning Agent at world position: ({start_x:.3f}, {start_z:.3f})")

    sim, agent, _ = init_sim(start_x=start_x, start_z=start_z)
    execute_waypoint_path(world_path, sim, agent,
                          SEMANTIC_DICTS["indices"][goal_prompt])


# =====================================================================
# Main
# =====================================================================

def main():
    """Entry point."""

    # === Step 1: Processing the 3D Map ===
    print("=== Step 1: Processing the 3D Map ===")
    map_img, occupancy_map, world_origin, resolution = load_and_filter_map(
        POINT_CLOUD_DATA, COLOR_DATA
    )
    print(f"Map size: {map_img.shape[1]}x{map_img.shape[0]} pixels, "
          f"resolution={resolution} px/m")

    # === Step 2: Select start & goal ===
    print("\n=== Step 2: Selecting Agent Start and Goal Positions ===")
    start = select_start(map_img)
    goal_prompt, goal = pick_goal(map_img)
    print(f"Start pixel: {start},  Goal pixel: {goal}")

    # === Step 3: RRT path planning ===
    print("\n=== Step 3: Executing Path Planning (RRT) ===")
    path = plan_path(start, goal, occupancy_map)
    if not path:
        print("Planner could not find a path.")
        sys.exit(1)

    # Optional: smooth the path to remove zig-zags
    path = smooth_path(path, occupancy_map)

    # === Step 4: Visualise ===
    print("\n=== Step 4: Visualizing the Planned Path ===")
    visualize_path(map_img, path, start, goal)

    # === Step 5: Convert to world coords & navigate ===
    print("\n=== Step 5: Translating Path to Habitat Simulator ===")
    world_path: List[Tuple[float, float]] = [
        pixel_to_world(px, py, world_origin, resolution)
        for (px, py) in path
    ]
    print(f"World path has {len(world_path)} waypoints.")

    run_in_sim(world_path[0], world_path, goal_prompt)


if __name__ == "__main__":
    main()