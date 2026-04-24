import math
import random
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np

from map_processor import (
    DISPLAY_SCALE,
    HEIGHT_FILTER_HIGH,
    HEIGHT_FILTER_LOW,
    MAP_RESOLUTION,
    OBSTACLE_INFLATE_RADIUS,
    OBSTACLE_POINT_THRESHOLD,
    load_and_filter_map,
    select_start,
    get_goal_pixels,
    pixel_to_world,
    semantic_map_to_uint8,
)
from navigator import init_sim, execute_waypoint_path
from navigator import MOVE_AMOUNT, TURN_AMOUNT


POINT_CLOUD_DATA = "semantic_3d_pointcloud/point.npy"
COLOR_DATA = "semantic_3d_pointcloud/color0255.npy"
RRT_MAX_ITER = 8000
RRT_STEP_SIZE = 15
RRT_GOAL_BIAS = 0.20
RRT_GOAL_TOLERANCE = 15
RANDOM_SEED = None  # Set an int, e.g. 7, when you want repeatable experiments.
VERBOSE = False
SHOW_TARGET_TABLE = True
SHOW_PARAMETERS = True
SHOW_WAYPOINTS = True
GOAL_MAX_SEARCH_RADIUS = 45
RRT_LAST_STATS = {}

# Semantic colour and index dictionaries for five required target categories.
# Colours come from the 101-category colour map; indices from info_semantic.json.
SEMANTIC_DICTS = {
    "colors": {
        "rack": [[0, 255, 133]],
        "cooktop": [[7, 255, 224]],
        "sofa": [[10, 0, 255]],
        "cushion": [[255, 9, 92]],
        "stair": [[173, 255, 0]],
    },
    "indices": {
        "rack": 8,
        "cooktop": 280,
        "sofa": 196,
        "cushion": [431, 430, 400, 350, 149, 151, 205, 251, 223],
        "stair": 192,
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
    global RRT_LAST_STATS
    RRT_LAST_STATS = {
        "success": False,
        "iterations": max_iter,
        "nodes": 1,
        "max_iter": max_iter,
        "step_size": step_size,
        "goal_bias": goal_bias,
        "goal_tolerance": goal_tolerance,
    }

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
            RRT_LAST_STATS.update({
                "success": True,
                "iterations": iteration + 1,
                "nodes": len(nodes),
                "raw_waypoints": len(path),
            })
            print(f"[RRT] Path found after {iteration + 1} iterations, "
                  f"{len(path)} waypoints.")
            return path

    RRT_LAST_STATS["nodes"] = len(nodes)
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


def path_length(path: List[Tuple[float, float]]) -> float:
    """Return total polyline length for 2D waypoints."""
    return sum(
        math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        for i in range(1, len(path))
    )


def print_waypoints(pixel_path: List[Tuple[int, int]],
                    world_path: List[Tuple[float, float]]) -> None:
    """Print route waypoints in both pixel and Habitat/world coordinates."""
    print("\nWaypoints on this route:")
    print(f"{'#':>3} {'pixel(x,y)':>14} {'world(x,z)':>20}")
    print("-" * 41)
    for i, ((px, py), (wx, wz)) in enumerate(zip(pixel_path, world_path)):
        print(f"{i:>3} ({px:>3},{py:>3}) ({wx:>8.3f},{wz:>8.3f})")


def print_experiment_summary(goal_prompt: str,
                             start: Tuple[int, int],
                             goal: Tuple[int, int],
                             raw_path: List[Tuple[int, int]],
                             smooth_path_px: List[Tuple[int, int]],
                             world_path: List[Tuple[float, float]],
                             occupancy_map: np.ndarray) -> None:
    """Print numbers that can be copied into Experiment.md/report."""
    free_regions, _, _, _ = cv2.connectedComponentsWithStats(
        (occupancy_map == 0).astype(np.uint8)
    )
    raw_len_px = path_length(raw_path)
    smooth_len_px = path_length(smooth_path_px)
    world_len_m = path_length(world_path)
    straight_m = math.hypot(
        world_path[-1][0] - world_path[0][0],
        world_path[-1][1] - world_path[0][1],
    )
    reduction = 0.0
    if len(raw_path) > 0:
        reduction = (len(raw_path) - len(smooth_path_px)) / len(raw_path) * 100.0

    print("\nExperiment summary:")
    print(f"  target                 : {goal_prompt}")
    print(f"  start_pixel / goal     : {start} -> {goal}")
    print(
        "  RRT params             : "
        f"max_iter={RRT_MAX_ITER}, step_size={RRT_STEP_SIZE}, "
        f"goal_bias={RRT_GOAL_BIAS}, goal_tolerance={RRT_GOAL_TOLERANCE}"
    )
    print(
        "  RRT result             : "
        f"iterations={RRT_LAST_STATS.get('iterations')}, "
        f"nodes={RRT_LAST_STATS.get('nodes')}, "
        f"raw_waypoints={len(raw_path)}, smoothed_waypoints={len(smooth_path_px)}"
    )
    print(f"  smoothing reduction    : {reduction:.1f}%")
    print(f"  raw path length        : {raw_len_px:.1f} px")
    print(f"  smoothed path length   : {smooth_len_px:.1f} px / {world_len_m:.2f} m")
    print(f"  straight-line distance : {straight_m:.2f} m")
    print(
        "  occupancy              : "
        f"free_regions={free_regions - 1}, obstacle_ratio={(occupancy_map > 0).mean():.3f}"
    )


# =====================================================================
# Visualisation
# =====================================================================

def visualize_path(map_img: np.ndarray,
                   path: List[Tuple[int, int]],
                   start: Tuple[int, int],
                   goal: Tuple[int, int],
                   window_name: str = "RRT Path") -> np.ndarray:
    """Draw the RRT path, start, and goal on a copy of the map."""
    base = semantic_map_to_uint8(map_img)
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    display = cv2.resize(
        base,
        None,
        fx=DISPLAY_SCALE,
        fy=DISPLAY_SCALE,
        interpolation=cv2.INTER_NEAREST,
    )

    def scale_point(pt: Tuple[int, int]) -> Tuple[int, int]:
        x, y = pt
        return (
            int(round((x + 0.5) * DISPLAY_SCALE)),
            int(round((y + 0.5) * DISPLAY_SCALE)),
        )

    display_path = [scale_point(pt) for pt in path]
    display_start = scale_point(start)
    display_goal = scale_point(goal)

    # Draw overlays after scaling so lines/text stay clean.
    for i in range(len(display_path) - 1):
        cv2.line(
            display,
            display_path[i],
            display_path[i + 1],
            (20, 160, 20),
            2,
            lineType=cv2.LINE_AA,
        )

    for pt in display_path:
        cv2.circle(display, pt, 3, (200, 0, 200), -1, lineType=cv2.LINE_AA)

    cv2.circle(display, display_start, 6, (0, 150, 0), -1, lineType=cv2.LINE_AA)
    cv2.circle(display, display_start, 8, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.circle(display, display_goal, 6, (0, 0, 230), -1, lineType=cv2.LINE_AA)
    cv2.circle(display, display_goal, 8, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    cv2.putText(
        display,
        "Start",
        (display_start[0] + 10, display_start[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 120, 0),
        1,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        display,
        "Goal",
        (display_goal[0] + 10, display_goal[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 180),
        1,
        lineType=cv2.LINE_AA,
    )

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display.shape[1], display.shape[0])
    cv2.imshow(window_name, display)
    print("Press any key on the visualisation window to continue...")
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)
    return display


# =====================================================================
# Goal selection helpers
# =====================================================================

def print_target_locations(map_img: np.ndarray,
                           world_origin: Tuple[float, float],
                           resolution: float) -> None:
    """Print approximate target locations in pixel and world coordinates."""
    print("\nTarget locations (semantic map centroid):")
    print(f"{'target':<10} {'pixels':>6} {'pixel(x,y)':>14} {'world(x,z)':>18}")
    print("-" * 52)
    for name in SEMANTIC_DICTS["colors"]:
        try:
            pixels = get_goal_pixels(map_img, SEMANTIC_DICTS["colors"], name)
        except ValueError:
            print(f"{name:<10} {'0':>6} {'N/A':>14} {'N/A':>18}")
            continue

        xs = np.array([p[0] for p in pixels])
        ys = np.array([p[1] for p in pixels])
        cx = int(round(float(xs.mean())))
        cy = int(round(float(ys.mean())))
        wx, wz = pixel_to_world(cx, cy, world_origin, resolution)
        print(
            f"{name:<10} {len(pixels):>6} "
            f"({cx:>3},{cy:>3}) "
            f"({wx:>6.2f},{wz:>6.2f})"
        )


def _nearest_free_pixel(occupancy_map: np.ndarray,
                        pixel: Tuple[int, int],
                        max_radius: int = 40,
                        labels: Optional[np.ndarray] = None,
                        target_region: Optional[int] = None) -> Optional[Tuple[int, int]]:
    """Return the nearest free pixel to `pixel` via expanding-ring search."""
    h, w = occupancy_map.shape[:2]
    x0, y0 = pixel
    if 0 <= x0 < w and 0 <= y0 < h and occupancy_map[y0, x0] == 0:
        if labels is None or target_region is None or labels[y0, x0] == target_region:
            return pixel
    for r in range(1, max_radius + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if max(abs(dx), abs(dy)) != r:
                    continue
                x, y = x0 + dx, y0 + dy
                if not (0 <= x < w and 0 <= y < h):
                    continue
                if occupancy_map[y, x] != 0:
                    continue
                if labels is not None and target_region is not None and labels[y, x] != target_region:
                    continue
                return (x, y)
    return None


def _line_clear(blocker_map: np.ndarray,
                start: Tuple[int, int],
                target: Tuple[int, int]) -> bool:
    """Return True when a 2D sightline does not cross non-target geometry."""
    x0, y0 = start
    x1, y1 = target
    dist = max(abs(x1 - x0), abs(y1 - y0))
    if dist == 0:
        return True
    h, w = blocker_map.shape[:2]
    for i in range(dist + 1):
        t = i / dist
        x = int(round(x0 + t * (x1 - x0)))
        y = int(round(y0 + t * (y1 - y0)))
        if not (0 <= x < w and 0 <= y < h):
            return False
        if blocker_map[y, x]:
            return False
    return True


def _find_visible_goal_pixel(map_img: np.ndarray,
                             occupancy_map: np.ndarray,
                             goal_pixels: List[Tuple[int, int]],
                             start: Tuple[int, int],
                             labels: np.ndarray,
                             target_region: Optional[int]) -> Optional[Tuple[int, int]]:
    """Find a nearby free-space point that has top-down visibility to the target."""
    if target_region is None:
        return None

    h, w = occupancy_map.shape[:2]
    target_mask = np.zeros((h, w), dtype=np.uint8)
    for x, y in goal_pixels:
        if 0 <= x < w and 0 <= y < h:
            target_mask[y, x] = 1

    # Use the rich semantic map, not the relaxed occupancy map, to block
    # line-of-sight.  This prevents choosing a goal on the wrong side of a wall
    # when occupancy was made permissive to keep narrow doorways connected.
    #
    # Important: only remove the exact target pixels from the blocker map.
    # A larger target neighbourhood can accidentally erase the wall behind a
    # wall-mounted object, making a free pixel in the next room look valid.
    semantic_blockers = np.any(map_img > (5 / 255.0), axis=2)
    blocker_map = semantic_blockers & ~(target_mask.astype(bool))
    blocker_map = cv2.dilate(
        blocker_map.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    ).astype(bool)

    best_score = float("inf")
    best_goal: Optional[Tuple[int, int]] = None
    sx, sy = start

    for tx, ty in goal_pixels:
        for r in range(1, GOAL_MAX_SEARCH_RADIUS + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    x, y = tx + dx, ty + dy
                    if not (0 <= x < w and 0 <= y < h):
                        continue
                    if occupancy_map[y, x] != 0:
                        continue
                    if labels[y, x] != target_region:
                        continue
                    if not _line_clear(blocker_map, (x, y), (tx, ty)):
                        continue

                    start_dist = math.hypot(x - sx, y - sy)
                    score = r * 2.0 + start_dist * 0.05
                    if score < best_score:
                        best_score = score
                        best_goal = (int(x), int(y))
    return best_goal


def pick_goal(map_img: np.ndarray,
              occupancy_map: np.ndarray,
              start: Tuple[int, int]) -> Tuple[str, Tuple[int, int]]:
    prompt = "Enter semantic destination (ex: 'rack', 'cushion', 'sofa', 'stair', 'cooktop'): "
    goal_prompt = input(prompt).strip().lower()
    if goal_prompt not in SEMANTIC_DICTS["colors"]:
        print(f"Goal '{goal_prompt}' is not valid.")
        sys.exit(1)

    goal_pixels = get_goal_pixels(map_img, SEMANTIC_DICTS["colors"], goal_prompt)
    _, labels, _, _ = cv2.connectedComponentsWithStats((occupancy_map == 0).astype(np.uint8))
    sx, sy = start
    start_region = int(labels[sy, sx]) if occupancy_map[sy, sx] == 0 else None
    # The target's own pixels are obstacles in the occupancy map, so snap to a
    # nearby free pixel.  Prefer points that are both reachable from the start
    # and have a clear top-down line-of-sight to the target object.
    visible_goal = _find_visible_goal_pixel(
        map_img,
        occupancy_map,
        goal_pixels,
        start,
        labels,
        start_region,
    )
    if visible_goal is not None:
        return goal_prompt, visible_goal

    # Fallback for unusual starts: choose any nearby free pixel.
    print("[Goal] WARNING: no visible goal pixel found; using nearest reachable free pixel.")
    random.shuffle(goal_pixels)
    if start_region:
        for candidate in goal_pixels:
            free = _nearest_free_pixel(
                occupancy_map,
                candidate,
                labels=labels,
                target_region=start_region,
            )
            if free is not None:
                return goal_prompt, free

    for candidate in goal_pixels:
        free = _nearest_free_pixel(occupancy_map, candidate)
        if free is not None:
            return goal_prompt, free
    print(f"Could not find a reachable free pixel near '{goal_prompt}'.")
    sys.exit(1)


def run_in_sim(start_world: Tuple[float, float],
               world_path: List[Tuple[float, float]],
               goal_prompt: str):
    start_x, start_z = start_world
    print(f"Spawning Agent at world position: ({start_x:.3f}, {start_z:.3f})")

    sim, agent, _ = init_sim(start_x=start_x, start_z=start_z)
    nav_stats = execute_waypoint_path(
        world_path,
        sim,
        agent,
        SEMANTIC_DICTS["indices"][goal_prompt],
    )
    if nav_stats:
        print(
            "Navigation summary: "
            f"move_amount={MOVE_AMOUNT}m, turn_amount={TURN_AMOUNT}deg, "
            f"frames={nav_stats['frames']}, target_seen_frames={nav_stats['seen_frames']}, "
            f"max_mask_pixels={nav_stats['max_mask_pixels']}, "
            f"actions=(forward={nav_stats['forward']}, left={nav_stats['left']}, "
            f"right={nav_stats['right']})"
        )


# =====================================================================
# Main
# =====================================================================

def main():
    """Entry point."""
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    # === Step 1: Processing the 3D Map ===
    print("=== Step 1: Processing the 3D Map ===")
    if SHOW_PARAMETERS or VERBOSE:
        print(
            "Parameters: "
            f"MAP_RESOLUTION={MAP_RESOLUTION}, "
            f"OBSTACLE_POINT_THRESHOLD={OBSTACLE_POINT_THRESHOLD}, "
            f"OBSTACLE_INFLATE_RADIUS={OBSTACLE_INFLATE_RADIUS}, "
            f"HEIGHT_FILTER=({HEIGHT_FILTER_LOW}, {HEIGHT_FILTER_HIGH}), "
            f"DISPLAY_SCALE={DISPLAY_SCALE}"
        )
    map_img, occupancy_map, world_origin, resolution = load_and_filter_map(
        POINT_CLOUD_DATA, COLOR_DATA
    )
    print(f"Map size: {map_img.shape[1]}x{map_img.shape[0]} pixels, "
          f"resolution={resolution} px/m")
    if SHOW_TARGET_TABLE or VERBOSE:
        print_target_locations(map_img, world_origin, resolution)

    # === Step 2: Select start & goal ===
    print("\n=== Step 2: Selecting Agent Start and Goal Positions ===")
    start = select_start(map_img)
    start_free = _nearest_free_pixel(occupancy_map, start)
    if start_free is None:
        print("Selected start is not near any free space.")
        sys.exit(1)
    if start_free != start:
        print(f"Start pixel was on an obstacle; using nearest free pixel {start_free}.")
        start = start_free
    goal_prompt, goal = pick_goal(map_img, occupancy_map, start)
    print(f"Start pixel: {start},  Goal pixel: {goal}")

    # === Step 3: RRT path planning ===
    print("\n=== Step 3: Executing Path Planning (RRT) ===")
    path = plan_path(
        start,
        goal,
        occupancy_map,
        max_iter=RRT_MAX_ITER,
        step_size=RRT_STEP_SIZE,
        goal_bias=RRT_GOAL_BIAS,
        goal_tolerance=RRT_GOAL_TOLERANCE,
    )
    if not path:
        print("Planner could not find a path.")
        sys.exit(1)
    raw_path = path

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
    if SHOW_WAYPOINTS or VERBOSE:
        print_waypoints(path, world_path)
    print_experiment_summary(
        goal_prompt,
        start,
        goal,
        raw_path,
        path,
        world_path,
        occupancy_map,
    )

    run_in_sim(world_path[0], world_path, goal_prompt)


if __name__ == "__main__":
    main()
