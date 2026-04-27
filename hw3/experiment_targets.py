"""Two-start × five-target sweep for the report's Results section.

For each combination (start ∈ {(95, 188), (145, 105)}, target ∈ {rack, cooktop,
sofa, cushion, stair}), runs RRT once with the current default parameters,
optionally smooths the path, saves the path-on-semantic-map image, and prints
a markdown summary.

Run:
    python experiment_targets.py
"""

import os
import random
import time
from typing import List, Tuple

import cv2
import numpy as np

import main
import map_processor as mp


STARTS = [(95, 188), (145, 105)]
TARGETS = ["rack", "cooktop", "sofa", "cushion", "stair"]
OUT_DIR = "experiments/start_target"
RANDOM_SEED = 42


def render_path_image(map_img, occupancy_map, path, start, goal):
    """Build the same visualisation that visualize_path() shows on screen."""
    base = mp.semantic_map_to_uint8(map_img)
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    free = occupancy_map == 0
    empty = np.all(base == 255, axis=2)
    base[free & empty] = np.array([245, 248, 230], dtype=np.uint8)
    obstacle = occupancy_map > 0
    base[obstacle] = (0.7 * base[obstacle] + 0.3 * np.array([80, 80, 80])).astype(np.uint8)

    display = cv2.resize(
        base, None,
        fx=mp.DISPLAY_SCALE, fy=mp.DISPLAY_SCALE,
        interpolation=cv2.INTER_NEAREST,
    )

    def scale(pt):
        x, y = pt
        return (int(round((x + 0.5) * mp.DISPLAY_SCALE)),
                int(round((y + 0.5) * mp.DISPLAY_SCALE)))

    if path:
        scaled = [scale(p) for p in path]
        for i in range(len(scaled) - 1):
            cv2.line(display, scaled[i], scaled[i + 1], (20, 160, 20),
                     2, cv2.LINE_AA)
        for pt in scaled:
            cv2.circle(display, pt, 3, (200, 0, 200), -1, cv2.LINE_AA)

    s = scale(start)
    g = scale(goal)
    cv2.circle(display, s, 6, (0, 150, 0), -1, cv2.LINE_AA)
    cv2.circle(display, s, 8, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(display, g, 6, (0, 0, 230), -1, cv2.LINE_AA)
    cv2.circle(display, g, 8, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(display, "Start", (s[0] + 10, s[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 120, 0), 1, cv2.LINE_AA)
    cv2.putText(display, "Goal", (g[0] + 10, g[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 180), 1, cv2.LINE_AA)
    return display


def run_one(map_img, occ, origin, res, labels, main_lbl, start_request, target):
    """Plan a path from start_request → target. Return metrics + image path."""
    # Snap the requested start to a free pixel in the main connected region.
    sx, sy = start_request
    if not (0 <= sx < occ.shape[1] and 0 <= sy < occ.shape[0]) or occ[sy, sx] != 0:
        snap = main._nearest_free_pixel(
            occ, start_request, labels=labels, target_region=main_lbl)
        if snap is None:
            return None, "start_unreachable"
        start = snap
    else:
        start = start_request
    start_region = int(labels[start[1], start[0]]) if occ[start[1], start[0]] == 0 else None

    # Goal selection (visibility BFS + standoff scoring).
    try:
        goal_pixels = mp.get_goal_pixels(map_img, main.SEMANTIC_DICTS["colors"], target)
    except ValueError as e:
        return None, f"goal_pixels_error: {e}"
    goal = main._find_visible_goal_pixel(
        map_img, occ, target, goal_pixels, start, labels, start_region)
    if goal is None:
        if target in main.TARGET_GOAL_DIRECTIONS:
            return None, "no_visible_goal"
        random.shuffle(goal_pixels)
        if start_region:
            for candidate in goal_pixels:
                goal = main._nearest_free_pixel(
                    occ, candidate, labels=labels, target_region=start_region)
                if goal is not None:
                    break
        if goal is None:
            for candidate in goal_pixels:
                goal = main._nearest_free_pixel(occ, candidate)
                if goal is not None:
                    break
        if goal is None:
            return None, "no_visible_goal"
    goal = (int(goal[0]), int(goal[1]))

    # Plan with current default parameters.
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    t0 = time.perf_counter()
    raw_path = main.plan_path(
        start, goal, occ,
        max_iter=main.RRT_MAX_ITER,
        step_size=main.RRT_STEP_SIZE,
        goal_bias=main.RRT_GOAL_BIAS,
        goal_tolerance=main.RRT_GOAL_TOLERANCE,
    )
    plan_time = time.perf_counter() - t0

    if raw_path is None:
        return None, "rrt_failed"

    iterations = main.RRT_LAST_STATS.get("iterations") or 0
    path = raw_path
    if main.USE_SMOOTH_PATH:
        path = main.smooth_path(raw_path, occ)

    world_path: List[Tuple[float, float]] = [
        mp.pixel_to_world(px, py, origin, res) for (px, py) in path
    ]
    path_len_m = main.path_length(world_path)
    turn_deg = main.path_turn_total_deg(world_path)
    _, _, predicted_actions = main.estimate_nav_actions(world_path)

    img = render_path_image(map_img, occ, path, start, goal)
    img_path = os.path.join(
        OUT_DIR,
        f"start_{start_request[0]:03d}-{start_request[1]:03d}_{target}.png",
    )
    cv2.imwrite(img_path, img)

    return {
        "start_request": start_request,
        "start": start,
        "goal": goal,
        "target": target,
        "iterations": iterations,
        "raw_waypoints": len(raw_path),
        "waypoints": len(path),
        "path_len": path_len_m,
        "turn_deg": turn_deg,
        "actions": predicted_actions,
        "plan_time": plan_time,
        "image": img_path,
    }, "ok"


def run_experiment():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading map (current map_processor parameters)...")
    map_img, occ, origin, res = mp.load_and_filter_map(
        main.POINT_CLOUD_DATA, main.COLOR_DATA,
    )
    print(f"Map: {occ.shape[1]}x{occ.shape[0]} px @ {res} px/m")

    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(
        (occ == 0).astype(np.uint8))
    main_lbl = int(stats[1:, 4].argmax() + 1)

    print(f"USE_SMOOTH_PATH = {main.USE_SMOOTH_PATH}")
    print(f"RRT params: step={main.RRT_STEP_SIZE}, bias={main.RRT_GOAL_BIAS}, "
          f"max_iter={main.RRT_MAX_ITER}, tol={main.RRT_GOAL_TOLERANCE}")
    print()

    rows = []
    for start in STARTS:
        for target in TARGETS:
            print(f"=== start={start}  target={target} ===")
            result, status = run_one(map_img, occ, origin, res, labels,
                                     main_lbl, start, target)
            if status != "ok":
                print(f"  FAILED: {status}")
                rows.append({
                    "start_request": start,
                    "target": target,
                    "status": status,
                })
                print()
                continue

            print(f"  start (snapped): {result['start']}  goal: {result['goal']}")
            print(f"  iter={result['iterations']}, raw_wp={result['raw_waypoints']}, "
                  f"wp={result['waypoints']}, len={result['path_len']:.2f} m, "
                  f"turn={result['turn_deg']:.1f}°, actions={result['actions']}")
            print(f"  image: {result['image']}")
            print()
            rows.append({**result, "status": "ok"})

    # Markdown table
    print("\n=== MARKDOWN TABLE ===\n")
    header = ("| Start | Target | Goal pixel | RRT Iter | waypoints | "
              "Path (m) | Turn° | predicted actions | Success |")
    sep = "|---|---|---|---|---|---|---|---|---|"
    print(header)
    print(sep)
    for r in rows:
        if r.get("status") != "ok":
            print(f"| {r['start_request']} | {r['target']} | — | — | — | — | "
                  f"— | — | ✗ ({r.get('status')}) |")
            continue
        print(f"| {r['start_request']} | {r['target']} | {r['goal']} | "
              f"{r['iterations']} | {r['waypoints']} | "
              f"{r['path_len']:.2f} | {r['turn_deg']:.1f}° | "
              f"{r['actions']} | ✓ |")

    print(f"\nAll path images saved under: {OUT_DIR}/")


if __name__ == "__main__":
    run_experiment()
