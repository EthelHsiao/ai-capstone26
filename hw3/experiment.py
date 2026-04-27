"""RRT parameter sweep — averaged over multiple trials per case.

Start fixed at (114, 46), goal = rack. Runs 7 (step_size, goal_bias)
combinations, each repeated TRIALS times with different random seeds.
Reports mean ± std and saves the path image from the first successful
trial of each case. Outputs a markdown table.

Run:
    python experiment.py
"""

import os
import random
import statistics
import time
from typing import List, Tuple

import cv2
import numpy as np

import main
import map_processor as mp


START = (114, 46)
TARGET = "rack"
OUT_DIR = "experiments/rrt_sweep"
TRIALS = 5
SEEDS = [0, 1, 2, 3, 4]   # one per trial; shared across cases for fairness

CASES = [
    {"id": "A1", "step": 5,  "bias": 0.20},
    {"id": "A2", "step": 15, "bias": 0.20},   # default — appears in both sweeps
    {"id": "A3", "step": 25, "bias": 0.20},
    {"id": "A4", "step": 35, "bias": 0.20},
    {"id": "B1", "step": 15, "bias": 0.05},
    {"id": "B2", "step": 15, "bias": 0.50},
    {"id": "B3", "step": 15, "bias": 0.80},
]


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


def run_single_trial(occupancy_map, world_origin, resolution, start, goal,
                     step_size, goal_bias, seed):
    """Run plan_path with a given seed and return per-trial metrics."""
    random.seed(seed)
    np.random.seed(seed)

    t0 = time.perf_counter()
    raw_path = main.plan_path(
        start, goal, occupancy_map,
        max_iter=main.RRT_MAX_ITER,
        step_size=step_size,
        goal_bias=goal_bias,
        goal_tolerance=main.RRT_GOAL_TOLERANCE,
    )
    plan_time = time.perf_counter() - t0

    success = raw_path is not None
    iterations = main.RRT_LAST_STATS.get("iterations") or 0

    path = raw_path
    if success and main.USE_SMOOTH_PATH:
        path = main.smooth_path(raw_path, occupancy_map)

    if success:
        world_path: List[Tuple[float, float]] = [
            mp.pixel_to_world(px, py, world_origin, resolution)
            for (px, py) in path
        ]
        path_len_m = main.path_length(world_path)
        turn_deg = main.path_turn_total_deg(world_path)
        _, _, predicted_actions = main.estimate_nav_actions(world_path)
        waypoints = len(path)
    else:
        path_len_m = 0.0
        turn_deg = 0.0
        predicted_actions = 0
        waypoints = 0

    return {
        "success": success,
        "iterations": iterations,
        "waypoints": waypoints,
        "path_len": path_len_m,
        "turn_deg": turn_deg,
        "actions": predicted_actions,
        "plan_time": plan_time,
        "path": path,
    }


def _agg(values):
    """Return (mean, std) for a list; std=0 if fewer than 2 samples."""
    if not values:
        return 0.0, 0.0
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def run_case(map_img, occupancy_map, world_origin, resolution, start, goal, case):
    """Run TRIALS trials of one (step, bias) combination and aggregate."""
    trials = []
    saved_image = None
    for trial_idx, seed in enumerate(SEEDS):
        r = run_single_trial(
            occupancy_map, world_origin, resolution, start, goal,
            case["step"], case["bias"], seed,
        )
        trials.append(r)
        succ = "✓" if r["success"] else "✗"
        print(f"  trial {trial_idx+1}/{TRIALS} seed={seed}: "
              f"iter={r['iterations']:>5}  wp={r['waypoints']:>3}  "
              f"len={r['path_len']:5.2f} m  turn={r['turn_deg']:6.1f}°  "
              f"actions={r['actions']:>4}  {succ}")

        if saved_image is None and r["success"]:
            img = render_path_image(map_img, occupancy_map, r["path"], start, goal)
            saved_image = os.path.join(
                OUT_DIR,
                f"{case['id']}_step{case['step']:02d}_bias{case['bias']:.2f}.png",
            )
            cv2.imwrite(saved_image, img)

    successes = [r for r in trials if r["success"]]
    n_succ = len(successes)

    iter_m, iter_s = _agg([r["iterations"] for r in successes])
    wp_m, wp_s = _agg([r["waypoints"] for r in successes])
    len_m, len_s = _agg([r["path_len"] for r in successes])
    turn_m, turn_s = _agg([r["turn_deg"] for r in successes])
    act_m, act_s = _agg([r["actions"] for r in successes])

    if saved_image is None:                # all trials failed
        img = render_path_image(map_img, occupancy_map, [], start, goal)
        saved_image = os.path.join(
            OUT_DIR,
            f"{case['id']}_step{case['step']:02d}_bias{case['bias']:.2f}_FAIL.png",
        )
        cv2.imwrite(saved_image, img)

    return {
        **case,
        "iter_mean": iter_m, "iter_std": iter_s,
        "wp_mean": wp_m, "wp_std": wp_s,
        "len_mean": len_m, "len_std": len_s,
        "turn_mean": turn_m, "turn_std": turn_s,
        "act_mean": act_m, "act_std": act_s,
        "success_count": n_succ, "trials": TRIALS,
        "image": saved_image,
    }


def run_experiment():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading map (current map_processor parameters)...")
    map_img, occ, origin, res = mp.load_and_filter_map(
        main.POINT_CLOUD_DATA, main.COLOR_DATA,
    )
    print(f"Map: {occ.shape[1]}x{occ.shape[0]} px @ {res} px/m  "
          f"(world origin = ({origin[0]:.2f}, {origin[1]:.2f}))")

    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(
        (occ == 0).astype(np.uint8))
    main_lbl = int(stats[1:, 4].argmax() + 1)

    sx, sy = START
    if not (0 <= sx < occ.shape[1] and 0 <= sy < occ.shape[0]) or occ[sy, sx] != 0:
        snap = main._nearest_free_pixel(
            occ, START, labels=labels, target_region=main_lbl)
        if snap is None:
            print(f"ERROR: cannot snap start {START} to a free pixel.")
            return
        print(f"Start {START} -> snapped to nearest free pixel {snap}")
        start = snap
    else:
        start = START

    pixels = mp.get_goal_pixels(map_img, main.SEMANTIC_DICTS["colors"], TARGET)
    goal = main._find_visible_goal_pixel(
        map_img, occ, TARGET, pixels, start, labels, main_lbl)
    if goal is None:
        print(f"ERROR: cannot find a visible goal pixel for '{TARGET}'.")
        return
    print(f"Start: {start},  Goal ({TARGET}): {goal}")
    print(f"USE_SMOOTH_PATH = {main.USE_SMOOTH_PATH}")
    print(f"OBSTACLE_INFLATE_RADIUS = {mp.OBSTACLE_INFLATE_RADIUS}")
    print(f"HEIGHT_FILTER_HIGH = {mp.HEIGHT_FILTER_HIGH}")
    print()

    results = []
    for case in CASES:
        print(f"=== Case {case['id']}: step={case['step']}, bias={case['bias']} "
              f"(× {TRIALS} trials) ===")
        r = run_case(map_img, occ, origin, res, start, goal, case)
        results.append(r)
        print(f"  AGGREGATE: iter={r['iter_mean']:.0f}±{r['iter_std']:.0f}  "
              f"wp={r['wp_mean']:.1f}±{r['wp_std']:.1f}  "
              f"len={r['len_mean']:.2f}±{r['len_std']:.2f} m  "
              f"turn={r['turn_mean']:.1f}±{r['turn_std']:.1f}°  "
              f"actions={r['act_mean']:.0f}±{r['act_std']:.0f}  "
              f"success={r['success_count']}/{r['trials']}")
        print(f"  image: {r['image']}")
        print()

    # Markdown table
    print("\n=== MARKDOWN TABLE (mean ± std over "
          f"{TRIALS} trials, seeds={SEEDS}) ===\n")
    header = ("| Case number | step size | goal bias | RRT Iterations | "
              "waypoints | Path length (m) | total turn angle | "
              "predicted actions | success |")
    sep = "|---|---|---|---|---|---|---|---|---|"
    print(header)
    print(sep)
    for r in results:
        if r["success_count"] == 0:
            print(f"| {r['id']} | {r['step']} | {r['bias']:.2f} | — | — | — | "
                  f"— | — | 0/{r['trials']} |")
            continue
        print(f"| {r['id']} | {r['step']} | {r['bias']:.2f} | "
              f"{r['iter_mean']:.0f} ± {r['iter_std']:.0f} | "
              f"{r['wp_mean']:.1f} ± {r['wp_std']:.1f} | "
              f"{r['len_mean']:.2f} ± {r['len_std']:.2f} | "
              f"{r['turn_mean']:.1f}° ± {r['turn_std']:.1f}° | "
              f"{r['act_mean']:.0f} ± {r['act_std']:.0f} | "
              f"{r['success_count']}/{r['trials']} |")

    print(f"\nAll path images saved under: {OUT_DIR}/")
    print("Each image is from the first successful trial of that case.")


if __name__ == "__main__":
    run_experiment()
