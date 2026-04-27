"""Compare original RRT paths with the smoothed RRT paths.

Runs the same two-start x five-target set used in the target experiment.
For each case, the script saves two semantic-map images:
    - raw RRT path
    - smoothed RRT path

It also prints a Markdown table and writes it to:
    experiments/smoothing/smoothing_results.md

Run:
    python experiment_smoothing.py
"""

import os
import random
import sys
import time
import types
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import map_processor as mp


def _import_main_for_planning():
    """Import main.py even in environments without Habitat/PIL installed."""
    try:
        import main
        return main
    except ModuleNotFoundError as exc:
        if exc.name not in {"PIL", "habitat_sim"}:
            raise

    sys.modules.pop("main", None)
    fake_nav = types.ModuleType("navigator")
    fake_nav.MOVE_AMOUNT = 0.05
    fake_nav.TURN_AMOUNT = 1.0

    def _unavailable(*_args, **_kwargs):
        raise RuntimeError("Habitat navigation is not needed for this experiment.")

    fake_nav.init_sim = _unavailable
    fake_nav.execute_waypoint_path = _unavailable
    sys.modules["navigator"] = fake_nav

    import main
    return main


main = _import_main_for_planning()


STARTS = [(95, 188), (145, 105)]
TARGETS = ["rack", "cooktop", "sofa", "cushion", "stair"]
OUT_DIR = "experiments/smoothing"
RANDOM_SEED = 42


def render_path_image(map_img, occupancy_map, path, start, goal, label):
    """Build a path visualisation on top of the semantic map."""
    base = mp.semantic_map_to_uint8(map_img)
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    free = occupancy_map == 0
    empty = np.all(base == 255, axis=2)
    base[free & empty] = np.array([245, 248, 230], dtype=np.uint8)

    obstacle = occupancy_map > 0
    base[obstacle] = (
        0.7 * base[obstacle] + 0.3 * np.array([80, 80, 80])
    ).astype(np.uint8)

    display = cv2.resize(
        base,
        None,
        fx=mp.DISPLAY_SCALE,
        fy=mp.DISPLAY_SCALE,
        interpolation=cv2.INTER_NEAREST,
    )

    def scale(pt):
        x, y = pt
        return (
            int(round((x + 0.5) * mp.DISPLAY_SCALE)),
            int(round((y + 0.5) * mp.DISPLAY_SCALE)),
        )

    if path:
        scaled = [scale(p) for p in path]
        for i in range(len(scaled) - 1):
            cv2.line(display, scaled[i], scaled[i + 1], (20, 160, 20), 2, cv2.LINE_AA)
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
    cv2.putText(display, label, (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2, cv2.LINE_AA)
    return display


def path_metrics(path, origin, res) -> Dict[str, float]:
    world_path = [mp.pixel_to_world(px, py, origin, res) for px, py in path]
    forward, turn, actions = main.estimate_nav_actions(world_path)
    return {
        "waypoints": len(path),
        "path_len": main.path_length(world_path),
        "turn_deg": main.path_turn_total_deg(world_path),
        "actions": actions,
        "forward_actions": forward,
        "turn_actions": turn,
    }


def choose_goal(map_img, occ, labels, start, target):
    start_region = int(labels[start[1], start[0]]) if occ[start[1], start[0]] == 0 else None
    goal_pixels = mp.get_goal_pixels(map_img, main.SEMANTIC_DICTS["colors"], target)
    goal = main._find_visible_goal_pixel(
        map_img, occ, target, goal_pixels, start, labels, start_region
    )
    if goal is None:
        if target in main.TARGET_GOAL_DIRECTIONS:
            return None
        random.shuffle(goal_pixels)
        if start_region:
            for candidate in goal_pixels:
                goal = main._nearest_free_pixel(
                    occ, candidate, labels=labels, target_region=start_region
                )
                if goal is not None:
                    break
        if goal is None:
            for candidate in goal_pixels:
                goal = main._nearest_free_pixel(occ, candidate)
                if goal is not None:
                    break
    if goal is None:
        return None
    return int(goal[0]), int(goal[1])


def run_one(map_img, occ, origin, res, labels, main_lbl, start_request, target):
    sx, sy = start_request
    if not (0 <= sx < occ.shape[1] and 0 <= sy < occ.shape[0]) or occ[sy, sx] != 0:
        start = main._nearest_free_pixel(occ, start_request, labels=labels, target_region=main_lbl)
        if start is None:
            return None, "start_unreachable"
    else:
        start = start_request

    goal = choose_goal(map_img, occ, labels, start, target)
    if goal is None:
        return None, "goal_unreachable"

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    t0 = time.perf_counter()
    raw_path = main.plan_path(
        start,
        goal,
        occ,
        max_iter=main.RRT_MAX_ITER,
        step_size=main.RRT_STEP_SIZE,
        goal_bias=main.RRT_GOAL_BIAS,
        goal_tolerance=main.RRT_GOAL_TOLERANCE,
    )
    plan_time = time.perf_counter() - t0
    if raw_path is None:
        return None, "rrt_failed"

    smooth_path = main.smooth_path(raw_path, occ)
    raw_metrics = path_metrics(raw_path, origin, res)
    smooth_metrics = path_metrics(smooth_path, origin, res)

    prefix = f"start_{start_request[0]:03d}-{start_request[1]:03d}_{target}"
    raw_img = os.path.join(OUT_DIR, f"{prefix}_raw.png")
    smooth_img = os.path.join(OUT_DIR, f"{prefix}_smooth.png")
    cv2.imwrite(raw_img, render_path_image(map_img, occ, raw_path, start, goal, "Raw RRT"))
    cv2.imwrite(
        smooth_img,
        render_path_image(map_img, occ, smooth_path, start, goal, "Smoothed RRT"),
    )

    return {
        "start_request": start_request,
        "start": start,
        "target": target,
        "goal": goal,
        "iterations": main.RRT_LAST_STATS.get("iterations") or 0,
        "nodes": main.RRT_LAST_STATS.get("nodes") or 0,
        "plan_time": plan_time,
        "raw": raw_metrics,
        "smooth": smooth_metrics,
        "wp_reduction_pct": 100.0 * (1.0 - smooth_metrics["waypoints"] / raw_metrics["waypoints"]),
        "len_reduction_pct": 100.0 * (1.0 - smooth_metrics["path_len"] / raw_metrics["path_len"]),
        "turn_reduction_pct": (
            0.0 if raw_metrics["turn_deg"] < 1e-9
            else 100.0 * (1.0 - smooth_metrics["turn_deg"] / raw_metrics["turn_deg"])
        ),
        "action_reduction_pct": 100.0 * (1.0 - smooth_metrics["actions"] / raw_metrics["actions"]),
        "raw_image": raw_img,
        "smooth_image": smooth_img,
    }, "ok"


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_markdown(rows):
    lines = []
    lines.append("### RRT Smoothing Comparison")
    lines.append("")
    lines.append(
        "| Start | Target | RRT iter | Raw wp | Smooth wp | "
        "Raw path (m) | Smooth path (m) | Raw turn (deg) | "
        "Smooth turn (deg) | Action reduction |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        if r.get("status") != "ok":
            lines.append(
                f"| {r['start_request']} | {r['target']} | - | - | - | - | - | - | - | "
                f"failed: {r['status']} |"
            )
            continue
        lines.append(
            f"| {r['start_request']} | {r['target']} | {r['iterations']} | "
            f"{r['raw']['waypoints']} | {r['smooth']['waypoints']} | "
            f"{r['raw']['path_len']:.2f} | {r['smooth']['path_len']:.2f} | "
            f"{r['raw']['turn_deg']:.1f} | {r['smooth']['turn_deg']:.1f} | "
            f"{r['action_reduction_pct']:.1f}% |"
        )

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if ok_rows:
        lines.append("")
        lines.append(
            f"Average waypoint reduction: {mean([r['wp_reduction_pct'] for r in ok_rows]):.1f}%"
        )
        lines.append(
            f"Average path length reduction: {mean([r['len_reduction_pct'] for r in ok_rows]):.1f}%"
        )
        lines.append(
            f"Average turn angle reduction: {mean([r['turn_reduction_pct'] for r in ok_rows]):.1f}%"
        )
        lines.append(
            f"Average predicted action reduction: {mean([r['action_reduction_pct'] for r in ok_rows]):.1f}%"
        )
    return "\n".join(lines)


def run_experiment():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading map...")
    map_img, occ, origin, res = mp.load_and_filter_map(main.POINT_CLOUD_DATA, main.COLOR_DATA)
    _, labels, stats, _ = cv2.connectedComponentsWithStats((occ == 0).astype(np.uint8))
    main_lbl = int(stats[1:, 4].argmax() + 1)

    rows = []
    for start in STARTS:
        for target in TARGETS:
            print(f"=== start={start} target={target} ===")
            result, status = run_one(map_img, occ, origin, res, labels, main_lbl, start, target)
            if status != "ok":
                print(f"  FAILED: {status}")
                rows.append({"start_request": start, "target": target, "status": status})
                continue
            rows.append({**result, "status": "ok"})
            print(
                f"  raw wp={result['raw']['waypoints']}, smooth wp={result['smooth']['waypoints']}, "
                f"path {result['raw']['path_len']:.2f}->{result['smooth']['path_len']:.2f} m, "
                f"turn {result['raw']['turn_deg']:.1f}->{result['smooth']['turn_deg']:.1f} deg"
            )
            print(f"  images: {result['raw_image']} / {result['smooth_image']}")

    markdown = build_markdown(rows)
    md_path = os.path.join(OUT_DIR, "smoothing_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown + "\n")

    print("\n" + markdown)
    print(f"\nSaved images and Markdown under: {OUT_DIR}/")


if __name__ == "__main__":
    run_experiment()
