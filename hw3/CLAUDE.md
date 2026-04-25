# HW3: Robot Navigation Framework — Context

> NYCU AI Capstone 2026 Spring
> Due: 2026/4/27

## Overview

Use a semantic point cloud of `apartment_0` (first floor only) to:
1. Build a 2D semantic map
2. Implement RRT path planning
3. Navigate a robot agent in Habitat simulator

Target objects: **rack, cushion, sofa, stair, cooktop**

---

## Part 1: 2D Semantic Map Construction

### Data
- `hw3/semantic_3d_pointcloud/point.npy` — 3D point coordinates (normalized 0–255)
- `hw3/semantic_3d_pointcloud/color0255.npy` — RGB colors in [0, 255] (use this to avoid float errors)
- `hw3/semantic_3d_pointcloud/color01.npy` — RGB colors in [0, 1] (alternative)

### Coordinate System
- Habitat sim: **x-z plane = horizontal**, y = vertical
- Scale: `world_coords = points_array * 10000.0 / 255.0`
- The map uses (x, z) scatter plot

### Steps
1. Filter out ceiling and floor points by color:
   - Ceiling color (RGB 0–255): `[8, 255, 214]`
   - Floor color (RGB 0–255): `[255, 194, 7]`
2. Project remaining points to 2D (x, z)
3. Additional processing: remove isolated points, inflate obstacles → occupancy map
4. `map_img` must be **float [0, 1]** for downstream visualization

### Critical: Pixel ↔ World Coordinate Mapping
You must establish a mapping between pixel coordinates on the 2D map and Habitat world coordinates. This is needed in Part 3 to convert RRT pixel paths to simulator waypoints.

---

## Part 2: RRT Algorithm

### Input
1. Target category string (e.g., "rack")
2. Start point selected by clicking on map GUI

### Output
1. Map image with path from start to a point **in front of** the target item
2. Waypoints along the route (in pixel coords, then converted to world coords)

### Semantic Color Dictionary
```python
SEMANTIC_DICTS = {
    "colors": {
        "rack": [[0, 255, 133]],
        "cooktop": [[7, 255, 224]],
        "sofa": [[10, 0, 255]],
        # TODO: add cushion, stair colors from the 101-category color map
    },
    "indices": {
        "rack": 8,
        "cooktop": 280,
        "sofa": 196,
        # TODO: add cushion, stair indices from info_semantic.json
    },
}
```

### RRT Parameters to Tune
- **Step size**: how far each RRT extension goes
- **Goal bias**: probability of sampling the goal vs random point (exploration vs exploitation)

---

## Part 3: Robot Navigation

### Agent Actions
- `move_forward` (default 0.1 m per step)
- `turn_left` (default 1° per step)
- `turn_right` (default 1° per step)
- Configurable via `habitat_sim.agent.ActuationSpec(amount=...)`

### Requirements
1. Convert pixel path → world coordinate waypoints
2. Execute waypoints: turn to face next waypoint, then walk forward
3. **Highlight target** with transparent red mask overlay during navigation (semantic sensor mask)

### Simulator Setup (navigator.py defaults)
- Scene: `../hw0/replica_v1/apartment_0/habitat/mesh_semantic.ply`
- Sensor height: 1.5 m, resolution: 512×512
- Initial heading: π (facing -z)
- Agent start: set via `agent_state.position = [x, 0.0, z]`

---

## Template Code Structure

```
hw3/
├── main.py              # Entry point, orchestrates all steps
├── map_processor.py     # TODO 1-1: load_and_filter_map()
├── navigator.py         # Sim init, navigation, waypoint execution (mostly provided)
└── semantic_3d_pointcloud/
    ├── point.npy
    ├── color0255.npy
    └── color01.npy
```

### TODOs in Template
| ID | Location | Task |
|----|----------|------|
| TODO 1-1 | `map_processor.py` | Implement `load_and_filter_map()`: filter ceiling/floor, project 2D, build occupancy map |
| TODO 1-2 | `main.py` | Call `load_and_filter_map()` and unpack results |
| TODO 2 | `main.py` | Implement `plan_path()` — RRT planner |
| TODO 3 | `main.py` | Visualize planned path on map |
| TODO 4 | `main.py` | Convert pixel path to world coordinates |

---

## Grading

### Online Demo (30%)
- 2 random targets picked from {rack, cushion, sofa, stair, cooktop} → 10 pts each
- Q&A → 10 pts
- Must show: start selection on map, RRT result, robot navigation along path

### Report (70%, must be in English)
1. **Implementation (50%)**: code explanation, screenshots
   - Point cloud processing approach
   - RRT algorithm details
2. **Results & Discussion**: different start/target combos, navigation results
3. **Questions (20%)**:
   - (15%) How do step size and goal bias affect RRT sampling?
   - (5%) Real-world indoor navigation challenges?
4. **References**

### Bonus (10%)
Improve RRT (e.g., RRT*, Bi-RRT, path smoothing) and compare with baseline

---

## Submission

```
{STUDENT_ID}_hw3.zip
├── README.md          # How to run
├── report.pdf
└── src/
    ├── main.py
    └── (other .py files)
```

- Wrong format: **-10 pts**
- Late: **-20 pts/day**
- Plagiarism: **0 score**
- Report not in English: **0 score**

---

## Implementation Notes & Gotchas

- `map_img` must be float [0, 1] — `select_start()` multiplies by 255 for display
- `get_goal_pixels()` compares map_img (float 0–1) against `gc / 255.0` with atol=10/255
- `execute_waypoint_path()` expects world coords as `List[Tuple[float, float]]` → (x, z)
- Agent y-position is always 0.0 (first floor)
- Navigator initial heading is π → agent initially faces -z direction
- Turn steps = degrees of angle difference (1° per step)
- Forward steps = distance / MOVE_AMOUNT (0.1 m per step)

---

## Experiment.md Maintenance Rules

`Experiment.md` is the running experiment/debug notebook for this homework. Keep it updated whenever code, parameters, map processing, path planning, visualization, or navigation behavior changes.

### When to Update

Update `Experiment.md` immediately after any of the following:

1. A bug or unexpected behavior is observed.
2. A parameter is changed.
3. A code fix is applied.
4. A fix partially works but does not fully solve the issue.
5. The debugging direction changes because the previous attempt failed.
6. A representative run produces useful logs, screenshots, or metrics.
7. A final configuration is chosen for demo/report.

### What to Record for Each Issue

For every debug item, write the following fields:

```markdown
### N. Short issue title

**狀態:** 已解決 / 部分解決 / 未解決 / 觀察中

**現象**
- What happened?
- Include exact console error/log if available.
- Include start pixel, target, goal pixel, and screenshot reference if relevant.

**原因**
- Current hypothesis or confirmed root cause.
- Mention whether the problem came from semantic map, occupancy map, goal selection, RRT, smoothing, coordinate conversion, or Habitat navigation.

**嘗試歷程**
1. First attempted fix and result.
2. If it failed or only partially worked, say so explicitly.
3. Explain why we changed direction.

**解決方法 / 目前做法**
- Code/parameter changes.
- File/function names if useful.
- Important parameter values.

**驗證**
- Logs/metrics proving whether it worked.
- If not fully verified, write what still needs to be checked.

**需要保留的 log / 截圖**
- Exact logs the user should paste.
- Screenshots needed for report/debug comparison.
```

### Required Logs to Preserve

For each meaningful run, preserve these when available:

- `Start pixel`, `Goal pixel`, and target name
- `[RRT] Path found...` or `[RRT] Failed...`
- `[Grid] Fallback...` if fallback is used
- `[Smooth] Reduced path...`
- `Waypoints on this route`
- `Experiment summary`
- `Navigation summary`
- `Target mask visible...` / `target_seen_frames`
- RRT Path screenshot, especially if the path looks wrong
- Habitat RGB screenshot when target red mask is visible

### How to Mark Incomplete Fixes

Do not write a failed or partial attempt as fully solved. Use language such as:

- `狀態: 部分解決`
- `此方法改善了 X，但仍然造成 Y`
- `因此改為嘗試 ...`
- `尚待確認: ...`
- `下一次請貼: ...`

Examples:

- If increasing `OBSTACLE_POINT_THRESHOLD` connects a route but causes path to cross stairs, record both the improvement and the new failure.
- If a goal-selection fix prevents false success but causes disconnected errors, mark it as partial and record the next direction.
- If a visual path looks like it crosses an object, record whether it crosses gray occupancy overlay or only colored semantic pixels.

### Current Important Debug Themes

Keep tracking these until final demo configuration is stable:

- Doorway connectivity vs. obstacle correctness
- Floor-based free-space quality
- Whether smoothing shortcuts through obstacles
- Rack goal side selection
- Stair blocking for first-floor-only navigation
- Whether target mask appears in Habitat navigation
 
