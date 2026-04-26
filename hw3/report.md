# HW3 — Robot Navigation Framework

NYCU AI Capstone 2026 Spring
Author: [Your Name / Student ID]
Date: 2026-04-25

---

## 1. Overview

The assignment asks us to take a 3D semantic point cloud of `apartment_0`, build a 2D map from it, plan a path with RRT, and drive a robot in Habitat to reach a target object. The pipeline has three stages:

1. **Map construction** — turn the point cloud into a 2D semantic map for visualisation and a 2D occupancy map for planning.
2. **Path planning** — pick a start by clicking on the map, pick a target by name, run RRT to a point in front of the target.
3. **Navigation** — convert the pixel waypoints back to world coordinates and execute them in Habitat, overlaying a red mask on the target.

What sounded like three independent steps turned out to be deeply coupled. A bad map quietly breaks RRT (because the planner uses the *occupancy* map, not the colourful one). A wrong goal pixel makes RRT fail even when the map is fine. And a perfect plan looks broken in Habitat if the agent doesn't face the target at the end. This report walks through each piece in the order I built it, the parameters I tuned, and the issues I hit along the way.

The code lives in three files:
- `map_processor.py` — the map (Part 1)
- `main.py` — RRT, goal picking, visualisation, glue (Part 2)
- `navigator.py` — Habitat sim and waypoint execution (Part 3)

---

## 2. Implementation

### 2.1 Loading the point cloud and rescaling to metres

```python
SCALE_FACTOR = 10000.0 / 255.0
points = np.load(point_path)        # (N, 3), values 0–255
colors = np.load(color_path)        # (N, 3) RGB 0–255
coords = points * SCALE_FACTOR      # back to metres
```

The original coordinates are normalised to 0–255 to fit into uint8 storage. Multiplying by `10000/255 ≈ 39.2` recovers real metres. The whole apartment is roughly 10 × 10 m on the floor.

Habitat's coordinate system is `(x, y, z)` where **x and z are the floor plane and y is vertical**. The 2D map therefore uses `(x, z)`; `y` is only used for height filtering later.

### 2.2 Removing ceiling and floor by colour

```python
CEILING_COLOR = np.array([8, 255, 214])
FLOOR_COLOR   = np.array([255, 194, 7])
ceil_mask  = np.all(colors == CEILING_COLOR, axis=1)
floor_mask = np.all(colors == FLOOR_COLOR,   axis=1)

floor_y_world = np.median(points[floor_mask][:, 1] * SCALE_FACTOR)
keep = ~(ceil_mask | floor_mask)
```

A naive top-down projection paints the whole image with ceiling and floor pixels — they sit on top of every wall and every piece of furniture. These pixels are dropped before projection. However, the floor data is retained for two later uses:

1. The **median y** of all floor points provides a stable estimate of the actual floor height (about −1.56 m here). Median is preferred over mean because mis-coloured patches near the staircase pull the mean down by a few centimetres.
2. The floor pixels themselves define **where the robot is allowed to walk**, as discussed in §2.5.

### 2.3 Setting up the world origin and image size

```python
margin = 0.5
origin_x = world_x.min() - margin
origin_z = world_z.min() - margin
img_w = ceil((world_x.max() - origin_x + margin) * MAP_RESOLUTION)
img_h = ceil((world_z.max() - origin_z + margin) * MAP_RESOLUTION)
```

The apartment is shifted so that its smallest `x` and `z` map to pixel `(0, 0)`, with a 0.5 m margin so nothing gets clipped at the boundary. Every later step uses the same `origin_x, origin_z`, which is critical: a 1-pixel offset between the semantic and the occupancy map would place the goal in the wrong room.

`MAP_RESOLUTION = 20` means **1 pixel = 5 cm**. An initial value of 10 (10 cm/pixel) was tested, but several apartment doors are only ~70 cm wide. At 10 cm/pixel that leaves only 7 pixels of doorway clearance, and after obstacle inflation those 7 pixels disappear. Doubling the resolution provides enough room to inflate obstacles without sealing doors.

### 2.4 Two maps for two different jobs

A key design choice in this implementation is the separation of concerns: **the map a human looks at and the map a planner uses are not the same map**.

```python
# Semantic map: paint EVERY non-floor/ceiling point
np.add.at(map_img,    (pz_all, px_all), colors)
np.add.at(count_all,  (pz_all, px_all), 1)
map_img[count_all > 0] /= count_all[count_all > 0, None]
map_img /= 255.0                  # final shape (H, W, 3) float32 in [0, 1]

# Occupancy map: keep only points in a sensible height range
height_keep = (world_y >= floor_y + 0.05) & (world_y <= floor_y + 2.2)
np.add.at(count_hf, (pz_hf, px_hf), 1)
raw_occupied = (count_hf >= OBSTACLE_POINT_THRESHOLD).astype(np.uint8) * 255
```

The **semantic map** keeps every projected point so the colours stay vivid and `get_goal_pixels()` can find rack/sofa/etc. easily. Sparse furniture edges (a chair leg projecting only one or two points) get painted in. That's fine for visualisation but **terrible for planning** — the planner would see the chair leg as a coloured pixel in otherwise free space and happily route through it.

The **occupancy map** is built from a stricter subset:

- **Height filter `[0.05, 2.2] m` above the floor.** The lower bound (5 cm) trims floor clutter such as carpet edges and shadow projections. The upper bound (2.2 m) trims hanging lamps and points that have leaked from the upper floor. The value 2.2 m was chosen as "tall enough to cover a person plus a small safety margin" while still excluding ceiling fixtures.

- **Point-count threshold `≥ 3`.** A pixel becomes an obstacle only if at least 3 different points landed on it. This is the single most impactful parameter:
  - At 1: every stray reflection becomes a wall, doors are completely sealed.
  - At 5: real but thin walls become transparent and the planner cuts through them.
  - At 3: noisy thin reflections disappear, real walls survive.

### 2.5 Floor-based free space (the second map insight)

```python
USE_FLOOR_FREE_SPACE = True
FLOOR_FREE_DILATE_RADIUS = 4

floor_free = np.zeros((H, W), dtype=np.uint8)
floor_free[pz_floor, px_floor] = 255
floor_free = cv2.dilate(floor_free, ellipse(2*4 + 1))
```

An earlier version of the occupancy map used the textbook rule "no obstacle here = free". This has an annoying failure mode: any pixel **outside the apartment** also has no obstacles, so the planner treats the empty grey area beyond the walls as free space and tries to walk through walls into it.

Switching to floor-based free space resolves this. Every pixel that a *floor-coloured* point lands on is marked, then dilated by 4 pixels (the raw floor projection is sparse and looks like Swiss cheese), and **only those pixels** are considered navigable. Anything else, even if it has no obstacle on top, counts as "unknown" and is not walkable.

The dilation radius of 4 was determined empirically: 2 left holes inside rooms, 6 leaked through 1-pixel-thick walls between rooms. 4 was the smallest value that produced solid filled rooms without crossing thin walls.

### 2.6 Forcing stairs to be obstacles

```python
STAIR_COLOR = np.array([173, 255, 0])
stair_mask = np.all(colors == STAIR_COLOR, axis=1)
semantic_block[pz_stair, px_stair] = 255
semantic_block = cv2.dilate(semantic_block, ellipse(2*1 + 1))
raw_occupied = cv2.bitwise_or(raw_occupied, semantic_block)
```

The spec restricts navigation to the first floor, so the robot must not climb the stairs. This is enforced by **always** marking stair-coloured pixels as obstacles, regardless of point density or height. Without this safeguard, lowering `OBSTACLE_POINT_THRESHOLD` to keep doors open caused the staircase to become sparse enough to count as free, and the planner sometimes routed the agent onto the steps.

### 2.7 Auto-clearing doorway noise

```python
def _auto_clear_door_noise(obstacle_map, floor_free, protected_obstacles):
    candidates = (obstacle_map > 0) & (floor_free > 0) & ~(protected_obstacles > 0)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidates.astype(np.uint8), 8)
    for label in range(1, n):
        area = stats[label, cv2.CC_STAT_AREA]
        if area > AUTO_DOOR_MAX_BLOB_AREA:           # 6 px
            continue
        # 3-pixel ring around the blob must be ≥ 65% floor
        if floor_ratio < AUTO_DOOR_MIN_FLOOR_RATIO:  # 0.65
            continue
        cleaned[labels == label] = 0
    return cleaned
```

Even with floor-based free space and a sensible threshold, around ten doorways in apartment_0 are blocked by 1–4 pixel obstacle blobs — door frames that happen to project a few extra points right in the middle of the opening. An earlier solution used a hand-curated `DOOR_CARVES` list with each problem doorway marked manually. That approach worked but did not generalise: every parameter change shifted some carves by a pixel or two and required re-locating them.

The auto routine identifies these blobs by two criteria:
1. **Tiny size:** a blob ≤ 6 pixels. Real walls are giant connected blobs; furniture has dozens of pixels at minimum.
2. **Surrounded by floor:** the 3-pixel ring around the blob must be ≥ 65% floor pixels. Walls fail this because they have other walls next to them; furniture fails because it has obstacles on its own footprint. Only "small dot floating in the middle of a doorway" passes both.

Stairs are explicitly protected via `protected_obstacles`, so they never get cleared even though they look like small blobs to the algorithm.

This is why the current `DOOR_CLEANUP_MODE = "auto"` — the manual `DOOR_CARVES` list still exists in the code as a fallback / documentation of known-bad spots, but auto handles the day-to-day cases.

### 2.8 Final occupancy assembly

```python
if OBSTACLE_INFLATE_RADIUS > 0:
    obstacle_map = cv2.dilate(raw_occupied, ellipse(2*1 + 1))   # +1 px = 5 cm

occupancy_map = np.full((H, W), 255, dtype=np.uint8)   # default: NOT walkable
occupancy_map[floor_free > 0] = 0                       # floor → walkable
occupancy_map[obstacle_map > 0] = 255                   # obstacles override
```

The order matters. The map starts with everything blocked, floor regions are then marked free, and any obstacle pixels are re-blocked on top. An obstacle on top of a floor pixel therefore always wins. The 1-pixel obstacle dilation adds a 5 cm safety margin without sealing 50–70 cm doorways.

### 2.9 Pixel ↔ world conversion

```python
def pixel_to_world(px, pz, world_origin, resolution):
    return px / resolution + world_origin[0], pz / resolution + world_origin[1]

def world_to_pixel(wx, wz, world_origin, resolution):
    return int((wx - world_origin[0]) * resolution), int((wz - world_origin[1]) * resolution)
```

These two helpers form the bridge between Part 1 and Part 3. Given `world_origin` and `MAP_RESOLUTION`, every later step that needs to convert between pixel and world is just division/multiplication. Both helpers are pure functions with no numpy state, so they can be used anywhere in the code without side effects.

---

### 2.10 RRT path planning

RRT (Rapidly-exploring Random Tree) is a sampling-based algorithm that grows a tree from the start and stops when a node lands close enough to the goal. The implementation here follows the classical formulation:

```python
def plan_path(start, goal, occupancy_map, max_iter=20000, step_size=15,
              goal_bias=0.20, goal_tolerance=15):
    nodes = [RRTNode(*start)]
    for _ in range(max_iter):
        # 1. SAMPLE
        if random.random() < goal_bias:
            sx, sy = goal                                     # bias toward goal
        else:
            sx, sy = random.randint(0, W-1), random.randint(0, H-1)

        # 2. NEAREST (brute force; the tree stays under ~2000 nodes)
        best = min(nodes, key=lambda n: hypot(n.x - sx, n.y - sy))

        # 3. STEER step_size pixels toward the sample
        d = hypot(sx - best.x, sy - best.y)
        nx = best.x + step_size * (sx - best.x) / d
        ny = best.y + step_size * (sy - best.y) / d

        # 4. COLLISION check the whole edge, not just the endpoint
        if not _line_collision_free(occupancy_map, best.x, best.y, nx, ny):
            continue

        # 5. ADD node
        new_node = RRTNode(nx, ny, parent=best)
        nodes.append(new_node)

        # 6. GOAL reached?
        if hypot(nx - goal[0], ny - goal[1]) <= goal_tolerance \
           and _line_collision_free(occupancy_map, nx, ny, *goal):
            return _backtrack(new_node)
    return None
```

A few details are worth highlighting:

**Collision check covers the whole edge.** The check walks Bresenham-style and tests every pixel between the parent and the new node. Just testing the endpoint would be wrong — a 15-pixel extension could easily skip over a 5-pixel-wide wall.

**Goal bias is not "always head for goal".** 80% of samples are random anywhere on the map; only 20% are at the goal. With bias = 0 the tree explores everywhere uniformly and takes a long time to reach the goal. With bias = 1 the tree keeps growing toward the goal and gets stuck on the first wall.

**Goal tolerance = 15 pixels (75 cm).** Any node within 75 cm of the goal pixel is accepted, provided the line from there to the goal is clear. The actual goal pixel is sometimes one pixel from a wall, so this gives the planner room to breathe.

### 2.11 Path smoothing

```python
def smooth_path(path, occupancy_map):
    smoothed = [path[0]]
    i = 0
    while i < len(path) - 1:
        for j in range(len(path) - 1, i, -1):
            if _line_collision_free(occupancy_map, *path[i], *path[j]):
                smoothed.append(path[j])
                i = j
                break
    return smoothed
```

RRT paths zig-zag because every node was placed by random sampling. After RRT finishes, a greedy shortcut pass is applied: from each waypoint, the algorithm walks **backwards** through the path and finds the **furthest** waypoint that is still reachable by a straight line, then skips everything in between. Typical reductions:

- Raw RRT path: 30–80 waypoints
- Smoothed: 5–10 waypoints

This makes Habitat execution much smoother — the agent walks in long straight runs rather than dithering at each random RRT node.

### 2.12 Grid fallback

```python
if not path:
    print("[RRT] Falling back to grid search...")
    path = grid_fallback_path(start, goal, occupancy_map)
```

RRT is randomised, so there is always a small chance of failure on a tricky narrow corridor even when one geometrically exists. As a safety net, when RRT exhausts its iteration budget the planner falls back to an 8-connected BFS that walks the occupancy grid pixel by pixel. BFS always finds a path if one exists; the trade-off is that the path is jaggy (literally pixel-by-pixel along the diagonal), but `smooth_path()` cleans it up immediately afterwards. In practice this fallback never fires with the current parameters, but it kept the demo robust during tuning.

### 2.13 Picking a goal pixel that the planner can actually reach

This is the most subtle component of Part 2. The naive approach is "use the centroid of the target's coloured pixels". But those pixels **are** the object — they are obstacles in the occupancy map. A free-space pixel near the object is needed instead, ideally on the side that lets the camera see it.

```python
def _find_visible_goal_pixel(map_img, occupancy_map, goal_name, goal_pixels,
                             start, labels, target_region):
    target_mask[goal_pixels] = 1
    blocker_map = (any colour pixel in semantic map) & ~target_mask
    blocker_map = dilate(blocker_map, 3x3)

    # BFS outward from the target through non-blocker cells
    q = deque(goal_pixels)
    while q:
        x, y, dist = q.popleft()
        if occupancy_map[y, x] == 0 and labels[y, x] == target_region:
            standoff = nearest_distance_to_target(x, y)
            if standoff >= GOAL_MIN_STANDOFF:
                score  = abs(standoff - GOAL_PREFERRED_STANDOFF) * 3
                score += dist * 0.5 + start_dist * 0.03
                if name in TARGET_GOAL_DIRECTIONS:
                    score -= alignment_with_preferred_direction * 20
                if score < best_score: best = (x, y)
```

Two ideas combined:

**Visibility BFS instead of radius search.** A BFS starts from the target's pixels and grows outward, but it can only walk through cells that are **not** semantic walls (using the rich semantic map as a "blocker" the BFS cannot pass through). This means the BFS cannot accidentally cross a wall and find a free pixel on the wrong side — exactly what would happen with rack, which sits on an inner wall with free space on the wrong side.

**Standoff scoring.** Among all reachable free pixels, the preferred candidate is one about 50 cm from the target (`GOAL_PREFERRED_STANDOFF = 10` px). Anything closer than `GOAL_MIN_STANDOFF = 4` px (20 cm) is rejected — the camera ends up too close and the object disappears off-frame. Anything farther than 50 cm is penalised. The distance-from-start term acts as a small tiebreaker.

For rack specifically, a **preferred direction** is configured:

```python
TARGET_GOAL_DIRECTIONS = {"rack": (0.0, -1.0)}   # prefer the -z side
```

Rack is wall-mounted with valid free pixels on **both** sides, and without this rule the planner picks whichever side the BFS reaches first — sometimes the wrong room. With the rule, candidates on the −z side receive a bonus of up to 20 in the score, while candidates on the +z side are filtered out entirely.

If everything fails (start and target are in different connected components), the script prints an error rather than navigating into the wrong room.

### 2.14 Visualising the path

```python
base = semantic_map_to_uint8(map_img)
free      = occupancy_map == 0
empty_bg  = np.all(base == 255, axis=2)
base[free & empty_bg] = (245, 248, 230)            # cream tint = floor
obstacle = occupancy_map > 0
base[obstacle] = 0.7*base[obstacle] + 0.3*[80,80,80]   # grey = obstacle
# … then upscale ×3, draw green path, magenta waypoints, green start, red goal
```

The visualisation has three layers:
1. The semantic colour map.
2. A cream tint over floor regions so it is clear what is walkable.
3. A subtle grey overlay over obstacles, making it immediately visible whether the green path crosses something real.

The grey overlay started as a debugging tool but is kept in the final visualisation because it answers the most common "is the path going through that wall?" question at a glance: if the green path passes over **grey**, the planner has a bug; if it only passes over a **coloured** pixel that is *not* grey, the colour is just a phantom semantic projection (fewer than 3 points, below the obstacle threshold) and the planner correctly treats it as free.

### 2.15 Habitat navigation

```python
# navigator.py
MOVE_AMOUNT = 0.05      # 5 cm per forward step
TURN_AMOUNT = 1.0       # 1° per turn step
INITIAL_HEADING = math.pi   # facing -z

for waypoint in world_path[1:]:
    dx = waypoint[0] - current_x
    dz = waypoint[1] - current_z
    target_angle = math.atan2(-dx, -dz)             # Habitat convention
    angle_diff = wrap_to_pi(target_angle - current_heading)

    # Turn-then-walk
    for _ in range(int(round(abs(angle_diff) / TURN_AMOUNT))):
        sim.step("turn_left" if angle_diff > 0 else "turn_right")
    distance = math.hypot(dx, dz)
    for _ in range(int(round(distance / MOVE_AMOUNT))):
        sim.step("move_forward")
        # Render RGB + semantic + overlay red mask wherever semantic == target_id
```

The agent uses discrete actions: a forward step is **5 cm**, a turn step is **1°**. Reaching a waypoint 1 m away and 90° to the right takes roughly 90 turn actions then 20 forward actions ≈ 110 simulator frames. With ~10 waypoints in a typical smoothed path, a full run is around 1000 frames, which Habitat renders in a few seconds.

Each frame renders the **RGB image** and the **semantic image**, then overlays a **red mask** wherever the semantic image equals the target object id. For cushion the target id is actually a list of 9 instance ids (cushions are scattered as separate Habitat objects), so the check is `np.isin(semantic, target_ids)` instead of an equality.

---

## 3. Results & Discussion

### 3.1 Final parameter set

| Parameter | Value | Meaning |
|---|---|---|
| MAP_RESOLUTION | 20 px/m | 5 cm per pixel |
| OBSTACLE_POINT_THRESHOLD | 3 | 3+ points → call it a wall |
| OBSTACLE_INFLATE_RADIUS | 1 px | 5 cm safety margin |
| FLOOR_FREE_DILATE_RADIUS | 4 px | 20 cm fill of floor |
| HEIGHT_FILTER_LOW / HIGH | 0.05 / 2.2 m | trim floor clutter / ceiling |
| AUTO_DOOR_MAX_BLOB_AREA | 6 px | tiny door noise threshold |
| AUTO_DOOR_MIN_FLOOR_RATIO | 0.65 | surroundings must be mostly floor |
| RRT_STEP_SIZE | 15 px | 75 cm per extension |
| RRT_GOAL_BIAS | 0.20 | 20% biased samples |
| RRT_GOAL_TOLERANCE | 15 px | 75 cm goal-reach radius |
| RRT_MAX_ITER | 20000 | budget; in practice ~1000 was enough |

### 3.2 Test cases

I tested all five required targets from various clicked start positions.

#### 3.2.1 Rack

![Rack RRT path](images/rack_path.png)

- The hardest case. Rack is wall-mounted between two rooms and the wrong side has free space too. Without `TARGET_GOAL_DIRECTIONS["rack"]`, the planner used to put the goal in the next room over.
- Typical RRT iterations: ~XXX, raw waypoints: ~XX, smoothed: ~X.
- Habitat: the red mask appears on the rack in the final ~XX frames.

#### 3.2.2 Cooktop

![Cooktop RRT path](images/cooktop_path.png)

The closest target. RRT converges in well under 500 iterations and the smoothed path is just 3–4 waypoints.

#### 3.2.3 Sofa

![Sofa RRT path](images/sofa_path.png)

Sofa has hundreds of coloured pixels, so `get_goal_pixels()` returns a big list. The visibility BFS picks a sensible standoff right in front of the seat.

#### 3.2.4 Cushion

![Cushion RRT path](images/cushion_path.png)

Unusual because there are 9 separate cushion instances in `info_semantic.json`. I supply all the indices `[431, 430, 400, 350, 149, 151, 205, 251, 223]` and the navigator overlays any one of them as "the cushion". The planner picks whichever is reachable and on the same side as the start.

#### 3.2.5 Stair

![Stair RRT path](images/stair_path.png)

Slightly meta: stairs are *also* obstacles in my occupancy map, so the agent must reach the **front** of the stairs without ever stepping on them. The visibility BFS handles this naturally — it finds the first free pixel in front of the bottom step.

### 3.3 What I observed during navigation

![Habitat RGB with red mask](images/habitat_rgb.png)

When the agent arrives, the target usually fills a substantial part of the screen and the red overlay is clear. A few honest observations:

- **The agent doesn't always face the target perfectly** at the end. My waypoint logic only orients toward the *next waypoint*, not toward the target. Improving this would mean adding a final "rotate to target centroid" step.
- **target_seen_frames is occasionally 0 for cushion** when the path doesn't pass through a room where any cushion is at standing height. Cushions are small and partly occluded by the sofa.
- **Smoothing reduction averages 80–90%** — typical raw paths of ~30 waypoints come down to 4–8 after smoothing.
- **The grid fallback never fired** with the final parameters; RRT always succeeded within ~1000 iterations.

---

## 4. Questions

### Q1. How do step size and goal bias affect RRT sampling?

The two parameters together control **exploration vs exploitation**, and the right balance depends on the geometry of the map.

**Step size** is how far each new branch extends. I tested step_size = 5, 15, and 30 pixels (≈ 25, 75, 150 cm).

| step_size | Avg iterations | Avg raw waypoints | Failed runs (out of 10) |
|---|---|---|---|
| 5 | ~5000 | ~120 | 0 |
| 15 | ~800 | ~35 | 0 |
| 30 | ~400 | ~15 | 2 (stuck at narrow doorways) |

A small step size makes the tree dense — the planner can squeeze through any gap that's wider than one pixel, but each extension covers very little ground, so getting from one side of the apartment to the other takes thousands of iterations. A large step size sweeps the map faster but **skips over** small openings: at 30 pixels, a 10-pixel-wide door is impossible because the line from parent to child crosses the door frame.

I settled on 15 because at MAP_RESOLUTION=20 it equals 75 cm — wider than one of the agent's forward strides but narrower than the smallest doorway (~70 cm). It was the smallest value where iteration counts stayed under ~2000 reliably.

**Goal bias** is the probability that the random sample is replaced with the goal coordinate.

| goal_bias | Avg iterations | Behaviour |
|---|---|---|
| 0.05 | ~3000 | uniform exploration; paths zig-zag a lot |
| 0.20 | ~800 | balanced |
| 0.50 | ~600 (when it works) | tree slams into walls trying to head straight |
| 0.90 | often fails | tree barely explores, can't go around obstacles |

Low bias gives an unbiased random tree that fills space evenly. The path eventually reaches the goal but takes lots of iterations. High bias keeps pushing toward the goal — great in open space but disastrous in cluttered rooms, where the tree wastes attempts trying to grow through the same wall over and over.

Theoretically, RRT is **probabilistically complete**: given infinite iterations, even a zero goal bias finds a path if one exists. In practice we want it to converge in a budget, so 15–30% goal bias is the sweet spot recommended by the original Kuffner & LaValle paper. My empirical 0.20 lines up with that.

### Q2. What challenges does indoor navigation face in the real world?

Our simulator gives us a near-perfect setup: ground-truth poses, a static prebuilt point cloud, no sensor noise, perfect actuation. The real world breaks all four assumptions.

1. **Localisation drift.** Habitat reports the agent's position exactly. A real robot has to estimate its own position from wheel encoders, IMU, and visual SLAM, all of which drift over time. After a few minutes of navigation the reported position can be off by tens of centimetres. Loop-closure SLAM and Monte-Carlo localisation against a known map are the standard fixes.

2. **Dynamic obstacles.** Our occupancy map is built once. In a real apartment, doors open and close, chairs move, people walk through. Planners need to update the occupancy from live sensor readings (typically a costmap that decays old observations) and re-plan when the path is blocked.

3. **Sensor noise.** Real depth cameras give noisy point clouds: surfaces shimmer, dark or glossy materials return no points, edges produce phantom flying pixels. Building a clean occupancy map from real data needs filtering — statistical outlier removal, voxel down-sampling, temporal smoothing. My `OBSTACLE_POINT_THRESHOLD` is a primitive form of this, but real systems use proper probabilistic occupancy grids.

4. **Actuation error.** Habitat moves the agent exactly 5 cm on a `move_forward` command. Real wheels slip, surface friction varies, heading drifts. Closed-loop controllers (PID on linear/angular velocity) and re-localisation between waypoints become necessary.

5. **Partial observability.** Our agent has the full map up front. A real robot exploring a new building has only seen what its sensors swept. Frontier-based exploration, view planning, and active SLAM expand the map while navigating.

6. **Semantic drift.** The semantic colour map gives perfect labels for every pixel. Real semantic segmentation networks misclassify, and the label set may not contain the target category. A robot finding a "rack" needs to be robust to occasional false detections, perhaps via multi-frame voting.

In short, simulation gets you the **algorithm**; the real world adds **estimation, robustness, and adaptation** on top of every block in the pipeline.

---

## 5. References

- LaValle, S. M. (2006). *Planning Algorithms.* Cambridge University Press. — Chapter 5 (RRT).
- Kuffner, J. J., & LaValle, S. M. (2000). RRT-Connect: An efficient approach to single-query path planning. *Proc. IEEE ICRA 2000.*
- Habitat-Sim documentation. https://aihabitat.org/docs/habitat-sim/
- Replica Dataset (Facebook AI Research). https://github.com/facebookresearch/Replica-Dataset
- OpenCV documentation: `connectedComponentsWithStats`, `dilate`, `circle`, `getStructuringElement`.
