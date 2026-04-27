# HW3 — Robot Navigation Framework

NYCU AI Capstone 2026 Spring
Author: 蕭宇岑 / 112550179

---

## 1. Implementation

### 1.1 Loading the point cloud and rescaling to metres

```python
SCALE_FACTOR = 10000.0 / 255.0
points = np.load(point_path)        # (N, 3), values 0–255
colors = np.load(color_path)        # (N, 3) RGB 0–255
coords = points * SCALE_FACTOR      # back to metres
```

The original coordinates are normalised to 0–255 to fit into uint8 storage. Multiplying by `10000/255 ≈ 39.2` recovers real metres. The whole apartment is roughly 10 × 10 m on the floor.

Habitat's coordinate system is `(x, y, z)` where **x and z are the floor plane and y is vertical**. The 2D map therefore uses `(x, z)`; `y` is only used for height filtering later.

### 1.2 Removing ceiling and floor by colour

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
2. The floor pixels themselves define **where the robot is allowed to walk**, as discussed in §1.5.

### 1.3 Setting up the world origin and image size

```python
margin = 0.5
origin_x = world_x.min() - margin
origin_z = world_z.min() - margin
img_w = ceil((world_x.max() - origin_x + margin) * MAP_RESOLUTION)
img_h = ceil((world_z.max() - origin_z + margin) * MAP_RESOLUTION)
```

The apartment is shifted so that its smallest `x` and `z` map to pixel `(0, 0)`, with a 0.5 m margin so nothing gets clipped at the boundary. Every later step uses the same `origin_x, origin_z`, which is critical: a 1-pixel offset between the semantic and the occupancy map would place the goal in the wrong room.

`MAP_RESOLUTION = 20` means **1 pixel = 5 cm**. An initial value of 10 (10 cm/pixel) was tested, but several apartment doors are only ~70 cm wide. At 10 cm/pixel that leaves only 7 pixels of doorway clearance, and after obstacle inflation those 7 pixels disappear. Doubling the resolution provides enough room to inflate obstacles without sealing doors.

### 1.4 Two maps for two different jobs

A key design choice in this implementation is the separation of concerns: **the map a human looks at and the map a planner uses are not the same map**.

```python
# Semantic map: paint EVERY non-floor/ceiling point
np.add.at(map_img,    (pz_all, px_all), colors)
np.add.at(count_all,  (pz_all, px_all), 1)
map_img[count_all > 0] /= count_all[count_all > 0, None]
map_img /= 255.0                  # final shape (H, W, 3) float32 in [0, 1]

# Occupancy map: keep only points in a sensible height range
height_keep = (world_y >= floor_y + HEIGHT_FILTER_LOW) & \
              (world_y <= floor_y + HEIGHT_FILTER_HIGH)
np.add.at(count_hf, (pz_hf, px_hf), 1)
raw_occupied = (count_hf >= OBSTACLE_POINT_THRESHOLD).astype(np.uint8) * 255
```

The **semantic map** keeps every projected point so the colours stay vivid and `get_goal_pixels()` can find rack/sofa/etc. easily. Sparse furniture edges (a chair leg projecting only one or two points) get painted in. That's fine for visualisation but **terrible for planning** — the planner would see the chair leg as a coloured pixel in otherwise free space and happily route through it.

The **occupancy map** is built from a height-filtered subset:

- **Height filter `[0.05, 1.5] m` above the floor.** The lower bound trims floor clutter. The upper bound is close to the camera/body height of the agent. This also removes high door-frame points, which opened doorways without manual carving.

- **Point-count threshold `≥ 1`.** After lowering the height filter, I could keep sparse wall and furniture points as obstacles. This is conservative and reduces the chance of cutting through thin objects.

### 1.5 Floor-based free space (the second map insight)

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

### 1.6 Forcing stairs to be obstacles

```python
STAIR_COLOR = np.array([173, 255, 0])
stair_mask = np.all(colors == STAIR_COLOR, axis=1)
semantic_block[pz_stair, px_stair] = 255
semantic_block = cv2.dilate(semantic_block, ellipse(2*2 + 1))
raw_occupied = cv2.bitwise_or(raw_occupied, semantic_block)
```

The spec restricts navigation to the first floor, so the robot must not climb the stairs. This is enforced by **always** marking stair-coloured pixels as obstacles, regardless of point density or height. Without this safeguard, lowering `OBSTACLE_POINT_THRESHOLD` to keep doors open caused the staircase to become sparse enough to count as free, and the planner sometimes routed the agent onto the steps.

### 1.7 Doorway cleanup attempts

```python
def _auto_clear_door_noise(obstacle_map, floor_free, protected_obstacles):
    candidates = (obstacle_map > 0) & (floor_free > 0) & ~(protected_obstacles > 0)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidates.astype(np.uint8), 8)
    for label in range(1, n):
        area = stats[label, cv2.CC_STAT_AREA]
        if area > AUTO_DOOR_MAX_BLOB_AREA:           # 18 px
            continue
        # 3-pixel ring around the blob must be ≥ 65% floor
        if floor_ratio < AUTO_DOOR_MIN_FLOOR_RATIO:  # 0.65
            continue
        cleaned[labels == label] = 0
    return cleaned
```

Doorways were one of the main map problems. My first solution was to manually find blocked doorway pixels and carve them open with `DOOR_CARVES`. This worked for this map, but it was not general: small changes in resolution, inflation, or filtering shifted the doorway positions.

I also tried an automatic cleanup routine. It removes small, thin obstacle blobs when the surrounding area is mostly floor. The idea was to clear noisy projected points inside doorways.

In practice, this was not reliable enough. Some blocked doors remained closed, while some real furniture edges and wall fragments looked similar to doorway noise. The final setting is therefore `DOOR_CLEANUP_MODE = "none"`. Instead of carving doors, I use the 1.5 m height filter to remove high door-frame points cleanly.

### 1.8 Final occupancy assembly

```python
if OBSTACLE_INFLATE_RADIUS > 0:
    obstacle_map = cv2.dilate(raw_occupied, ellipse(2*2 + 1))   # +2 px = 10 cm

occupancy_map = np.full((H, W), 255, dtype=np.uint8)   # default: NOT walkable
occupancy_map[floor_free > 0] = 0                       # floor → walkable
occupancy_map[obstacle_map > 0] = 255                   # obstacles override
```

The order matters. The map starts with everything blocked, floor regions are then marked free, and any obstacle pixels are re-blocked on top. An obstacle on top of a floor pixel therefore always wins. The final obstacle dilation is 2 pixels, or 10 cm. This kept the Habitat navigation path farther from furniture and reduced collisions.

### 1.9 Pixel ↔ world conversion

```python
def pixel_to_world(px, pz, world_origin, resolution):
    return px / resolution + world_origin[0], pz / resolution + world_origin[1]

def world_to_pixel(wx, wz, world_origin, resolution):
    return int((wx - world_origin[0]) * resolution), int((wz - world_origin[1]) * resolution)
```

These two helpers form the bridge between Part 1 and Part 3. Given `world_origin` and `MAP_RESOLUTION`, every later step that needs to convert between pixel and world is just division/multiplication. Both helpers are pure functions with no numpy state, so they can be used anywhere in the code without side effects.

---

### 1.10 RRT path planning

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

### 1.11 Path smoothing

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

RRT paths zig-zag because every node was placed by random sampling. After RRT finishes, a greedy shortcut pass is applied: from each waypoint, the algorithm walks **backwards** through the path and finds the **furthest** waypoint that is still reachable by a straight line, then skips everything in between. In my final 10-case experiment, smoothing reduced the waypoint count by 61.7% on average.

This makes Habitat execution much smoother — the agent walks in long straight runs rather than dithering at each random RRT node.

### 1.12 Grid fallback

```python
if not path:
    print("[RRT] Falling back to grid search...")
    path = grid_fallback_path(start, goal, occupancy_map)
```

RRT is randomised, so there is always a small chance of failure on a tricky narrow corridor even when one geometrically exists. As a safety net, when RRT exhausts its iteration budget the planner falls back to an 8-connected BFS that walks the occupancy grid pixel by pixel. BFS always finds a path if one exists; the trade-off is that the path is jaggy (literally pixel-by-pixel along the diagonal), but `smooth_path()` cleans it up immediately afterwards. In practice this fallback never fires with the current parameters, but it kept the demo robust during tuning.

### 1.13 Picking a goal pixel that the planner can actually reach

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

Stair needed a different rule. The stair object is large, so searching from all stair pixels can choose a side point instead of the desired free space below the stairs. For `stair`, I keep only the bottom band of stair pixels as BFS seeds:

```python
STAIR_BOTTOM_BAND_PX = 6
search_points = [(x, y) for x, y in stair_pixels
                 if y >= stair_bottom_y - STAIR_BOTTOM_BAND_PX]
```

Candidate goals must also be below the stair bottom. This makes the goal land near the free space at the bottom of the stairs. If this BFS still fails, stair uses a deterministic fallback that searches only below the stair bottom, so the general nearest-free fallback does not choose a random side point around the large stair region.

If the visible-goal search fails for normal objects, the code falls back to the nearest reachable free pixel near the target. For objects with strict side requirements such as rack, it reports an error instead of navigating to the wrong side.

### 1.14 Visualising the path

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

### 1.15 Habitat navigation

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

## 2. Results & Discussion

### 2.1 Final parameter set

| Parameter | Value | Meaning |
|---|---|---|
| MAP_RESOLUTION | 20 px/m | 5 cm per pixel |
| OBSTACLE_POINT_THRESHOLD | 1 | keep sparse wall/furniture points |
| OBSTACLE_INFLATE_RADIUS | 2 px | 10 cm safety margin |
| SEMANTIC_BLOCK_INFLATE_RADIUS | 2 px | keep stairs safely blocked |
| FLOOR_FREE_DILATE_RADIUS | 4 px | 20 cm fill of floor |
| HEIGHT_FILTER_LOW / HIGH | 0.05 / 1.5 m | trim floor clutter / high door-frame points |
| DOOR_CLEANUP_MODE | none | final map uses height filtering instead of carving |
| RRT_STEP_SIZE | 15 px | 75 cm per extension |
| RRT_GOAL_BIAS | 0.20 | 20% biased samples |
| RRT_GOAL_TOLERANCE | 15 px | 75 cm goal-reach radius |
| RRT_MAX_ITER | 20000 | planning budget |

### 2.2 Test cases

I tested all five required targets from two start positions to test different levels of planning difficulty.

**Start point =(95, 188)**
The first start point is placed in a more open area of the apartment, so many targets can be reached without passing through several narrow doorways. This gives a baseline case where RRT mainly needs to find a direct collision-free path.
| cooktop |cushion|rack|sofa|stair|
|---|---|---|---|---|
|![image](https://hackmd.io/_uploads/Syt8PJpTZg.png)|![image](https://hackmd.io/_uploads/Sk3FDy6TWg.png) | ![image](https://hackmd.io/_uploads/ByuqPJap-x.png)| ![image](https://hackmd.io/_uploads/S1UsvJ6pZx.png)|![image](https://hackmd.io/_uploads/Sk4nP1ppWe.png)|

**Start point =(145, 105)**
The second start point is placed inside a smaller room. From this position, the planner often needs to leave the room through a narrow doorway and navigate around walls before reaching the target. This tests whether the occupancy map, goal selection, and RRT planner can still work in a more constrained environment.
| cooktop |cushion|rack|sofa|stair|
|---|---|---|---|---|
|![image](https://hackmd.io/_uploads/rJzpP1TTWx.png)|![image](https://hackmd.io/_uploads/HyJAPy66be.png)|![image](https://hackmd.io/_uploads/B1c0P1aTZe.png)| ![image](https://hackmd.io/_uploads/Bkv1uy6TWl.png)|![image](https://hackmd.io/_uploads/rJ4m5yTaZl.png)|


**Conclusion**
Overall, the experiments show that RRT can find valid paths for different start-target combinations, but the path quality depends strongly on the indoor layout. Open-space cases are usually direct, while room-to-room cases require more exploration and more turns.


### 2.3 Discussions
#### Door not open causing disconnection
##### manually carve door
At first, I manually located several blocked doorways on the occupancy map and carved them open by hand. This helped reconnect some rooms and allowed RRT to find paths that were previously impossible.

However, this approach was not general. The door positions had to be found manually, and small changes in map resolution or obstacle inflation could shift the blocked pixels. Because of this, manual carving was hard to maintain and not reliable for other maps, so I quickly moved away from this solution.

occupancy map
![image](https://hackmd.io/_uploads/B1rN2Ja6Wl.png)

```python=
DOOR_CLEANUP_MODE = "manual"  # "none", "manual", "auto", or "both"
DOOR_CARVES = [   
    # (pixel_xy, radius) — carve real doorways blocked by sparse projection points
    ((50, 15), 2),   # top room ↔ main
    ((81, 21), 2),   # upper-right pocket ↔ main
    ((68, 44), 2),   # upper-right room ↔ main
    ((19, 58), 2),   # left corridor ↔ main
    ((62, 64), 2),   # central doorway (was radius 1)
    ((85, 59), 1),   # rack/stair right strip ↔ main (1.4px gap)
    ((55, 93), 2),   # rack room ↔ main
    ((73, 112), 2),  # rack room lower ↔ main
    ((146, 121), 3), # user-provided camera pose doorway world=(3.753, 0.245)
    ((94, 142), 2),  # bottom-right room ↔ main
    ((56, 147), 2),  # bottom strip ↔ main
]
```

##### Auto Cleanup: Poor Result

I tried to automatically remove small obstacle blobs that appeared inside floor-connected regions. The goal was to clear noisy projection points that blocked doorways.

However, the result was poor. Many real walls, furniture edges, and object boundaries also appear as small thin blobs in the map, so the rule could not distinguish real obstacles from doorway noise. Some doors still remained blocked, while some useful obstacle details could be removed accidentally. Because of this, auto cleanup was not reliable enough for the final map.

![image](https://hackmd.io/_uploads/Hk06nk6p-e.png)

##### Adjusting the Ceiling Filter to Match the Camera Height

Finally, I realized that the upper part of the door frame is similar to ceiling-level geometry. Since the robot camera does not need to collide with or navigate around objects above its height, these high points should not block the 2D occupancy map.

Therefore, I lowered the ceiling/height filter so that high door-frame points were removed together with ceiling points. This opened the doorways without manually carving them or using unreliable auto cleanup rules.

This worked much better than the previous methods and became the final approach used in my map construction.
```python
HEIGHT_FILTER_HIGH = 1.5 # 2.2 -> 1.5
```

occupancy map
![image](https://hackmd.io/_uploads/Syy9RJpabg.png)


--- 

#### In Actual Navigation, the Agent Bumps into Furniture

During Habitat navigation, I found that the agent sometimes bumped into furniture. Although the simulator can slide the agent out of a stuck position and still eventually reach the target, this would be unsafe for a real-world robot. Therefore, I tried to reduce these collisions.

| Case | Path smoothing | Obstacle inflation radius | Collision while navigating |
|---|---|---|---|
| 1 | No | 5 cm | Yes |
| 2 | Yes | 5 cm | Yes |
| 3 | No | 10 cm | No |
| 4 | Yes | 10 cm | No |

At first, I thought the collision was mainly caused by path smoothing, because smoothing may shortcut the path and bring it too close to furniture. However, the experiments showed that the obstacle inflation radius had a larger effect.

When the obstacle inflation radius was only 5 cm, collisions still happened with or without smoothing. After increasing the radius to 10 cm, the planned path kept a safer distance from furniture, and the collision problem almost disappeared. Therefore, I used a 10 cm obstacle inflation radius in the final setting.


```python
OBSTACLE_INFLATE_RADIUS = 2 #originally 1
```
|case1|case2|
|---|---|
| ![image](https://hackmd.io/_uploads/rJQyAFn6Ze.png)|![image](https://hackmd.io/_uploads/Hy48-c2p-e.png) 

|case3|case4|
|---|---|
|![image](https://hackmd.io/_uploads/SkA7wh3aWe.png)| ![image](https://hackmd.io/_uploads/Bkpi3h3p-x.png)|



--- 
#### Decide the Target Point

The target object itself cannot be used directly as the RRT goal. Objects such as the sofa, rack, cooktop, and stair are obstacles in the occupancy map, so their pixels are not reachable. Therefore, the goal for RRT should be a nearby free-space point in front of or beside the target object.

##### Phase 1: Search for the Nearest Free Point

My original method was to search around the target object and choose the first nearby free pixel as the goal. This worked for some simple cases, but it failed when the object was close to a thin wall.

For example, when navigating to the rack from the bottom-right room, the nearest free point could be selected on the wrong side of the wall. The robot could still plan a path to that point, but the point was not actually in front of the rack. This violates the requirement that the target point should be near the front of the target item.

##### Phase 2: Rack Direction Bias

To fix the rack case, I added a direction preference for the rack target. Since the correct standing position is on the front side of the rack, the goal selection gives preference to free pixels in that direction. This prevents the planner from choosing a reachable point on the wrong side of the wall.

With this change, the rack target point is selected in front of the rack instead of across the wall.
```python=
TARGET_GOAL_DIRECTIONS = {
    "rack": (0.0, -1.0),
}
```

##### Phase 3: Stair Goal Selection

For `stair`, the desired target point should be on the free space below the stairs. However, because the stair object covers a relatively large area, the free space at the bottom can be farther from the stair pixels than the closer side regions. When I used a strict direction constraint, the planner sometimes could not find a valid reachable point and reported the target as disconnected.

To solve this, I used a more specific rule for stairs. Instead of searching from the whole stair region, I searched from the lower part of the stair pixels and only accepted free-space points below the stair. This allowed the planner to choose a reachable goal at the bottom of the stairs.


##### Phase 3 ：stair - softer rule
I changed the stair goal selection to use a softer and more specific rule. Instead of searching outward from all stair pixels, the algorithm first finds the bottom part of the stair region and starts the search from there. It then only accepts free-space points below the stair bottom. This makes the selected goal closer to the actual place where the agent should stand, and the planner successfully chooses a point below the stairs

| original | now |
|----- |----- |
| |![image](https://hackmd.io/_uploads/rk53GkpTWl.png)|

---
### 2.4 Bonus - Smoothing
#### RRT Improvement: Path Smoothing

The original RRT algorithm can find a valid collision-free path, but the path is often jagged because the tree is built from random samples. This creates many unnecessary intermediate waypoints and sharp turns. Although the path is valid on the occupancy map, it is not ideal for actual robot navigation because the agent needs to turn many times.

To improve this, I added a greedy path smoothing step after RRT finds a path. Starting from each waypoint, the smoother tries to connect directly to a later waypoint. If the straight shortcut is collision-free, all intermediate waypoints are removed. This keeps the path valid while reducing unnecessary zig-zag movement.


#### Result
I used the same start-target combinations as the previous RRT result section: two start points and five semantic targets, for a total of 10 test cases. For each case, I compared the original raw RRT path with the smoothed path. Both paths use the same start point, target point, RRT parameters, and occupancy map, so the difference comes only from the smoothing step.

Only two representative examples are shown in the report figures, but the quantitative table includes all 10 cases.
|Original|Smoothed|
|---|---|
|![image](https://hackmd.io/_uploads/ryRYre6aZx.png)|![image](https://hackmd.io/_uploads/HkcqHlap-x.png)|

|Original|Smoothed|
|---|---|
|![image](https://hackmd.io/_uploads/SkKDDe6TZg.png)|![image](https://hackmd.io/_uploads/r1mODlTa-e.png)|

 Start | Target | RRT iter | Raw wp | Smooth wp | Raw path (m) | Smooth path (m) | Raw turn (deg) | Smooth turn (deg) | Action reduction |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| (95, 188) | rack | 76 | 5 | 3 | 2.39 | 2.32 | 45.8 | 13.0 | 37.2% |
| (95, 188) | cooktop | 842 | 23 | 7 | 15.22 | 11.07 | 827.0 | 332.8 | 51.0% |
| (95, 188) | sofa | 13 | 7 | 2 | 4.10 | 3.75 | 118.8 | 0.0 | 62.7% |
| (95, 188) | cushion | 14 | 8 | 2 | 4.90 | 4.69 | 103.1 | 0.0 | 53.2% |
| (95, 188) | stair | 200 | 15 | 4 | 10.11 | 7.84 | 507.7 | 185.2 | 51.8% |
| (145, 105) | rack | 13 | 7 | 5 | 4.36 | 3.84 | 205.9 | 93.2 | 42.0% |
| (145, 105) | cooktop | 13 | 8 | 4 | 4.89 | 4.89 | 93.8 | 84.5 | 4.7% |
| (145, 105) | sofa | 195 | 23 | 9 | 15.74 | 14.31 | 800.7 | 274.5 | 49.7% |
| (145, 105) | cushion | 85 | 17 | 5 | 11.53 | 9.63 | 624.1 | 94.8 | 66.3% |
| (145, 105) | stair | 287 | 22 | 5 | 14.72 | 11.09 | 907.8 | 215.1 | 63.6% |

Average waypoint reduction: 61.7%
Average path length reduction: 12.8%
Average turn angle reduction: 68.6%
Average predicted action reduction: 48.2%
#### Conclusion
The improved RRT path shows clear benefits. On average, the number of waypoints was reduced by 61.7%, the path length was reduced by 12.8%, the total turning angle was reduced by 68.6%, and the predicted number of navigation actions was reduced by 48.2%. This means the improved version produces paths that are shorter, smoother, and more practical for navigation.

Overall, the original RRT is useful for finding a valid route, but its raw path is not very efficient for execution. Adding path smoothing improves the quality of the RRT output while keeping the collision-free constraint, so I used the smoothed RRT path in the final navigation pipeline.



---

## 3. Questions

### Q1. How do step size and goal bias affect RRT sampling?
All measurements use a fixed start (114, 46) and rack as the target. Smoothing is disabled so waypoint counts reflect the raw RRT tree. 
I do each setup 5 times and caluculated the variance and standard.
#### Result

| Case number | step size | goal bias | RRT Iterations | waypoints | Path length (m) | total turn angle | predicted actions | success |
|---|---|---|---|---|---|---|---|---|
| A1 | 5 | 0.20 | 962 ± 642 | 50.8 ± 23.1 | 12.92 ± 5.85 | 1527.4° ± 713.0° | 1786 ± 828 | 5/5 |
| A2 | 15 | 0.20 | 381 ± 307 | 22.6 ± 9.7 | 15.32 ± 6.63 | 824.5° ± 470.2° | 1131 ± 597 | 5/5 |
| A3 | 25 | 0.20 | 109 ± 76 | 8.2 ± 0.4 | 7.89 ± 0.33 | 252.4° ± 37.5° | 410 ± 41 | 5/5 |
| A4 | 35 | 0.20 | 643 ± 344 | 12.8 ± 4.0 | 14.73 ± 5.26 | 544.5° ± 167.3° | 839 ± 248 | 5/5 |
| B1 | 15 | 0.05 | 221 ± 121 | 19.4 ± 8.9 | 13.04 ± 5.86 | 772.7° ± 424.2° | 1033 ± 538 | 5/5 |
| B2 | 15 | 0.50 | 593 ± 437 | 13.6 ± 1.5 | 8.56 ± 0.89 | 433.5° ± 152.3° | 605 ± 170 | 5/5 |
| B3 | 15 | 0.80 | 2483 ± 3059 | 22.2 ± 9.4 | 14.94 ± 6.73 | 770.5° ± 465.1° | 1069 ± 599 | 5/5 |

| A1 |A2|A3|A4|
|---|---|---|---|
|![image](https://hackmd.io/_uploads/rkIrjCnabg.png)|![image](https://hackmd.io/_uploads/H1MIiAhp-x.png) | ![image](https://hackmd.io/_uploads/HJaKo03a-g.png)|![image](https://hackmd.io/_uploads/BkO5iRh6Wg.png)
|

| B1 |B2|B3|
|---|---|---|
| ![image](https://hackmd.io/_uploads/ry9njR2abe.png)|![image](https://hackmd.io/_uploads/SJrajCh6Wl.png) |![image](https://hackmd.io/_uploads/SJlAiA3T-l.png)
|
#### Discussion
***Step size***
Step size shows the expected U-shaped behaviour. When the step is too small, the tree needs many small extensions to cross the apartment. When the step is too large, many proposed edges are rejected because they collide with walls or fail to pass through narrow doorways. The most interesting case is A3 (step_size=25): its standard deviations are much smaller than the other settings, suggesting that this value gives a good balance between making progress and preserving enough resolution to pass through constrained spaces. In contrast, both smaller and larger steps produce paths that are more sensitive to the random seed.

***Goal bias***
Goal bias is affected more by variance than by the mean alone. A high bias such as B3 (goal_bias=0.80) can work very well when the direct direction to the goal is useful, but it becomes unstable when walls block that direction. In those cases, the tree repeatedly tries to grow toward the goal and can waste thousands of iterations. B2 (goal_bias=0.50) produced the cleanest path in this experiment, but it is closer to this greedy failure mode than the lower-bias setting.

***Final Selection***
Therefore, the implementation keeps step_size=15 and goal_bias=0.20. Although A3 and B2 perform better on some individual metrics, they are less conservative. For a demo that needs to work across many start/target pairs on the first attempt, the chosen parameters are a safer trade-off between speed, path quality, and robustness.

### Q2. What challenges does indoor navigation face in the real world?

Our simulator gives us a near-perfect setup: ground-truth poses, a static prebuilt point cloud, no sensor noise, perfect actuation. The real world breaks all these assumptions.

1. **Localisation drift.** Habitat always knows the exact agent position. A real robot must estimate its pose using wheel odometry, IMU, depth, or visual SLAM. Small errors accumulate over time, so the robot may think it is following the planned path while its real position is shifted.

3. **Dynamic obstacles.** The occupancy map in this project is static. In a real apartment, doors may open or close, furniture can move, and people may block the path. The robot would need live sensing, local costmap updates, and re-planning when the original RRT path becomes invalid.

4. **Actuation error.** In Habitat, each `move_forward` or `turn` command is executed exactly. Real robots have wheel slip, uneven floor contact, and imperfect turning. Therefore, waypoint following needs feedback control, such as PID control and periodic re-localisation.
5. **Sensor noise.** Real depth cameras give noisy point clouds: surfaces shimmer, dark or glossy materials return no points, edges produce phantom flying pixels. Building a clean occupancy map from real data needs filtering — statistical outlier removal, voxel down-sampling, temporal smoothing. Though I use `OBSTACLE_POINT_THRESHOLD` and other proccesses to successfully build a valid occupancy map for the robot, but it's not general enough for a real system.


---

## 4. References

- LaValle, S. M. (2006). *Planning Algorithms.* Cambridge University Press. — Chapter 5 (RRT).
- Kuffner, J. J., & LaValle, S. M. (2000). RRT-Connect: An efficient approach to single-query path planning. *Proc. IEEE ICRA 2000.*
- Habitat-Sim documentation. https://aihabitat.org/docs/habitat-sim/
- Replica Dataset (Facebook AI Research). https://github.com/facebookresearch/Replica-Dataset
- OpenCV documentation: `connectedComponentsWithStats`, `dilate`, `circle`, `getStructuringElement`.
