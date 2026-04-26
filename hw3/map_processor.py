import cv2
import numpy as np
from typing import List, Tuple

SCALE_FACTOR = 10000.0 / 255.0
CEILING_COLOR = np.array([8, 255, 214])
FLOOR_COLOR = np.array([255, 194, 7])
STAIR_COLOR = np.array([173, 255, 0])

MAP_RESOLUTION = 20  # pixels per world-meter
OBSTACLE_POINT_THRESHOLD = 1  # Conservative: keep sparse furniture as obstacles; open verified doors manually.
OBSTACLE_INFLATE_RADIUS = 1  # Keep this small; large inflation closes narrow doorways.
SEMANTIC_BLOCK_INFLATE_RADIUS = 1  # Always block stairs/unsafe semantic regions.
DISPLAY_SCALE = 3  # Enlarge OpenCV map windows without changing map coordinates.
DOOR_CLEANUP_MODE = "auto"  # "none", "manual", "auto", or "both"
DOOR_CARVES = [   # (pixel_xy, radius) — carve real doorways blocked by sparse projection points
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
USE_FLOOR_FREE_SPACE = True
FLOOR_FREE_DILATE_RADIUS = 4
AUTO_DOOR_MAX_BLOB_AREA = 18
AUTO_DOOR_MAX_BLOB_THICKNESS = 3
AUTO_DOOR_CONTEXT_RADIUS = 3
AUTO_DOOR_MIN_FLOOR_RATIO = 0.65
# Height above floor level to include as obstacles (metres).
# Excludes floor clutter below 5 cm and anything above the agent's camera/body
# height (1.5 m = SENSOR_HEIGHT in navigator.py) — points above that cannot
# physically collide with the agent, and removing them opens up door-frame tops.
HEIGHT_FILTER_LOW = 0.05
HEIGHT_FILTER_HIGH = 1.5


def _auto_clear_door_noise(obstacle_map: np.ndarray,
                           floor_free: np.ndarray,
                           protected_obstacles: np.ndarray) -> np.ndarray:
    """Remove tiny obstacle blobs that only interrupt continuous floor space."""
    floor = floor_free > 0
    protected = protected_obstacles > 0
    candidates = (obstacle_map > 0) & floor & ~protected
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidates.astype(np.uint8),
        connectivity=8,
    )

    cleaned = obstacle_map.copy()
    cleared = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0 or area > AUTO_DOOR_MAX_BLOB_AREA:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if min(w, h) > AUTO_DOOR_MAX_BLOB_THICKNESS:
            continue

        x0 = max(0, x - AUTO_DOOR_CONTEXT_RADIUS)
        y0 = max(0, y - AUTO_DOOR_CONTEXT_RADIUS)
        x1 = min(obstacle_map.shape[1], x + w + AUTO_DOOR_CONTEXT_RADIUS)
        y1 = min(obstacle_map.shape[0], y + h + AUTO_DOOR_CONTEXT_RADIUS)

        context = floor[y0:y1, x0:x1]
        floor_ratio = float(np.count_nonzero(context)) / float(context.size)
        if floor_ratio < AUTO_DOOR_MIN_FLOOR_RATIO:
            continue

        cleaned[labels == label] = 0
        cleared += area

    if cleared:
        print(f"[Map] Auto-cleared {cleared} doorway-noise obstacle pixels.")
    return cleaned


def load_and_filter_map(point_path: str, color_path: str):
    """
    Load the semantic point cloud, filter ceiling/floor, project to 2D,
    and build both a coloured semantic map image and a binary occupancy map.

    Returns
    -------
    map_img : np.ndarray, float32, shape (H, W, 3), values in [0, 1]
        Semantic colour map where each pixel keeps the colour of the point
        cloud projected onto it.  Background (free space) is black (0,0,0).
    occupancy_map : np.ndarray, uint8, shape (H, W)
        0 = free, 255 = obstacle (inflated walls / furniture edges).
    world_origin : tuple (origin_x, origin_z)
        World-coordinate of pixel (0, 0) – needed for pixel↔world conversion.
    resolution : float
        Pixels-per-world-meter used when rasterizing.
    """

    points = np.load(point_path)   # (N, 3) normalised 0-255
    colors = np.load(color_path)   # (N, 3) uint8 0-255

    # ------------------------------------------------------------------
    # 1. Filter ceiling and floor by colour
    # ------------------------------------------------------------------
    ceil_mask = np.all(colors == CEILING_COLOR, axis=1)
    floor_mask = np.all(colors == FLOOR_COLOR, axis=1)

    # Determine floor height from floor-coloured points so we can apply a
    # height filter that keeps only obstacle-level geometry (walls, furniture).
    floor_coords = points[floor_mask] * SCALE_FACTOR
    floor_y_world = np.median(floor_coords[:, 1])
    floor_world_x = floor_coords[:, 0]
    floor_world_z = floor_coords[:, 2]

    keep = ~(ceil_mask | floor_mask)
    points = points[keep]
    colors = colors[keep]

    # ------------------------------------------------------------------
    # 2. Convert to real-world metres (Habitat coordinate system)
    #    x-z = horizontal plane, y = vertical
    # ------------------------------------------------------------------
    coords = points * SCALE_FACTOR  # (N, 3) in metres

    world_x = coords[:, 0]
    world_y = coords[:, 1]
    world_z = coords[:, 2]

    # ------------------------------------------------------------------
    # 3. Establish a shared coordinate origin from ALL points (full extent),
    #    then rasterize two separate maps:
    #      • semantic map  – all non-ceiling/floor points (rich colours)
    #      • occupancy map – height-filtered points only (doorways stay open)
    # ------------------------------------------------------------------
    margin = 0.5  # metres
    origin_x = world_x.min() - margin
    origin_z = world_z.min() - margin

    img_w = int(np.ceil((world_x.max() - origin_x + margin) * MAP_RESOLUTION))
    img_h = int(np.ceil((world_z.max() - origin_z + margin) * MAP_RESOLUTION))

    def _to_px(wx, wz):
        px_ = np.clip(((wx - origin_x) * MAP_RESOLUTION).astype(int), 0, img_w - 1)
        pz_ = np.clip(((wz - origin_z) * MAP_RESOLUTION).astype(int), 0, img_h - 1)
        return px_, pz_

    # --- Semantic colour map (all points) ---
    px_all, pz_all = _to_px(world_x, world_z)
    map_img = np.zeros((img_h, img_w, 3), dtype=np.float64)
    count_all = np.zeros((img_h, img_w), dtype=np.float64)
    np.add.at(map_img, (pz_all, px_all), colors.astype(np.float64))
    np.add.at(count_all, (pz_all, px_all), 1.0)
    occupied_mask = count_all > 0
    map_img[occupied_mask] /= count_all[occupied_mask, np.newaxis]
    map_img /= 255.0
    map_img = map_img.astype(np.float32)

    # --- Height-filtered points for occupancy ---
    height_keep = (world_y >= floor_y_world + HEIGHT_FILTER_LOW) & \
                  (world_y <= floor_y_world + HEIGHT_FILTER_HIGH)
    px_hf, pz_hf = _to_px(world_x[height_keep], world_z[height_keep])
    count_hf = np.zeros((img_h, img_w), dtype=np.float64)
    np.add.at(count_hf, (pz_hf, px_hf), 1.0)
    raw_occupied = (count_hf >= OBSTACLE_POINT_THRESHOLD).astype(np.uint8) * 255

    # Use observed floor as the navigable prior.  This makes doorways clearer:
    # free space is where first-floor floor points exist, then obstacles are
    # subtracted from it.  Unknown empty space outside rooms is not navigable.
    if USE_FLOOR_FREE_SPACE:
        px_floor, pz_floor = _to_px(floor_world_x, floor_world_z)
        floor_free = np.zeros((img_h, img_w), dtype=np.uint8)
        floor_free[pz_floor, px_floor] = 255
        if FLOOR_FREE_DILATE_RADIUS > 0:
            floor_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    2 * FLOOR_FREE_DILATE_RADIUS + 1,
                    2 * FLOOR_FREE_DILATE_RADIUS + 1,
                ),
            )
            floor_free = cv2.dilate(floor_free, floor_kernel, iterations=1)

    # Stairs lead away from the first floor, so keep them non-navigable even
    # when the generic point-count threshold is relaxed to preserve doorways.
    semantic_block = np.zeros((img_h, img_w), dtype=np.uint8)
    stair_mask = np.all(colors == STAIR_COLOR, axis=1)
    if np.any(stair_mask):
        px_stair, pz_stair = _to_px(world_x[stair_mask], world_z[stair_mask])
        semantic_block[pz_stair, px_stair] = 255
        if SEMANTIC_BLOCK_INFLATE_RADIUS > 0:
            block_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    2 * SEMANTIC_BLOCK_INFLATE_RADIUS + 1,
                    2 * SEMANTIC_BLOCK_INFLATE_RADIUS + 1,
                ),
            )
            semantic_block = cv2.dilate(semantic_block, block_kernel, iterations=1)
        raw_occupied = cv2.bitwise_or(raw_occupied, semantic_block)

    # Doorway projection noise is easiest to identify before dilation: after
    # inflation a tiny false obstacle can merge into real walls or furniture.
    if USE_FLOOR_FREE_SPACE and DOOR_CLEANUP_MODE in ("auto", "both"):
        raw_occupied = _auto_clear_door_noise(
            raw_occupied,
            floor_free,
            semantic_block,
        )

    # Invert: for navigation the *empty* space is navigable.
    # But "obstacle" in the occupancy map should be the walls/objects.
    # The projected points *are* the walls/furniture surfaces, so they
    # are the obstacles.  Free space has no points.
    # We inflate the obstacle regions so the agent keeps a safe distance.
    if OBSTACLE_INFLATE_RADIUS > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * OBSTACLE_INFLATE_RADIUS + 1, 2 * OBSTACLE_INFLATE_RADIUS + 1),
        )
        obstacle_map = cv2.dilate(raw_occupied, kernel, iterations=1)
    else:
        obstacle_map = raw_occupied

    if USE_FLOOR_FREE_SPACE:
        occupancy_map = np.full((img_h, img_w), 255, dtype=np.uint8)
        occupancy_map[floor_free > 0] = 0
        occupancy_map[obstacle_map > 0] = 255
    else:
        occupancy_map = obstacle_map

    # The point-cloud projection can leave one-pixel false obstacles in narrow
    # doorways.  Carve only verified doorway pixels instead of relaxing the
    # whole map, so walls and stairs remain blocked.
    if DOOR_CLEANUP_MODE in ("manual", "both"):
        for (cx, cy), radius in DOOR_CARVES:
            cv2.circle(occupancy_map, (cx, cy), radius, 0, -1)

    return map_img, occupancy_map, (origin_x, origin_z), MAP_RESOLUTION


# ------------------------------------------------------------------
# Pixel ↔ World conversion helpers
# ------------------------------------------------------------------

def pixel_to_world(px: int, pz: int, world_origin: Tuple[float, float],
                   resolution: float) -> Tuple[float, float]:
    """Convert a pixel coordinate (col, row) to world (x, z)."""
    origin_x, origin_z = world_origin
    world_x = px / resolution + origin_x
    world_z = pz / resolution + origin_z
    return world_x, world_z


def world_to_pixel(world_x: float, world_z: float,
                   world_origin: Tuple[float, float],
                   resolution: float) -> Tuple[int, int]:
    """Convert world (x, z) to pixel coordinate (col, row)."""
    origin_x, origin_z = world_origin
    px = int((world_x - origin_x) * resolution)
    pz = int((world_z - origin_z) * resolution)
    return px, pz


# ------------------------------------------------------------------
# Existing template functions (unchanged)
# ------------------------------------------------------------------

def semantic_map_to_uint8(map_img: np.ndarray,
                          white_background: bool = True,
                          remove_isolated: bool = True) -> np.ndarray:
    """Convert a float semantic map to uint8 for visualization."""
    img = (map_img * 255).astype(np.uint8)
    if remove_isolated:
        occupied = ~np.all(img == 0, axis=2)
        neighbor_count = cv2.filter2D(
            occupied.astype(np.uint8),
            -1,
            np.ones((3, 3), dtype=np.uint8),
            borderType=cv2.BORDER_CONSTANT,
        )
        img[occupied & (neighbor_count <= 1)] = 0
    if white_background:
        empty = np.all(img == 0, axis=2)
        img[empty] = 255
    return img


def _display_image(map_img: np.ndarray) -> np.ndarray:
    """Return an enlarged uint8 image for OpenCV display."""
    img = semantic_map_to_uint8(map_img)
    if DISPLAY_SCALE == 1:
        return img
    return cv2.resize(
        img,
        None,
        fx=DISPLAY_SCALE,
        fy=DISPLAY_SCALE,
        interpolation=cv2.INTER_NEAREST,
    )


def select_start(map_img: np.ndarray) -> Tuple[int, int]:
    """Display map and return user-clicked start coordinate."""
    start_point = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            map_x = min(map_img.shape[1] - 1, max(0, x // DISPLAY_SCALE))
            map_y = min(map_img.shape[0] - 1, max(0, y // DISPLAY_SCALE))
            start_point.append((map_x, map_y))
            print(f"Start selected: ({map_x}, {map_y})")

    display = _display_image(map_img)
    cv2.namedWindow("Select Start", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select Start", display.shape[1], display.shape[0])
    cv2.setMouseCallback("Select Start", mouse_callback)
    print("Click on the map window to select a start location...")

    while True:
        cv2.imshow("Select Start", display)
        key = cv2.waitKey(1) & 0xFF
        if start_point:
            break
        if key == ord("q"):
            raise RuntimeError("No start selected. Exiting.")

    cv2.destroyWindow("Select Start")
    return start_point[0]


def get_goal_pixels(map_img: np.ndarray, semantic_dict: dict, goal_name: str) -> List[Tuple[int, int]]:
    """Find all pixels corresponding to the goal object based on colour matching."""

    if goal_name.lower() not in semantic_dict:
        raise ValueError(f"Unknown semantic object: {goal_name}. Available options: {list(semantic_dict.keys())}")

    goal_colors = semantic_dict[goal_name.lower()]
    goal_pixels: List[Tuple[float, float]] = []

    for gc in goal_colors:
        gc_norm = np.array(gc) / 255.0
        mask_goal = np.all(np.isclose(map_img, gc_norm, atol=10 / 255.0), axis=2)
        zs, xs = np.where(mask_goal)
        goal_pixels.extend(list(zip(xs, zs)))

    if not goal_pixels:
        raise ValueError(f"No valid pixels found for '{goal_name}'.")

    return goal_pixels
