import cv2
import numpy as np
from typing import List, Tuple

SCALE_FACTOR = 10000.0 / 255.0
CEILING_COLOR = np.array([8, 255, 214])
FLOOR_COLOR = np.array([255, 194, 7])

# Resolution of the rasterized 2D map (pixels per world-meter).
# Higher → finer map but larger image; 10 px/m is a good balance.
MAP_RESOLUTION = 10  # pixels per world-meter
OBSTACLE_INFLATE_RADIUS = 3  # pixels – how much to inflate obstacles


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
    keep = ~(ceil_mask | floor_mask)

    points = points[keep]
    colors = colors[keep]

    # ------------------------------------------------------------------
    # 2. Convert to real-world metres (Habitat coordinate system)
    #    x-z = horizontal plane, y = vertical
    # ------------------------------------------------------------------
    coords = points * SCALE_FACTOR  # (N, 3) in metres

    world_x = coords[:, 0]
    world_z = coords[:, 2]
    # We don't need world_y any more (vertical) after filtering

    # ------------------------------------------------------------------
    # 3. Rasterize to a 2D image  (x → column, z → row)
    # ------------------------------------------------------------------
    x_min, x_max = world_x.min(), world_x.max()
    z_min, z_max = world_z.min(), world_z.max()

    # Add a small margin so edge points don't fall exactly on the border
    margin = 0.5  # metres
    origin_x = x_min - margin
    origin_z = z_min - margin

    img_w = int(np.ceil((x_max - origin_x + margin) * MAP_RESOLUTION))
    img_h = int(np.ceil((z_max - origin_z + margin) * MAP_RESOLUTION))

    # Pixel indices for every point
    px = ((world_x - origin_x) * MAP_RESOLUTION).astype(int)
    pz = ((world_z - origin_z) * MAP_RESOLUTION).astype(int)

    # Clip to valid range
    px = np.clip(px, 0, img_w - 1)
    pz = np.clip(pz, 0, img_h - 1)

    # Build the colour map image  (row = z-pixel, col = x-pixel)
    map_img = np.zeros((img_h, img_w, 3), dtype=np.float64)
    count = np.zeros((img_h, img_w), dtype=np.float64)

    # Use np.add.at so overlapping points accumulate; we average later.
    np.add.at(map_img, (pz, px), colors.astype(np.float64))
    np.add.at(count, (pz, px), 1.0)

    # Average and normalise to [0, 1]
    occupied_mask = count > 0
    map_img[occupied_mask] /= count[occupied_mask, np.newaxis]
    map_img /= 255.0  # → [0, 1]
    map_img = map_img.astype(np.float32)

    # ------------------------------------------------------------------
    # 4. Build occupancy map
    #    Occupied pixel = any projected point landed there → obstacle.
    #    Free pixel = nothing projected → navigable.
    #    Then we inflate obstacles so the agent doesn't clip walls.
    # ------------------------------------------------------------------
    raw_occupied = (count > 0).astype(np.uint8) * 255

    # Invert: for navigation the *empty* space is navigable.
    # But "obstacle" in the occupancy map should be the walls/objects.
    # The projected points *are* the walls/furniture surfaces, so they
    # are the obstacles.  Free space has no points.
    # We inflate the obstacle regions so the agent keeps a safe distance.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * OBSTACLE_INFLATE_RADIUS + 1, 2 * OBSTACLE_INFLATE_RADIUS + 1),
    )
    occupancy_map = cv2.dilate(raw_occupied, kernel, iterations=1)

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

def select_start(map_img: np.ndarray) -> Tuple[int, int]:
    """Display map and return user-clicked start coordinate."""
    start_point = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            start_point.append((x, y))
            print(f"Start selected: ({x}, {y})")

    cv2.namedWindow("Select Start")
    cv2.setMouseCallback("Select Start", mouse_callback)
    print("Click on the map window to select a start location...")

    while True:
        cv2.imshow("Select Start", (map_img * 255).astype(np.uint8))
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