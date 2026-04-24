import cv2
import numpy as np
from typing import List, Tuple

SCALE_FACTOR = 10000.0 / 255.0
CEILING_COLOR = np.array([8, 255, 214])
FLOOR_COLOR = np.array([255, 194, 7])

MAP_RESOLUTION = 10  # pixels per world-meter
OBSTACLE_POINT_THRESHOLD = 3  # Ignore very sparse projected points in occupancy.
OBSTACLE_INFLATE_RADIUS = 0  # Keep doorways open; increase only if navigation clips walls.
DISPLAY_SCALE = 3  # Enlarge OpenCV map windows without changing map coordinates.
# Height above floor level to include as obstacles (metres).
# Excludes floor clutter below 5 cm and anything above standing height.
HEIGHT_FILTER_LOW = 0.05
HEIGHT_FILTER_HIGH = 2.2


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
    floor_y_world = np.median((points[floor_mask] * SCALE_FACTOR)[:, 1])

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
        occupancy_map = cv2.dilate(raw_occupied, kernel, iterations=1)
    else:
        occupancy_map = raw_occupied

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
