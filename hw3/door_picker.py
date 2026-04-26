import cv2
import numpy as np

import map_processor as mp
from map_processor import pixel_to_world, semantic_map_to_uint8


POINT_PATH = "semantic_3d_pointcloud/point.npy"
COLOR_PATH = "semantic_3d_pointcloud/color0255.npy"


def make_display(map_img, occupancy_map, selected, radius):
    semantic = semantic_map_to_uint8(map_img)
    display = semantic.copy()

    free = occupancy_map == 0
    obstacles = occupancy_map > 0
    display[free & np.all(display == 255, axis=2)] = (235, 255, 255)
    display[obstacles] = (
        0.55 * display[obstacles].astype(np.float32)
        + 0.45 * np.array([145, 145, 145], dtype=np.float32)
    ).astype(np.uint8)

    if selected is not None:
        x, y = selected
        cv2.circle(display, (x, y), radius, (0, 0, 255), 1)
        cv2.drawMarker(
            display,
            (x, y),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=8,
            thickness=1,
        )

    return cv2.resize(
        display,
        (display.shape[1] * mp.DISPLAY_SCALE, display.shape[0] * mp.DISPLAY_SCALE),
        interpolation=cv2.INTER_NEAREST,
    )


def main():
    original_mode = mp.DOOR_CLEANUP_MODE
    mp.DOOR_CLEANUP_MODE = "none"
    map_img, occupancy_map, origin, resolution = mp.load_and_filter_map(
        POINT_PATH,
        COLOR_PATH,
    )
    mp.DOOR_CLEANUP_MODE = original_mode

    selected = None
    radius = 2

    print("Door picker")
    print("  Left click: choose the doorway pixel to carve")
    print("  +/-       : change carve radius")
    print("  q or Esc  : quit")
    print()
    print("The display uses DOOR_CLEANUP_MODE='none' temporarily, so blocked doors")
    print("are visible. Copy the printed DOOR_CARVES line into map_processor.py.")
    print()

    def on_mouse(event, x_display, y_display, flags, params):
        nonlocal selected
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        x = int(x_display // mp.DISPLAY_SCALE)
        y = int(y_display // mp.DISPLAY_SCALE)
        if not (0 <= x < occupancy_map.shape[1] and 0 <= y < occupancy_map.shape[0]):
            return

        selected = (x, y)
        world_x, world_z = pixel_to_world(x, y, origin, resolution)
        state = "free" if occupancy_map[y, x] == 0 else "obstacle"
        rgb = tuple(int(v) for v in (map_img[y, x] * 255).round())

        print(f"clicked pixel=({x}, {y}), world=({world_x:.3f}, {world_z:.3f}), "
              f"state={state}, semantic_rgb={rgb}")
        print(f"DOOR_CARVES entry: (({x}, {y}), {radius}),")

    window = "Pick Doorway"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        cv2.imshow(window, make_display(map_img, occupancy_map, selected, radius))
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            break
        if key in (ord("+"), ord("=")):
            radius = min(8, radius + 1)
            print(f"radius={radius}")
        elif key in (ord("-"), ord("_")):
            radius = max(1, radius - 1)
            print(f"radius={radius}")
        elif selected is not None and key == ord("p"):
            x, y = selected
            print(f"DOOR_CARVES entry: (({x}, {y}), {radius}),")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
