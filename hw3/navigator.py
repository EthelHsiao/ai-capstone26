import math
import os
from contextlib import contextmanager
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb

SCENE_PATH = "../hw0/replica_v1/apartment_0/habitat/mesh_semantic.ply"
SENSOR_HEIGHT = 1.5
SENSOR_WIDTH = 512
SENSOR_HEIGHT_PX = 512
SENSOR_PITCH = 0.0
MOVE_AMOUNT = 0.05
TURN_AMOUNT = 1.0
INITIAL_HEADING = math.pi
SUPPRESS_HABITAT_INIT_LOGS = True

# Default action names
MOVE_FORWARD = "move_forward"
TURN_LEFT = "turn_left"
TURN_RIGHT = "turn_right"

# =============================
# Image Formatting
# =============================
def _transform_rgb_bgr(image: np.ndarray) -> np.ndarray:
    """Convert RGB to BGR for OpenCV display."""
    return image[:, :, [2, 1, 0]]

def _transform_depth(image: np.ndarray) -> np.ndarray:
    """Normalize and convert depth to a displayable uint8 image."""
    return (image / 10.0 * 255).astype(np.uint8)

def _transform_semantic(semantic_obs: np.ndarray) -> np.ndarray:
    """Convert raw semantic map to a colorized image."""
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    return cv2.cvtColor(np.asarray(semantic_img), cv2.COLOR_RGB2BGR)

# =============================
# Simulator Core
# =============================
@contextmanager
def _suppress_native_output(enabled: bool = True):
    """Temporarily silence C/C++ stdout/stderr noise during simulator setup."""
    if not enabled:
        yield
        return

    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)
        os.close(devnull_fd)


def init_sim(scene_path: str = SCENE_PATH, start_x: float = 0.9, start_z: float = 4.6):
    """Initialize the Habitat simulator environment and set the agent's start state."""
    sim_settings = {
        "scene": scene_path,
        "default_agent": 0,
        "sensor_height": SENSOR_HEIGHT,
        "width": SENSOR_WIDTH,
        "height": SENSOR_HEIGHT_PX,
        "sensor_pitch": SENSOR_PITCH,
    }

    # Global Config
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = sim_settings["scene"]

    # Agent Config
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    specs = []

    # Define sensors
    for uuid, stype in [
        ("color_sensor", habitat_sim.SensorType.COLOR),
        ("depth_sensor", habitat_sim.SensorType.DEPTH),
        ("semantic_sensor", habitat_sim.SensorType.SEMANTIC),
    ]:
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = stype
        spec.resolution = [sim_settings["height"], sim_settings["width"]]
        spec.position = [0.0, sim_settings["sensor_height"], 0.0]
        spec.orientation = [sim_settings["sensor_pitch"], 0.0, 0.0]
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        specs.append(spec)

    agent_cfg.sensor_specifications = specs

    # Define action space
    agent_cfg.action_space = {
        MOVE_FORWARD: habitat_sim.agent.ActionSpec(
            MOVE_FORWARD, habitat_sim.agent.ActuationSpec(amount=MOVE_AMOUNT)
        ),
        TURN_LEFT: habitat_sim.agent.ActionSpec(
            TURN_LEFT, habitat_sim.agent.ActuationSpec(amount=TURN_AMOUNT)
        ),
        TURN_RIGHT: habitat_sim.agent.ActionSpec(
            TURN_RIGHT, habitat_sim.agent.ActuationSpec(amount=TURN_AMOUNT)
        ),
    }

    with _suppress_native_output(SUPPRESS_HABITAT_INIT_LOGS):
        # Initialize Simulator
        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
        sim = habitat_sim.Simulator(cfg)

        # Initialize Agent at starting coordinates
        agent = sim.initialize_agent(sim_settings["default_agent"])
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array([start_x, 0.0, start_z])  # World translation
        agent.set_state(agent_state)

    print("Habitat simulator initialized successfully.")
    return sim, agent, list(agent_cfg.action_space.keys())


def navigate_and_see(sim, agent, action: str, goal_index: int = None):
    """
    Step the simulator by one action, display the sensor observations in OpenCV,
    and visually highlight the physical destination.
    """
    if action not in [MOVE_FORWARD, TURN_LEFT, TURN_RIGHT]:
        return

    obs = sim.step(action)
    rgb = _transform_rgb_bgr(obs["color_sensor"])
    depth = _transform_depth(obs["depth_sensor"])
    semantic_labels = obs["semantic_sensor"]
    target_mask_pixels = 0

    # Overlay goal label if provided
    if goal_index is not None:
        goal_indices = np.atleast_1d(goal_index).astype(np.uint32)
        mask = np.isin(semantic_labels, goal_indices)
        target_mask_pixels = int(np.count_nonzero(mask))
        if np.any(mask):
            overlay = rgb.copy()
            overlay[mask] = np.array([0, 0, 255], dtype=overlay.dtype)
            rgb = cv2.addWeighted(overlay, 0.3, rgb, 0.7, 0)
    obs["target_mask_pixels"] = target_mask_pixels

    cv2.imshow("RGB", rgb)
    cv2.imshow("Depth", depth)
    cv2.imshow("Semantic", _transform_semantic(semantic_labels))
    cv2.waitKey(1)

    return obs


def execute_waypoint_path(path_world: List[Tuple[float, float]], sim, agent, goal_idx: int = None):
    """
    Convert a sequence of world 3D waypoints into simulator actuation actions
    (turning and moving forward).
    """
    heading = INITIAL_HEADING
    frames = 0
    seen_frames = 0
    max_mask_pixels = 0
    action_counts = {
        MOVE_FORWARD: 0,
        TURN_LEFT: 0,
        TURN_RIGHT: 0,
    }

    def step_and_count(action: str):
        nonlocal frames, seen_frames, max_mask_pixels
        obs = navigate_and_see(sim, agent, action, goal_idx)
        if obs is None:
            return
        action_counts[action] += 1
        frames += 1
        mask_pixels = int(obs.get("target_mask_pixels", 0))
        if mask_pixels > 0:
            seen_frames += 1
            max_mask_pixels = max(max_mask_pixels, mask_pixels)

    for i in range(1, len(path_world)):
        x0, z0 = path_world[i - 1]
        x1, z1 = path_world[i]
        dx, dz = x1 - x0, z1 - z0

        target_angle = math.atan2(dx, dz)
        dtheta = (target_angle - heading + math.pi) % (2 * math.pi) - math.pi

        # 1. Turn to align the agent towards the target angle
        turn_steps = int(round(abs(math.degrees(dtheta)) / TURN_AMOUNT))
        action = TURN_LEFT if dtheta > 0 else TURN_RIGHT
        for _ in range(turn_steps):
            step_and_count(action)

        # 2. Step forward physically toward the waypoint
        steps_forward = int(round(math.sqrt(dx**2 + dz**2) / MOVE_AMOUNT))
        for _ in range(steps_forward):
            step_and_count(MOVE_FORWARD)

        # Update current heading tracker
        heading = target_angle

    if goal_idx is not None and seen_frames == 0:
        print("Target mask was not visible during motion; scanning in place...")
        for _ in range(360):
            step_and_count(TURN_LEFT)
            if seen_frames > 0:
                break

    print("Agent path execution completed.")
    return {
        "frames": frames,
        "seen_frames": seen_frames,
        "max_mask_pixels": max_mask_pixels,
        "forward": action_counts[MOVE_FORWARD],
        "left": action_counts[TURN_LEFT],
        "right": action_counts[TURN_RIGHT],
    }
