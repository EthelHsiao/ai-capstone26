import numpy as np
import cv2
from PIL import Image
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb

# =============================
# Scene & Simulator Settings
# =============================
test_scene = "replica_v1/apartment_0/habitat/mesh_semantic.ply"

sim_settings = {
    "scene_id": test_scene,
    "default_agent": 0,
    "sensor_height": 1.5,
    "width": 512,
    "height": 512,
    "sensor_pitch": 0.0,  # x rotation in radians
}

# =============================
# Image Transformation Helpers
# =============================
def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def transform_depth(image):
    depth_img = (image / 10 * 255).astype(np.uint8)
    return depth_img

def transform_semantic(semantic_obs):
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_img = cv2.cvtColor(np.asarray(semantic_img), cv2.COLOR_RGB2BGR)
    return semantic_img

# =============================
# Simulator & Agent Setup
# =============================
def make_simple_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene_id"]

    # Define sensors
    sensor_specs = []
    for sensor_type, uuid in [
        (habitat_sim.SensorType.COLOR, "color_sensor"),
        (habitat_sim.SensorType.DEPTH, "depth_sensor"),
        (habitat_sim.SensorType.SEMANTIC, "semantic_sensor"),
    ]:
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = sensor_type
        spec.resolution = [settings["height"], settings["width"]]
        spec.position = [0.0, settings["sensor_height"], 0.0]
        spec.orientation = [settings["sensor_pitch"], 0.0, 0.0]  # Euler angles
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(spec)

    # Define agent actions
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.1)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

# Create simulator
cfg = make_simple_cfg(sim_settings)
sim = habitat_sim.Simulator(cfg)

# Initialize agent
agent = sim.initialize_agent(sim_settings["default_agent"])

# Set initial agent state
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([0.0, 0.0, 0.0])
agent.set_state(agent_state)

# =============================
# Action space and keys
# =============================
action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
print("Discrete action space: ", action_names)

FORWARD_KEY = "w"
LEFT_KEY = "a"
RIGHT_KEY = "d"
FINISH = "f"

print("#############################")
print("Use keyboard to control the agent:")
print(" w - move forward")
print(" a - turn left")
print(" d - turn right")
print(" f - finish and quit")
print("#############################")

# =============================
# Navigation & Visualization
# =============================
def navigateAndSee(action=""):
    if action in action_names:
        observations = sim.step(action)
        
        cv2.imshow("RGB", transform_rgb_bgr(observations["color_sensor"]))
        cv2.imshow("Depth", transform_depth(observations["depth_sensor"]))
        cv2.imshow("Semantic", transform_semantic(observations["semantic_sensor"]))
        
        # Print camera pose
        agent_state = agent.get_state()
        sensor_state = agent_state.sensor_states["color_sensor"]
        pos = sensor_state.position
        rot = sensor_state.rotation
        print("camera pose: x y z rw rx ry rz")
        print(pos[0], pos[1], pos[2], rot.w, rot.x, rot.y, rot.z)

# =============================
# Main loop
# =============================
action = "move_forward"
navigateAndSee(action)

while True:
    keystroke = cv2.waitKey(0)
    if keystroke == ord(FORWARD_KEY):
        action = "move_forward"
        navigateAndSee(action)
        print("action: FORWARD")
    elif keystroke == ord(LEFT_KEY):
        action = "turn_left"
        navigateAndSee(action)
        print("action: LEFT")
    elif keystroke == ord(RIGHT_KEY):
        action = "turn_right"
        navigateAndSee(action)
        print("action: RIGHT")
    elif keystroke == ord(FINISH):
        print("action: FINISH")
        break
    else:
        print("INVALID KEY")
        continue

cv2.destroyAllWindows()
sim.close()
