import habitat_sim
import copy
from habitat.sims.habitat_simulator.actions import HabitatSimActions

DEFAULT_SETTINGS = {
    "default_agent": 0,
    "enable_physics": False,
    "rgb_height": 480,
    "rgb_width": 640,
    "depth_height": 480,
    "depth_width": 640,
    "sensor_height": 0.88,
    "hfov": 90
}



def make_cfg(scene_file_glb, settings):
    cfg = copy.deepcopy(DEFAULT_SETTINGS)
    cfg.update(settings)

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = scene_file_glb
    sim_cfg.enable_physics = cfg["enable_physics"]

    # Note: all sensors must have the same resolution
    sensor_specs = []
    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [cfg["rgb_height"], cfg["rgb_width"]]
    color_sensor_spec.position = [0.0, cfg["sensor_height"], 0.0]
    color_sensor_spec.hfov = cfg['hfov']
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(color_sensor_spec)

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [cfg["depth_height"], cfg["depth_width"]]
    depth_sensor_spec.position = [0.0, cfg["sensor_height"], 0.0]
    depth_sensor_spec.hfov = cfg['hfov']
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(depth_sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "stop": habitat_sim.ActionSpec("stop"),
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=15.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=15.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])