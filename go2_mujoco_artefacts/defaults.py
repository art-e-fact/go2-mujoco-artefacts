import os
import platform as _platform

ROBOT = "go2"
BASE_DIR = os.path.dirname(__file__)
ROBOT_DIR = os.path.normpath(
    os.path.join(BASE_DIR, "../src/unitree_mujoco/unitree_robots", ROBOT)
)
box_scene = os.path.join(ROBOT_DIR, "scene.xml")
rail_scene = os.path.join(BASE_DIR, "../resources/scene_rail_track.xml")
DEFAULT_SCENE = rail_scene
DEFAULT_WTW_DIR = os.path.join(BASE_DIR, "../src/unitree_mujoco/simulate_python/wtw")
DEFAULT_WTW_CFG = os.path.join(DEFAULT_WTW_DIR, "parameters_cpu.pkl")
DEFAULT_DOMAIN_ID = 1
DEFAULT_INTERFACE = "lo0" if _platform.system() == "Darwin" else "lo"

SIMULATE_DT = 0.005
VIEWER_DT = 0.02
TRANSITION_DURATION = 2.0
WTW_HZ = 50
WTW_STEP_EVERY = max(1, round(1.0 / (WTW_HZ * SIMULATE_DT)))
IDLE_SETTLE_TICKS = round(0.5 / SIMULATE_DT)
 
