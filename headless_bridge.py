#!/usr/bin/env python3
"""
Headless MuJoCo bridge for testing — mirrors unitree_mujoco.py without the viewer.

Usage:
  python headless_bridge.py [--seconds N]   # run for N seconds then exit (0 = forever)
"""

import sys
import os
import time
import argparse
import threading
import mujoco

# Bridge lives in simulate_python alongside config.py
BRIDGE_DIR = os.path.join(os.path.dirname(__file__),
                          "src", "unitree_mujoco", "simulate_python")
sys.path.insert(0, BRIDGE_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "src", "unitree_sdk2_python"))


from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge
import config


def main():
    parser = argparse.ArgumentParser(description="Headless MuJoCo bridge")
    parser.add_argument("--seconds", type=float, default=0,
                        help="Stop after N seconds (0 = run forever)")
    parser.add_argument("--interface", default=config.INTERFACE)
    parser.add_argument("--domain",    default=config.DOMAIN_ID, type=int)
    args = parser.parse_args()

    # config.ROBOT_SCENE is relative to the simulate_python directory
    scene_path = os.path.join(BRIDGE_DIR, config.ROBOT_SCENE)
    mj_model = mujoco.MjModel.from_xml_path(scene_path)
    mj_data  = mujoco.MjData(mj_model)
    mj_model.opt.timestep = config.SIMULATE_DT

    ChannelFactoryInitialize(args.domain, args.interface)
    bridge = UnitreeSdk2Bridge(mj_model, mj_data)

    print("[headless_bridge] Running. Publishing rt/lowstate + rt/sportmodestate",
          flush=True)

    deadline = time.perf_counter() + args.seconds if args.seconds > 0 else None

    while True:
        step_start = time.perf_counter()
        mujoco.mj_step(mj_model, mj_data)
        if deadline and time.perf_counter() >= deadline:
            print("[headless_bridge] Done.", flush=True)
            break
        elapsed = time.perf_counter() - step_start
        sleep = config.SIMULATE_DT - elapsed
        if sleep > 0:
            time.sleep(sleep)


if __name__ == "__main__":
    main()
