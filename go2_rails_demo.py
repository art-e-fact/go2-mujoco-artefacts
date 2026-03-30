#!/usr/bin/env python3
"""
Go2 Rail-Following Demo

Generates a procedural rail network and drives the Go2 along a chosen road
using a simple pursuit controller via the high-level SportClient API.

Usage:
  python go2_rails_demo.py              # MuJoCo viewer (UI)
  python go2_rails_demo.py --headless   # no display (CI / testing)

"""

import sys
import os
import math
import time
import threading
import subprocess
from utils import get_python_executable, sim_sleep, FrontCameraRecorder

_HERE    = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR = os.path.join(_HERE, "src", "unitree_mujoco", "simulate_python")
_SDK_DIR = os.path.join(_HERE, "src", "unitree_sdk2_python")

sys.path.insert(0, _SDK_DIR)
sys.path.insert(0, _SIM_DIR)


def _drain(proc, events):
    """Print subprocess stdout; set events[i] when events[i][0] marker appears."""
    for line in proc.stdout:
        print(f"  [sim] {line.rstrip()}")
        for marker, event in events:
            if marker and marker in line:
                event.set()


def _stop(procs):
    for proc in reversed(procs):
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def _build_waypoints(road):
    """Extract (x, y, heading_rad) waypoints from a road."""
    return [(x, y, math.radians(h)) for x, y, h in road]


def _pursuit_step(robot_x, robot_y, robot_yaw, waypoints, lookahead=1.0):
    """Compute (vx, vyaw) to steer toward the nearest waypoint + lookahead.

    Returns (vx, vyaw, done). done=True when past the last waypoint.
    """
    # Find closest waypoint
    best_i, best_d2 = 0, float("inf")
    for i, (wx, wy, _) in enumerate(waypoints):
        d2 = (wx - robot_x) ** 2 + (wy - robot_y) ** 2
        if d2 < best_d2:
            best_i, best_d2 = i, d2

    # Target a point ~lookahead ahead on the path
    target_i = best_i
    for i in range(best_i, len(waypoints)):
        dx = waypoints[i][0] - robot_x
        dy = waypoints[i][1] - robot_y
        if math.sqrt(dx * dx + dy * dy) >= lookahead:
            target_i = i
            break
    else:
        target_i = len(waypoints) - 1

    tx, ty, _ = waypoints[target_i]
    dx, dy = tx - robot_x, ty - robot_y
    desired_yaw = math.atan2(dy, dx)

    # Heading error (wrap to [-pi, pi])
    err = desired_yaw - robot_yaw
    err = (err + math.pi) % (2 * math.pi) - math.pi

    done = best_i >= len(waypoints) - 2 and best_d2 < lookahead ** 2
    return err, done, tx, ty


def main():
    import argparse
    import numpy as np
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.sport.sport_client import SportClient
    import config
    from rail_gen import RailwayScene, TerrainSpec

    parser = argparse.ArgumentParser(description="Go2 Rail-Following Demo")
    parser.add_argument("--seed",      type=int,   default=None,  help="Random seed")
    parser.add_argument("--n-roads",   type=int,   default=1,     help="Number of rail roads")
    parser.add_argument("--interface", default=config.INTERFACE,   help="Network interface")
    parser.add_argument("--domain",    type=int,   default=0,     help="DDS domain ID")
    parser.add_argument("--headless",  action="store_true",        help="No viewer (CI / testing)")
    parser.add_argument("--telemetry", metavar="PATH", default=None,
                        help="Write simulation state (qpos/qvel) as JSONL to PATH")
    parser.add_argument("--record",       metavar="PATH", default=None,
                        help="Save spectator-view recording (passed to sport_mujoco.py)")
    parser.add_argument("--record-front", metavar="PATH", default=None,
                        help="Save front-camera recording to PATH")
    parser.add_argument("--heightmap",       action="store_true",
                        help="Enable HeightMap_ DDS publishing in the sim")
    parser.add_argument("--heightmap-debug", action="store_true",
                        help="Visualise height map rays in the viewer")
    parser.add_argument("--v-forward",  type=float, default=0.4,  help="Forward velocity (m/s)")
    parser.add_argument("--yaw-gain",   type=float, default=2.0,  help="Proportional yaw gain")
    parser.add_argument("--lookahead",  type=float, default=1.5,  help="Pursuit lookahead distance (m)")
    parser.add_argument("--terrain",    action="store_true",       help="Enable terrain heightfield (off by default)")
    args = parser.parse_args()

    # --- Generate rail scene ---
    rng = np.random.default_rng(args.seed)
    terrain = TerrainSpec() if args.terrain else None
    scene = RailwayScene.build(rng, n_roads=args.n_roads, terrain=terrain)

    waypoints = _build_waypoints(scene.net.roads[0])
    start_pos = waypoints[0] if waypoints else None
    print(f"Rail scene: {len(scene.net.roads)} roads, following road 0 "
          f"({len(waypoints)} waypoints)")

    scene_path = scene.save_mujoco_scene(_HERE, start_pos=start_pos)
    print(f"Saved MuJoCo scene to {scene_path}")

    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": _SDK_DIR}
    procs = []
    recorder = None

    try:
        # --- sport_mujoco.py ---
        sim_cmd = [get_python_executable(), "-u", os.path.join(_SIM_DIR, "sport_mujoco.py"),
                   "--interface", args.interface, "--domain", str(args.domain),
                   "--scene", scene_path, "--keyframe", "rail_start"]
        if args.headless:
            sim_cmd.append("--headless")
        if args.record:
            sim_cmd += ["--record", os.path.abspath(args.record)]
        if args.telemetry:
            sim_cmd += ["--telemetry", os.path.abspath(args.telemetry)]
        if args.heightmap:
            sim_cmd.append("--heightmap")
        if args.heightmap_debug:
            sim_cmd.append("--heightmap-debug")

        sim_proc = subprocess.Popen(
            sim_cmd, cwd=_SIM_DIR,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=env,
        )
        sim_ready    = threading.Event()
        sim_standing = threading.Event()
        threading.Thread(target=_drain, args=(sim_proc,
                                              [("Serving sport RPC", sim_ready),
                                               ("Standing complete.", sim_standing)]),
                         daemon=True).start()
        procs.append(sim_proc)

        assert sim_ready.wait(timeout=60), "sport_mujoco did not load in time"
        if not args.headless:
            time.sleep(1.0)
        assert sim_standing.wait(timeout=20), "sport_mujoco did not reach standing pose"
        time.sleep(1.5)  # WTW pre-warming

        # --- SportClient ---
        ChannelFactoryInitialize(args.domain, args.interface)
        client = SportClient()
        client.SetTimeout(10.0)
        client.Init()

        telemetry_path = os.path.abspath(args.telemetry) if args.telemetry else None
        sleep = (lambda dt: sim_sleep(dt, telemetry_path)) if telemetry_path else time.sleep

        if args.record_front:
            recorder = FrontCameraRecorder(args.record_front)
            recorder.start()

        print(f"\n=== Go2 Rail-Following Demo ===")
        print(f"v_forward={args.v_forward}  yaw_gain={args.yaw_gain}  "
              f"lookahead={args.lookahead}")
        print("=" * 50 + "\n")

        # --- Set up high-state subscriber to get robot pose ---
        from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
        from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Pose_, Point_, Quaternion_

        marker_pub = ChannelPublisher("rt/human_marker_pose", Pose_)
        marker_pub.Init()

        robot_state = {"x": 0.0, "y": 0.0, "yaw": 0.0, "ready": False}
        state_lock = threading.Lock()

        def _on_high_state(msg):
            with state_lock:
                robot_state["x"] = msg.position[0]
                robot_state["y"] = msg.position[1]
                robot_state["yaw"] = msg.imu_state.rpy[2]
                robot_state["ready"] = True

        state_sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        state_sub.Init(_on_high_state, 10)

        # Wait for first state update
        for _ in range(100):
            with state_lock:
                if robot_state["ready"]:
                    break
            time.sleep(0.05)

        # --- Pursuit loop ---
        dt = 0.1
        done = False
        step_count = 0
        while not done:
            with state_lock:
                rx, ry, ryaw = robot_state["x"], robot_state["y"], robot_state["yaw"]

            yaw_err, done, tx, ty = _pursuit_step(rx, ry, ryaw, waypoints,
                                                  lookahead=args.lookahead)
            marker_pub.Write(Pose_(Point_(tx, ty, 0.9), Quaternion_(0, 0, 0, 1)))
            print(f"Target: ({tx:.2f}, {ty:.2f})  Robot: ({rx:.2f}, {ry:.2f})  ")
            vyaw = args.yaw_gain * yaw_err
            vyaw = max(-2.5, min(2.5, vyaw))  # clamp rotation speed
            
            client.Move(args.v_forward, 0.0, vyaw * 2)
            sleep(dt)
            step_count += 1

            if step_count % 50 == 0:
                print(f"  step {step_count}: pos=({rx:.2f}, {ry:.2f}) "
                      f"yaw={math.degrees(ryaw):.1f}° err={math.degrees(yaw_err):.1f}°")

        client.StopMove()
        print(f"\nReached end of road after {step_count} steps.")

    finally:
        if recorder is not None:
            recorder.stop()
        _stop(procs)
        # Clean up temp scene file
        if os.path.exists(scene_path):
            os.remove(scene_path)
            print(f"Cleaned up {scene_path}")


if __name__ == "__main__":
    main()
