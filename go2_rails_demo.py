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


def _build_path(waypoints):
    """Return cumulative arc-length distances and (x,y) arrays for waypoints."""
    xs = [w[0] for w in waypoints]
    ys = [w[1] for w in waypoints]
    dists = [0.0]
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i - 1]
        dy = ys[i] - ys[i - 1]
        dists.append(dists[-1] + math.sqrt(dx * dx + dy * dy))
    return dists, xs, ys


def _sample_path(dists, xs, ys, s):
    """Interpolate (x, y) at arc-length distance s along the path."""
    s = max(0.0, min(s, dists[-1]))
    for i in range(1, len(dists)):
        if dists[i] >= s:
            t = (s - dists[i - 1]) / (dists[i] - dists[i - 1]) if dists[i] > dists[i - 1] else 0.0
            return xs[i - 1] + t * (xs[i] - xs[i - 1]), ys[i - 1] + t * (ys[i] - ys[i - 1])
    return xs[-1], ys[-1]


def _path_tangent(dists, xs, ys, s):
    """Return the tangent angle (rad) of the path at arc-length s."""
    s = max(0.0, min(s, dists[-1]))
    for i in range(1, len(dists)):
        if dists[i] >= s:
            return math.atan2(ys[i] - ys[i - 1], xs[i] - xs[i - 1])
    return math.atan2(ys[-1] - ys[-2], xs[-1] - xs[-2])


def _closest_path_s(dists, xs, ys, px, py):
    """Return the arc-length s of the point on the path closest to (px, py)."""
    best_s, best_d2 = 0.0, float("inf")
    for i in range(1, len(dists)):
        # Project (px,py) onto segment [i-1, i]
        ax, ay = xs[i - 1], ys[i - 1]
        bx, by = xs[i], ys[i]
        dx, dy = bx - ax, by - ay
        seg_len2 = dx * dx + dy * dy
        if seg_len2 < 1e-12:
            continue
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len2))
        cx, cy = ax + t * dx, ay + t * dy
        d2 = (px - cx) ** 2 + (py - cy) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_s = dists[i - 1] + t * (dists[i] - dists[i - 1])
    return best_s


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
    parser.add_argument("--v-forward",     type=float, default=0.4,  help="Forward velocity (m/s)")
    parser.add_argument("--yaw-gain",      type=float, default=2.0,  help="Heading alignment gain")
    parser.add_argument("--lateral-gain",  type=float, default=1.5,  help="Lateral centering gain")
    parser.add_argument("--target-speed",  type=float, default=0.3,  help="Target speed along path (m/s)")
    parser.add_argument("--target-lead",   type=float, default=1.0,  help="Initial target lead distance (m)")
    parser.add_argument("--terrain",       action="store_true",       help="Enable terrain heightfield (off by default)")
    parser.add_argument("--teleop",        action="store_true",       help="Control robot with gamepad instead of auto")
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
        sim_cmd.append("--uwb")

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
              f"target_speed={args.target_speed}")
        print("=" * 50 + "\n")

        # --- Gamepad (teleop mode) ------------------------------------------
        joy = None
        if args.teleop:
            import pygame
            pygame.init()
            if pygame.joystick.get_count() == 0:
                print("ERROR: --teleop requested but no gamepad found")
                return
            joy = pygame.joystick.Joystick(0)
            joy.init()
            print(f"[teleop] Using gamepad: {joy.get_name()}")

        # --- Set up UWB subscriber to get robot pose ---
        from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import UwbState_
        from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Pose_, Point_, Quaternion_

        marker_pub = ChannelPublisher("rt/human_marker_pose", Pose_)
        marker_pub.Init()

        uwb = {"az": 0.0, "pitch": 0.0, "dist": 0.0, "yaw": 0.0, "ready": False}
        uwb_lock = threading.Lock()

        def _on_uwb(msg):
            with uwb_lock:
                uwb["az"] = msg.orientation_est
                uwb["pitch"] = msg.pitch_est
                uwb["dist"] = msg.distance_est
                uwb["yaw"] = msg.base_yaw
                uwb["ready"] = True

        uwb_sub = ChannelSubscriber("rt/uwbstate", UwbState_)
        uwb_sub.Init(_on_uwb, 10)

        # Publish initial marker so UWB has a target to measure
        path_dists, path_xs, path_ys = _build_path(waypoints)
        path_s = args.target_lead  # current arc-length position of target
        tx, ty = _sample_path(path_dists, path_xs, path_ys, path_s)
        marker_pub.Write(Pose_(Point_(tx, ty, 0.9), Quaternion_(0, 0, 0, 1)))

        # Wait for first UWB update
        for _ in range(100):
            with uwb_lock:
                if uwb["ready"]:
                    break
            time.sleep(0.05)

        # --- Control loop ---
        DEADZONE = 0.1
        dt = 0.1
        step_count = 0
        MIN_DIST = 1.0

        if joy:
            # --- Teleop mode: gamepad controls the robot directly ---
            print("[teleop] Gamepad active. Left stick = move, right stick X = rotate.")
            import pygame
            while True:
                pygame.event.pump()
                jlx = joy.get_axis(0)   # left stick X → lateral
                jly = -joy.get_axis(1)  # left stick Y → forward (inverted)
                jrx = joy.get_axis(3)   # right stick X → yaw

                vx = jly * args.v_forward if abs(jly) > DEADZONE else 0.0
                vy = -jlx * 0.3 if abs(jlx) > DEADZONE else 0.0
                vyaw = -jrx * 2.5 if abs(jrx) > DEADZONE else 0.0

                client.Move(vx, vy, vyaw)
                sleep(dt)
                step_count += 1

                if step_count % 50 == 0:
                    print(f"  step {step_count}: vx={vx:.2f} vy={vy:.2f} vyaw={vyaw:.2f}")
        else:
            # --- Auto mode: follow path with UWB-based control ---
            while path_s < path_dists[-1]:
                # Advance target along the path at constant speed
                path_s += args.target_speed * dt
                tx, ty = _sample_path(path_dists, path_xs, path_ys, path_s)
                marker_pub.Write(Pose_(Point_(tx, ty, 0.9), Quaternion_(0, 0, 0, 1)))

                with uwb_lock:
                    tag_dist = uwb["dist"]
                    ryaw = uwb["yaw"]

                # Reconstruct robot world position from UWB
                az_r = uwb["az"]
                cos_y, sin_y = math.cos(ryaw), math.sin(ryaw)
                lx = tag_dist * math.cos(az_r)
                ly = tag_dist * math.sin(az_r)
                rx = tx - (cos_y * lx - sin_y * ly)
                ry = ty - (sin_y * lx + cos_y * ly)

                # Find closest point on path and its tangent
                robot_s = _closest_path_s(path_dists, path_xs, path_ys, rx, ry)
                tangent = _path_tangent(path_dists, path_xs, path_ys, robot_s)
                cx, cy = _sample_path(path_dists, path_xs, path_ys, robot_s)

                # Lateral offset: positive = robot is to the left of the path
                dx, dy = rx - cx, ry - cy
                lateral_err = -math.sin(tangent) * dx + math.cos(tangent) * dy

                # Heading error: difference between robot yaw and path tangent
                heading_err = (tangent - ryaw + math.pi) % (2 * math.pi) - math.pi

                # Forward: slow down / stop when too close to tag
                vx = 0.0 if tag_dist < MIN_DIST else args.v_forward * min(1.0, (tag_dist - MIN_DIST) / MIN_DIST)
                # Lateral: push robot back toward path center
                vy = -args.lateral_gain * lateral_err
                vy = max(-0.3, min(0.3, vy))
                # Yaw: align with path tangent
                vyaw = args.yaw_gain * heading_err
                vyaw = max(-2.5, min(2.5, vyaw))

                client.Move(vx, vy, vyaw)
                sleep(dt)
                step_count += 1

                if step_count % 50 == 0:
                    print(f"  step {step_count}: target=({tx:.2f}, {ty:.2f}) "
                          f"lat={lateral_err:+.2f}m hdg={math.degrees(heading_err):+.1f}°")

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
