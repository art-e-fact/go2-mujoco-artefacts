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


def _analyze_rails(data, width, height, resolution, forward_yaw, radius=1.0):
    """Detect rail heading and lateral offset from a heightmap.

    Args:
        data: flat float32 array (width*height), EMPTY=1e9 for missing cells.
        width, height, resolution: grid parameters.
        forward_yaw: approximate forward direction (rad, world frame) to
                     disambiguate the ±180° PCA ambiguity.
        radius: only consider cells within this distance (m) from the robot.

    Returns:
        (rail_heading_world, lateral_offset) or None if insufficient data.
        rail_heading_world: heading of the rails in world frame (rad).
        lateral_offset: signed distance from robot to rail midline (m);
                        positive = robot is to the left of center.
    """
    import numpy as np

    valid = data[data < 1e8]
    if len(valid) < 20:
        return None
    # Rails are the highest features — keep cells above the 90th percentile
    thresh = np.percentile(valid, 99)
    rail_mask = (data >= thresh) & (data < 1e8)
    if rail_mask.sum() < 10:
        return None

    # Cell positions relative to grid center (= robot).
    # The grid is axis-aligned with the world frame, so these offsets
    # are world-frame relative displacements from the robot.
    idx = np.argwhere(rail_mask.reshape(height, width))  # (N, 2) as [iy, ix]
    cx = (idx[:, 1] - width / 2.0 + 0.5) * resolution   # world-x offset
    cy = (idx[:, 0] - height / 2.0 + 0.5) * resolution   # world-y offset

    # Only keep cells within `radius` of the robot to ignore distant turns.
    near = cx**2 + cy**2 <= radius**2
    cx, cy = cx[near], cy[near]
    if len(cx) < 10:
        return None

    # Split into two rail clusters using the gap in perpendicular projection.
    perp_fwd = -np.sin(forward_yaw) * cx + np.cos(forward_yaw) * cy
    order = np.argsort(perp_fwd)
    sorted_perp = perp_fwd[order]
    gaps = np.diff(sorted_perp)
    split = np.argmax(gaps)
    if gaps[split] < 2 * resolution or min(split + 1, len(cx) - split - 1) < 3:
        return None  # no clear two-rail separation

    # Recenter each rail cluster to remove inter-rail lateral spread.
    mask_a, mask_b = order[:split + 1], order[split + 1:]
    ca = np.column_stack([cx[mask_a] - cx[mask_a].mean(),
                          cy[mask_a] - cy[mask_a].mean()])
    cb = np.column_stack([cx[mask_b] - cx[mask_b].mean(),
                          cy[mask_b] - cy[mask_b].mean()])
    pts = np.vstack([ca, cb])

    # PCA on recentered points: principal axis = rail direction
    cov = np.cov(pts[:, 0], pts[:, 1])  # 2×2
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, 1]  # largest eigenvalue
    rail_heading = np.arctan2(principal[1], principal[0])

    # Disambiguate ±180°: pick the direction closer to forward_yaw
    diff = (rail_heading - forward_yaw + np.pi) % (2 * np.pi) - np.pi
    if abs(diff) > np.pi / 2:
        rail_heading += np.pi
    rail_heading = (rail_heading + np.pi) % (2 * np.pi) - np.pi

    # Lateral offset: average of both rail centroids projected onto perp axis.
    perp_x, perp_y = -np.sin(rail_heading), np.cos(rail_heading)
    mid_x = (cx[mask_a].mean() + cx[mask_b].mean()) / 2
    mid_y = (cy[mask_a].mean() + cy[mask_b].mean()) / 2
    lateral_offset = -(mid_x * perp_x + mid_y * perp_y)

    return rail_heading, lateral_offset, np.column_stack([cx, cy])


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
    parser.add_argument("--no-heightmap",     action="store_false", dest="heightmap", default=True,
                        help="Disable HeightMap_ DDS publishing in the sim")
    parser.add_argument("--heightmap-debug", action="store_true",
                        help="Visualise height map rays in the viewer")
    parser.add_argument("--v-forward",     type=float, default=1.0,  help="Forward velocity (m/s)")
    parser.add_argument("--yaw-gain",      type=float, default=1.0,  help="Heading alignment gain")
    parser.add_argument("--lateral-gain",  type=float, default=1.0,  help="Lateral centering gain")
    parser.add_argument("--target-speed",  type=float, default=0.3,  help="Target speed along path (m/s)")
    parser.add_argument("--target-lead",   type=float, default=1.0,  help="Initial target lead distance (m)")
    parser.add_argument("--no-terrain",       action="store_false", dest="terrain", default=True, help="Disable terrain heightfield")
    parser.add_argument("--teleop",        action="store_true",       help="Control robot with gamepad instead of auto")
    parser.add_argument("--rerun",         action="store_true",       help="Stream data to Rerun viewer")
    parser.add_argument("--heightmap-nav", action="store_true",
                        help="Steer using heightmap rail detection instead of path geometry")
    parser.add_argument("--policy", choices=["wtw", "rsl_rl"], default="wtw",
                        help="Locomotion policy (default: wtw)")
    args = parser.parse_args()

    if args.heightmap_nav:
        args.heightmap = True

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

    if args.rerun:
        import rerun as rr
        rr.init("go2_rails_demo", spawn=True)
        scene.log_rerun()

    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": _SDK_DIR}
    procs = []
    recorder = None

    try:
        # --- sport_mujoco.py ---
        sim_cmd = [get_python_executable(), "-u", os.path.join(_SIM_DIR, "sport_mujoco.py"),
                   "--interface", args.interface, "--domain", str(args.domain),
                   "--scene", scene_path, "--keyframe", "rail_start",
                   "--policy", args.policy]
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
            text=True, env=env, start_new_session=True,
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

        # --- Heightmap subscriber (for --rerun and/or --heightmap-nav) ---
        hm_sub = None
        hm_data = None   # latest heightmap data array
        hm_msg = None    # latest HeightMap_ message
        if args.rerun or args.heightmap_nav:
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import HeightMap_
            hm_sub = ChannelSubscriber("rt/utlidar/height_map_array", HeightMap_)
            hm_sub.Init()

        def _poll_heightmap():
            """Read latest HeightMap_, update hm_data/hm_msg, optionally log to Rerun."""
            nonlocal hm_data, hm_msg
            if hm_sub is None:
                return
            msg = hm_sub.Read()
            if msg is None:
                return
            hm_msg = msg
            hm_data = np.array(msg.data, dtype=np.float32)
            if args.rerun:
                mask = hm_data < 1e8
                if mask.any():
                    ix = np.arange(len(hm_data)) % msg.width
                    iy = np.arange(len(hm_data)) // msg.width
                    rr.set_time("sim_time", timestamp=msg.stamp)
                    rr.log("heightmap", rr.Points3D(
                        np.column_stack([
                            msg.origin[0] + ix[mask] * msg.resolution,
                            msg.origin[1] + iy[mask] * msg.resolution,
                            hm_data[mask],
                        ]),
                        radii=0.015,
                    ))

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
                _poll_heightmap()
                step_count += 1
        elif args.heightmap_nav:
            # --- Heightmap-nav mode: steer using rail detection ---
            while path_s < path_dists[-1]:
                path_s += args.target_speed * dt
                tx, ty = _sample_path(path_dists, path_xs, path_ys, path_s)
                marker_pub.Write(Pose_(Point_(tx, ty, 0.9), Quaternion_(0, 0, 0, 1)))

                with uwb_lock:
                    tag_dist = uwb["dist"]
                    ryaw = uwb["yaw"]
                    az = uwb["az"]

                _poll_heightmap()

                # Approximate forward direction in world frame
                forward_yaw = ryaw + az

                result = None
                if hm_data is not None and hm_msg is not None:
                    result = _analyze_rails(hm_data, hm_msg.width, hm_msg.height,
                                            hm_msg.resolution, forward_yaw)

                if result is None:
                    client.StopMove()
                    if step_count % 20 == 0:
                        print("  [heightmap-nav] WARNING: no rails detected, stopping")
                    sleep(dt)
                    step_count += 1
                    continue

                rail_heading, lateral_err, rail_xy = result
                heading_err = (rail_heading - ryaw + math.pi) % (2 * math.pi) - math.pi

                vx = 0.0 if tag_dist < MIN_DIST else args.v_forward * min(1.0, (tag_dist - MIN_DIST) / MIN_DIST)
                vy = max(-0.6, min(0.6, -args.lateral_gain * lateral_err))
                vyaw = max(-2.5, min(2.5, args.yaw_gain * heading_err))


                client.Move(vx, vy, vyaw)

                if args.rerun:
                    # Robot world position = grid center
                    rx = hm_msg.origin[0] + hm_msg.width * hm_msg.resolution / 2
                    ry = hm_msg.origin[1] + hm_msg.height * hm_msg.resolution / 2
                    rz = 0.35
                    L = 2.0  # arrow length
                    rr.log("rail_analysis/forward_yaw", rr.Arrows3D(
                        origins=[[rx, ry, rz]],
                        vectors=[[L * math.cos(forward_yaw), L * math.sin(forward_yaw), 0]],
                        colors=[[0, 200, 0]],
                    ))
                    rr.log("rail_analysis/rail_heading", rr.Arrows3D(
                        origins=[[rx, ry, rz]],
                        vectors=[[L * math.cos(rail_heading), L * math.sin(rail_heading), 0]],
                        colors=[[255, 255, 0]],
                    ))
                    rr.log("rail_analysis/robot_yaw", rr.Arrows3D(
                        origins=[[rx, ry, rz]],
                        vectors=[[L * math.cos(ryaw), L * math.sin(ryaw), 0]],
                        colors=[[100, 100, 255]],
                    ))
                    # Rail cells (world coords)
                    rr.log("rail_analysis/cells", rr.Points3D(
                        np.column_stack([rx + rail_xy[:, 0], ry + rail_xy[:, 1],
                                         np.full(len(rail_xy), rz)]),
                        colors=[[255, 50, 50]], radii=0.03,
                    ))
                    # Lateral offset line (robot → track center)
                    px, py = -math.sin(rail_heading), math.cos(rail_heading)
                    rr.log("rail_analysis/lateral", rr.LineStrips3D(
                        [[[rx, ry, rz],
                          [rx - lateral_err * px, ry - lateral_err * py, rz]]],
                        colors=[[255, 128, 0]], radii=0.02,
                    ))

                sleep(dt)
                step_count += 1

                if step_count % 50 == 0:
                    print(f"  step {step_count}: rail_hdg={math.degrees(rail_heading):.1f}° "
                          f"lat={lateral_err:+.3f}m hdg_err={math.degrees(heading_err):+.1f}°")
        else:
            # --- Path-based auto mode: follow path with UWB-based control ---
            while path_s < path_dists[-1]:
                path_s += args.target_speed * dt
                tx, ty = _sample_path(path_dists, path_xs, path_ys, path_s)
                marker_pub.Write(Pose_(Point_(tx, ty, 0.9), Quaternion_(0, 0, 0, 1)))

                with uwb_lock:
                    tag_dist = uwb["dist"]
                    ryaw = uwb["yaw"]

                az_r = uwb["az"]
                cos_y, sin_y = math.cos(ryaw), math.sin(ryaw)
                lx = tag_dist * math.cos(az_r)
                ly = tag_dist * math.sin(az_r)
                rx = tx - (cos_y * lx - sin_y * ly)
                ry = ty - (sin_y * lx + cos_y * ly)

                robot_s = _closest_path_s(path_dists, path_xs, path_ys, rx, ry)
                tangent = _path_tangent(path_dists, path_xs, path_ys, robot_s)
                cx, cy = _sample_path(path_dists, path_xs, path_ys, robot_s)

                dx, dy = rx - cx, ry - cy
                lateral_err = -math.sin(tangent) * dx + math.cos(tangent) * dy
                heading_err = (tangent - ryaw + math.pi) % (2 * math.pi) - math.pi

                vx = 0.0 if tag_dist < MIN_DIST else args.v_forward * min(1.0, (tag_dist - MIN_DIST) / MIN_DIST)
                vy = max(-0.3, min(0.3, -args.lateral_gain * lateral_err))
                vyaw = max(-2.5, min(2.5, args.yaw_gain * heading_err))

                client.Move(vx, vy, vyaw)
                sleep(dt)
                _poll_heightmap()
                step_count += 1

        client.StopMove()
        print(f"\nReached end of road after {step_count} steps.")

    except KeyboardInterrupt:
        print("\nInterrupted.")
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
