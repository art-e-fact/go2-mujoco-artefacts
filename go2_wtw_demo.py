#!/usr/bin/env python3
"""
Go2 Walk-These-Ways Demo

Drives the Go2 along a square path using the high-level SportClient API.
Starts the full sim stack automatically.

Usage:
  python go2_wtw_demo.py              # MuJoCo viewer (UI)
  python go2_wtw_demo.py --headless   # no display (CI / testing)

"""

import sys
import os
import time
import threading
import subprocess
from utils import get_python_executable, sim_sleep, last_sim_time, FrontCameraRecorder

_HERE    = os.path.dirname(os.path.abspath(__file__))
_SDK_DIR = os.path.join(_HERE, "src", "unitree_sdk2_python")

sys.path.insert(0, _SDK_DIR)


def _drain(proc, events):
    """Print subprocess stdout; set events[i] when events[i][0] marker appears.

    events: list of (marker_str_or_None, threading.Event)
    """
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


def main():
    import argparse
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.sport.sport_client import SportClient
    from unitree_mujoco import config

    parser = argparse.ArgumentParser(description="Go2 Walk-These-Ways Demo")
    parser.add_argument("--cycles",    type=int,   default=1,   help="Number of square-path cycles")
    parser.add_argument("--interface", default=config.INTERFACE, help="Network interface")
    parser.add_argument("--domain",    type=int,   default=0,   help="DDS domain ID")
    parser.add_argument("--headless",  action="store_true",      help="No viewer (use for testing/CI)")
    parser.add_argument("--scene",         metavar="PATH", default=None,
                        help="MuJoCo scene XML (passed to sport_mujoco.py; defaults to config.ROBOT_SCENE)")
    parser.add_argument("--telemetry",      metavar="PATH", default=None,
                        help="Write simulation state (qpos/qvel) as JSONL to PATH")
    parser.add_argument("--record",       metavar="PATH", default=None,
                        help="Save spectator-view recording (passed to sport_mujoco.py)")
    parser.add_argument("--record-front", metavar="PATH", default=None,
                        help="Save front-camera recording to PATH (e.g. front.mp4)")
    parser.add_argument("--heightmap",  action="store_true",
                        help="Enable HeightMap_ DDS publishing in the sim")
    parser.add_argument("--heightmap-debug", action="store_true",
                        help="Visualise height map rays in the viewer (implies --heightmap)")
    parser.add_argument("--v-forward",      type=float, default=0.4,  help="Forward velocity (m/s)")
    parser.add_argument("--v-lateral",      type=float, default=0.0,  help="Lateral velocity (m/s)")
    parser.add_argument("--rotation-speed", type=float, default=2.5,  help="Rotation speed (rad/s)")
    args = parser.parse_args()

    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": _SDK_DIR}
    procs = []
    recorder = None

    try:
        # --- sport_mujoco.py: unified sim + WTW + RPC server in one process ---
        _sport_mujoco = os.path.join(os.path.dirname(sys.executable), "sport-mujoco")
        sim_cmd = [_sport_mujoco,
                   "--interface", args.interface, "--domain", str(args.domain)]
        if args.headless:
            sim_cmd.append("--headless")
        if args.scene:
            sim_cmd += ["--scene", os.path.abspath(args.scene)]
        if args.record:
            sim_cmd += ["--record", os.path.abspath(args.record)]
        if args.telemetry:
            sim_cmd += ["--telemetry", os.path.abspath(args.telemetry)]
        if args.heightmap:
            sim_cmd.append("--heightmap")
        if args.heightmap_debug:
            sim_cmd.append("--heightmap-debug")

        sim_proc = subprocess.Popen(
            sim_cmd,
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
            time.sleep(1.0)  # give the viewer time to finish opening
        assert sim_standing.wait(timeout=20), "sport_mujoco did not reach standing pose"
        time.sleep(1.5)  # WTW pre-warming

        # --- SportClient ---
        ChannelFactoryInitialize(args.domain, args.interface)
        client = SportClient()
        client.SetTimeout(10.0)
        client.Init()

        # --- HeightMap subscriber (verification) ---
        if args.heightmap:
            from unitree_sdk2py.core.channel import ChannelSubscriber
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import HeightMap_

            def _on_heightmap(msg):
                import numpy as _np
                arr = _np.array(msg.data, dtype=_np.float32)
                filled = arr[arr < 1.0e9]
                if len(filled) > 0:
                    print(f"[heightmap] t={msg.stamp:.2f} {msg.width}x{msg.height} "
                          f"origin=({msg.origin[0]:.2f},{msg.origin[1]:.2f}) "
                          f"cells={len(filled)}/{len(arr)} "
                          f"h=[{filled.min():.3f}, {filled.max():.3f}] "
                          f"avg={filled.mean():.3f} std={filled.std():.3f} "
                          f"median={_np.median(filled):.3f} "
                          f"p95={_np.percentile(filled, 95):.3f} "
                          f"p99={_np.percentile(filled, 99):.3f}")

            hmap_sub = ChannelSubscriber("rt/utlidar/height_map_array", HeightMap_)
            hmap_sub.Init(_on_heightmap, 10)

        telemetry_path = os.path.abspath(args.telemetry) if args.telemetry else None
        sleep = (lambda dt: sim_sleep(dt, telemetry_path)) if telemetry_path else time.sleep

        if args.record_front:
            recorder = FrontCameraRecorder(args.record_front)
            recorder.start()

        print(f"\n=== Walk-These-Ways Go2 Square Demo ({args.cycles} cycle(s)) ===")
        print("Sequence per cycle: turn 5 s → forward 8 s → turn 3 s → forward 5 s")
        print("=================================================================\n")


        v_forward      = args.v_forward
        v_lateral      = args.v_lateral
        rotation_speed = args.rotation_speed
        print(f"Params: v_forward={v_forward} v_lateral={v_lateral} rotation_speed={rotation_speed}")

        for cycle in range(args.cycles):
            print(f"Cycle {cycle + 1}/{args.cycles}")
            client.Move(0.0,       0.0, rotation_speed);  sleep(3.2)   # turn left
            client.Move(v_forward, 0.0, 0.0);             sleep(4.0)   # forward
            client.Move(0.0,       0.0, -rotation_speed); sleep(3.0)   # turn right
            client.Move(v_forward, 0.0, 0.0);             sleep(3.0)   # forward

        client.StopMove()
        print(f"\nCompleted {args.cycles} cycle(s).")

    finally:
        if recorder is not None:
            recorder.stop()
        _stop(procs)


if __name__ == "__main__":
    main()
