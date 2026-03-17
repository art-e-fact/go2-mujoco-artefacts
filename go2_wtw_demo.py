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
from utils import get_python_executable, sim_sleep, last_sim_time

_HERE    = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR = os.path.join(_HERE, "src", "unitree_mujoco", "simulate_python")
_SDK_DIR = os.path.join(_HERE, "src", "unitree_sdk2_python")

sys.path.insert(0, _SDK_DIR)
sys.path.insert(0, _SIM_DIR)


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
    import config

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
    parser.add_argument("--v-forward",      type=float, default=0.4,  help="Forward velocity (m/s)")
    parser.add_argument("--v-lateral",      type=float, default=0.0,  help="Lateral velocity (m/s)")
    parser.add_argument("--rotation-speed", type=float, default=2.5,  help="Rotation speed (rad/s)")
    args = parser.parse_args()

    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": _SDK_DIR}
    procs = []
    front_ffmpeg = None
    front_stop   = None
    front_thread = None

    try:
        # --- sport_mujoco.py: unified sim + WTW + RPC server in one process ---
        sim_cmd = [get_python_executable(), "-u", os.path.join(_SIM_DIR, "sport_mujoco.py"),
                   "--interface", args.interface, "--domain", str(args.domain)]
        if args.headless:
            sim_cmd.append("--headless")
        if args.scene:
            sim_cmd += ["--scene", os.path.abspath(args.scene)]
        if args.record:
            sim_cmd += ["--record", os.path.abspath(args.record)]
        if args.telemetry:
            sim_cmd += ["--telemetry", os.path.abspath(args.telemetry)]

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
            time.sleep(1.0)  # give the viewer time to finish opening
        assert sim_standing.wait(timeout=20), "sport_mujoco did not reach standing pose"
        time.sleep(1.5)  # WTW pre-warming

        # --- SportClient ---
        ChannelFactoryInitialize(args.domain, args.interface)
        client = SportClient()
        client.SetTimeout(10.0)
        client.Init()

        # --- Front-camera recorder ------------------------------------------
        if args.record_front:
            from unitree_sdk2py.go2.video.video_client import VideoClient
            front_stop = threading.Event()
            front_ffmpeg = subprocess.Popen([
                "ffmpeg", "-y",
                "-f", "mjpeg", "-r", "30", "-i", "pipe:",
                "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                os.path.abspath(args.record_front),
            ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

            def _record_front():
                vclient = VideoClient()
                vclient.SetTimeout(3.0)
                vclient.Init()
                while not front_stop.is_set():
                    code, data = vclient.GetImageSample()
                    if code == 0 and data:
                        try:
                            front_ffmpeg.stdin.write(bytes(data))
                        except BrokenPipeError:
                            break
                    time.sleep(1 / 30)

            front_thread = threading.Thread(target=_record_front, daemon=True)
            front_thread.start()
            print(f"[demo] Front camera recording → {os.path.abspath(args.record_front)}")

        telemetry_path = os.path.abspath(args.telemetry) if args.telemetry else None
        sleep = (lambda dt: sim_sleep(dt, telemetry_path)) if telemetry_path else time.sleep

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
        if front_stop is not None:
            front_stop.set()
        if front_thread is not None:
            front_thread.join(timeout=2)
        if front_ffmpeg is not None:
            front_ffmpeg.stdin.close()
            front_ffmpeg.wait()
            print(f"[demo] Front camera recording saved: {os.path.abspath(args.record_front)}")
        _stop(procs)


if __name__ == "__main__":
    main()
