"""
Integration test for the Go2 WTW demo.

go2_wtw_demo.py --headless manages the full stack internally
(sport_mujoco.py + SportClient).

Run with: pytest tests/test_demo.py -v -s
"""

import json
import os
import sys
import subprocess
import pytest

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from utils import get_python_executable, make_jsonl_chart

SDK_PATH    = os.path.join(PROJECT_DIR, "src", "unitree_sdk2_python")
SCENE_PATH  = os.path.join(PROJECT_DIR, "resources", "scene_rail_track.xml")
OUTPUT_DIR  = os.environ.get("ARTEFACTS_SCENARIO_UPLOAD_DIR", os.path.join(PROJECT_DIR, "output"))
ENV         = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": SDK_PATH}


def _body_up_z(qpos):
    """Z component of the robot's up-axis from quaternion [w, x, y, z]. >0 = upright."""
    w, x, y, z = qpos[3], qpos[4], qpos[5], qpos[6]
    return 1.0 - 2.0 * (x*x + y*y)


def test_square_path_one_cycle():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    spectator_mp4   = os.path.join(OUTPUT_DIR, "run.mp4")
    front_mp4       = os.path.join(OUTPUT_DIR, "front.mp4")
    telemetry_jsonl = os.path.join(OUTPUT_DIR, "telemetry.jsonl")

    v_forward      = os.environ.get("v_forward",      "0.4")
    v_lateral      = os.environ.get("v_lateral",      "0.0")
    rotation_speed = os.environ.get("rotation_speed", "2.5")

    result = subprocess.run(
        [get_python_executable(), "-u", "go2_wtw_demo.py",
         "--headless", "--cycles", "1",
         "--scene",          SCENE_PATH,
         "--record",         spectator_mp4,
         "--record-front",   front_mp4,
         "--telemetry",      telemetry_jsonl,
         "--v-forward",      v_forward,
         "--v-lateral",      v_lateral,
         "--rotation-speed", rotation_speed],
        cwd=PROJECT_DIR,
        capture_output=True, text=True,
        timeout=120,
        env=ENV,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    assert result.returncode == 0, f"Demo exited with code {result.returncode}"
    assert "Completed 1 cycle(s)." in result.stdout
    assert os.path.getsize(spectator_mp4) > 0, "Spectator recording is empty"
    assert os.path.getsize(front_mp4) > 0, "Front camera recording is empty"

    with open(telemetry_jsonl) as f:
        snapshots = [json.loads(l) for l in f if l.strip()]
    assert snapshots, "No telemetry snapshots written"
    flipped = [s for s in snapshots if _body_up_z(s["qpos"]) <= 0]
    assert not flipped, (
        f"Robot was upside down at t={flipped[0]['t']}s "
        f"(body_up_z={_body_up_z(flipped[0]['qpos']):.3f})"
    )

    TARGET_X, TARGET_Y, TARGET_TOL = 0.0, 0.0, 5.0  # update after a known-good run
    final = snapshots[-1]
    fx, fy = final["qpos"][0], final["qpos"][1]
    delta = ((fx - TARGET_X)**2 + (fy - TARGET_Y)**2) ** 0.5
    assert delta < TARGET_TOL, (
        f"Robot finished {delta:.2f}m from target ({TARGET_X}, {TARGET_Y}) "
        f"at t={final['t']}s (x={fx:.2f}, y={fy:.2f})"
    )

    make_jsonl_chart(
        telemetry_jsonl,
        attr_x="qpos.0", attr_y="qpos.1",
        output_dir=OUTPUT_DIR,
        chart_name="trajectory_xy",
        field_unit="m",
    )
