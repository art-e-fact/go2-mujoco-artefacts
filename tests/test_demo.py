"""
Integration test for the Go2 WTW demo.

go2_wtw_demo.py --headless manages the full stack internally
(headless_bridge + sport_sim_server + SportClient).

Run with: pytest tests/test_demo.py -v -s
"""

import os
import sys
import subprocess
import pytest

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SDK_PATH    = os.path.join(PROJECT_DIR, "src", "unitree_sdk2_python")
OUTPUT_DIR  = os.environ.get("ARTEFACTS_SCENARIO_UPLOAD_DIR", os.path.join(PROJECT_DIR, "output"))
ENV         = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": SDK_PATH}


def test_square_path_one_cycle():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    spectator_mp4 = os.path.join(OUTPUT_DIR, "run.mp4")
    front_mp4     = os.path.join(OUTPUT_DIR, "front.mp4")

    result = subprocess.run(
        [sys.executable, "-u", "go2_wtw_demo.py",
         "--headless", "--cycles", "1",
         "--record",       spectator_mp4,
         "--record-front", front_mp4,
         "--interface", "lo", "--domain", "0"],
        cwd=PROJECT_DIR,
        capture_output=True, text=True,
        timeout=60,
        env=ENV,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    assert result.returncode == 0, f"Demo exited with code {result.returncode}"
    assert "Completed 1 cycle(s)." in result.stdout
    assert os.path.getsize(spectator_mp4) > 0, "Spectator recording is empty"
    assert os.path.getsize(front_mp4) > 0, "Front camera recording is empty"
