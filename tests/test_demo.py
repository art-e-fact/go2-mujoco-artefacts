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
ENV         = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": SDK_PATH}


def test_square_path_one_cycle():
    result = subprocess.run(
        [sys.executable, "-u", "go2_wtw_demo.py",
         "--headless", "--cycles", "1",
         "--interface", "lo", "--domain", "0"],
        cwd=PROJECT_DIR,
        capture_output=True, text=True,
        timeout=60,   # bridge + server start + 3 s wait + 16 s cycle + buffer
        env=ENV,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    assert result.returncode == 0, f"Demo exited with code {result.returncode}"
    assert "Completed 1 cycle(s)." in result.stdout
