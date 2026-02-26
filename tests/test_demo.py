"""
Integration test for Go2 MuJoCo demo.

Runs go2_wtw_demo.py for one cycle and verifies it exits cleanly.

Run with: pytest tests/ -v
"""

import pytest
import subprocess
import sys
import os


def test_demo_runs_one_cycle():
    """Run go2_wtw_demo.py for one cycle and check it exits cleanly."""
    result = subprocess.run(
        [sys.executable, "go2_wtw_demo.py", "--cycles", "1"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        capture_output=True,
        text=True,
        timeout=60  # 18 seconds per cycle + buffer
    )
    
    print(f"stdout: {result.stdout}")
    print(f"stderr: {result.stderr}")
    
    assert result.returncode == 0, f"Demo failed with code {result.returncode}: {result.stderr}"
    assert "Completed 1 cycle" in result.stdout, "Demo should complete one cycle"
