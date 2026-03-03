"""
Integration test for Go2 MuJoCo demo.

Runs go2_wtw_demo.py for one cycle with GUI and video recording.
The video is saved for upload to artefacts.

Run with: pytest tests/ -v
"""

import pytest
import subprocess
import sys
import os
import platform
import shutil


def get_python_executable():
    """Get the appropriate Python executable for the platform."""
    if platform.system() == "Darwin":
        # macOS: use mjpython for MuJoCo GUI support
        mjpython = shutil.which("mjpython")
        if mjpython:
            return mjpython
    return sys.executable


def test_demo_runs_one_cycle():
    """Run go2_wtw_demo.py for one cycle with video recording."""
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, "demo_recording.mp4")

    cmd = [get_python_executable(), "go2_wtw_demo.py", "--cycles", "1", "--headless", "--record", video_path]

    result = subprocess.run(
        cmd,
        cwd=project_dir,
        capture_output=True,
        text=True,
        timeout=60  # 18 seconds per cycle + buffer
    )
    
    print(f"stdout: {result.stdout}")
    print(f"stderr: {result.stderr}")
    
    assert result.returncode == 0, f"Demo failed with code {result.returncode}: {result.stderr}"
    assert "Completed 1 cycle" in result.stdout, "Demo should complete one cycle"
    assert os.path.exists(video_path), "Video file should be created"
    assert os.path.getsize(video_path) > 0, "Video file should not be empty"
