import platform
import shutil
import sys


def get_python_executable():
    """Return mjpython on macOS (needed for MuJoCo viewer), else sys.executable."""
    if platform.system() == "Darwin":
        mjpython = shutil.which("mjpython")
        if mjpython:
            return mjpython
    return sys.executable
