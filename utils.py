import json
import time
import traceback
import platform
import shutil
import sys
from contextlib import suppress
from typing import Optional

import numpy as np


def _get_deep_attr_from_dict(d, keys):
    """Traverse a nested dict/list by a sequence of string keys (or int indices)."""
    val = d
    for k in keys:
        if isinstance(val, list):
            val = val[int(k)]
        else:
            val = val[k]
    return val


def last_sim_time(path):
    """Return the last sim time from a telemetry JSONL file using a tail seek."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 2048))
            tail = f.read().decode("utf-8", errors="ignore")
        last_line = [l for l in tail.splitlines() if l.strip()][-1]
        return json.loads(last_line)["t"]
    except (FileNotFoundError, IndexError, KeyError, json.JSONDecodeError):
        return None


def sim_sleep(dt, telemetry_path, poll=0.05):
    """Sleep for `dt` simulated seconds by polling the telemetry file."""
    t0 = None
    while t0 is None:
        t0 = last_sim_time(telemetry_path)
        if t0 is None:
            time.sleep(poll)
    target = t0 + dt
    while True:
        t = last_sim_time(telemetry_path)
        if t is not None and t >= target:
            break
        time.sleep(poll)


def get_python_executable():
    """Return mjpython on macOS (needed for MuJoCo viewer), else sys.executable."""
    if platform.system() == "Darwin":
        mjpython = shutil.which("mjpython")
        if mjpython:
            return mjpython
    return sys.executable
