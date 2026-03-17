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


def make_jsonl_chart(
    filepath: str,
    attr_x: str,
    attr_y: str,
    output_dir: str,
    chart_name: str,
    field_unit: Optional[str] = None,
):
    try:
        x_plot_name = f"{attr_x} ({field_unit})" if field_unit else attr_x
        y_plot_name = f"{attr_y} ({field_unit})" if field_unit else attr_y

        topic_x = attr_x.split(".")
        topic_y = attr_y.split(".")
        if "time" in topic_x:
            x_plot_name = "Time (s)"
        if "time" in topic_y:
            y_plot_name = "Time (s)"

        x_data, y_data = [], []
        with open(filepath) as f:
            for line in f:
                jdict = json.loads(line)
                x = y = None
                with suppress(KeyError, IndexError):
                    x = _get_deep_attr_from_dict(jdict, topic_x)
                with suppress(KeyError, IndexError):
                    y = _get_deep_attr_from_dict(jdict, topic_y)
                if x is None and y is None:
                    continue
                if x is None and x_data:
                    x = x_data[-1]
                if y is None and y_data:
                    y = y_data[-1]
                x_data.append(x)
                y_data.append(y)

        output_filepath = f"{output_dir}/{chart_name}.csv"
        with open(output_filepath, "w") as f:
            f.write(f"{x_plot_name},{y_plot_name}\n")
            for x, y in zip(x_data, y_data):
                f.write(f"{x},{y}\n")
    except Exception as e:
        print(f"ERROR: Unable to create chart for {chart_name}.", *traceback.format_exception(e))


def get_python_executable():
    """Return mjpython on macOS (needed for MuJoCo viewer), else sys.executable."""
    if platform.system() == "Darwin":
        mjpython = shutil.which("mjpython")
        if mjpython:
            return mjpython
    return sys.executable
