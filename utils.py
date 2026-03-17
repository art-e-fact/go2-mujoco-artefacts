import json
import traceback
import platform
import shutil
import sys
from contextlib import suppress
from typing import Optional

import numpy as np
import plotly.graph_objects as go


def _get_deep_attr_from_dict(d, keys):
    """Traverse a nested dict/list by a sequence of string keys (or int indices)."""
    val = d
    for k in keys:
        if isinstance(val, list):
            val = val[int(k)]
        else:
            val = val[k]
    return val


def _plot_data(x_data, y_data, x_label, y_label, output_filepath):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_data, y=y_data, mode="lines+markers", name=y_label))
    fig.update_layout(xaxis_title=x_label, yaxis_title=y_label)
    fig.write_html(output_filepath)


def make_jsonl_chart(
    filepath: str,
    attr_x: str,
    attr_y: str,
    output_dir: str,
    chart_name: str,
    field_unit: Optional[str] = None,
):
    try:
        x_data, y_data = [], []
        x_plot_name = f"{attr_x} ({field_unit})" if field_unit else attr_x
        y_plot_name = f"{attr_y} ({field_unit})" if field_unit else attr_y

        topic_x = attr_x.split(".")
        topic_y = attr_y.split(".")
        if "time" in topic_x:
            x_plot_name = "Time (s)"
        if "time" in topic_y:
            y_plot_name = "Time (s)"

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

        _plot_data(
            np.array(x_data, dtype=float),
            np.array(y_data, dtype=float),
            x_plot_name, y_plot_name,
            f"{output_dir}/{chart_name}.html",
        )
    except Exception as e:
        print(f"ERROR: Unable to create chart for {chart_name}.", *traceback.format_exception(e))


def get_python_executable():
    """Return mjpython on macOS (needed for MuJoCo viewer), else sys.executable."""
    if platform.system() == "Darwin":
        mjpython = shutil.which("mjpython")
        if mjpython:
            return mjpython
    return sys.executable
