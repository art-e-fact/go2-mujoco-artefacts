from __future__ import annotations

import re

import mujoco
import numpy as np
import rerun as rr


def _sanitize_path_component(s: str) -> str:
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^a-zA-Z0-9_\-./]", "_", s)
    s = re.sub(r"_+", "_", s)
    return s or "unnamed"


def camera_images(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
) -> dict[str, rr.Image]:
    out: dict[str, rr.Image] = {}

    for cam_id in range(int(model.ncam)):
        try:
            cam_name = model.cam(cam_id).name
        except Exception:
            cam_name = ""

        name = _sanitize_path_component(str(cam_name)) if cam_name else f"camera_{cam_id}"

        renderer.update_scene(data, camera=cam_id)
        image = np.asarray(renderer.render(), dtype=np.uint8)

        out[name] = rr.Image(image)

    return out
