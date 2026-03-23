from __future__ import annotations

import re
from collections.abc import Iterable

import mujoco
import numpy as np
import rerun as rr


def _sanitize_path_component(s: str) -> str:
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^a-zA-Z0-9_\-./]", "_", s)
    s = re.sub(r"_+", "_", s)
    return s or "unnamed"


def _mjt_sensor_name(sensor_type: int) -> str:
    names = {
        int(mujoco.mjtSensor.mjSENS_TOUCH): "touch",
        int(mujoco.mjtSensor.mjSENS_ACCELEROMETER): "accelerometer",
        int(mujoco.mjtSensor.mjSENS_VELOCIMETER): "velocimeter",
        int(mujoco.mjtSensor.mjSENS_GYRO): "gyro",
        int(mujoco.mjtSensor.mjSENS_FORCE): "force",
        int(mujoco.mjtSensor.mjSENS_TORQUE): "torque",
        int(mujoco.mjtSensor.mjSENS_MAGNETOMETER): "magnetometer",
        int(mujoco.mjtSensor.mjSENS_RANGEFINDER): "rangefinder",
        int(mujoco.mjtSensor.mjSENS_CAMPROJECTION): "camprojection",
        int(mujoco.mjtSensor.mjSENS_JOINTPOS): "jointpos",
        int(mujoco.mjtSensor.mjSENS_JOINTVEL): "jointvel",
        int(mujoco.mjtSensor.mjSENS_TENDONPOS): "tendonpos",
        int(mujoco.mjtSensor.mjSENS_TENDONVEL): "tendonvel",
        int(mujoco.mjtSensor.mjSENS_ACTUATORPOS): "actuatorpos",
        int(mujoco.mjtSensor.mjSENS_ACTUATORVEL): "actuatorvel",
        int(mujoco.mjtSensor.mjSENS_ACTUATORFRC): "actuatorfrc",
        int(mujoco.mjtSensor.mjSENS_JOINTACTFRC): "jointactfrc",
        int(mujoco.mjtSensor.mjSENS_TENDONACTFRC): "tendonactfrc",
        int(mujoco.mjtSensor.mjSENS_BALLQUAT): "ballquat",
        int(mujoco.mjtSensor.mjSENS_BALLANGVEL): "ballangvel",
        int(mujoco.mjtSensor.mjSENS_JOINTLIMITPOS): "jointlimitpos",
        int(mujoco.mjtSensor.mjSENS_JOINTLIMITVEL): "jointlimitvel",
        int(mujoco.mjtSensor.mjSENS_JOINTLIMITFRC): "jointlimitfrc",
        int(mujoco.mjtSensor.mjSENS_TENDONLIMITPOS): "tendonlimitpos",
        int(mujoco.mjtSensor.mjSENS_TENDONLIMITVEL): "tendonlimitvel",
        int(mujoco.mjtSensor.mjSENS_TENDONLIMITFRC): "tendonlimitfrc",
        int(mujoco.mjtSensor.mjSENS_FRAMEPOS): "framepos",
        int(mujoco.mjtSensor.mjSENS_FRAMEQUAT): "framequat",
        int(mujoco.mjtSensor.mjSENS_FRAMEXAXIS): "framexaxis",
        int(mujoco.mjtSensor.mjSENS_FRAMEYAXIS): "frameyaxis",
        int(mujoco.mjtSensor.mjSENS_FRAMEZAXIS): "framezaxis",
        int(mujoco.mjtSensor.mjSENS_FRAMELINVEL): "framelinvel",
        int(mujoco.mjtSensor.mjSENS_FRAMEANGVEL): "frameangvel",
        int(mujoco.mjtSensor.mjSENS_FRAMELINACC): "framelinacc",
        int(mujoco.mjtSensor.mjSENS_FRAMEANGACC): "frameangacc",
        int(mujoco.mjtSensor.mjSENS_SUBTREECOM): "subtreecom",
        int(mujoco.mjtSensor.mjSENS_SUBTREELINVEL): "subtreelinvel",
        int(mujoco.mjtSensor.mjSENS_SUBTREEANGMOM): "subtreeangmom",
        int(mujoco.mjtSensor.mjSENS_INSIDESITE): "insidesite",
        int(mujoco.mjtSensor.mjSENS_GEOMDIST): "geomdist",
        int(mujoco.mjtSensor.mjSENS_GEOMNORMAL): "geomnormal",
        int(mujoco.mjtSensor.mjSENS_GEOMFROMTO): "geomfromto",
        int(mujoco.mjtSensor.mjSENS_CONTACT): "contact",
        int(mujoco.mjtSensor.mjSENS_E_POTENTIAL): "potential_energy",
        int(mujoco.mjtSensor.mjSENS_E_KINETIC): "kinetic_energy",
        int(mujoco.mjtSensor.mjSENS_CLOCK): "clock",
        int(mujoco.mjtSensor.mjSENS_TACTILE): "tactile",
        int(mujoco.mjtSensor.mjSENS_PLUGIN): "plugin",
        int(mujoco.mjtSensor.mjSENS_USER): "user",
    }
    return names.get(int(sensor_type), f"sensor_{int(sensor_type)}")


def _mjt_obj_name(obj_type: int) -> str:
    names = {
        int(mujoco.mjtObj.mjOBJ_UNKNOWN): "unknown",
        int(mujoco.mjtObj.mjOBJ_BODY): "body",
        int(mujoco.mjtObj.mjOBJ_XBODY): "xbody",
        int(mujoco.mjtObj.mjOBJ_JOINT): "joint",
        int(mujoco.mjtObj.mjOBJ_DOF): "dof",
        int(mujoco.mjtObj.mjOBJ_GEOM): "geom",
        int(mujoco.mjtObj.mjOBJ_SITE): "site",
        int(mujoco.mjtObj.mjOBJ_CAMERA): "camera",
        int(mujoco.mjtObj.mjOBJ_LIGHT): "light",
        int(mujoco.mjtObj.mjOBJ_FLEX): "flex",
        int(mujoco.mjtObj.mjOBJ_MESH): "mesh",
        int(mujoco.mjtObj.mjOBJ_SKIN): "skin",
        int(mujoco.mjtObj.mjOBJ_HFIELD): "hfield",
        int(mujoco.mjtObj.mjOBJ_TEXTURE): "texture",
        int(mujoco.mjtObj.mjOBJ_MATERIAL): "material",
        int(mujoco.mjtObj.mjOBJ_PAIR): "pair",
        int(mujoco.mjtObj.mjOBJ_EXCLUDE): "exclude",
        int(mujoco.mjtObj.mjOBJ_EQUALITY): "equality",
        int(mujoco.mjtObj.mjOBJ_TENDON): "tendon",
        int(mujoco.mjtObj.mjOBJ_ACTUATOR): "actuator",
        int(mujoco.mjtObj.mjOBJ_SENSOR): "sensor",
        int(mujoco.mjtObj.mjOBJ_NUMERIC): "numeric",
        int(mujoco.mjtObj.mjOBJ_TEXT): "text",
        int(mujoco.mjtObj.mjOBJ_TUPLE): "tuple",
        int(mujoco.mjtObj.mjOBJ_KEY): "key",
        int(mujoco.mjtObj.mjOBJ_PLUGIN): "plugin",
    }
    return names.get(int(obj_type), f"obj_{int(obj_type)}")


def _obj_name(model: mujoco.MjModel, obj_type: int, obj_id: int) -> str | None:
    if obj_id < 0:
        return None
    try:
        name = mujoco.mj_id2name(model, obj_type, obj_id)
        if name:
            return str(name)
    except Exception:
        pass
    return None


def _component_names(sensor_type: int, dim: int, data_type: int) -> list[str]:
    if data_type == int(mujoco.mjtDataType.mjDATATYPE_QUATERNION) or sensor_type in {
        int(mujoco.mjtSensor.mjSENS_BALLQUAT),
        int(mujoco.mjtSensor.mjSENS_FRAMEQUAT),
    }:
        base = ["w", "x", "y", "z"]
        return base[:dim] if dim <= 4 else [f"q{i}" for i in range(dim)]

    if data_type == int(mujoco.mjtDataType.mjDATATYPE_AXIS):
        base = ["x", "y", "z"]
        return base[:dim] if dim <= 3 else [f"axis_{i}" for i in range(dim)]

    vector3_types = {
        int(mujoco.mjtSensor.mjSENS_ACCELEROMETER),
        int(mujoco.mjtSensor.mjSENS_VELOCIMETER),
        int(mujoco.mjtSensor.mjSENS_GYRO),
        int(mujoco.mjtSensor.mjSENS_FORCE),
        int(mujoco.mjtSensor.mjSENS_TORQUE),
        int(mujoco.mjtSensor.mjSENS_MAGNETOMETER),
        int(mujoco.mjtSensor.mjSENS_BALLANGVEL),
        int(mujoco.mjtSensor.mjSENS_FRAMEPOS),
        int(mujoco.mjtSensor.mjSENS_FRAMEXAXIS),
        int(mujoco.mjtSensor.mjSENS_FRAMEYAXIS),
        int(mujoco.mjtSensor.mjSENS_FRAMEZAXIS),
        int(mujoco.mjtSensor.mjSENS_FRAMELINVEL),
        int(mujoco.mjtSensor.mjSENS_FRAMEANGVEL),
        int(mujoco.mjtSensor.mjSENS_FRAMELINACC),
        int(mujoco.mjtSensor.mjSENS_FRAMEANGACC),
        int(mujoco.mjtSensor.mjSENS_SUBTREECOM),
        int(mujoco.mjtSensor.mjSENS_SUBTREELINVEL),
        int(mujoco.mjtSensor.mjSENS_SUBTREEANGMOM),
        int(mujoco.mjtSensor.mjSENS_GEOMNORMAL),
    }
    if sensor_type in vector3_types and dim == 3:
        return ["x", "y", "z"]

    if sensor_type == int(mujoco.mjtSensor.mjSENS_CAMPROJECTION) and dim == 2:
        return ["u", "v"]

    if sensor_type == int(mujoco.mjtSensor.mjSENS_GEOMFROMTO) and dim == 6:
        return ["from_x", "from_y", "from_z", "to_x", "to_y", "to_z"]

    if dim == 1:
        return ["value"]

    return [f"v{i}" for i in range(dim)]


def _sensor_base_name(model: mujoco.MjModel, sensor_id: int) -> str:
    sensor = model.sensor(sensor_id)
    if sensor.name:
        sp = sensor.name.split("_")
        ret = "/".join([sp[-1]] + sp[:-1])
        return ret

    sensor_type = int(model.sensor_type[sensor_id])
    obj_type = int(model.sensor_objtype[sensor_id])
    obj_id = int(model.sensor_objid[sensor_id])

    sensor_type_name = _mjt_sensor_name(sensor_type)
    obj_type_name = _mjt_obj_name(obj_type)
    obj_name = _obj_name(model, obj_type, obj_id)

    parts = ["sensor", sensor_type_name]
    if obj_name:
        parts.append(f"{obj_type_name}/{obj_name}")
    elif obj_id >= 0:
        parts.append(f"{obj_type_name}/{obj_id}")
    else:
        parts.append(str(sensor_id))
    return "/".join(_sanitize_path_component(p) for p in parts)


def log_all_sensors_to_rerun(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    root: str = "sensors",
    _styled_paths: set[str] | None = None,
) -> None:
    """
    Log all MuJoCo sensors into Rerun as time series.

    Call this after your physics step or after mj_forward/mj_step has produced
    fresh sensor values.
    """
    if _styled_paths is None:
        _styled_paths = set()

    sensordata = np.asarray(data.sensordata, dtype=np.float64)

    for sensor_id in range(int(model.nsensor)):
        adr = int(model.sensor_adr[sensor_id])
        dim = int(model.sensor_dim[sensor_id])
        sensor_type = int(model.sensor_type[sensor_id])
        data_type = int(model.sensor_datatype[sensor_id])

        values = sensordata[adr : adr + dim]
        if values.shape[0] != dim:
            continue

        base_name = _sensor_base_name(model, sensor_id)
        path = f"{root}/{base_name}"

        component_names = _component_names(sensor_type, dim, data_type)

        rr.log(path, rr.AnyValues(**dict(zip(component_names, values))))
