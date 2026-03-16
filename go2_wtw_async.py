import asyncio
import io
import os
import platform as _platform
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass, field, fields, is_dataclass
from pprint import pprint
from typing import Any, Mapping

import asyncio_for_robotics as afor
import asyncio_for_robotics.textio as aforio
import av
import cyclonedds.idl as idl
import numpy as np
import rerun as rr
import unitree_sdk2py
from asyncio_for_robotics.core.sub import BaseSub
from PIL import Image
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.idl.builtin_interfaces.msg.dds_._Time_ import Time_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_

from proc_utils import finish_process, ignore_interupt
from utils import get_python_executable

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR = os.path.join(_HERE, "src", "unitree_mujoco", "simulate_python")


@dataclass
class DDSTransform(idl.IdlStruct):
    frame: str
    parent: str = "world"
    time: Time_ = field(default_factory=lambda *_: Time_(0, 0))
    x: float = 0
    y: float = 0
    z: float = 0
    mat: idl.types.array[float, 9] = field(default_factory=lambda *_: [0.0] * 9)


def sim_subproc() -> subprocess.Popen[str]:
    sim_cmd = f"python sport_mujoco.py"
    return subprocess.Popen(
        sim_cmd.split(" "),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=_SIM_DIR,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )


async def moves():
    client = SportClient()
    client.SetTimeout(10.0)
    client.Init()
    await asyncio.sleep(5)
    print("Hello")
    client.Move(0.0, 0.0, 2.5)
    await asyncio.sleep(5.0)  # forward
    client.Move(0.4, 0.0, 0.0)
    await asyncio.sleep(8.0)  # forward
    client.Move(0.0, 0.0, -2.5)
    await asyncio.sleep(3.0)  # forward + turn right
    client.Move(0.4, 0.0, 0.0)
    await asyncio.sleep(5.0)  # forward
    print("done")


def _to_builtin(obj: Any) -> Any:
    """Convert CycloneDDS dataclasses to plain Python objects."""
    if is_dataclass(obj):
        return {f.name: _to_builtin(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, (bytes, bytearray)):
        return list(obj)  # easier to inspect than raw bytes in generic views
    return obj


def _flatten_dict(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts for rr.AnyValues."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}__{k}" if prefix else k
        if isinstance(v, Mapping):
            out.update(_flatten_dict(v, key))
        else:
            out[key] = v
    return out


def log_lowstate(msg: LowState_) -> None:
    """
    Log one Unitree LowState_ sample to Rerun.

    Expected msg type:
        unitree_sdk2py.idl.unitree_go.msg.dds_.LowState_
    """
    builtin = _to_builtin(msg)
    flat = _flatten_dict(builtin)
    rr.log(f"/raw", rr.AnyValues(**flat))

    qw, qx, qy, qz = [float(x) for x in msg.imu_state.quaternion]

    rr.log(
        f"/imu",
        rr.Transform3D(
            quaternion=rr.Quaternion(xyzw=[qx, qy, qz, qw]),
        ),
        rr.AnyValues(
            gyroscope=[float(x) for x in msg.imu_state.gyroscope],
            accelerometer=[float(x) for x in msg.imu_state.accelerometer],
            rpy=[float(x) for x in msg.imu_state.rpy],
            temperature=int(msg.imu_state.temperature),
        ),
    )

    ms = {
        "mode": [],
        "q": [],
        "dq": [],
        "ddq": [],
        "tau_est": [],
        "q_raw": [],
        "dq_raw": [],
        "ddq_raw": [],
        "temperature": [],
        "lost": [],
    }
    for i in msg.motor_state:
        for k in ms.keys():
            ms[k].append(getattr(i, k))

    rr.log(f"/motor_states", rr.AnyValues(**ms))
    rr.log(f"/foot", rr.AnyValues(force=msg.foot_force, force_est=msg.foot_force_est))
    rr.log(f"/fan", rr.AnyValues(frequency=msg.fan_frequency))
    rr.log(
        f"/power",
        rr.AnyValues(tension=float(msg.power_v), current=float(msg.power_a)),
    )


async def lowstate_rerun():
    dds_sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub = afor.BaseSub()
    dds_sub.Init(sub.input_data)
    async for msg in sub.listen_reliable():
        log_lowstate(msg)


async def images_rerun():
    vclient = VideoClient()
    vclient.SetTimeout(3.0)
    vclient.Init()
    counter = 0

    async for tns in afor.Rate(30).listen():
        code, data = await asyncio.to_thread(vclient.GetImageSample)
        rr.log(
            "/video",
            rr.EncodedImage(
                contents=bytes(data),
                media_type=rr.MediaType.JPEG,
                draw_order=rr.components.DrawOrder(counter),
            ),
            rr.AnyValues(code=code),
        )
        counter += 1


async def tf_rerun():
    dds_sub = ChannelSubscriber("/sim/tf", DDSTransform)
    sub: BaseSub[DDSTransform] = afor.BaseSub()
    dds_sub.Init(sub.input_data)
    async for msg in sub.listen_reliable(queue_size=1000):
        rr.log(
            f"tf",
            rr.Transform3D(
                translation=[msg.x, msg.y, msg.z],
                mat3x3=np.array(msg.mat).reshape(3, 3),
                parent_frame=msg.parent,
                child_frame=msg.frame,
            ),
        )


async def time_rerun():
    dds_sub = ChannelSubscriber("/sim/time", Time_)
    sub: BaseSub[Time_] = afor.BaseSub()
    dds_sub.Init(sub.input_data)
    async for msg in sub.listen_reliable():
        rr.log("sim_time", rr.Scalars(msg.sec * 1e9 + msg.nanosec))


async def ready_up(process_sub: aforio.Sub):
    async for line in process_sub.listen_reliable():
        if "[sport_mujoco] DDS" in line:
            # [sport_mujoco] DDS domain=1 interface=lo
            domain = int(line.split("domain=")[1].split(" ")[0])
            interface = line.split("interface=")[1].split(" ")[0]
            ChannelFactoryInitialize(domain, interface)
            break  # sim is ready


async def pre_main():
    p = sim_subproc()
    sub = aforio.from_proc_stdout(p)
    rr.init("rerun_example_dna_abacus")
    rr.save("data.rrd")
    rr.connect_grpc()
    try:
        await ready_up(sub)
        async with asyncio.TaskGroup() as tg:
            tg.create_task(time_rerun())
            tg.create_task(lowstate_rerun())
            tg.create_task(images_rerun())
            tg.create_task(tf_rerun())
            # tg.create_task(videostream_rerun())
            await moves()
            for t in tg._tasks:
                t.cancel()
    finally:
        print("finish_process")
        sub.close()
        with ignore_interupt():
            finish_process(p)


if __name__ == "__main__":
    with suppress(KeyboardInterrupt):
        asyncio.run(pre_main())
