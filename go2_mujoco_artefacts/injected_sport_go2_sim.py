import argparse
import asyncio
import json
import logging
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, AsyncIterable, Callable, Self

import asyncio_for_robotics as afor
import mujoco
import mujoco.viewer
import numpy as np
import rerun as rr

from . import defaults
from .camera_utils import camera_images
from .mesh_utils import mujoco_mesh_view_to_rerun
from .sensor_utils import log_all_sensors_to_rerun
from .simple_sport_go2_sim import Simulation

logger = logging.getLogger()
logger.addHandler(rr.LoggingHandler(f"logs/{__name__}"))
logger.setLevel(-1)


class InjectedSimulation(Simulation):
    def __init__(
        self,
        scene: str = ...,
        model_dir: str = ...,
        cfg_path: str = ...,
        domain: int = ...,
        interface: str = ...,
        headless: bool = False,
    ):
        self.step_count = 0
        super().__init__(scene, model_dir, cfg_path, domain, interface, headless)
        self.sim_sub: afor.BaseSub[Self] = afor.BaseSub()
        self.time_sub: afor.BaseSub[float] = afor.BaseSub()

    def get_time(self) -> int:
        return int(self.data.time * 1e9)

    def step(self):
        """We add async subscribers to the sim, at each step we put something
        into the sub. Aside from afor.sub we can also use an asyncio.Queue to
        not miss any data and not bother with QoS.

        Very important is that sim  and other objects are mutable and updated
        at each step. So we have to be carefull and return copies of the state
        when necessary.
        """
        self.step_count += 1
        rr.set_time("sim_step", sequence=self.step_count)
        rr.set_time("sim_time", duration=self.data.time)
        original = super().step()
        self.sim_sub._input_data_asyncio(self)
        self.time_sub._input_data_asyncio(self.data.time)
        return original


def log_tf(sim: InjectedSimulation):
    """Logs the tf of every objext in mujoco"""
    mj_model: mujoco.MjModel = sim.model
    mj_data: mujoco.MjData = sim.data
    for i in range(mj_model.nbody):
        parent = "world"
        frame = mj_model.body(i).name
        pos = mj_data.xpos[i]  # world position of body frame
        xmat = mj_data.xmat[i]  # world orientation matrix, flattened 3x3
        # print(frame, pos, xmat)
        rr.log(
            (f"/tf/{parent}/{frame}" if frame != "world" else "/tf/world"),
            rr.Transform3D(
                translation=pos,
                mat3x3=xmat,
                # parent_frame=parent if frame != "world" else "",
                # child_frame=frame,
            ),
            rr.TransformAxes3D(axis_length=0.1),
            # static=True
        )


def load_mesh(sim: InjectedSimulation):
    """Loads all the meshes of mujoco in rerun."""
    logger.debug("loading mesh")
    for id in range(sim.model.ngeom):
        body_id = sim.model.geom_bodyid[id]
        body = sim.model.body(body_id)

        body_tf_path = Path(
            f"/tf/{"world"}/{body.name}" if body.name != "world" else "/tf/world"
        )

        mesh_id = sim.model.geom_dataid[id]
        geom_type = sim.model.geom_type[id]
        geom_size = sim.model.geom_size[id]
        logger.debug((id, geom_type, mujoco.mjtGeom(geom_type), geom_size))

        fill_mode = (
            rr.components.FillMode.Solid
            if body.name == "world"
            else rr.components.FillMode.MajorWireframe
        )
        name = ""
        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            name = "box"
            rr_3d_object = rr.Boxes3D(sizes=geom_size * 2, fill_mode=fill_mode)
        elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            name = "sphere"
            rr_3d_object = rr.Ellipsoids3D(
                radii=geom_size[[0] * 3], fill_mode=fill_mode
            )
        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            name = "cylinder"
            rr_3d_object = rr.Cylinders3D(
                lengths=geom_size[1] * 2, radii=geom_size[0], fill_mode=fill_mode
            )
        else:
            if mesh_id < 0:
                continue
            mesh = sim.model.mesh(mesh_id)
            name = mesh.name
            rr_3d_object = mujoco_mesh_view_to_rerun(sim.model, mesh)

        tf_path = body_tf_path / f"{name}" / f"{id}"

        transf = rr.Transform3D(
            translation=sim.model.geom_pos[id],
            quaternion=sim.model.geom_quat[id][[1, 2, 3, 0]],
        )
        rr.log(tf_path.as_posix(), rr_3d_object, transf, static=True)


def log_sensors(sim: InjectedSimulation):
    """Logs all the sensors of mujoco in rerun."""
    mj_model: mujoco.MjModel = sim.model
    mj_data: mujoco.MjData = sim.data
    log_all_sensors_to_rerun(mj_model, mj_data)


def sim_blocking_callback(sim: InjectedSimulation):
    """Blocking callback to be sure to perform something at each sim step."""
    log_tf(sim)
    log_sensors(sim)


async def run_viewport(sim: InjectedSimulation):
    """Updates the viewport.

    If this returns, it means the viewport has been closed by the user.
    """
    if sim.headless:
        await asyncio.Future()  # pauses here
        return
    logger.debug("starting viewport")
    sim_sub: afor.BaseSub[InjectedSimulation] = sim.sim_sub
    viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
    try:
        # we can have the viewport running at a different rate than the sim!
        # sim is running at 50Hz viewport at 10Hz
        async for _ in afor.Rate(30).listen():
            if not viewer.is_running():
                logger.debug("Viewer window closed manually")
                return
            viewer.sync()
    finally:
        logger.debug("Viewer closing")
        viewer.close()


async def record_cameras(sim: InjectedSimulation):
    """Continuously logs every camera in the sim to rerun.

    async to execute at 10Hz, and not at every sim step.
    """
    model: mujoco.MjModel = sim.model
    data: mujoco.MjData = sim.data
    renderer = mujoco.Renderer(model, height=500, width=700)

    # This executes at 10Hz, or simulation rate, which ever is slower
    next_tick = sim.sim_sub.wait_for_new()
    async for _ in afor.Rate(10, time_source=sim.get_time).listen():
        await next_tick
        next_tick = sim.sim_sub.wait_for_new()

        cam_images = camera_images(model, data, renderer)
        for name, image in cam_images.items():
            rr.log(f"cam/{name}", image)


async def go_to_abs_pos_controller(target: complex, sim: InjectedSimulation):
    """Moves the robot to a target position then returns."""
    logger.info(f"Going to {target}")
    rr.log(
        "tf/world/target",
        rr.Transform3D(translation=[target.real, target.imag, 0.5]),
        rr.Cylinders3D(
            radii=0.6/2,
            lengths=1,
            colors=[0, 255, 0, 50],
            fill_mode=rr.components.FillMode.Solid,
        ),
    )
    async for _ in afor.Rate(10, time_source=sim.get_time).listen():
        pos_vec = sim.data.body("base_link").xpos
        x_dir = sim.data.body("base_link").xmat[[0, 3, 6]]
        rr.log("cmd/mat", rr.Scalars(sim.data.body("base_link").xmat))
        dir = x_dir[0] + x_dir[1] * 1j
        dir = dir / abs(dir)
        pos = pos_vec[0] + pos_vec[1] * 1j
        rel_pos = (target - pos) / dir

        rr.log("cmd/pos", rr.Arrows2D(vectors=[pos.real, pos.imag]))
        rr.log(
            "cmd/dir",
            rr.Arrows2D(origins=[pos.real, pos.imag], vectors=[dir.real, dir.imag]),
        )
        rr.log(
            "cmd/rel_pos",
            rr.Arrows2D(
                origins=[pos.real, pos.imag],
                vectors=[(target - pos).real, (target - pos).imag],
            ),
        )

        angle: float = np.angle(rel_pos)
        dist: float = abs(rel_pos)

        speed_cmd = np.clip(dist, -1, 1) / 2

        if dist <= 0.6:
            logger.info(f"Target {target} reached")
            sim.server._handle_move(json.dumps({"x": 0, "y": 0, "z": 0}))
            rr.log(
                "tf/world/target",
                rr.Transform3D(translation=[target.real, target.imag, 0.5]),
                rr.Cylinders3D(
                    radii=0.1,
                    lengths=1,
                    colors=[0, 255, 0, 0],
                    fill_mode=rr.components.FillMode.Solid,
                ),
            )
            return

        sim.server._handle_move(
            json.dumps(
                {
                    "x": (rel_pos / dist).real * speed_cmd,
                    "y": (rel_pos / dist).imag * speed_cmd,
                    "z": angle * 2,
                }
            )
        )


async def task(sim: InjectedSimulation):
    """Kinda state machine"""
    logger.debug("Started Task")
    await sim.sim_sub.wait_for_value()
    await asyncio.sleep(5)

    await go_to_abs_pos_controller(1 - 1j, sim)
    await go_to_abs_pos_controller(1 + 1.7j, sim)
    await go_to_abs_pos_controller(1 - 1j, sim)

    await go_to_abs_pos_controller(-0.5 - 1.0j, sim)
    await go_to_abs_pos_controller(-0.5 + 1.7j, sim)
    await go_to_abs_pos_controller(-0.5 - 1.0j, sim)

    await go_to_abs_pos_controller(2 - 1.0j, sim)
    await go_to_abs_pos_controller(2 + 3j, sim)
    await go_to_abs_pos_controller(2 - 1.0j, sim)

    await go_to_abs_pos_controller(0 + 0.0j, sim)
    logger.debug("Task done")
    quit()


def setup_rerun():
    rr.init("newton-dls")
    rr.save("data.rrd")
    rr.connect_grpc()
    rr.set_time("sim_time", duration=0)
    rr.set_time("sim_step", sequence=0)
    rr.log(
        "cmd/orix",
        rr.Arrows2D(origins=[0, 0], vectors=[1, 0], colors=[255, 0, 0]),
        static=True,
    )
    rr.log(
        "cmd/oriy",
        rr.Arrows2D(origins=[0, 0], vectors=[0, 1], colors=[0, 255, 0]),
        static=True,
    )
    logger.debug("Rerun started")


async def setup_and_start(sim: InjectedSimulation):
    setup_rerun()
    load_mesh(sim)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(run_viewport(sim))
        tg.create_task(record_cameras(sim))
        tg.create_task(task(sim))

        await arun(
            sim, sim_blocking_callback, afor.Rate(1 / defaults.SIMULATE_DT).listen()
        )
        for t in tg._tasks:
            t.cancel()
        for t in tg._tasks:
            await t


async def arun(
    sim: InjectedSimulation,
    callback: Callable[[InjectedSimulation], Any],
    async_iterator: AsyncIterable,
):
    async for _ in async_iterator:
        sim.step()
        callback(sim)


async def main():
    parser = argparse.ArgumentParser(description="MuJoCo GO2 sim + sport RPC")
    parser.add_argument("--scene", default=defaults.DEFAULT_SCENE)
    parser.add_argument("--model-dir", default=defaults.DEFAULT_WTW_DIR)
    parser.add_argument("--cfg-path", default=defaults.DEFAULT_WTW_CFG)
    parser.add_argument("--domain", default=defaults.DEFAULT_DOMAIN_ID, type=int)
    parser.add_argument("--interface", default=defaults.DEFAULT_INTERFACE)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    sim = InjectedSimulation(
        scene=args.scene,
        model_dir=args.model_dir,
        cfg_path=args.cfg_path,
        domain=args.domain,
        interface=args.interface,
        headless=True,
    )
    await setup_and_start(sim)


if __name__ == "__main__":
    with suppress(KeyboardInterrupt):
        asyncio.run(main())
    try:
        rr.get_data_recording().flush()
    except:
        pass
