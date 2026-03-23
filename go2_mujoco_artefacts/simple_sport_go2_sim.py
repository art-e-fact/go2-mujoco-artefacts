import argparse
import json
import os
import platform as _platform
import time
import xml.etree.ElementTree as ET
from enum import Enum
from threading import Lock

import mujoco
import mujoco.viewer
import numpy as np
import torch
import unitree_sdk2py.go2.sport.sport_api as sport_api
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_ as LowState_default
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.rpc.internal import RPC_ERR_SERVER_API_NOT_IMPL
from unitree_sdk2py.rpc.server import Server

from .defaults import (
    BASE_DIR,
    DEFAULT_DOMAIN_ID,
    DEFAULT_INTERFACE,
    DEFAULT_SCENE,
    DEFAULT_WTW_CFG,
    DEFAULT_WTW_DIR,
    IDLE_SETTLE_TICKS,
    ROBOT,
    ROBOT_DIR,
    SIMULATE_DT,
    TRANSITION_DURATION,
    VIEWER_DT,
    WTW_HZ,
    WTW_STEP_EVERY,
)
from .wtw_controller import (
    DEFAULT_JOINT_ANGLES_WTW,
    WTW_TO_MUJOCO_CTRL,
    WalkTheseWaysController,
)

_WTW_STAND_POS = np.zeros(12, dtype=np.float64)
for i in range(12):
    _WTW_STAND_POS[WTW_TO_MUJOCO_CTRL[i]] = DEFAULT_JOINT_ANGLES_WTW[i]

STAND_UP_POS = _WTW_STAND_POS
STAND_DOWN_POS = np.array(
    [
        0.0473455,
        1.22187,
        -2.44375,
        -0.0473455,
        1.22187,
        -2.44375,
        0.0473455,
        1.22187,
        -2.44375,
        -0.0473455,
        1.22187,
        -2.44375,
    ],
    dtype=np.float64,
)


UNSUPPORTED_API_IDS = (
    sport_api.SPORT_API_ID_EULER,
    sport_api.SPORT_API_ID_SIT,
    sport_api.SPORT_API_ID_RISESIT,
    sport_api.SPORT_API_ID_SPEEDLEVEL,
    sport_api.SPORT_API_ID_HELLO,
    sport_api.SPORT_API_ID_STRETCH,
    sport_api.SPORT_API_ID_CONTENT,
    sport_api.SPORT_API_ID_DANCE1,
    sport_api.SPORT_API_ID_DANCE2,
    sport_api.SPORT_API_ID_SWITCHJOYSTICK,
    sport_api.SPORT_API_ID_POSE,
    sport_api.SPORT_API_ID_SCRAPE,
    sport_api.SPORT_API_ID_FRONTFLIP,
    sport_api.SPORT_API_ID_FRONTJUMP,
    sport_api.SPORT_API_ID_FRONTPOUNCE,
    sport_api.SPORT_API_ID_HEART,
    sport_api.SPORT_API_ID_STATICWALK,
    sport_api.SPORT_API_ID_TROTRUN,
    sport_api.SPORT_API_ID_ECONOMICGAIT,
    sport_api.SPORT_API_ID_LEFTFLIP,
    sport_api.SPORT_API_ID_BACKFLIP,
    sport_api.SPORT_API_ID_HANDSTAND,
    sport_api.SPORT_API_ID_FREEWALK,
    sport_api.SPORT_API_ID_FREEBOUND,
    sport_api.SPORT_API_ID_FREEJUMP,
    sport_api.SPORT_API_ID_FREEAVOID,
    sport_api.SPORT_API_ID_CLASSICWALK,
    sport_api.SPORT_API_ID_WALKUPRIGHT,
    sport_api.SPORT_API_ID_CROSSSTEP,
    sport_api.SPORT_API_ID_AUTORECOVERY_SET,
    sport_api.SPORT_API_ID_AUTORECOVERY_GET,
    sport_api.SPORT_API_ID_SWITCHAVOIDMODE,
)


class State(Enum):
    IDLE_CONNECTED = "idle_connected"
    STANDING = "standing"
    STANDING_UP = "standing_up"
    STANDING_DOWN = "standing_down"
    WALKING = "walking"
    DAMP = "damp"


class SportDirectController(WalkTheseWaysController):
    def step_from_mujoco(
        self,
        sensordata: np.ndarray,
        num_motor: int,
        dim_motor_sensor: int,
        commands: np.ndarray,
    ) -> np.ndarray:
        joint_pos_ctrl = sensordata[:num_motor]
        joint_vel_ctrl = sensordata[num_motor : 2 * num_motor]
        quat = sensordata[dim_motor_sensor : dim_motor_sensor + 4].astype(np.float32)

        joint_pos_wtw = np.array(
            [joint_pos_ctrl[WTW_TO_MUJOCO_CTRL[i]] for i in range(12)],
            dtype=np.float32,
        )
        joint_vel_wtw = np.array(
            [joint_vel_ctrl[WTW_TO_MUJOCO_CTRL[i]] for i in range(12)],
            dtype=np.float32,
        )

        obs = self._build_obs(quat, joint_pos_wtw, joint_vel_wtw, commands)
        self.update_history(obs)

        with torch.no_grad():
            latent = self.adaptation_module(self.obs_history)
            action = self.body(torch.cat([self.obs_history, latent], dim=1))

        self.last_actions = self.actions.clone()
        self.actions = action[0].clone()

        target_pos_wtw = action[0].numpy() * self.action_scale
        target_pos_wtw[[0, 3, 6, 9]] *= self.hip_scale_reduction
        target_pos_wtw += DEFAULT_JOINT_ANGLES_WTW

        self.gait_index = (self.gait_index + self.dt * commands[4]) % 1.0

        target_ctrl = np.zeros(12, dtype=np.float64)
        for i in range(12):
            target_ctrl[WTW_TO_MUJOCO_CTRL[i]] = target_pos_wtw[i]
        return target_ctrl

    def _build_obs(
        self,
        quat: np.ndarray,
        joint_pos_wtw: np.ndarray,
        joint_vel_wtw: np.ndarray,
        commands: np.ndarray,
    ) -> torch.Tensor:
        obs = np.zeros(self.num_obs, dtype=np.float32)
        obs[0:3] = self.get_gravity_vector(quat)
        obs[3:18] = commands * self.commands_scale
        obs[18:30] = (joint_pos_wtw - DEFAULT_JOINT_ANGLES_WTW) * self.obs_scales[
            "dof_pos"
        ]
        obs[30:42] = joint_vel_wtw * self.obs_scales["dof_vel"]
        obs[42:54] = torch.clip(
            self.actions, -self.clip_actions, self.clip_actions
        ).numpy()
        obs[54:66] = self.last_actions.numpy()
        obs[66:70] = self.get_clock_inputs(commands)
        return torch.tensor(obs, dtype=torch.float32).unsqueeze(0)


class SportMuJoCoServer(Server):
    def __init__(self, controller: SportDirectController, num_motor: int):
        super().__init__(sport_api.SPORT_SERVICE_NAME)
        self.controller = controller
        self.num_motor = num_motor
        self.lock = Lock()

        self.state = State.IDLE_CONNECTED
        self.vx = 0.0
        self.vy = 0.0
        self.vyaw = 0.0

        self.transition_start_step = 0
        self.transition_from = np.zeros(num_motor, dtype=np.float64)
        self.transition_to = np.zeros(num_motor, dtype=np.float64)

        self.current_q = np.zeros(num_motor, dtype=np.float64)
        self.sim_step = 0
        self.last_wtw_ctrl: np.ndarray | None = None

    def Init(self):
        self._SetApiVersion(sport_api.SPORT_API_VERSION)
        self._RegistHandler(
            sport_api.SPORT_API_ID_STANDUP, self._handle_stand_up, False
        )
        self._RegistHandler(
            sport_api.SPORT_API_ID_STANDDOWN, self._handle_stand_down, False
        )
        self._RegistHandler(sport_api.SPORT_API_ID_MOVE, self._handle_move, False)
        self._RegistHandler(
            sport_api.SPORT_API_ID_STOPMOVE, self._handle_stop_move, False
        )
        self._RegistHandler(sport_api.SPORT_API_ID_DAMP, self._handle_damp, False)
        self._RegistHandler(
            sport_api.SPORT_API_ID_BALANCESTAND, self._handle_stub, False
        )
        self._RegistHandler(
            sport_api.SPORT_API_ID_RECOVERYSTAND, self._handle_stub, False
        )
        for api_id in UNSUPPORTED_API_IDS:
            self._RegistHandler(api_id, self._handle_not_impl, False)

    def _handle_stand_up(self, _parameter: str):
        with self.lock:
            self.transition_from = self.current_q.copy()
            self.transition_to = STAND_UP_POS.copy()
            self.transition_start_step = self.sim_step
            self.state = State.STANDING_UP
            self.controller.reset()
        print("[sport_mujoco] StandUp")
        return 0, ""

    def _handle_stand_down(self, _parameter: str):
        with self.lock:
            self.transition_from = self.current_q.copy()
            self.transition_to = STAND_DOWN_POS.copy()
            self.transition_start_step = self.sim_step
            self.state = State.STANDING_DOWN
        print("[sport_mujoco] StandDown")
        return 0, ""

    def _handle_move(self, parameter: str):
        payload = json.loads(parameter) if parameter else {}
        vx = float(payload.get("x", 0.0))
        vy = float(payload.get("y", 0.0))
        vyaw = float(payload.get("z", 0.0))
        with self.lock:
            self.vx = vx
            self.vy = vy
            self.vyaw = vyaw
            self.state = (
                State.WALKING
                if any(abs(v) > 1e-6 for v in (vx, vy, vyaw))
                else State.STANDING
            )
        print(f"[sport_mujoco] Move vx={vx:.2f} vy={vy:.2f} vyaw={vyaw:.2f}")
        return 0, ""

    def _handle_stop_move(self, _parameter: str):
        with self.lock:
            self.vx = 0.0
            self.vy = 0.0
            self.vyaw = 0.0
            if self.state == State.WALKING:
                self.state = State.STANDING
        print("[sport_mujoco] StopMove")
        return 0, ""

    def _handle_damp(self, _parameter: str):
        with self.lock:
            self.state = State.DAMP
        print("[sport_mujoco] Damp")
        return 0, ""

    def _handle_stub(self, _parameter: str):
        return 0, ""

    def _handle_not_impl(self, _parameter: str):
        return RPC_ERR_SERVER_API_NOT_IMPL, ""

    def tick(
        self,
        sensordata: np.ndarray,
        num_motor: int,
        dim_motor_sensor: int,
    ) -> tuple[np.ndarray, float, float]:
        with self.lock:
            self.current_q = sensordata[:num_motor].copy()

            state = self.state
            step = self.sim_step
            vx = self.vx
            vy = self.vy
            vyaw = self.vyaw
            transition_start_step = self.transition_start_step
            transition_from = self.transition_from.copy()
            transition_to = self.transition_to.copy()

            self.sim_step += 1

        ctrl_target = self.current_q.copy()
        kp = 50.0
        kd = 3.5

        if state == State.IDLE_CONNECTED:
            if step >= IDLE_SETTLE_TICKS:
                with self.lock:
                    self.controller.reset()
                    self.state = State.STANDING
                print("[sport_mujoco] Standing complete")

        elif state == State.DAMP:
            kp = 0.0
            kd = 2.0

        elif state in (State.STANDING_UP, State.STANDING_DOWN):
            elapsed = (step - transition_start_step) * SIMULATE_DT
            progress = min(max(elapsed / TRANSITION_DURATION, 0.0), 1.0)
            phase = 0.5 - 0.5 * np.cos(np.pi * progress)
            ctrl_target = (1.0 - phase) * transition_from + phase * transition_to
            kp = 20.0 + 30.0 * phase
            if progress >= 1.0:
                next_state = (
                    State.STANDING
                    if state == State.STANDING_UP
                    else State.IDLE_CONNECTED
                )
                with self.lock:
                    self.state = next_state
                print(f"[sport_mujoco] Transition done -> {next_state.value}")

        elif state in (State.STANDING, State.WALKING):
            commands = self.controller.get_commands(
                vx if state == State.WALKING else 0.0,
                vy if state == State.WALKING else 0.0,
                vyaw if state == State.WALKING else 0.0,
            )
            if step % WTW_STEP_EVERY == 0:
                self.last_wtw_ctrl = self.controller.step_from_mujoco(
                    sensordata,
                    num_motor,
                    dim_motor_sensor,
                    commands,
                )
            if self.last_wtw_ctrl is not None:
                ctrl_target = self.last_wtw_ctrl
            kp = self.controller.stiffness
            kd = self.controller.damping

        return ctrl_target, kp, kd


class Simulation:
    def __init__(
        self,
        scene: str = DEFAULT_SCENE,
        model_dir: str = DEFAULT_WTW_DIR,
        cfg_path: str = DEFAULT_WTW_CFG,
        domain: int = DEFAULT_DOMAIN_ID,
        interface: str = DEFAULT_INTERFACE,
        headless: bool = False,
    ):
        self.scene = scene
        self.model_dir = model_dir
        self.cfg_path = cfg_path
        self.domain = domain
        self.interface = interface
        self.headless = headless

        self.model = _load_scene(scene)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.model.opt.timestep = SIMULATE_DT

        self.num_motor = self.model.nu
        self.dim_motor_sensor = 3 * self.num_motor

        self.controller = SportDirectController(model_dir, cfg_path)
        self.server = SportMuJoCoServer(self.controller, self.num_motor)

        ChannelFactoryInitialize(domain, interface)
        self.server.Init()
        self.server.Start()

        self.low_state = LowState_default()
        self.low_state_pub = ChannelPublisher("rt/lowstate", LowState_)
        self.low_state_pub.Init()

        print(f"[sport_mujoco] DDS domain={domain} interface={interface}")
        print(f"[sport_mujoco] Scene={scene}")
        print(
            f"[sport_mujoco] WTW every {WTW_STEP_EVERY} steps -> {WTW_HZ} Hz sim-time"
        )

    def step(self):
        ctrl_target, kp, kd = self.server.tick(
            self.data.sensordata,
            self.num_motor,
            self.dim_motor_sensor,
        )
        self._apply_pd_control(ctrl_target, kp, kd)
        mujoco.mj_step(self.model, self.data)
        self._publish_low_state()

    def _apply_pd_control(self, ctrl_target: np.ndarray, kp: float, kd: float):
        joint_q = self.data.sensordata[: self.num_motor]
        joint_dq = self.data.sensordata[self.num_motor : 2 * self.num_motor]
        self.data.ctrl[:] = kp * (ctrl_target - joint_q) + kd * (-joint_dq)

    def _publish_low_state(self):
        for i in range(self.num_motor):
            self.low_state.motor_state[i].q = self.data.sensordata[i]
            self.low_state.motor_state[i].dq = self.data.sensordata[self.num_motor + i]
            self.low_state.motor_state[i].tau_est = self.data.sensordata[
                2 * self.num_motor + i
            ]

        imu_offset = self.dim_motor_sensor
        self.low_state.imu_state.quaternion[0] = self.data.sensordata[imu_offset + 0]
        self.low_state.imu_state.quaternion[1] = self.data.sensordata[imu_offset + 1]
        self.low_state.imu_state.quaternion[2] = self.data.sensordata[imu_offset + 2]
        self.low_state.imu_state.quaternion[3] = self.data.sensordata[imu_offset + 3]
        self.low_state_pub.Write(self.low_state)


def _load_scene(scene_path: str) -> mujoco.MjModel:
    scene_path = os.path.abspath(scene_path)
    scene_dir = os.path.dirname(scene_path)

    if scene_dir == ROBOT_DIR:
        return mujoco.MjModel.from_xml_path(scene_path)

    tree = ET.parse(scene_path)
    root = tree.getroot()
    for elem in root.iter():
        file_attr = elem.get("file")
        if file_attr and not os.path.isabs(file_attr):
            elem.set("file", os.path.normpath(os.path.join(scene_dir, file_attr)))

    tmp_scene = os.path.join(ROBOT_DIR, "_tmp_scene.xml")
    try:
        tree.write(tmp_scene, encoding="unicode", xml_declaration=False)
        return mujoco.MjModel.from_xml_path(tmp_scene)
    finally:
        if os.path.exists(tmp_scene):
            os.remove(tmp_scene)


def run(sim: Simulation):
    if sim.headless:
        while True:
            loop_start = time.perf_counter()
            sim.step()
            _sleep_to_rate(loop_start, SIMULATE_DT)

    viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
    next_viewer_sync = time.perf_counter()

    while viewer.is_running():
        loop_start = time.perf_counter()
        sim.step()

        if loop_start >= next_viewer_sync:
            viewer.sync()
            next_viewer_sync = loop_start + VIEWER_DT

        _sleep_to_rate(loop_start, SIMULATE_DT)


def _sleep_to_rate(loop_start: float, dt: float):
    remaining = dt - (time.perf_counter() - loop_start)
    if remaining > 0.0:
        time.sleep(remaining)


def main():
    parser = argparse.ArgumentParser(description="MuJoCo GO2 sim + sport RPC")
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--model-dir", default=DEFAULT_WTW_DIR)
    parser.add_argument("--cfg-path", default=DEFAULT_WTW_CFG)
    parser.add_argument("--domain", default=DEFAULT_DOMAIN_ID, type=int)
    parser.add_argument("--interface", default=DEFAULT_INTERFACE)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    sim = Simulation(
        scene=args.scene,
        model_dir=args.model_dir,
        cfg_path=args.cfg_path,
        domain=args.domain,
        interface=args.interface,
        headless=args.headless,
    )
    run(sim)


if __name__ == "__main__":
    main()
