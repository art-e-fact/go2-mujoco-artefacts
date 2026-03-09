#!/usr/bin/env python3
"""
Go2 Walk-These-Ways Demo

Drives the Go2 along a square path using the high-level SportClient API.
Starts the full sim stack automatically.

Usage:
  python go2_wtw_demo.py              # MuJoCo viewer (UI)
  python go2_wtw_demo.py --headless   # no display (CI / testing)

WalkTheseWaysController is also defined here and imported by sport_sim_server.
"""

import sys
import os
import time
import threading
import subprocess
import numpy as np
import torch
import mujoco
import pickle
import io

_HERE    = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR = os.path.join(_HERE, "src", "unitree_mujoco", "simulate_python")
_SDK_DIR = os.path.join(_HERE, "src", "unitree_sdk2_python")

sys.path.insert(0, _SDK_DIR)

# In order to run on non-Cuda machines
class CPUUnpickler(pickle.Unpickler):
    """Unpickler that maps CUDA tensors to CPU"""
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
        return super().find_class(module, name)


# Maps WTW joint indices to MuJoCo ctrl indices (different ordering)
# 4 Legs with 3 joints each: Hip, Thigh, Calf
WTW_TO_MUJOCO_CTRL = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

# Default joint angles in WTW order i.e standing pose
DEFAULT_JOINT_ANGLES_WTW = np.array([
    0.1, 0.8, -1.5,   # FL
    -0.1, 0.8, -1.5,  # FR
    0.1, 1.0, -1.5,   # RL
    -0.1, 1.0, -1.5,  # RR
], dtype=np.float32)


class WalkTheseWaysController:
    def __init__(self, model_dir: str, cfg_path: str):
        """
        Initialize the Walk-These-Ways controller.
        
        Args:
            model_dir: Path to checkpoints directory
            cfg_path: Path to parameters_cpu.pkl
        """
        # Load config
        with open(cfg_path, 'rb') as f:
            pkl_cfg = CPUUnpickler(f).load()
        self.cfg = pkl_cfg['Cfg']
        
        # Load models
        self.body = torch.jit.load(f"{model_dir}/body_latest.jit", map_location='cpu')
        self.adaptation_module = torch.jit.load(f"{model_dir}/adaptation_module_latest.jit", map_location='cpu')
        self.body.eval()
        self.adaptation_module.eval()
        
        # Config values
        self.num_obs = self.cfg['env']['num_observations']  # 70
        self.num_hist = self.cfg['env']['num_observation_history']  # 30
        self.num_actions = self.cfg['env']['num_actions']  # 12
        self.num_commands = self.cfg['commands']['num_commands']  # 15
        
        # Observation scales
        self.obs_scales = self.cfg['obs_scales']
        self.clip_actions = self.cfg['normalization']['clip_actions']
        
        # Control params
        self.action_scale = self.cfg['control']['action_scale']  # 0.25
        self.hip_scale_reduction = self.cfg['control']['hip_scale_reduction']  # 0.5
        self.stiffness = self.cfg['control']['stiffness']['joint']  # 20.0
        self.damping = self.cfg['control']['damping']['joint']  # 0.5
        
        # Command scales (15 commands)
        self.commands_scale = np.array([
            self.obs_scales['lin_vel'],          # 0: lin_vel_x
            self.obs_scales['lin_vel'],          # 1: lin_vel_y  
            self.obs_scales['ang_vel'],          # 2: ang_vel_yaw
            self.obs_scales['body_height_cmd'],  # 3: body height
            1.0,  # 4: gait freq
            1.0,  # 5: gait phase
            1.0,  # 6: gait offset
            1.0,  # 7: gait bound
            1.0,  # 8: gait duration
            self.obs_scales['footswing_height_cmd'],  # 9: footswing height
            self.obs_scales['body_pitch_cmd'],   # 10: body pitch
            self.obs_scales['body_roll_cmd'],    # 11: body roll
            self.obs_scales['stance_width_cmd'], # 12: stance width
            self.obs_scales['stance_length_cmd'],# 13: stance length
            self.obs_scales['aux_reward_cmd'],   # 14: aux reward
        ], dtype=np.float32)
        
        # State buffers
        self.obs_history = torch.zeros(1, self.num_obs * self.num_hist, dtype=torch.float32)
        self.actions = torch.zeros(self.num_actions, dtype=torch.float32)
        self.last_actions = torch.zeros(self.num_actions, dtype=torch.float32)
        
        # Gait parameters
        self.gait_index = 0.0
        self.dt = 0.02  # 50Hz control
        
        print(f"Loaded Walk-These-Ways policy:")
        print(f"  num_obs: {self.num_obs}, num_hist: {self.num_hist}")
        print(f"  action_scale: {self.action_scale}, hip_scale: {self.hip_scale_reduction}")
        print(f"  stiffness: {self.stiffness}, damping: {self.damping}")
    
    def get_gravity_vector(self, quat: np.ndarray) -> np.ndarray:
        """
        Get gravity vector in body frame from quaternion.
        MuJoCo quat: [w, x, y, z]
        
        Projects world gravity [0, 0, -1] into body frame using R.T @ gravity
        """
        w, x, y, z = quat
        
        # Build rotation matrix from quaternion
        # This is the rotation from body to world frame
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
        ], dtype=np.float32)
        
        # Project gravity into body frame: R.T @ [0, 0, -1]
        gravity_world = np.array([0, 0, -1], dtype=np.float32)
        gravity_body = R.T @ gravity_world
        
        return gravity_body
    
    def get_clock_inputs(self, commands: np.ndarray) -> np.ndarray:
        """Get clock inputs for gait phase using command parameters"""
        # Extract gait parameters from commands
        phases = commands[5]   # phase offset between legs
        offsets = commands[6]  # additional offset
        bounds = commands[7]   # bound parameter
        
        # Compute foot indices (matching walk-these-ways)
        foot_indices = [
            self.gait_index + phases + offsets + bounds,  # FL
            self.gait_index + offsets,                     # FR
            self.gait_index + bounds,                      # RL
            self.gait_index + phases                       # RR
        ]
        
        clock = np.zeros(4, dtype=np.float32)
        clock[0] = np.sin(2 * np.pi * foot_indices[0])
        clock[1] = np.sin(2 * np.pi * foot_indices[1])
        clock[2] = np.sin(2 * np.pi * foot_indices[2])
        clock[3] = np.sin(2 * np.pi * foot_indices[3])
        
        return clock
    
    def get_commands(self, vx: float = 0.0, vy: float = 0.0, vyaw: float = 0.0) -> np.ndarray:
        """
        Get command vector (15 dims).
        Default is a trotting gait.
        """
        commands = np.zeros(self.num_commands, dtype=np.float32)
        
        # Velocity commands
        commands[0] = vx  # lin_vel_x
        commands[1] = vy  # lin_vel_y
        commands[2] = vyaw  # ang_vel_yaw
        
        # Body pose commands
        commands[3] = 0.0  # body height (relative)
        
        # Gait parameters (trotting defaults)
        commands[4] = 3.0  # gait frequency
        commands[5] = 0.5  # phase
        commands[6] = 0.0  # offset
        commands[7] = 0.0  # bound
        commands[8] = 0.5  # duration
        
        # Footswing and body orientation
        commands[9] = 0.08  # footswing height
        commands[10] = 0.0  # body pitch
        commands[11] = 0.0  # body roll
        
        # Stance params
        commands[12] = 0.0  # stance width
        commands[13] = 0.0  # stance length
        commands[14] = 0.0  # aux reward
        
        return commands
    
    def build_observation(self, data: mujoco.MjData, commands: np.ndarray) -> torch.Tensor:
        """
        Build 70-dim observation vector.
        
        Structure:
        - gravity_vector: 3
        - commands * scale: 15
        - dof_pos (relative): 12
        - dof_vel: 12
        - actions: 12
        - last_actions: 12
        - clock_inputs: 4
        Total: 70
        """
        obs = np.zeros(self.num_obs, dtype=np.float32)
        idx = 0
        
        # 1. Gravity vector (3)
        quat = data.qpos[3:7]  # w, x, y, z
        gravity = self.get_gravity_vector(quat)
        obs[idx:idx+3] = gravity
        idx += 3
        
        # 2. Commands * scale (15)
        scaled_commands = commands * self.commands_scale
        obs[idx:idx+self.num_commands] = scaled_commands
        idx += self.num_commands
        
        # 3. Joint positions relative to default (12)
        joint_pos = data.qpos[7:19]
        dof_pos_rel = (joint_pos - DEFAULT_JOINT_ANGLES_WTW) * self.obs_scales['dof_pos']
        obs[idx:idx+12] = dof_pos_rel
        idx += 12
        
        # 4. Joint velocities (12)
        joint_vel = data.qvel[6:18]
        dof_vel = joint_vel * self.obs_scales['dof_vel']
        obs[idx:idx+12] = dof_vel
        idx += 12
        
        # 5. Current actions (12)
        clipped_actions = torch.clip(self.actions, -self.clip_actions, self.clip_actions)
        obs[idx:idx+12] = clipped_actions.numpy()
        idx += 12
        
        # 6. Last actions (12)
        obs[idx:idx+12] = self.last_actions.numpy()
        idx += 12
        
        # 7. Clock inputs (4)
        clock = self.get_clock_inputs(commands)
        obs[idx:idx+4] = clock
        idx += 4
        
        return torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    
    def update_history(self, obs: torch.Tensor):
        """Update observation history buffer (FIFO)"""
        # Shift history and add new observation
        self.obs_history = torch.roll(self.obs_history, -self.num_obs, dims=1)
        self.obs_history[0, -self.num_obs:] = obs[0]
    
    def step(self, data: mujoco.MjData, commands: np.ndarray) -> np.ndarray:
        """
        Run one policy step.
        
        Returns: target joint positions in MuJoCo order
        """
        # Build observation
        obs = self.build_observation(data, commands)
        
        # Update history
        self.update_history(obs)
        
        # Run adaptation module
        with torch.no_grad():
            latent = self.adaptation_module(self.obs_history)
            
            # Run body policy
            body_input = torch.cat([self.obs_history, latent], dim=1)
            action = self.body(body_input)
        
        # Update action buffers
        self.last_actions = self.actions.clone()
        self.actions = action[0].clone()
        
        # Convert action to target positions
        action_np = action[0].numpy()
        
        # Scale actions
        scaled_action = action_np * self.action_scale
        
        # Hip scale reduction
        scaled_action[[0, 3, 6, 9]] *= self.hip_scale_reduction
        
        # Add to default angles (WTW order)
        target_pos_wtw = scaled_action + DEFAULT_JOINT_ANGLES_WTW
        
        # Update gait index
        gait_freq = commands[4]
        self.gait_index = (self.gait_index + self.dt * gait_freq) % 1.0
        
        return target_pos_wtw
    
    def reset(self):
        """Reset controller state"""
        self.obs_history.zero_()
        self.actions.zero_()
        self.last_actions.zero_()
        self.gait_index = 0.0


def _drain(proc, events):
    """Print subprocess stdout; set events[i] when events[i][0] marker appears.

    events: list of (marker_str_or_None, threading.Event)
    """
    for line in proc.stdout:
        print(f"  [sim] {line.rstrip()}")
        for marker, event in events:
            if marker and marker in line:
                event.set()


def _stop(procs):
    for proc in reversed(procs):
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def main():
    import argparse
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.sport.sport_client import SportClient

    parser = argparse.ArgumentParser(description="Go2 Walk-These-Ways Demo")
    parser.add_argument("--cycles",    type=int,   default=1,   help="Number of square-path cycles")
    parser.add_argument("--interface", default="lo",             help="Network interface")
    parser.add_argument("--domain",    type=int,   default=0,   help="DDS domain ID")
    parser.add_argument("--headless",  action="store_true",      help="No viewer (use for testing/CI)")
    args = parser.parse_args()

    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": _SDK_DIR}
    procs = []

    try:
        # --- Sport server first (WTW model load takes ~5-10 s) ---
        # Starting it before the bridge means the policy is fully loaded and
        # ready to receive the first rt/lowstate, so the robot never goes limp.
        server_proc  = subprocess.Popen(
            [sys.executable, "-u", os.path.join(_SIM_DIR, "sport_sim_server.py"),
             "--interface", args.interface, "--domain", str(args.domain)],
            cwd=_HERE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=env,
        )
        server_ready    = threading.Event()
        server_standing = threading.Event()
        threading.Thread(target=_drain, args=(server_proc,
                                              [("Serving sport RPC", server_ready),
                                               ("Standing complete.", server_standing)]),
                         daemon=True).start()
        procs.append(server_proc)
        assert server_ready.wait(timeout=60), "sport_sim_server did not load in time"

        # --- Bridge (viewer or headless) — started only after WTW is loaded ---
        if args.headless:
            bridge_cmd = [sys.executable, "-u",
                          os.path.join(_HERE, "headless_bridge.py"),
                          "--interface", args.interface, "--domain", str(args.domain)]
            bridge_cwd    = _HERE
            bridge_marker = "Running"
        else:
            bridge_cmd = [sys.executable, "-u",
                          os.path.join(_SIM_DIR, "unitree_mujoco.py")]
            bridge_cwd    = _SIM_DIR   # config.ROBOT_SCENE is relative to this dir
            bridge_marker = None

        bridge_proc  = subprocess.Popen(bridge_cmd, cwd=bridge_cwd,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, env=env)
        bridge_ready = threading.Event()
        threading.Thread(target=_drain, args=(bridge_proc, [(bridge_marker, bridge_ready)]),
                         daemon=True).start()
        procs.append(bridge_proc)

        if args.headless:
            assert bridge_ready.wait(timeout=15), "headless_bridge did not start in time"
        else:
            time.sleep(2.0)   # give the viewer time to open and initialise DDS

        # Wait for sport server to see the bridge and reach standing.
        assert server_standing.wait(timeout=20), "sport_sim_server did not reach standing pose"
        time.sleep(1.5)  # WTW pre-warming accumulates during STANDING state

        # --- SportClient ---
        ChannelFactoryInitialize(args.domain, args.interface)
        client = SportClient()
        client.SetTimeout(10.0)
        client.Init()

        print(f"\n=== Walk-These-Ways Go2 Square Demo ({args.cycles} cycle(s)) ===")
        print("Sequence per cycle: forward 4 s → turn 4 s → forward 4 s → turn 4 s")
        print("=================================================================\n")

        for cycle in range(args.cycles):
            print(f"Cycle {cycle + 1}/{args.cycles}")
            client.Move(0.5, 0.0, 0.0);  time.sleep(4.0)   # forward
            client.Move(0.2, 0.0, 1.5);  time.sleep(4.0)   # forward + turn left
            client.Move(0.5, 0.0, 0.0);  time.sleep(4.0)   # forward
            client.Move(0.2, 0.0, 1.5);  time.sleep(4.0)   # forward + turn left

        client.StopMove()
        print(f"\nCompleted {args.cycles} cycle(s).")

    finally:
        _stop(procs)


if __name__ == "__main__":
    main()
