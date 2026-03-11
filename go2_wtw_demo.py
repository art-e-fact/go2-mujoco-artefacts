#!/usr/bin/env python3
"""
Go2 Walk-These-Ways Demo for MuJoCo

Uses the pretrained gait-conditioned policy from walk-these-ways-go2.
Observation: 70 dims per timestep, 30-step history (2100 total)
Architecture: adaptation_module(2100) -> latent(2), body(2102) -> action(12)
"""

import time
import numpy as np
import torch
import mujoco
import mujoco.viewer
import pickle
import io

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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Go2 Walk-These-Ways Demo")
    parser.add_argument("--cycles", type=int, default=0, help="Exit after N cycles (0 = run forever)")
    parser.add_argument("--record", type=str, default=None, help="Record video to this file path (e.g. output/demo.mp4)")
    parser.add_argument("--headless", action="store_true", help="Run without viewer (no display required)")
    args = parser.parse_args()
    
    # Paths
    model_dir = "src/walk-these-ways-go2/runs/gait-conditioned-agility/pretrain-go2/train/142238.667503/checkpoints"
    cfg_path = "src/walk-these-ways-go2/runs/gait-conditioned-agility/pretrain-go2/train/142238.667503/parameters_cpu.pkl"
    xml_path = "src/unitree_mujoco/unitree_robots/go2/scene_flat.xml"  # Flat ground scene (copied from resources/)
    max_cycles = args.cycles
    
    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Simulation params
    model.opt.timestep = 0.005  # 200 Hz physics
    control_decimation = 4  # 50 Hz control
    
    # Initialize controller
    controller = WalkTheseWaysController(model_dir, cfg_path)
    
    # PD gains (from walk-these-ways config)
    kp = np.full(12, controller.stiffness, dtype=np.float32)
    kd = np.full(12, controller.damping, dtype=np.float32)
    
    # State
    target_pos = np.zeros(12, dtype=np.float32)
    step_count = 0
    
    # Video recording setup
    renderer = None
    record_cam = None
    frames = []
    if args.record:
        model.vis.global_.offwidth = 1280
        model.vis.global_.offheight = 720
        renderer = mujoco.Renderer(model, width=1280, height=720)
        # Camera that tracks the robot
        record_cam = mujoco.MjvCamera()
        record_cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        record_cam.trackbodyid = model.body('base_link').id
        record_cam.distance = 3.0
        record_cam.elevation = -30.0
        record_cam.azimuth = 135.0
        print(f"Recording video to: {args.record}")

    print("\n=== Walk-These-Ways Go2 Square Demo ===")
    print("Simple sequence: Forward -> Turn -> Forward -> Turn (repeat)")
    print("==========================================\n")
    
    def sim_step():
        nonlocal step_count, target_pos
        sim_time = step_count * model.opt.timestep
        cycle_num = int(sim_time // 18.0)
        cycle_time = sim_time % 18.0  # 18 second cycle

        if max_cycles > 0 and cycle_num >= max_cycles:
            print(f"\nCompleted {max_cycles} cycle(s). Exiting.")
            return False

        # Build commands for square path
        if cycle_time < 1.0:
            commands = controller.get_commands()  # Stand still
        elif cycle_time < 5.0:
            commands = controller.get_commands(vx=0.5)  # Walk forward
        elif cycle_time < 9.0:
            commands = controller.get_commands(vx=0.2, vyaw=1.5)  # Forward + turn left
        elif cycle_time < 13.0:
            commands = controller.get_commands(vx=0.5)  # Walk forward
        elif cycle_time < 17.0:
            commands = controller.get_commands(vx=0.2, vyaw=1.5)  # Forward + turn left
        else:
            commands = controller.get_commands()  # Brief pause before cycle repeats

        # Policy step (at control frequency)
        if step_count % control_decimation == 0:
            target_pos = controller.step(data, commands)

        # PD control
        current_pos = data.qpos[7:19]
        current_vel = data.qvel[6:18]
        torques_wtw = kp * (target_pos - current_pos) - kd * current_vel
        data.ctrl[:] = torques_wtw[WTW_TO_MUJOCO_CTRL]

        # Physics step
        mujoco.mj_step(model, data)
        step_count += 1

        # Capture frame for video (at 50 fps)
        if renderer and step_count % control_decimation == 0:
            renderer.update_scene(data, record_cam)
            frames.append(renderer.render().copy())

        return True

    if args.headless:
        while sim_step():
            pass
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                step_start = time.time()
                if not sim_step():
                    break
                viewer.sync()
                elapsed_sim = time.time() - step_start
                sleep_time = model.opt.timestep - elapsed_sim
                if sleep_time > 0:
                    time.sleep(sleep_time)

    # Save video after sim loop ends
    if renderer and frames:
        import mediapy as media
        media.write_video(args.record, frames, fps=50)
        print(f"Saved video: {args.record} ({len(frames)} frames)")
        renderer.close()


if __name__ == "__main__":
    main()
