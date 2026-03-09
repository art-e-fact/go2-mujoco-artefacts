#!/usr/bin/env python3
"""
Record rt/lowstate joint positions and velocities to a CSV file.

Usage (while the sim stack is running):
  python record_joints.py                  # records until Ctrl+C
  python record_joints.py --seconds 10     # records for 10 seconds
  python record_joints.py --out joints.csv
"""

import sys, os, time, argparse, csv, signal

_SDK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "src", "unitree_sdk2_python")
sys.path.insert(0, _SDK_DIR)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

# Motor names in ctrl order (FR, FL, RR, RL) × (hip, thigh, calf)
JOINT_NAMES = [
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
]

rows = []
t0 = None
_done = False


def _on_lowstate(msg: LowState_):
    global t0
    now = time.perf_counter()
    if t0 is None:
        t0 = now
    row = [f"{now - t0:.4f}"]
    for i in range(12):
        row.append(f"{msg.motor_state[i].q:.5f}")
        row.append(f"{msg.motor_state[i].dq:.5f}")
    row.append(f"{msg.imu_state.quaternion[0]:.5f}")  # w
    row.append(f"{msg.imu_state.quaternion[1]:.5f}")  # x
    row.append(f"{msg.imu_state.quaternion[2]:.5f}")  # y
    row.append(f"{msg.imu_state.quaternion[3]:.5f}")  # z
    rows.append(row)


def main():
    global _done
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--domain",    type=int, default=0)
    parser.add_argument("--seconds",   type=float, default=0,
                        help="Stop after N seconds (0 = Ctrl+C)")
    parser.add_argument("--out",       default="joints.csv")
    args = parser.parse_args()

    signal.signal(signal.SIGINT,  lambda *_: globals().update(_done=True))
    signal.signal(signal.SIGTERM, lambda *_: globals().update(_done=True))

    ChannelFactoryInitialize(args.domain, args.interface)
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(_on_lowstate, 10)

    deadline = time.perf_counter() + args.seconds if args.seconds > 0 else None
    print(f"Recording to {args.out} … (Ctrl+C to stop)")
    while not _done:
        if deadline and time.perf_counter() >= deadline:
            break
        time.sleep(0.05)

    header = ["t_s"]
    for name in JOINT_NAMES:
        header += [f"{name}_q", f"{name}_dq"]
    header += ["imu_qw", "imu_qx", "imu_qy", "imu_qz"]

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    print(f"Saved {len(rows)} rows → {args.out}")


if __name__ == "__main__":
    main()
