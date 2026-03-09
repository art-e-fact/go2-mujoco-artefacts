# Running the High-Level SDK with MuJoCo

Three terminals, all run from the project root.

## Prerequisites

The following manual changes are required after cloning `src/unitree_mujoco` (only once):

**`src/unitree_mujoco/simulate_python/config.py`** — disable joystick (otherwise the sim thread exits if no gamepad is connected):
```python
USE_JOYSTICK = 0
```

**`src/unitree_mujoco/simulate_python/unitree_mujoco.py`** — reset to standing keyframe on startup. Add after `mj_data = mujoco.MjData(mj_model)`:
```python
mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)  # start from "home" pose
```

## Terminal 1 — Sport server (start this first)

```bash
cd src/unitree_mujoco/simulate_python
PYTHONPATH=../../../src/unitree_sdk2_python python sport_sim_server.py
```

Wait for `Serving sport RPC. Waiting for bridge…` — WTW model is fully loaded.

## Terminal 2 — MuJoCo simulator (start after sport server is ready)

```bash
cd src/unitree_mujoco/simulate_python
PYTHONPATH=../../../src/unitree_sdk2_python python unitree_mujoco.py
```

The viewer opens. Sport server prints `Bridge connected` then `Standing complete.`
within ~0.5 s. The robot holds its keyframe pose and WTW takes over immediately.

## Terminal 3 — SDK client

```bash
cd src/unitree_sdk2_python
PYTHONPATH=. python example/go2/high_level/go2_sport_client.py lo
```

At the `Enter id or name:` prompt, enter a command ID or name:

| ID | Name | Effect |
|----|------|--------|
| `1` | stand_up | stand upright |
| `2` | stand_down | crouch |
| `3` | move forward | walk forward at 0.3 m/s |
| `4` | move lateral | strafe left at 0.3 m/s |
| `5` | move rotate | rotate at 0.5 rad/s |
| `6` | stop_move | stop walking |
| `0` | damp | cut motor torque (robot falls) |
| `9` | balanced stand | stubbed (returns ok) |

Suggested sequence: `1` → `3` → `6` → `2` → `1`
