# Running the High-Level SDK with MuJoCo

## Prerequisites

### 1 — WTW policy checkpoints

Place the three checkpoint files from the Walk-These-Ways pretrained Go2 policy into:

```
src/unitree_mujoco/simulate_python/wtw/
  body_latest.jit
  adaptation_module_latest.jit
  parameters_cpu.pkl
```

Using sparse-checkout to avoid downloading the full repository:

```bash
bash scripts/fetch_wtw_checkpoints.sh
```

### 2 — unitree_mujoco config

**`src/unitree_mujoco/simulate_python/config.py`** — disable joystick (otherwise the sim thread exits if no gamepad is connected):
```python
USE_JOYSTICK = 0
```

**`src/unitree_mujoco/simulate_python/unitree_mujoco.py`** — reset to standing keyframe on startup. Add after `mj_data = mujoco.MjData(mj_model)`:
```python
mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)  # start from "home" pose
```
(Not needed for `sport_mujoco.py` — it handles this itself.)

---

## Option A — Direct integration (recommended, used by go2_wtw_demo.py)

Two terminals. WTW runs inside the sim loop — no sync issues at any real-time factor.

### Terminal 1 — Unified sim + sport server

```bash
cd src/unitree_mujoco/simulate_python
PYTHONPATH=../../../src/unitree_sdk2_python python sport_mujoco.py
```

Wait for `Serving sport RPC.` — WTW model is loaded and viewer is open.

### Terminal 2 — SDK client

```bash
cd src/unitree_sdk2_python
PYTHONPATH=. python example/go2/high_level/go2_sport_client.py lo
```

---

## Option B — Bridge-based (two-process, original approach)

Three terminals. Requires manual start ordering. May desync if sim runs slower than real-time.

### Terminal 1 — Sport server (start this first)

```bash
cd src/unitree_mujoco/simulate_python
PYTHONPATH=../../../src/unitree_sdk2_python python sport_sim_server.py
```

Wait for `Serving sport RPC. Waiting for bridge…` — WTW model is fully loaded.

### Terminal 2 — MuJoCo simulator (start after sport server is ready)

```bash
cd src/unitree_mujoco/simulate_python
PYTHONPATH=../../../src/unitree_sdk2_python python unitree_mujoco.py
```

The viewer opens. Sport server prints `Bridge connected` then `Standing complete.`

### Terminal 3 — SDK client

```bash
cd src/unitree_sdk2_python
PYTHONPATH=. python example/go2/high_level/go2_sport_client.py lo
```

---

## SDK commands

At the `Enter id or name:` prompt:

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
