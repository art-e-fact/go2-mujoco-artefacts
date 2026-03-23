# Go2 MuJoCo Walk Demo (Artefacts)

Unitree Go2 walking demo using MuJoCo and the Walk-These-Ways pretrained policy.
The demo runs on ubuntu, but also MacOS natively without vms or containerization.

A flat scene is added to the `unitree_mujoco` package (see copy stage) to provide a simple demo for the [Go2](https://www.unitree.com/go2) to move.

With thanks to [Teddy Liao](https://github.com/Teddy-Liao) for the pretrained policy in the [walk-these-ways-go2](https://github.com/Teddy-Liao/walk-these-ways-go2) repository.

## Setup

```bash
# Clone required repos
mkdir -p src
git clone --depth=1 -b high-level-direct-elian https://github.com/art-e-fact/unitree_mujoco.git src/unitree_mujoco
git clone --depth=1 https://github.com/unitreerobotics/unitree_sdk2_python.git src/unitree_sdk2_python
git clone --depth=1 https://github.com/unitreerobotics/unitree_ros.git src/go2_urdf

# Download WTW policy checkpoints (3 files only, no full repo clone)
bash scripts/fetch_wtw_checkpoints.sh
```

### Pixi

```bash
# cyclonedds dependencies are buggy, this fixes it
bash ./scripts/fix_unitree_sdk2_for_pixi.sh
# enter the shell of the environment
pixi install
```

## Visualize (Rerun)

The archive is bugged, so you have to launch rerun first.

```bash
pixi run rerun
```

The rerun template is available at `./rerun_template.rbl`

## Run

```bash
pixi run sim
```

## Atributions

Rail scene based on work from sBjamms https://sketchfab.com/3d-models/train-track-dff84793ce2f4d7ca19e67b5194eeca2
