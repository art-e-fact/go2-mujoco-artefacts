# Go2 MuJoCo Walk Demo (Artefacts)

Unitree Go2 walking demo using MuJoCo.
The demo runs on ubuntu, but also MacOS natively without vms or containerization.

A flat scene is added to the `unitree_mujoco` package (see copy stage) to provide a simple demo for the [Go2](https://www.unitree.com/go2) to move.


## Setup

### Prerequisites

This project uses `uv` for Python environment management. For installation instructions, see https://docs.astral.sh/uv/getting-started/installation/


## Run
```bash
uv run go2_rails_demo.py  --rerun --seed 123 --policy rsl_rl --heightmap-nav
```
The robot should walk on the rails following the target object.

### Run test with artefacts

1. Create a project at app.artefacts.com
2. Rename `project:` in the `artefacts.yaml` file to your `<org_name>/<project_name>
3. Run `artefacts config add <org_name>/<project_name>` , create the ApiKey, and paste into the terminal
4. Select `N` when you are prompted to whether you would like a new `artefacts.yaml` file

You will be able to run the test with the following command
```
artefacts run basic_test
```

Once the test has finished, test result (and a video) will be uploaded to your project page on the artefacts dashboard


## Atributions

Rail scene based on work from sBjamms https://sketchfab.com/3d-models/train-track-dff84793ce2f4d7ca19e67b5194eeca2
