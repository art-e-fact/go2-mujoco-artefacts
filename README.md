# Go2 MuJoCo Walk Demo (Artefacts)

Unitree Go2 walking demo using MuJoCo and the Walk-These-Ways pretrained policy.
The demo runs on ubuntu, but also MacOS natively without vms or containerization.

A flat scene is added to the `unitree_mujoco` package (see copy stage) to provide a simple demo for the [Go2](https://www.unitree.com/go2) to move.

With thanks to [Teddy Liao](https://github.com/Teddy-Liao) for the pretrained policy in the [walk-these-ways-go2](https://github.com/Teddy-Liao/walk-these-ways-go2) repository.

## Setup

```bash
# Clone required repos
mkdir -p src && cd src
git clone https://github.com/unitreerobotics/unitree_mujoco.git
git clone https://github.com/Teddy-Liao/walk-these-ways-go2.git
cd ..

# Copy flat scene to unitree_mujoco
cp resources/scene_flat.xml src/unitree_mujoco/unitree_robots/go2/

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
# Install dependencies
pip install -r requirements.txt
```
## Run (Linux)
```bash
source venv/bin/activate # if not already
python go2_wtw_demo.py
```
## Run (MacOS)
```bash
source venv/bin/activate # if not already
mjpython go2_wtw_demo.py
```
The robot will walk forward and turn left in a loop.

### Run test with artefacts

```
artefacts run basic_test
```
