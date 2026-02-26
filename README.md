# Go2 MuJoCo Walk Demo

Unitree Go2 walking demo using MuJoCo and the Walk-These-Ways pretrained policy.

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
