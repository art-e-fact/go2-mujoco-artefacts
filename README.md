# Go2 MuJoCo Walk Demo

Unitree Go2 walking demo using MuJoCo and the Walk-These-Ways pretrained policy.

## Docker (Recommended)

```bash
# Build
docker build -t go2-mujoco-demo .

# Run (Linux with X11)
xhost +local:docker
docker run --rm -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix go2-mujoco-demo
# Run (Mac with XQuartz - enable "Allow connections from network clients" in XQuartz preferences)
docker run --rm -it -e DISPLAY=host.docker.internal:0 go2-mujoco-demo
# Run forever (no auto-exit)
docker run --rm -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix go2-mujoco-demo python3 go2_wtw_demo.py --cycles 0
```

## Manual Setup

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

# Install CPU-only PyTorch (recommended for inference)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Run

```bash
source venv/bin/activate
python go2_wtw_demo.py
```

The robot will walk forward and turn left in a loop.

## Project Structure

```
go2-mujuco/
├── go2_wtw_demo.py      # Main demo script
├── requirements.txt     # Python dependencies
├── README.md
├── resources/
│   └── scene_flat.xml   # Flat ground scene (references unitree_mujoco)
└── src/
    ├── unitree_mujoco/      # Robot models (clone from GitHub)
    └── walk-these-ways-go2/ # Pretrained policy (clone from GitHub)
```
