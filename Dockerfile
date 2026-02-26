# Go2 MuJoCo Walk-These-Ways Demo
# Build: docker build -t go2-mujoco-demo .
# 
# Linux:
#   xhost +local:docker && docker run --rm -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix go2-mujoco-demo
#
# Mac (with XQuartz running, allow network clients):
#   docker run --rm -it -e DISPLAY=host.docker.internal:0 go2-mujoco-demo

FROM ubuntu:22.04

# Avoid interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive
# Force software rendering (works in Docker without GPU passthrough)
ENV LIBGL_ALWAYS_SOFTWARE=1

# Install Python and system dependencies for MuJoCo and OpenGL/X11
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3-pip \
    python3-venv \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libgl1-mesa-glx \
    libglfw3 \
    libglew2.2 \
    libosmesa6 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    libxxf86vm1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
# Note: On ARM (Apple Silicon), torch installs from regular PyPI
COPY requirements.txt .
RUN pip3 install -r requirements.txt \
    && pip3 install torch

# Clone required repos
RUN mkdir -p src && cd src \
    && git clone --depth 1 https://github.com/unitreerobotics/unitree_mujoco.git \
    && git clone --depth 1 https://github.com/Teddy-Liao/walk-these-ways-go2.git

# Copy application files
COPY go2_wtw_demo.py .
COPY resources/scene_flat.xml src/unitree_mujoco/unitree_robots/go2/

# Default: run 4 cycles then exit (or run forever with --cycles 0)
CMD ["python3", "go2_wtw_demo.py", "--cycles", "4"]
