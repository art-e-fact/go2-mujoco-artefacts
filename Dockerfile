FROM python:3.12-slim

# System deps for MuJoCo rendering (EGL/offscreen) and video encoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglfw3 \
    libgles2 \
    libegl1 \
    libglib2.0-0 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /ws

# MuJoCo needs a display context for passive viewer; use EGL for offscreen
ENV MUJOCO_GL=egl

CMD artefacts run $ARTEFACTS_JOB_NAME
