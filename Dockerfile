FROM nvidia/cuda:12.8.1-base-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV MUJOCO_GL=egl
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglfw3 libgles2 libegl1 libglib2.0-0 \
    python3-pip python3-dev ffmpeg git curl cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws
COPY pyproject.toml uv.lock .python-version ./
# Use cpu-torch (for smaller image size)
RUN uv lock --index pytorch=https://download.pytorch.org/whl/cpu
RUN uv sync

COPY artefacts.yaml go2_wtw_demo.py utils.py ./
COPY tests/ tests/
COPY resources/ resources/

CMD uv run artefacts run $ARTEFACTS_JOB_NAME