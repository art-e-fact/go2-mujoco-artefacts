#!/usr/bin/env bash
set -euo pipefail

BASE="https://raw.githubusercontent.com/Teddy-Liao/walk-these-ways-go2/main"
CKPT="runs/gait-conditioned-agility/pretrain-go2/train/142238.667503"
DEST="$(dirname "$0")/../src/unitree_mujoco/simulate_python/wtw"

mkdir -p "$DEST"

echo "Downloading WTW checkpoints..."
curl -fL "$BASE/$CKPT/checkpoints/body_latest.jit"              -o "$DEST/body_latest.jit"
curl -fL "$BASE/$CKPT/checkpoints/adaptation_module_latest.jit" -o "$DEST/adaptation_module_latest.jit"
curl -fL "$BASE/$CKPT/parameters_cpu.pkl"                       -o "$DEST/parameters_cpu.pkl"
echo "Done. Files written to $DEST/"
