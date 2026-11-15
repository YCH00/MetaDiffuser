#!/usr/bin/env bash
# Source this file before running the pinned CORRO legacy environments.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MUJOCO_PY_MUJOCO_PATH="$PROJECT_ROOT/.runtime/legacy/mujoco200_linux"
export MUJOCO_PY_MJPRO_PATH="$PROJECT_ROOT/.runtime/legacy/mjpro131"
export MUJOCO_PY_MJKEY_PATH="$PROJECT_ROOT/.runtime/legacy/mjkey.txt"
export LD_LIBRARY_PATH="$PROJECT_ROOT/.runtime/sysroot/usr/lib/x86_64-linux-gnu:$MUJOCO_PY_MUJOCO_PATH/bin:$MUJOCO_PY_MJPRO_PATH/bin:${LD_LIBRARY_PATH:-}"
export CPATH="$PROJECT_ROOT/.runtime/sysroot/usr/include:${CPATH:-}"
export LIBRARY_PATH="$PROJECT_ROOT/.runtime/sysroot/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
export PATH="$PROJECT_ROOT/.runtime/sysroot/usr/bin:$PATH"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

