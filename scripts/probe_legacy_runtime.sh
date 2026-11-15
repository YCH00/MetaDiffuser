#!/usr/bin/env bash
set -eu
source "$(dirname "$0")/legacy_env.sh"
ldd "$MUJOCO_PY_MJPRO_PATH/bin/libmujoco131.so"
ldd "$MUJOCO_PY_MUJOCO_PATH/bin/libmujoco200.so"
