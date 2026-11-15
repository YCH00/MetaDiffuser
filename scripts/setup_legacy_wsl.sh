#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
source scripts/legacy_env.sh
export UV_CACHE_DIR="$PROJECT_ROOT/.runtime/uv-cache"
export UV_PYTHON_INSTALL_DIR="$PROJECT_ROOT/.runtime/python"
UV_BIN="$PROJECT_ROOT/.runtime/legacy/uv-x86_64-unknown-linux-gnu/uv"
chmod +x "$UV_BIN"
"$UV_BIN" python install 3.10
if [ ! -x .venv-legacy/bin/python ]; then
    "$UV_BIN" venv --python 3.10 .venv-legacy
fi
"$UV_BIN" pip install --python .venv-legacy/bin/python 'pip<24.1' 'setuptools<70' 'wheel<0.40' 'numpy==1.23.5' 'Cython==0.29.37'
"$UV_BIN" pip install --python .venv-legacy/bin/python 'torch==2.6.0+cpu' --index-url https://download.pytorch.org/whl/cpu
"$UV_BIN" pip install --python .venv-legacy/bin/python 'gym==0.17.3' 'scipy==1.10.1' 'matplotlib==3.7.5' 'six==1.17.0' 'glfw==2.6.5' 'imageio==2.31.6' 'fasteners==0.19' 'cffi==1.16.0' 'pycparser==2.22' 'cloudpickle==1.6.0' 'requests==2.32.3' 'PyOpenGL==3.1.7'
"$UV_BIN" pip install --python .venv-legacy/bin/python --no-build-isolation 'mujoco-py==2.0.2.13'
.venv-legacy/bin/python -m pip freeze > requirements-legacy.lock.txt
