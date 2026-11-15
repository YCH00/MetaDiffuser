#!/usr/bin/env bash
# Extract runtime/build libraries locally; no apt system installation.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$PROJECT_ROOT/.runtime/debs" "$PROJECT_ROOT/.runtime/sysroot"
cd "$PROJECT_ROOT/.runtime/debs"
apt-get download libglu1-mesa libopengl0 libgl1 libgl-dev libglx-dev libglvnd-core-dev libosmesa6=24.0.5-1ubuntu1 libosmesa6-dev=24.0.5-1ubuntu1 patchelf
for package in ./*.deb; do
    dpkg-deb -x "$package" "$PROJECT_ROOT/.runtime/sysroot"
done
