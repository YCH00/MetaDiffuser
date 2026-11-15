#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/legacy_env.sh"
cd "$PROJECT_ROOT"
exec .venv-legacy/bin/python "$@"
