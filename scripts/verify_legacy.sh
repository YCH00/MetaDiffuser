#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/legacy_env.sh"
cd "$PROJECT_ROOT"
.venv-legacy/bin/python scripts/verify_environments.py "$@"
