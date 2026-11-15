"""Read-only runtime probe; runnable with Windows Python or WSL Python."""
import importlib.util
import json
from pathlib import Path
import platform

print(json.dumps({"python": platform.python_version(), "platform": platform.platform(),
                  "modules": {m: bool(importlib.util.find_spec(m)) for m in ("torch", "numpy", "mujoco_py", "gym")},
                  "legacy_mujoco_dirs": [str(p) for p in (Path.home() / ".mujoco").glob("*")]}, indent=2))

