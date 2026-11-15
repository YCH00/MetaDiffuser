"""Restore pinned runtime/reference sources after a clean Git clone."""
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
lock = json.loads((root / "references/sources.lock.json").read_text())
for name, url, key in [
    ("CORRO", "https://github.com/PKU-RL/CORRO.git", "corro_commit"),
    ("diffuser", "https://github.com/jannerm/diffuser.git", "diffuser_commit"),
]:
    destination = root / "third_party" / name
    if not destination.exists():
        subprocess.run(["git", "clone", url, str(destination)], check=True)
        subprocess.run(["git", "-C", str(destination), "checkout", "--detach", lock[key]], check=True)
    actual = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
    if actual != lock[key]:
        raise RuntimeError(f"Existing {name} checkout differs from lock; preserve it and resolve explicitly")
    print(f"Verified {name}: {actual}")
