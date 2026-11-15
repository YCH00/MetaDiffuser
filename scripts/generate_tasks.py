"""Generate explicit task parameters without importing the training runtime."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metadiffuser.config import Config, ENV_SPECS
from metadiffuser.envs import make_tasks


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", nargs="+", choices=list(ENV_SPECS), required=True)
    p.add_argument("--output", default="datasets")
    args = p.parse_args()
    for name in args.env:
        cfg = Config.load(Path("configs") / f"{name}.json")
        tasks = make_tasks(cfg)
        destination = Path(args.output) / f"{name}_tasks.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if json.loads(destination.read_text(encoding="utf-8")) != tasks:
                raise ValueError(f"Existing manifest differs: {destination}")
        else:
            destination.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
        print(f"Verified {len(tasks)} explicit tasks: {destination}", flush=True)
