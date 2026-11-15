"""Run independent per-task CORRO SAC collectors in bounded subprocesses."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--tasks", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--iterations", type=int, required=True)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--exclude", nargs="*", default=[])
    args = p.parse_args()
    tasks = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    pending = []
    for t in tasks:
        name = t["task_id"]
        if name in args.exclude:
            continue
        directory = root / name
        if directory.exists():
            metrics = directory / "evaluation.jsonl"
            records = [json.loads(s) for s in metrics.read_text().splitlines()] if metrics.exists() else []
            if records and records[-1]["iteration"] >= args.iterations and (directory / "best.pt").exists():
                print(f"Already complete: {name}", flush=True)
                continue
            raise RuntimeError(f"Incomplete existing collector run: {directory}; preserve it and choose a new output")
        pending.append(name)

    def execute(name):
        command = [sys.executable, "-m", "metadiffuser.cli", "--device", "cpu", "--threads", "1",
                   "expert", "--collector", "corro", "--config", args.config, "--tasks", args.tasks,
                   "--output", args.output, "--task-id", name, "--iterations", str(args.iterations)]
        with (root / f"{name}.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"Finished: {name}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for _ in executor.map(execute, pending):
            pass


if __name__ == "__main__":
    main()

