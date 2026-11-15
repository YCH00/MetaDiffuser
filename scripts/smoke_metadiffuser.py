"""Run the main method end to end without baselines or MuJoCo dependencies."""
import argparse
from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", required=True, help="New empty output directory")
    args = p.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    root = Path(__file__).resolve().parents[1]
    config = str(root / "configs/smoke.json")
    base = [sys.executable, "-m", "metadiffuser.cli", "--device", args.device, "--threads", "1"]
    data, pre, diffusion = [str(output / part) for part in ("data", "pretrain", "diffusion")]
    commands = [
        ["collect", "--config", config, "--output", data, "--controller-smoke"],
        ["pretrain", "--config", config, "--dataset", data, "--output", pre, "--max-steps", "100"],
        ["train", "--config", config, "--dataset", data, "--pretrained", str(Path(pre) / "latest.pt"),
         "--output", diffusion, "--max-steps", "100"],
        ["evaluate", "--checkpoint", str(Path(diffusion) / "latest.pt"), "--dataset", data,
         "--output", str(output / "evaluation"), "--episodes", "1"],
    ]
    for command in commands:
        subprocess.run(base + command, cwd=root, check=True)
    print(f"Main-method smoke completed: {output / 'evaluation/summary.json'}")
