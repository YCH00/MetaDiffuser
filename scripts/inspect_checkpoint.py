"""Inspect a trusted local checkpoint in a second runtime before evaluation."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metadiffuser.runtime import load_checkpoint

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    args = p.parse_args()
    ck = load_checkpoint(args.checkpoint, "cpu")
    print(json.dumps({k: ck.get(k) for k in ("stage", "step", "state_dim", "action_dim", "training_budget_complete")}), flush=True)
