"""Export the exact selected episodes to baseline exchange formats.

This does not run the baseline or change its environment definitions. Consumers
must use tasks.json and distinguish terminations from timeouts.
"""
import argparse
import json
from pathlib import Path
import pickle
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metadiffuser.data import file_hash, load_split
from metadiffuser.runtime import json_write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quality", choices=("expert", "medium", "random"), default="expert")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    index = []
    for split in ("train", "validation", "warm_start"):
        episodes, manifest = load_split(args.dataset, split, args.quality if split == "warm_start" else "expert")
        for task in manifest["tasks"]:
            selected = [e for e in episodes if e["metadata"]["task_id"] == task["task_id"]]
            if not selected:
                continue
            folder = output / split / task["task_id"]
            folder.mkdir(parents=True)
            for source, target in (("observations", "obs"), ("actions", "actions"), ("rewards", "rewards"),
                                   ("next_observations", "next_obs"), ("terminated", "terminated"), ("truncated", "truncated")):
                np.save(folder / f"{target}.npy", np.concatenate([e[source] for e in selected]))
            # CORRO's terminals encode episode boundaries; retain separate flags too.
            np.save(folder / "terminals.npy", np.concatenate([e["terminated"] | e["truncated"] for e in selected])[:, None])
            trajectories = [{"observations": e["observations"], "actions": e["actions"],
                             "rewards": e["rewards"].reshape(-1), "next_observations": e["next_observations"],
                             "terminals": e["terminated"], "timeouts": e["truncated"]} for e in selected]
            # Prompt-DT's public loader expects a trusted pickle list of trajectories.
            with (folder / "prompt_dt.pkl").open("wb") as stream:
                pickle.dump(trajectories, stream, protocol=4)
            for i, e in enumerate(selected):
                focal = np.empty((len(e["actions"]), 4), dtype=object)
                for j, field in enumerate(("observations", "actions", "rewards", "next_observations")):
                    focal[:, j] = list(e[field])
                np.save(folder / f"focal_episode_{i:05d}.npy", focal, allow_pickle=True)
            json_write(folder / "episode_metadata.json", [e["metadata"] for e in selected])
            index.append({"split": split, "task_id": task["task_id"], "directory": str(folder), "episodes": len(selected)})
    json_write(output / "tasks.json", manifest["tasks"])
    json_write(output / "export_manifest.json", {"source_manifest_sha256": file_hash(Path(args.dataset) / "manifest.json"),
               "files": {str(p.relative_to(output)): file_hash(p) for p in output.rglob("*") if p.is_file()}, "index": index,
               "warning": "Raw coordinates; baseline-specific train-only normalization and task/evaluation adapters still required."})


if __name__ == "__main__":
    main()

