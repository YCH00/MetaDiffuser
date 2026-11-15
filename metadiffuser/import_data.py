"""Strict import of CORRO flat buffers using explicit episode boundaries.

CORRO ring-buffer dumps alone do NOT encode the wrap pointer or distinguish
timeouts. This converter intentionally requires that information rather than
inventing continuous episodes or parsing rounded goal values from directory names.
"""
import json
from pathlib import Path
import numpy as np
from .data import file_hash, load_split, save_episode
from .runtime import json_write


def import_corro(index_path, output):
    index_path, output = Path(index_path), Path(output)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "env": index["env"], "tasks": index["tasks"],
                "action_low": index["action_low"], "action_high": index["action_high"],
                "source": "imported_corro_explicit_episode_index", "episodes": [],
                "source_index_sha256": file_hash(index_path)}
    task_map = {t["task_id"]: t for t in index["tasks"]}
    used = set()
    for source in index["sources"]:
        path = (index_path.parent / source["directory"]).resolve()
        arrays = {name: np.load(path / filename, allow_pickle=False) for name, filename in
                  (("observations", "obs.npy"), ("actions", "actions.npy"), ("rewards", "rewards.npy"),
                   ("next_observations", "next_obs.npy"))}
        n = len(arrays["observations"])
        if any(len(v) != n for v in arrays.values()):
            raise ValueError("Source buffer length mismatch")
        for episode_id, entry in enumerate(source["episodes"]):
            indices = np.asarray(entry["indices"], dtype=np.int64)
            if indices.ndim != 1 or not len(indices) or np.any(indices < 0) or np.any(indices >= n):
                raise ValueError("Invalid explicit source indices")
            identities = {(str(path), int(i)) for i in indices}
            if len(identities) != len(indices) or used.intersection(identities):
                raise ValueError("Source transition duplicated across episodes/splits")
            used.update(identities)
            ep = {k: v[indices].astype(np.float32) for k, v in arrays.items()}
            ep["rewards"] = ep["rewards"].reshape(-1, 1)
            ep["terminated"] = np.zeros(len(indices), bool)
            ep["truncated"] = np.zeros(len(indices), bool)
            if entry["end_reason"] not in ("terminated", "truncated"):
                raise ValueError("Explicit end_reason is required")
            ep[entry["end_reason"]][-1] = True
            task = task_map[source["task_id"]]
            expected = "test" if entry["split"] == "warm_start" else "train"
            if task["split"] != expected:
                raise ValueError("Source task/split mismatch")
            meta = {"task_id": task["task_id"], "task_parameters": task["parameters"],
                    "split": entry["split"], "quality": entry["quality"],
                    "collection_seed": entry["collection_seed"], "source": manifest["source"],
                    "policy_checkpoint": entry["policy_checkpoint"], "return": float(ep["rewards"].sum()),
                    "source_directory": str(path), "source_indices": indices.tolist()}
            relative = Path(entry["split"]) / task["task_id"] / entry["quality"] / f"episode_{len(manifest['episodes']):07d}.npz"
            save_episode(output / relative, ep, meta)
            manifest["episodes"].append({"path": relative.as_posix(), "sha256": file_hash(output / relative),
                                          "task_id": task["task_id"], "split": entry["split"], "quality": entry["quality"]})
    manifest["collection_complete"] = True
    json_write(output / "manifest.json", manifest)
    for split in ("train", "validation", "warm_start"):
        load_split(output, split)
    return output / "manifest.json"

