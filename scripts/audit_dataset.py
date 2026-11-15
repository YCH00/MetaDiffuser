"""Report per-task realized policy quality and exact trajectory duplication."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metadiffuser.data import load_split, file_hash
from metadiffuser.runtime import json_write


def audit(root, output):
    groups = {}
    for split, quality in [("train", "expert"), ("validation", "expert"),
                           ("warm_start", "expert"), ("warm_start", "medium"), ("warm_start", "random")]:
        episodes, manifest = load_split(root, split, quality)
        for ep in episodes:
            meta = ep["metadata"]
            key = (meta["task_id"], split, quality)
            groups.setdefault(key, []).append(ep)
    rows = []
    for (task, split, quality), episodes in groups.items():
        returns = [float(e["rewards"].sum()) for e in episodes]
        signatures = {hashlib.sha256(e["observations"].tobytes() + e["actions"].tobytes()).hexdigest() for e in episodes}
        rows.append({"task_id": task, "split": split, "quality": quality,
                     "episodes": len(episodes), "transitions": sum(len(e["rewards"]) for e in episodes),
                     "return_mean": float(np.mean(returns)), "return_std": float(np.std(returns)),
                     "return_min": min(returns), "return_max": max(returns),
                     "unique_trajectories": len(signatures), "duplicate_fraction": 1 - len(signatures) / len(episodes),
                     "policy_sampling": sorted({e["metadata"].get("policy_sampling", "not_recorded") for e in episodes})})
    comparisons = []
    for task in manifest["tasks"]:
        if task["split"] != "test":
            continue
        values = {r["quality"]: r["return_mean"] for r in rows if r["task_id"] == task["task_id"]}
        comparisons.append({"task_id": task["task_id"], "expert_above_medium_above_random":
                            values["expert"] > values["medium"] > values["random"], **values})
    summary = {"manifest_sha256": file_hash(Path(root) / "manifest.json"), "source": manifest["source"],
               "total_episodes": sum(r["episodes"] for r in rows),
               "total_transitions": sum(r["transitions"] for r in rows),
               "rows": rows, "warm_start_quality_order": comparisons,
               "note": "Quality ordering is measured, not inferred from checkpoint age. No claim of matching author data."}
    json_write(output, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    audit(args.dataset, args.output)
