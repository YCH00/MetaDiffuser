"""Non-pickle episode format, task-balanced windows, train-only statistics."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


FIELDS = ("observations", "actions", "rewards", "next_observations", "terminated", "truncated")


def file_hash(path):
    sha = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def validate_episode(ep):
    n = len(ep["observations"])
    if n < 1 or any(len(ep[k]) != n for k in FIELDS):
        raise ValueError("Inconsistent episode lengths")
    if ep["observations"].shape != ep["next_observations"].shape:
        raise ValueError("Inconsistent state dimensions")
    if ep["observations"].ndim != 2 or ep["actions"].ndim != 2 or ep["rewards"].shape != (n, 1):
        raise ValueError("Expected [L,Ds], [L,Da], [L,1]")
    if ep["terminated"].shape != (n,) or ep["truncated"].shape != (n,):
        raise ValueError("End flags must have shape [L]")
    if any(not np.isfinite(ep[k]).all() for k in FIELDS):
        raise ValueError("Nonfinite episode values")
    done = ep["terminated"].astype(bool) | ep["truncated"].astype(bool)
    if done[:-1].any() or not done[-1]:
        raise ValueError("Expected one complete episode without internal boundaries")
    if not np.allclose(ep["next_observations"][:-1], ep["observations"][1:], atol=1e-5):
        raise ValueError("Discontinuous episode")


def save_episode(path, ep, metadata):
    validate_episode(ep)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    np.savez_compressed(path, **{k: ep[k] for k in FIELDS}, metadata=json.dumps(metadata))


def load_episode(path):
    with np.load(path, allow_pickle=False) as f:
        ep = {k: f[k] for k in FIELDS}
        ep["metadata"] = json.loads(str(f["metadata"]))
    validate_episode(ep)
    return ep


def load_split(root, split, quality="expert"):
    manifest = json.loads((Path(root) / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("collection_complete", False):
        raise ValueError("Dataset collection is incomplete")
    records = [r for r in manifest["episodes"] if r["split"] == split and r["quality"] == quality]
    if not records:
        raise ValueError(f"No {quality} episodes for split {split}")
    task_map = {t["task_id"]: t for t in manifest["tasks"]}
    if len(task_map) != len(manifest["tasks"]):
        raise ValueError("Duplicate task identifiers")
    if len({r["path"] for r in records}) != len(records):
        raise ValueError("Duplicate episode paths")
    expected_ids = {t["task_id"] for t in manifest["tasks"] if t["split"] == ("test" if split == "warm_start" else "train")}
    if {r["task_id"] for r in records} != expected_ids:
        raise ValueError("Missing tasks in dataset split")
    episodes = []
    for r in records:
        p = (Path(root) / r["path"]).resolve()
        if not p.is_relative_to(Path(root).resolve()) or file_hash(p) != r["sha256"]:
            raise ValueError("Invalid episode path or checksum")
        ep = load_episode(p)
        meta = ep["metadata"]
        if meta["task_id"] != r["task_id"] or meta["split"] != split:
            raise ValueError("Episode/manifest split mismatch")
        task = task_map[meta["task_id"]]
        expected_task_split = "test" if split == "warm_start" else "train"
        if task["split"] != expected_task_split or meta["task_parameters"] != task["parameters"]:
            raise ValueError("Task split or parameter mismatch")
        episodes.append(ep)
    return episodes, manifest


def check_dataset_config(manifest, cfg):
    if manifest["env"] != cfg.env:
        raise ValueError("Configuration and dataset environment differ")
    for split, expected in (("train", cfg.train_tasks), ("test", cfg.test_tasks)):
        if sum(t["split"] == split for t in manifest["tasks"]) != expected:
            raise ValueError(f"Unexpected {split} task count")
    if any(t["task_seed"] != cfg.task_seed for t in manifest["tasks"]):
        raise ValueError("Task sampling seed differs from configuration")


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray
    action_low: np.ndarray
    action_high: np.ndarray
    reward_scale: float

    @classmethod
    def fit(cls, episodes, action_low, action_high, reward_scale=1.):
        if any(e["metadata"]["split"] != "train" for e in episodes):
            raise ValueError("Normalization must use training episodes only")
        states = np.concatenate([np.concatenate([e["observations"], e["next_observations"]]) for e in episodes])
        low, high = np.asarray(action_low, np.float32), np.asarray(action_high, np.float32)
        if np.any(high <= low):
            raise ValueError("Invalid action bounds")
        return cls(states.mean(0), states.std(0).clip(1e-6), low, high, reward_scale)

    def state(self, s):
        return (s - self.mean) / self.std

    def action(self, a):
        return 2 * (a - self.action_low) / (self.action_high - self.action_low) - 1

    def inverse_action(self, a):
        return self.action_low + (a + 1) * (self.action_high - self.action_low) / 2

    def context(self, ep, start, length):
        sl = slice(start, start + length)
        return np.concatenate([self.state(ep["observations"][sl]), self.action(ep["actions"][sl]),
                               ep["rewards"][sl] * self.reward_scale,
                               self.state(ep["next_observations"][sl])], -1).astype(np.float32)

    def state_dict(self):
        return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in vars(self).items()}

    @classmethod
    def from_state_dict(cls, d):
        return cls(**{k: np.asarray(v, np.float32) if k != "reward_scale" else v for k, v in d.items()})


class WindowSampler:
    def __init__(self, episodes, normalizer, cfg, seed=0, include_trajectory=True):
        self.normalizer, self.cfg = normalizer, cfg
        self.include_trajectory = include_trajectory
        self.rng = np.random.default_rng(seed)
        self.tasks = {}
        required = max(cfg.history, cfg.horizon) if include_trajectory else cfg.history
        for ep in episodes:
            if len(ep["observations"]) >= required:
                self.tasks.setdefault(ep["metadata"]["task_id"], []).append(ep)
        if not self.tasks:
            raise ValueError("No complete windows")
        expected = {e["metadata"]["task_id"] for e in episodes}
        if set(self.tasks) != expected:
            raise ValueError("A task has no sufficiently long episodes; refusing to silently drop it")
        self.keys = sorted(self.tasks)
        self.window_count = sum(len(e["observations"]) - required + 1 for es in self.tasks.values() for e in es)

    def batch(self, count, device, independent_context=False):
        n, cfg = self.normalizer, self.cfg
        batch = {k: [] for k in ("context", "states", "actions", "rewards", "next_states")}
        if self.include_trajectory:
            batch["trajectory"] = []
        for _ in range(count):
            task = self.tasks[self.keys[self.rng.integers(len(self.keys))]]
            ep = task[self.rng.integers(len(task))]
            ctx_ep = task[self.rng.integers(len(task))] if independent_context else ep
            cs = int(self.rng.integers(len(ctx_ep["observations"]) - cfg.history + 1))
            contexts = [n.context(ctx_ep, cs, cfg.history)]
            for _ in range(cfg.context_windows - 1):
                extra_start = int(self.rng.integers(len(ctx_ep["observations"]) - cfg.history + 1))
                contexts.append(n.context(ctx_ep, extra_start, cfg.history))
            batch["context"].append(contexts[0] if cfg.context_windows == 1 else np.stack(contexts))
            if self.include_trajectory:
                start = int(self.rng.integers(len(ep["observations"]) - cfg.horizon + 1))
                sl = slice(start, start + cfg.horizon)
                batch["trajectory"].append(np.concatenate([n.state(ep["observations"][sl]), n.action(ep["actions"][sl])], -1))
            # Eq. (1): supervise all transitions in the same historical window.
            target_ep = ep if independent_context else ctx_ep
            ts = int(self.rng.integers(len(ep["observations"]) - cfg.history + 1)) if independent_context else cs
            target = slice(ts, ts + cfg.history)
            batch["states"].append(n.state(target_ep["observations"][target]))
            batch["actions"].append(n.action(target_ep["actions"][target]))
            batch["rewards"].append(target_ep["rewards"][target] * n.reward_scale)
            batch["next_states"].append(n.state(target_ep["next_observations"][target]))
        return {k: torch.as_tensor(np.stack(v), dtype=torch.float32, device=device) for k, v in batch.items()}
