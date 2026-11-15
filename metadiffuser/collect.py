import json
from pathlib import Path
import numpy as np
import torch
from .data import file_hash, save_episode
from .envs import make_env, make_tasks
from .runtime import json_write, load_checkpoint
from .sac import Actor


def load_actor(path, device, task=None):
    ck = load_checkpoint(path, device)
    if task is not None and ck["task"]["parameters"] != task["parameters"]:
        raise ValueError("Policy checkpoint belongs to another task")
    if ck.get("source") == "pinned_corro_sac":
        from .corro_collector import CorroActor
        actor = CorroActor(ck["state_dim"], ck["action_dim"], ck["width"], ck["low"], ck["high"], device)
    else:
        actor = Actor(ck["state_dim"], ck["action_dim"], ck["width"], ck["low"], ck["high"]).to(device)
    actor.load_state_dict(ck["actor"])
    return actor.eval()


def rollout(env, policy, seed):
    s, _ = env.reset(seed=seed)
    rows = []
    while True:
        a = policy(s)
        ns, r, term, trunc, _ = env.step(a)
        rows.append((s, a, [r], ns, term, trunc))
        s = ns
        if term or trunc:
            break
    keys = ("observations", "actions", "rewards", "next_observations", "terminated", "truncated")
    return {k: np.asarray(v, dtype=bool if k in ("terminated", "truncated") else np.float32)
            for k, v in zip(keys, zip(*rows))}


def collect_dataset(cfg, output, tasks=None, policy_root=None, controller_smoke=False, device="cpu"):
    """SAC expert/return-selected medium and random warm-start data.

    Controller data is explicitly smoke-only in every metadata record.
    """
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    if controller_smoke and cfg.env != "point_robot":
        raise ValueError("The smoke controller is only available for Point-Robot")
    if not controller_smoke and policy_root is None:
        raise ValueError("Formal collection requires per-task SAC checkpoints")
    if cfg.episodes_per_task < 2:
        raise ValueError("At least two episodes/task required for held-out validation")
    tasks = make_tasks(cfg) if tasks is None else tasks
    output.mkdir(parents=True, exist_ok=True)
    cfg.save(output / "config.json")
    manifest = {"schema_version": 1, "env": cfg.env, "tasks": tasks, "episodes": [],
                "source": "controller_smoke_only" if controller_smoke else "checkpointed_sac_collection",
                "task_seed": cfg.task_seed}
    for ti, task in enumerate(tasks):
        env = make_env(cfg, task["parameters"])
        manifest["action_low"] = env.action_space.low.tolist()
        manifest["action_high"] = env.action_space.high.tolist()
        qualities = ["expert"] if task["split"] == "train" else ["expert", "medium", "random"]
        expert_path = None if controller_smoke else Path(policy_root) / task["task_id"] / "best.pt"
        expert_actor = None if controller_smoke else load_actor(expert_path, device, task)
        collector_implementation = "controller_smoke_only" if controller_smoke else load_checkpoint(expert_path, "cpu")["source"]
        seed_base = cfg.seed * 10_000_000 + ti * 100_000
        random_returns = []
        for i in range(5):
            rng = np.random.default_rng(seed_base + 90_000 + i)
            random_returns.append(float(rollout(env, lambda s: rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32),
                                                seed_base + 90_000 + i)["rewards"].sum()))
        medium_path = None
        if not controller_smoke and task["split"] == "test":
            policies = json.loads((Path(policy_root) / task["task_id"] / "policies.json").read_text())
            expert_return = max(p["mean_return"] for p in policies)
            target_return = (expert_return + float(np.mean(random_returns))) / 2
            medium_record = min(policies, key=lambda p: abs(p["mean_return"] - target_return))
            medium_path = Path(policy_root) / task["task_id"] / medium_record["path"]
        for qi, quality in enumerate(qualities):
            actor = load_actor(medium_path, device, task) if medium_path and quality == "medium" else expert_actor
            source_path = medium_path if quality == "medium" else expert_path
            for i in range(cfg.episodes_per_task):
                seed = seed_base + qi * 10_000 + i
                rng = np.random.default_rng(seed)
                torch.manual_seed(seed)
                def policy(s):
                    if quality == "random":
                        return rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
                    if controller_smoke:
                        action = np.clip(env.goal - s, env.action_space.low, env.action_space.high)
                        if quality == "medium":
                            action = 0.5 * action + 0.5 * rng.uniform(env.action_space.low, env.action_space.high)
                        return action.astype(np.float32)
                    with torch.no_grad():
                        return actor(torch.as_tensor(s, dtype=torch.float32, device=device)[None], cfg.collection_deterministic)[0][0].cpu().numpy()
                ep = rollout(env, policy, seed)
                validation_count = max(1, int(cfg.episodes_per_task * cfg.validation_fraction))
                split = ("warm_start" if task["split"] == "test" else
                         "validation" if i < validation_count else "train")
                meta = {"task_id": task["task_id"], "task_parameters": task["parameters"], "split": split,
                        "quality": quality, "collection_seed": seed, "return": float(ep["rewards"].sum()),
                        "policy_checkpoint": str(source_path) if source_path and quality != "random" else None,
                        "policy_sha256": file_hash(source_path) if source_path and quality != "random" else None,
                        "source": manifest["source"],
                        "exploration_noise": quality == "random" or (not controller_smoke and not cfg.collection_deterministic) or (controller_smoke and quality == "medium"),
                        "policy_sampling": "uniform_random" if quality == "random" else "controller" if controller_smoke else "deterministic_mean" if cfg.collection_deterministic else "sac_distribution",
                        "collector_implementation": collector_implementation,
                        "random_policy_reference": float(np.mean(random_returns))}
                relative = Path(split) / task["task_id"] / quality / f"episode_{i:05d}.npz"
                save_episode(output / relative, ep, meta)
                manifest["episodes"].append({"path": relative.as_posix(), "sha256": file_hash(output / relative),
                                              "task_id": task["task_id"], "split": split, "quality": quality})
        env.close()
        # Interrupted collection remains inspectable; completeness checked by doctor.
        json_write(output / "manifest.json", manifest)
        print(f"collected {task['task_id']}", flush=True)
    manifest["collection_complete"] = True
    json_write(output / "manifest.json", manifest)
    return output / "manifest.json"
