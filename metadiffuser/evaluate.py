import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from .config import Config
from .data import Normalizer, file_hash, load_split
from .diffusion import DualGuide, GaussianDiffusion
from .envs import make_env
from .models import ConditionalUNet, WorldModel
from .runtime import json_write, load_checkpoint, log_record, seed_all


PAPER_TABLE_1 = {"point_robot": -4.48, "ant_dir": 247.7, "cheetah_dir": 936.2,
                 "cheetah_vel": -45.9, "walker_param": 368.3, "hopper_param": 356.4}
CVAE_TABLE_1 = {"point_robot": -6.11, "ant_dir": 130.6, "cheetah_dir": 759.4,
                "cheetah_vel": -87.1, "walker_param": 187.9, "hopper_param": 157.3}


def evaluate(checkpoint, dataset, output, device, quality="expert", episodes=None, overrides=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    ck = load_checkpoint(checkpoint, device)
    if ck["stage"] not in ("diffusion", "cvae"):
        raise ValueError("Evaluation requires a planner checkpoint")
    cfg = Config(**ck["config"])
    allowed = {"dynamics_sign", "dynamics_weight", "reward_weight", "guide_scale", "cfg_scale", "temperature", "context_mode"}
    overrides = overrides or {}
    if ck["stage"] == "cvae" and set(overrides) - {"context_mode", "temperature"}:
        raise ValueError("CVAE does not support diffusion-guidance overrides")
    if set(overrides) - allowed:
        raise ValueError("Only sampling ablations may override a trained checkpoint; diffusion K requires retraining")
    context_mode = overrides.get("context_mode", "normal")
    if context_mode not in ("normal", "zero", "shuffled"):
        raise ValueError("Invalid context mode")
    for key, value in overrides.items():
        if key != "context_mode":
            setattr(cfg, key, value)
    cfg.validate()
    experiment_config = asdict(cfg)
    experiment_config.pop("seed")
    experiment_config["context_mode"] = context_mode
    experiment_config["method"] = ck["stage"]
    experiment_id = hashlib.sha256(json.dumps(experiment_config, sort_keys=True).encode()).hexdigest()
    seed_all(cfg.seed + 2_000_000)
    warm, manifest = load_split(dataset, "warm_start", quality)
    if manifest["tasks"] != ck["task_manifest"] or file_hash(Path(dataset) / "manifest.json") != ck["dataset_sha256"]:
        raise ValueError("Evaluation tasks/data differ from the recorded training dataset")
    sd, ad = ck["state_dim"], ck["action_dim"]
    normalizer = Normalizer.from_state_dict(ck["normalizer"])
    world = WorldModel(sd, ad, cfg).to(device)
    world.load_state_dict(ck["world"])
    world.freeze()
    if ck["stage"] == "diffusion":
        planner = GaussianDiffusion(ConditionalUNet(sd + ad, cfg), sd, cfg).to(device)
        planner.load_state_dict(ck["diffusion"])
        if ck["ema"] is not None:
            planner.model.load_state_dict(ck["ema"])
    else:
        from .cvae import CVAEPlanner
        planner = CVAEPlanner(sd, ad, cfg).to(device)
        planner.load_state_dict(ck["cvae"])
    planner.eval().requires_grad_(False)
    guide = DualGuide(world, cfg)
    tasks = [t for t in manifest["tasks"] if t["split"] == "test"]
    by_task = {t["task_id"]: [e for e in warm if e["metadata"]["task_id"] == t["task_id"] and len(e["observations"]) >= cfg.history] for t in tasks}
    records = []
    episodes = cfg.eval_episodes if episodes is None else episodes
    cfg.save(output / "config.json")
    json_write(output / "provenance.json", {"checkpoint_sha256": file_hash(checkpoint), "dataset_sha256": file_hash(Path(dataset) / "manifest.json"),
               "source": manifest["source"], "overrides": overrides, "quality": quality,
               "pretraining_budget_complete": ck["pretraining_budget_complete"],
               "diffusion_budget_complete": ck["training_budget_complete"]})
    for ti, task in enumerate(tasks):
        env = make_env(cfg, task["parameters"])
        for episode_id in range(episodes):
            eval_seed = cfg.seed * 1_000_000 + ti * 1000 + episode_id
            # Indices fixed by seed and task, independent of warm-start quality.
            rng = np.random.default_rng(eval_seed + 7_000_000)
            context_task = tasks[(ti + 1) % len(tasks)]["task_id"] if context_mode == "shuffled" else task["task_id"]
            pool = by_task[context_task]
            if not pool:
                raise ValueError(f"No sufficiently long warm-start episodes for {context_task}")
            contexts, indices = [], []
            for _ in range(cfg.context_windows):
                idx = int(rng.integers(len(pool)))
                start = int(rng.integers(len(pool[idx]["observations"]) - cfg.history + 1))
                contexts.append(normalizer.context(pool[idx], start, cfg.history))
                indices.append({"episode_index": idx, "start": start, "collection_seed": pool[idx]["metadata"]["collection_seed"]})
            with torch.no_grad():
                z = world.encoder(torch.as_tensor(np.stack(contexts), device=device)).mean(0, keepdim=True)
                if context_mode == "zero":
                    z.zero_()
            s, _ = env.reset(seed=eval_seed)
            total, length, clips, predicted, errors = 0., 0, 0, [], []
            generator = torch.Generator(device=device).manual_seed(eval_seed + 5_000_000)
            start_time = time.perf_counter()
            while True:
                initial = torch.as_tensor(normalizer.state(s), dtype=torch.float32, device=device)[None]
                trajectory = planner.sample(initial, z, ad, guide if ck["stage"] == "diffusion" else None, generator)
                with torch.no_grad():
                    reward, error = guide.components(trajectory, z)
                predicted.append(float(reward.item() / cfg.reward_scale))
                errors.append(float(error.item()))
                raw = normalizer.inverse_action(trajectory[0, 0, sd:].cpu().numpy())
                action = np.clip(raw, env.action_space.low, env.action_space.high)
                clips += int(np.count_nonzero(np.abs(raw - action) > 1e-6))
                s, r, term, trunc, _ = env.step(action)
                total += r
                length += 1
                if term or trunc or length >= cfg.episode_limit:
                    break
            record = {"env": cfg.env, "task_id": task["task_id"], "task_parameters": task["parameters"],
                      "method": ck["stage"],
                      "experiment_id": experiment_id, "dataset_sha256": ck["dataset_sha256"],
                      "checkpoint_step": ck["step"], "data_source": manifest["source"],
                      "model_seed": cfg.seed, "episode_seed": eval_seed, "warm_start_quality": quality,
                      "context_task": context_task, "warm_start_indices": indices,
                      "return": total, "length": length, "inference_seconds": time.perf_counter() - start_time,
                      "action_clip_fraction": clips / (length * ad), "predicted_plan_return_mean": float(np.mean(predicted)),
                      "dynamics_error_mean": float(np.mean(errors)), "dynamics_sign": cfg.dynamics_sign}
            records.append(record)
            log_record(output / "episodes.jsonl", record)
            print(f"eval {task['task_id']} episode={episode_id} return={total:.3f}", flush=True)
        env.close()
    task_means = {t["task_id"]: float(np.mean([r["return"] for r in records if r["task_id"] == t["task_id"]])) for t in tasks}
    mean = float(np.mean(list(task_means.values())))
    reference = (PAPER_TABLE_1 if ck["stage"] == "diffusion" else CVAE_TABLE_1)[cfg.env]
    summary = {"env": cfg.env, "method": ck["stage"], "model_seed": cfg.seed, "task_means": task_means, "mean_return": mean,
               "paper_table_1_literature_value": reference,
               "difference_from_literature": mean - reference,
               "comparable_to_paper": False, "source": manifest["source"],
               "note": "Unresolved paper details and data provenance require review before numerical equivalence claims."}
    json_write(output / "summary.json", summary)
    return summary


def aggregate(paths, output):
    all_records = []
    for path in paths:
        all_records.extend(json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())
    groups, seen = {}, set()
    for r in all_records:
        key = (r["env"], r["warm_start_quality"], r["dynamics_sign"], r["experiment_id"],
               r["dataset_sha256"], r["checkpoint_step"], r.get("method", "diffusion"))
        identity = (*key, r["model_seed"], r["task_id"], r["episode_seed"])
        if identity in seen:
            raise ValueError("Duplicate evaluation episodes; refusing double-counted results")
        seen.add(identity)
        groups.setdefault(key, {}).setdefault(r["model_seed"], {}).setdefault(r["task_id"], []).append(r["return"])
    result = []
    for key, seeds in groups.items():
        values = [float(np.mean([np.mean(x) for x in tasks.values()])) for tasks in seeds.values()]
        result.append({"env": key[0], "quality": key[1], "dynamics_sign": key[2], "seeds": sorted(seeds),
                       "experiment_id": key[3], "dataset_sha256": key[4], "checkpoint_step": key[5],
                       "method": key[6],
                       "mean": float(np.mean(values)), "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                       "seed_returns": values, "statistic": "episode mean, then equal-weight task mean, then seed mean/sample SD"})
    json_write(output, result)
    return result

