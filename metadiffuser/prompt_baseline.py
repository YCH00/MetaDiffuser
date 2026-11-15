"""Pinned Prompt-DT model with explicit dataset and task-isolation adapters.

The public model is imported unchanged. Training settings follow pdt_main.py;
prompt length, train-only normalization and return target are disclosed choices
for the MetaDiffuser task manifests, not recovered author hyperparameters.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from .config import Config
from .data import Normalizer, check_dataset_config, file_hash, load_split
from .envs import make_env
from .runtime import atomic_save, json_write, load_checkpoint, log_record, restore_rng, rng_state, seed_all
from .train import prepare_run


@dataclass
class PromptSettings:
    context_length: int = 20
    hidden_size: int = 128
    layers: int = 3
    heads: int = 1
    dropout: float = .1
    batch: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 10000
    train_steps: int = 50000
    return_scale: float = 1.
    target_return: float | None = None


def make_model(sd, ad, cfg, settings):
    if cfg.context_windows != 1:
        raise ValueError("Prompt-DT adapter currently requires exactly one warm-start window")
    if cfg.history not in (4, 10, 20) or settings.context_length in (6, 12, 22):
        raise ValueError("Prompt layout must avoid the original model's length-equality branch")
    if settings.return_scale <= 0 or settings.warmup_steps < 1:
        raise ValueError("Invalid Prompt-DT return scale or warmup budget")
    root = Path(__file__).resolve().parents[1] / "third_party" / "prompt-dt"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from prompt_dt.prompt_decision_transformer import PromptDecisionTransformer
    return PromptDecisionTransformer(state_dim=sd, act_dim=ad, hidden_size=settings.hidden_size,
        max_length=settings.context_length, max_ep_len=cfg.episode_limit,
        n_layer=settings.layers, n_head=settings.heads, n_inner=4 * settings.hidden_size,
        activation_function="relu", n_positions=1024,
        resid_pdrop=settings.dropout, attn_pdrop=settings.dropout)


def sequence(ep, start, length, normalizer, scale, prompt=False):
    """Left padding and undiscounted RTG; prompt RTG sees its h rewards only."""
    stop = min(start + length, len(ep["observations"]))
    size = stop - start
    rewards = ep["rewards"][start:stop].astype(np.float32)
    future = rewards if prompt else ep["rewards"][start:]
    rtg = np.cumsum(future[::-1], axis=0)[::-1][:size].copy() / scale
    arrays = [normalizer.state(ep["observations"][start:stop]),
              normalizer.action(ep["actions"][start:stop]), rewards,
              (ep["terminated"][start:stop] | ep["truncated"][start:stop]).astype(np.float32),
              rtg, np.arange(start, stop, dtype=np.int64), np.ones(size, np.float32)]
    # Original model guesses prompt layout from equality to K and RTG length
    # modulo 10. Two masked tokens avoid both branches for paper h=4/10/20.
    pad = length - size + (2 if prompt else 0)
    return [np.pad(a, [(pad, 0)] + [(0, 0)] * (a.ndim - 1)) for a in arrays]


class PromptSampler:
    def __init__(self, episodes, normalizer, cfg, settings):
        self.groups = {}
        for e in episodes:
            if len(e["observations"]) >= cfg.history:
                self.groups.setdefault(e["metadata"]["task_id"], []).append(e)
        if len(self.groups) != cfg.train_tasks:
            raise ValueError("Every training task needs a prompt-length episode")
        self.ids = sorted(self.groups)
        self.rng = np.random.default_rng(cfg.seed)
        self.n, self.cfg, self.settings = normalizer, cfg, settings

    def batch(self, device):
        prompts, batches = [], []
        for _ in range(self.settings.batch):
            group = self.groups[self.ids[int(self.rng.integers(len(self.ids)))]]
            ep, context = [group[int(self.rng.integers(len(group)))] for _ in range(2)]
            start = int(self.rng.integers(len(ep["observations"])))
            cstart = int(self.rng.integers(len(context["observations"]) - self.cfg.history + 1))
            batches.append(sequence(ep, start, self.settings.context_length, self.n, self.settings.return_scale))
            prompts.append(sequence(context, cstart, self.cfg.history, self.n, self.settings.return_scale, True))
        def stack(rows):
            return tuple(torch.as_tensor(np.stack(values), device=device,
                         dtype=torch.long if i == 5 else torch.float32) for i, values in enumerate(zip(*rows)))
        return stack(prompts), stack(batches)


def train(cfg, settings, dataset, output, device, resume=None, max_steps=None):
    seed_all(cfg.seed)
    episodes, manifest = load_split(dataset, "train")
    check_dataset_config(manifest, cfg)
    n = Normalizer.fit(episodes, manifest["action_low"], manifest["action_high"])
    sd, ad = episodes[0]["observations"].shape[1], episodes[0]["actions"].shape[1]
    model = make_model(sd, ad, cfg, settings).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min((s + 1) / settings.warmup_steps, 1))
    sampler = PromptSampler(episodes, n, cfg, settings)
    fingerprint = file_hash(Path(dataset) / "manifest.json")
    target = float(np.quantile([e["rewards"].sum() for e in episodes], .95)) if settings.target_return is None else settings.target_return
    start = 0
    if resume:
        ck = load_checkpoint(resume, device)
        if ck["stage"] != "prompt_dt" or asdict(Config(**ck["config"])) != asdict(cfg) or ck["settings"] != asdict(settings) or ck["dataset_sha256"] != fingerprint:
            raise ValueError("Prompt-DT resume configuration or dataset mismatch")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        sampler.rng.bit_generator.state = ck["sampler_rng"]
        restore_rng(ck["rng"])
        start = ck["step"]
    output = prepare_run(output, cfg, resume)
    json_write(output / "prompt_settings.json", asdict(settings))
    total = settings.train_steps if max_steps is None else max_steps
    timer = time.perf_counter()
    for step in range(start + 1, total + 1):
        model.train()
        prompt, batch = sampler.batch(device)
        s, a, r, _, rtg, t, mask = batch
        predicted = model(s, a, r, rtg, t, attention_mask=mask, prompt=prompt)[1]
        loss = (predicted[mask.bool()] - a[mask.bool()]).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite Prompt-DT loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), .25, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if step == start + 1 or step % 100 == 0 or step == total:
            log_record(output / "metrics.jsonl", {"step": step, "action_mse": loss.item(), "elapsed_seconds": time.perf_counter() - timer})
            print(f"prompt_dt step={step}/{total} action_mse={loss.item():.6f}", flush=True)
        if step % cfg.checkpoint_every == 0 or step == total:
            atomic_save({"stage": "prompt_dt", "config": asdict(cfg), "settings": asdict(settings), "step": step,
                         "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                         "rng": rng_state(), "sampler_rng": sampler.rng.bit_generator.state,
                         "normalizer": n.state_dict(), "state_dim": sd, "action_dim": ad,
                         "task_manifest": manifest["tasks"], "dataset_sha256": fingerprint,
                         "target_return": target, "training_budget_complete": step >= settings.train_steps,
                         "source_commit": "5f665e50f6218a4df97c3f12e3e556c08bac9041"}, output / "latest.pt")
    return output / "latest.pt"


def evaluate(checkpoint, dataset, output, device, quality="expert", episodes=None):
    ck = load_checkpoint(checkpoint, device)
    if ck["stage"] != "prompt_dt":
        raise ValueError("Expected a Prompt-DT checkpoint")
    cfg, settings = Config(**ck["config"]), PromptSettings(**ck["settings"])
    warm, manifest = load_split(dataset, "warm_start", quality)
    if manifest["tasks"] != ck["task_manifest"] or file_hash(Path(dataset) / "manifest.json") != ck["dataset_sha256"]:
        raise ValueError("Prompt-DT evaluation dataset differs from training")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    n = Normalizer.from_state_dict(ck["normalizer"])
    model = make_model(ck["state_dim"], ck["action_dim"], cfg, settings).to(device)
    model.load_state_dict(ck["model"])
    model.eval().requires_grad_(False)
    tasks = [t for t in manifest["tasks"] if t["split"] == "test"]
    task_returns = {}
    for ti, task in enumerate(tasks):
        env = make_env(cfg, task["parameters"])
        pool = [e for e in warm if e["metadata"]["task_id"] == task["task_id"] and len(e["observations"]) >= cfg.history]
        if not pool:
            raise ValueError(f"No warm-start windows for {task['task_id']}")
        returns = []
        for episode_id in range(cfg.eval_episodes if episodes is None else episodes):
            eval_seed = cfg.seed * 1_000_000 + ti * 1000 + episode_id
            rng = np.random.default_rng(eval_seed + 7_000_000)
            index = int(rng.integers(len(pool)))
            start = int(rng.integers(len(pool[index]["observations"]) - cfg.history + 1))
            arrays = sequence(pool[index], start, cfg.history, n, settings.return_scale, True)
            prompt = tuple(torch.as_tensor(a[None], device=device, dtype=torch.long if i == 5 else torch.float32)
                           for i, a in enumerate(arrays))
            state, _ = env.reset(seed=eval_seed)
            states, actions, targets = [], [], []
            total, clips = 0., 0
            timer = time.perf_counter()
            for step in range(cfg.episode_limit):
                states.append(n.state(state))
                actions.append(np.zeros(ck["action_dim"], np.float32))
                targets.append([(ck["target_return"] - total) / settings.return_scale])
                with torch.no_grad():
                    unit = model.get_action(torch.as_tensor(np.asarray(states), device=device, dtype=torch.float32),
                        torch.as_tensor(np.asarray(actions), device=device, dtype=torch.float32), None,
                        torch.as_tensor(targets, device=device, dtype=torch.float32),
                        torch.arange(len(states), device=device)[None], prompt=prompt).cpu().numpy()
                raw = n.inverse_action(unit)
                action = np.clip(raw, env.action_space.low, env.action_space.high)
                clips += int(np.count_nonzero(np.abs(raw - action) > 1e-6))
                actions[-1] = n.action(action)
                state, reward, term, trunc, _ = env.step(action)
                total += reward
                if term or trunc:
                    break
            record = {"env": cfg.env, "method": "prompt_dt", "task_id": task["task_id"],
                      "task_parameters": task["parameters"], "model_seed": cfg.seed,
                      "episode_seed": eval_seed, "warm_start_quality": quality,
                      "warm_start_indices": [{"episode_index": index, "start": start,
                          "collection_seed": pool[index]["metadata"]["collection_seed"]}],
                      "return": total, "length": step + 1, "target_return": ck["target_return"],
                      "action_clip_fraction": clips / ((step + 1) * ck["action_dim"]),
                      "inference_seconds": time.perf_counter() - timer,
                      "dataset_sha256": ck["dataset_sha256"], "checkpoint_step": ck["step"]}
            log_record(output / "episodes.jsonl", record)
            returns.append(total)
            print(f"prompt_dt eval {task['task_id']} episode={episode_id} return={total:.3f}", flush=True)
        env.close()
        task_returns[task["task_id"]] = float(np.mean(returns))
    summary = {"env": cfg.env, "method": "prompt_dt", "model_seed": cfg.seed,
               "task_means": task_returns, "mean_return": float(np.mean(list(task_returns.values()))),
               "training_budget_complete": ck["training_budget_complete"], "comparable_to_paper": False}
    json_write(output / "summary.json", summary)
    json_write(output / "provenance.json", {"checkpoint_sha256": file_hash(checkpoint),
               "dataset_sha256": ck["dataset_sha256"], "config": asdict(cfg), "settings": asdict(settings),
               "source_commit": ck["source_commit"], "quality": quality})
    return summary
