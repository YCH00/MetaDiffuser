import copy
import math
import time
from dataclasses import asdict
from pathlib import Path
import torch
from .config import Config
from .data import Normalizer, WindowSampler, check_dataset_config, file_hash, load_split
from .diffusion import GaussianDiffusion
from .models import ConditionalUNet, WorldModel
from .runtime import atomic_save, json_write, load_checkpoint, log_record, restore_rng, rng_state, seed_all, system_info, snapshot_source


def world_loss(world, batch, cfg, shuffle=False):
    z = world.encode(batch["context"])
    if shuffle:
        z = z.roll(1, dims=0)
    r = world.reward(batch["states"], batch["actions"], z)
    s = world.dynamics(batch["states"], batch["actions"], z)
    # Fixed-variance Gaussian NLL, summed over event dimensions, averaged over samples.
    reward_nll = 0.5 * ((r - batch["rewards"]).square() / cfg.reward_variance + math.log(2 * math.pi * cfg.reward_variance)).sum(-1).mean()
    dynamics_nll = 0.5 * ((s - batch["next_states"]).square() / cfg.dynamics_variance + math.log(2 * math.pi * cfg.dynamics_variance)).sum(-1).mean()
    metrics = {"reward_mse": (r - batch["rewards"]).square().mean().item(),
               "dynamics_mse": (s - batch["next_states"]).square().mean().item(),
               "context_variance": z.var(0, unbiased=False).mean().item()}
    return reward_nll + dynamics_nll, metrics


def gradient_step(loss, module, optimizer, clip):
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite training loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(module.parameters(), float("inf") if clip is None else clip,
                                        error_if_nonfinite=True)
    optimizer.step()
    return float(norm)


def prepare_run(run, cfg, resume):
    run = Path(run)
    if run.exists() and any(run.iterdir()) and not resume:
        raise FileExistsError(f"Use a new run directory or --resume: {run}")
    run.mkdir(parents=True, exist_ok=True)
    cfg.save(run / "config.json")
    json_write(run / "system.json", system_info())
    snapshot_source(run)
    return run


def pretrain(cfg, dataset, run, device, max_steps=None, resume=None):
    cfg.validate()
    seed_all(cfg.seed)
    episodes, manifest = load_split(dataset, "train")
    check_dataset_config(manifest, cfg)
    val_episodes, _ = load_split(dataset, "validation")
    n = Normalizer.fit(episodes, manifest["action_low"], manifest["action_high"], cfg.reward_scale)
    sd, ad = episodes[0]["observations"].shape[-1], episodes[0]["actions"].shape[-1]
    world = WorldModel(sd, ad, cfg).to(device)
    optimizer = torch.optim.Adam(world.parameters(), lr=cfg.pretrain_lr)
    sampler = WindowSampler(episodes, n, cfg, cfg.seed, include_trajectory=False)
    valid = WindowSampler(val_episodes, n, cfg, cfg.seed + 10_000, include_trajectory=False)
    steps_per_epoch = math.ceil(sampler.window_count / cfg.pretrain_batch)
    total = steps_per_epoch * cfg.pretrain_epochs if max_steps is None else max_steps
    start = 0
    fingerprint = file_hash(Path(dataset) / "manifest.json")
    if resume:
        ck = load_checkpoint(resume, device)
        if asdict(Config(**ck["config"])) != asdict(cfg) or ck["dataset_sha256"] != fingerprint or ck["stage"] != "pretrain":
            raise ValueError("Resume configuration/data/stage mismatch")
        world.load_state_dict(ck["world"])
        optimizer.load_state_dict(ck["optimizer"])
        sampler.rng.bit_generator.state = ck["sampler_rng"]
        valid.rng.bit_generator.state = ck["validation_rng"]
        restore_rng(ck["rng"])
        start = ck["step"]
    run = prepare_run(run, cfg, resume)
    timer = time.perf_counter()
    for step in range(start + 1, total + 1):
        world.train()
        batch = sampler.batch(cfg.pretrain_batch, device)
        loss, metrics = world_loss(world, batch, cfg)
        grad_norm = gradient_step(loss, world, optimizer, cfg.train_grad_clip)
        if step == start + 1 or step % 100 == 0 or step == total:
            world.eval()
            with torch.no_grad():
                b = valid.batch(cfg.pretrain_batch, device)
                _, validation = world_loss(world, b, cfg)
                _, shuffled = world_loss(world, b, cfg, shuffle=True)
                independent = valid.batch(cfg.pretrain_batch, device, independent_context=True)
                _, independent_metrics = world_loss(world, independent, cfg)
            record = {"step": step, "epoch_equivalent": step / steps_per_epoch, "loss": loss.item(),
                      "grad_norm": grad_norm, "train": metrics, "validation": validation,
                      "shuffled_context": shuffled, "independent_context": independent_metrics,
                      "elapsed_seconds": time.perf_counter() - timer}
            log_record(run / "metrics.jsonl", record)
            print(f"pretrain step={step}/{total} loss={loss.item():.5f} val={validation}", flush=True)
        if step % cfg.checkpoint_every == 0 or step == total:
            atomic_save({"stage": "pretrain", "config": asdict(cfg), "step": step,
                         "steps_per_epoch": steps_per_epoch, "state_dim": sd, "action_dim": ad,
                         "world": world.state_dict(), "normalizer": n.state_dict(),
                         "optimizer": optimizer.state_dict(), "rng": rng_state(),
                         "sampler_rng": sampler.rng.bit_generator.state,
                         "validation_rng": valid.rng.bit_generator.state,
                         "dataset_sha256": fingerprint, "task_manifest": manifest["tasks"],
                         "training_budget_complete": step >= steps_per_epoch * cfg.pretrain_epochs}, run / "latest.pt")
    return run / "latest.pt"


def train_diffusion(cfg, dataset, pretrained, run, device, max_steps=None, resume=None):
    cfg.validate()
    seed_all(cfg.seed)
    ck = load_checkpoint(pretrained, device)
    if ck["stage"] != "pretrain":
        raise ValueError("Expected pretraining checkpoint")
    for key in ("env", "history", "context_dim", "encoder_width", "channels", "embedding_dim", "embedding_hidden", "predictor", "reward_scale"):
        if ck["config"][key] != getattr(cfg, key):
            raise ValueError(f"Pretraining architecture/config mismatch: {key}")
    fingerprint = file_hash(Path(dataset) / "manifest.json")
    if ck["dataset_sha256"] != fingerprint:
        raise ValueError("Pretraining and diffusion datasets differ")
    episodes, manifest = load_split(dataset, "train")
    check_dataset_config(manifest, cfg)
    n = Normalizer.from_state_dict(ck["normalizer"])
    sd, ad = ck["state_dim"], ck["action_dim"]
    world = WorldModel(sd, ad, cfg).to(device)
    world.load_state_dict(ck["world"])
    world.freeze()
    model = ConditionalUNet(sd + ad, cfg).to(device)
    diffusion = GaussianDiffusion(model, sd, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.diffusion_lr)
    ema = copy.deepcopy(model) if cfg.ema_decay is not None else None
    sampler = WindowSampler(episodes, n, cfg, cfg.seed)
    start = 0
    if resume:
        saved = load_checkpoint(resume, device)
        if asdict(Config(**saved["config"])) != asdict(cfg) or saved["dataset_sha256"] != fingerprint or saved["pretrain_sha256"] != file_hash(pretrained):
            raise ValueError("Resume configuration/data/pretraining mismatch")
        diffusion.load_state_dict(saved["diffusion"])
        optimizer.load_state_dict(saved["optimizer"])
        if ema is not None:
            ema.load_state_dict(saved["ema"])
        sampler.rng.bit_generator.state = saved["sampler_rng"]
        restore_rng(saved["rng"])
        start = saved["step"]
    run = prepare_run(run, cfg, resume)
    total = cfg.diffusion_train_steps if max_steps is None else max_steps
    timer = time.perf_counter()
    for step in range(start + 1, total + 1):
        b = sampler.batch(cfg.diffusion_batch, device)
        with torch.no_grad():
            z = world.encode(b["context"])
        loss = diffusion.loss(b["trajectory"], z)
        grad_norm = gradient_step(loss, model, optimizer, cfg.train_grad_clip)
        if ema is not None:
            with torch.no_grad():
                for p, q in zip(ema.parameters(), model.parameters()):
                    p.lerp_(q, 1 - cfg.ema_decay)
        if step == start + 1 or step % 100 == 0 or step == total:
            log_record(run / "metrics.jsonl", {"step": step, "noise_mse": loss.item(), "grad_norm": grad_norm,
                                               "elapsed_seconds": time.perf_counter() - timer})
            print(f"diffusion step={step}/{total} loss={loss.item():.5f}", flush=True)
        if step % cfg.checkpoint_every == 0 or step == total:
            atomic_save({"stage": "diffusion", "config": asdict(cfg), "step": step,
                         "state_dim": sd, "action_dim": ad, "world": world.state_dict(),
                         "diffusion": diffusion.state_dict(), "ema": ema.state_dict() if ema else None,
                         "normalizer": n.state_dict(), "optimizer": optimizer.state_dict(),
                         "rng": rng_state(), "sampler_rng": sampler.rng.bit_generator.state,
                         "dataset_sha256": fingerprint, "pretrain_sha256": file_hash(pretrained),
                         "task_manifest": manifest["tasks"],
                         "pretraining_budget_complete": ck["training_budget_complete"],
                         "training_budget_complete": step >= cfg.diffusion_train_steps}, run / "latest.pt")
    return run / "latest.pt"
