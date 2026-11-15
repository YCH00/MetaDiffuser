from dataclasses import asdict
from pathlib import Path
import torch
from .cvae import CVAEPlanner
from .data import Normalizer, WindowSampler, check_dataset_config, file_hash, load_split
from .models import WorldModel
from .runtime import atomic_save, load_checkpoint, log_record, restore_rng, rng_state, seed_all
from .train import gradient_step, prepare_run


def train_cvae(cfg, dataset, pretrained, run, device, max_steps=None, resume=None):
    cfg.validate()
    seed_all(cfg.seed)
    ck = load_checkpoint(pretrained, device)
    if ck["stage"] != "pretrain":
        raise ValueError("Expected a world-model pretraining checkpoint")
    fingerprint = file_hash(Path(dataset) / "manifest.json")
    if ck["dataset_sha256"] != fingerprint:
        raise ValueError("CVAE and world model datasets differ")
    for key in ("env", "history", "context_dim", "encoder_width", "channels", "embedding_dim", "embedding_hidden", "predictor", "reward_scale"):
        if ck["config"][key] != getattr(cfg, key):
            raise ValueError(f"Pretrained architecture mismatch: {key}")
    episodes, manifest = load_split(dataset, "train")
    check_dataset_config(manifest, cfg)
    n = Normalizer.from_state_dict(ck["normalizer"])
    sd, ad = ck["state_dim"], ck["action_dim"]
    world = WorldModel(sd, ad, cfg).to(device)
    world.load_state_dict(ck["world"])
    world.freeze()
    model = CVAEPlanner(sd, ad, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.diffusion_lr)
    sampler = WindowSampler(episodes, n, cfg, cfg.seed)
    start = 0
    if resume:
        saved = load_checkpoint(resume, device)
        if saved["stage"] != "cvae" or saved["config"] != asdict(cfg) or saved["dataset_sha256"] != fingerprint or saved["pretrain_sha256"] != file_hash(pretrained):
            raise ValueError("CVAE resume metadata mismatch")
        model.load_state_dict(saved["cvae"])
        optimizer.load_state_dict(saved["optimizer"])
        sampler.rng.bit_generator.state = saved["sampler_rng"]
        restore_rng(saved["rng"])
        start = saved["step"]
    run = prepare_run(run, cfg, resume)
    total = cfg.diffusion_train_steps if max_steps is None else max_steps
    for step in range(start + 1, total + 1):
        b = sampler.batch(cfg.diffusion_batch, device)
        with torch.no_grad():
            z = world.encode(b["context"])
        loss, metrics = model.loss(b["trajectory"], z)
        norm = gradient_step(loss, model, optimizer, cfg.train_grad_clip)
        if step == start + 1 or step % 100 == 0 or step == total:
            log_record(run / "metrics.jsonl", {"step": step, "loss": loss.item(), "grad_norm": norm, **metrics})
            print(f"CVAE step={step}/{total} loss={loss.item():.5f}", flush=True)
        if step % cfg.checkpoint_every == 0 or step == total:
            atomic_save({"stage": "cvae", "config": asdict(cfg), "step": step,
                         "state_dim": sd, "action_dim": ad, "world": world.state_dict(),
                         "cvae": model.state_dict(), "normalizer": n.state_dict(),
                         "optimizer": optimizer.state_dict(), "rng": rng_state(),
                         "sampler_rng": sampler.rng.bit_generator.state, "dataset_sha256": fingerprint,
                         "pretrain_sha256": file_hash(pretrained), "task_manifest": manifest["tasks"],
                         "pretraining_budget_complete": ck["training_budget_complete"],
                         "training_budget_complete": step >= cfg.diffusion_train_steps}, run / "latest.pt")
    return run / "latest.pt"
