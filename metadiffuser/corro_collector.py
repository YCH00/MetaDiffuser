"""Use the pinned CORRO SAC networks and update implementation directly.

Only the task manifest/environment API and output format are adapted. Collection
budgets below come from CORRO's released data_collection_config, not an assertion
that MetaDiffuser authors used the identical budgets.
"""
from dataclasses import asdict
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from .envs import make_env
from .runtime import atomic_save, json_write, log_record, rng_state, seed_all
from .sac import Replay, evaluate_actor


DEFAULTS = {
    "point_robot": dict(width=32, alpha=.01, gamma=.9, iterations=100, initial_episodes=10, updates=10),
    "ant_dir": dict(width=128, alpha=.2, gamma=.99, iterations=200, initial_episodes=50, updates=200),
    "cheetah_vel": dict(width=128, alpha=.2, gamma=.99, iterations=500, initial_episodes=50, updates=200),
    "cheetah_dir": dict(width=128, alpha=.2, gamma=.99, iterations=500, initial_episodes=50, updates=200),
    "hopper_param": dict(width=128, alpha=.2, gamma=.99, iterations=700, initial_episodes=50, updates=200),
    "walker_param": dict(width=128, alpha=.2, gamma=.99, iterations=3000, initial_episodes=50, updates=200),
}


def original_classes(device):
    root = Path(__file__).resolve().parents[1] / "third_party" / "CORRO"
    if not root.exists():
        raise RuntimeError("Pinned CORRO checkout is required")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from algorithms.sac import SAC
    from models.policy import TanhGaussianPolicy
    from torchkit.networks import FlattenMlp
    from torchkit import pytorch_utils as ptu
    ptu.set_gpu_mode(str(device).startswith("cuda"))
    ptu.device = torch.device(device)
    return SAC, TanhGaussianPolicy, FlattenMlp


class CorroActor(nn.Module):
    def __init__(self, sd, ad, width, low, high, device):
        super().__init__()
        _, Policy, _ = original_classes(device)
        self.policy = Policy(obs_dim=sd, action_dim=ad, hidden_sizes=[width, width]).to(device)
        self.register_buffer("low", torch.as_tensor(low, dtype=torch.float32, device=device))
        self.register_buffer("high", torch.as_tensor(high, dtype=torch.float32, device=device))

    def forward(self, s, deterministic=False):
        unit, _, _, _ = self.policy(s, deterministic=deterministic)
        return self.low + (unit + 1) * .5 * (self.high - self.low), None


def train_corro_expert(cfg, task, run, device, iterations=None):
    run = Path(run)
    if run.exists() and any(run.iterdir()):
        raise FileExistsError(run)
    run.mkdir(parents=True, exist_ok=True)
    seed_all(cfg.seed)
    settings = dict(DEFAULTS[cfg.env])
    if iterations is not None:
        settings["iterations"] = iterations
    env, evaluation = make_env(cfg, task["parameters"]), make_env(cfg, task["parameters"])
    sd, ad = int(np.prod(env.observation_space.shape)), int(np.prod(env.action_space.shape))
    SAC, _, Q = original_classes(device)
    actor = CorroActor(sd, ad, settings["width"], env.action_space.low, env.action_space.high, device)
    q1, q2 = [Q(input_size=sd + ad, output_size=1, hidden_sizes=[settings["width"]] * 2) for _ in range(2)]
    agent = SAC(actor.policy, q1, q2, gamma=settings["gamma"], entropy_alpha=settings["alpha"],
                automatic_entropy_tuning=False).to(device)
    replay = Replay(1_000_000, sd, ad, cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    steps, best, records = 0, -float("inf"), []
    json_write(run / "sac_config.json", {**settings, "episodes_per_iteration": 2, "batch": 256,
                                         "actor_lr": .0003, "critic_lr": .0003, "tau": .005,
                                         "source_commit": "210ad407a0a7807b7789ab426d8b5be6c72268a7",
                                         "cheetah_dir_configuration_borrowed_from_vel": cfg.env == "cheetah_dir"})
    cfg.save(run / "config.json")

    def collect(count, initial=False):
        nonlocal steps
        for _ in range(count):
            s, _ = env.reset(seed=int(rng.integers(2**31)))
            while True:
                if initial:
                    # Match CORRO VariBadWrapper: the advertised space remains the
                    # raw environment space, and its samples are then affine-mapped.
                    unit = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
                else:
                    with torch.no_grad():
                        unit = agent.act(torch.as_tensor(s, device=device)[None])[0][0].cpu().numpy()
                a = env.action_space.low + (unit + 1) * .5 * (env.action_space.high - env.action_space.low)
                ns, r, term, trunc, _ = env.step(a)
                # Original CORRO collection treats all non-goal states as nonterminal,
                # including timeouts (and Walker fall terminations). Preserve and disclose.
                replay.add(s, unit, r, ns, False)
                steps += 1
                s = ns
                if term or trunc:
                    break

    collect(settings["initial_episodes"], initial=True)
    for iteration in range(1, settings["iterations"] + 1):
        collect(2)
        metrics = {}
        for _ in range(settings["updates"]):
            b = replay.sample(256, device)
            metrics = agent.update(b["s"], b["a"], b["r"], b["ns"], b["terminal"])
        if iteration % 20 == 0 or iteration == settings["iterations"]:
            returns = evaluate_actor(actor, evaluation, 10, cfg.seed + 1_000_000, device)
            mean = float(np.mean(returns))
            path = run / f"policy_{steps:09d}.pt"
            payload = {"source": "pinned_corro_sac", "actor": actor.state_dict(), "state_dim": sd,
                       "action_dim": ad, "width": settings["width"], "low": env.action_space.low.tolist(),
                       "high": env.action_space.high.tolist(), "task": task, "seed": cfg.seed, "steps": steps,
                       "iterations": iteration, "returns": returns, "config": asdict(cfg), "sac": settings,
                       "agent": agent.state_dict(), "optimizers": {"q1": agent.qf1_optim.state_dict(),
                       "q2": agent.qf2_optim.state_dict(), "policy": agent.policy_optim.state_dict()}, "rng": rng_state()}
            atomic_save(payload, path)
            if mean > best:
                best = mean
                atomic_save(payload, run / "best.pt")
            records.append({"path": path.name, "step": steps, "mean_return": mean})
            json_write(run / "policies.json", records)
            log_record(run / "evaluation.jsonl", {"iteration": iteration, "environment_steps": steps,
                       "optimizer_steps": iteration * settings["updates"], "returns": returns, "mean": mean, "losses": metrics})
            print(f"CORRO SAC {task['task_id']} iteration={iteration} steps={steps} return={mean:.3f}", flush=True)
    env.close()
    evaluation.close()
    return run / "best.pt"

