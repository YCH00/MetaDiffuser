"""Single-task SAC for explicitly sourced expert data.

This is an independently implemented SAC collector, not a claim of matching
CORRO's undisclosed MetaDiffuser collection budgets. Timeouts bootstrap.
"""
import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from .envs import make_env
from .models import mlp
from .runtime import atomic_save, json_write, log_record, seed_all


@dataclass
class SACConfig:
    steps: int = 100_000
    batch: int = 256
    width: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    start_steps: int = 1000
    update_after: int = 1000
    replay_size: int = 1_000_000
    eval_every: int = 5000
    eval_episodes: int = 10


class Actor(nn.Module):
    def __init__(self, sd, ad, width, low, high):
        super().__init__()
        self.net = mlp(sd, 2 * ad, width, layers=2)
        self.register_buffer("scale", torch.as_tensor((np.asarray(high) - np.asarray(low)) / 2, dtype=torch.float32))
        self.register_buffer("bias", torch.as_tensor((np.asarray(high) + np.asarray(low)) / 2, dtype=torch.float32))

    def forward(self, states, deterministic=False):
        mean, log_std = self.net(states).chunk(2, -1)
        dist = Normal(mean, log_std.clamp(-20, 2).exp())
        raw = mean if deterministic else dist.rsample()
        unit = raw.tanh()
        action = unit * self.scale + self.bias
        # Stable tanh Jacobian; include the environment action affine transform.
        log_prob = (dist.log_prob(raw) - 2 * (math.log(2) - raw - nn.functional.softplus(-2 * raw))
                    - self.scale.log()).sum(-1, keepdim=True)
        return action, log_prob


class Replay:
    def __init__(self, capacity, sd, ad, seed):
        self.storage = {k: np.empty((capacity, dim), np.float32)
                        for k, dim in (("s", sd), ("a", ad), ("r", 1), ("ns", sd), ("terminal", 1))}
        self.capacity, self.size, self.pos = capacity, 0, 0
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, ns, terminal):
        for k, v in zip(self.storage, (s, a, r, ns, terminal)):
            self.storage[k][self.pos] = v
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch, device):
        idx = self.rng.integers(self.size, size=batch)
        return {k: torch.as_tensor(v[idx], device=device) for k, v in self.storage.items()}


def evaluate_actor(actor, env, episodes, seed, device):
    returns = []
    for i in range(episodes):
        s, _ = env.reset(seed=seed + i)
        total = 0.
        while True:
            with torch.no_grad():
                a, _ = actor(torch.as_tensor(s, dtype=torch.float32, device=device)[None], deterministic=True)
            s, r, terminal, timeout, _ = env.step(a[0].cpu().numpy())
            total += r
            if terminal or timeout:
                break
        returns.append(total)
    return returns


def train_expert(cfg, task, run, device, sac=None):
    sac = sac or SACConfig()
    run = Path(run)
    if run.exists() and any(run.iterdir()):
        raise FileExistsError(run)
    run.mkdir(parents=True, exist_ok=True)
    seed_all(cfg.seed)
    env, evaluation = make_env(cfg, task["parameters"]), make_env(cfg, task["parameters"])
    sd, ad = int(np.prod(env.observation_space.shape)), int(np.prod(env.action_space.shape))
    actor = Actor(sd, ad, sac.width, env.action_space.low, env.action_space.high).to(device)
    q1, q2 = [mlp(sd + ad, 1, sac.width, 2).to(device) for _ in range(2)]
    target1, target2 = copy.deepcopy(q1).requires_grad_(False), copy.deepcopy(q2).requires_grad_(False)
    log_alpha = torch.zeros((), device=device, requires_grad=True)
    optim_a = torch.optim.Adam(actor.parameters(), lr=sac.learning_rate)
    optim_q = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=sac.learning_rate)
    optim_alpha = torch.optim.Adam([log_alpha], lr=sac.learning_rate)
    replay = Replay(min(sac.replay_size, sac.steps), sd, ad, cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    s, _ = env.reset(seed=cfg.seed)
    evaluation_records, best = [], -float("inf")
    json_write(run / "sac_config.json", asdict(sac))
    cfg.save(run / "config.json")
    for step in range(1, sac.steps + 1):
        if step <= sac.start_steps:
            a = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
        else:
            with torch.no_grad():
                a = actor(torch.as_tensor(s, dtype=torch.float32, device=device)[None])[0][0].cpu().numpy()
        ns, r, term, trunc, _ = env.step(a)
        replay.add(s, a, r, ns, term)
        s = ns
        if term or trunc:
            s, _ = env.reset()
        if step >= sac.update_after:
            b = replay.sample(sac.batch, device)
            alpha = log_alpha.exp().detach()
            with torch.no_grad():
                na, lp = actor(b["ns"])
                nx = torch.cat([b["ns"], na], -1)
                target = b["r"] + sac.gamma * (1 - b["terminal"]) * (torch.minimum(target1(nx), target2(nx)) - alpha * lp)
            x = torch.cat([b["s"], b["a"]], -1)
            q_loss = (q1(x) - target).square().mean() + (q2(x) - target).square().mean()
            optim_q.zero_grad(set_to_none=True)
            q_loss.backward()
            optim_q.step()
            q1.requires_grad_(False)
            q2.requires_grad_(False)
            new_a, lp = actor(b["s"])
            new_x = torch.cat([b["s"], new_a], -1)
            actor_loss = (alpha * lp - torch.minimum(q1(new_x), q2(new_x))).mean()
            optim_a.zero_grad(set_to_none=True)
            actor_loss.backward()
            optim_a.step()
            q1.requires_grad_(True)
            q2.requires_grad_(True)
            # Differential entropy target accounts for the affine action scaling.
            entropy_target = -ad + actor.scale.log().sum()
            alpha_loss = -(log_alpha * (lp.detach() + entropy_target)).mean()
            optim_alpha.zero_grad(set_to_none=True)
            alpha_loss.backward()
            optim_alpha.step()
            with torch.no_grad():
                for target_net, online in ((target1, q1), (target2, q2)):
                    for p, q in zip(target_net.parameters(), online.parameters()):
                        p.lerp_(q, sac.tau)
        if step % sac.eval_every == 0 or step == sac.steps:
            returns = evaluate_actor(actor, evaluation, sac.eval_episodes, cfg.seed + 1_000_000, device)
            mean = float(np.mean(returns))
            path = run / f"policy_{step:09d}.pt"
            payload = {"actor": actor.state_dict(), "state_dim": sd, "action_dim": ad, "width": sac.width,
                       "low": env.action_space.low.tolist(), "high": env.action_space.high.tolist(),
                       "task": task, "seed": cfg.seed, "steps": step, "returns": returns,
                       "source": "independent_sac_collector", "config": asdict(cfg), "sac": asdict(sac)}
            atomic_save(payload, path)
            evaluation_records.append({"path": path.name, "mean_return": mean, "step": step})
            if mean > best:
                best = mean
                atomic_save(payload, run / "best.pt")
            log_record(run / "evaluation.jsonl", {"step": step, "returns": returns, "mean": mean})
            print(f"SAC {task['task_id']} step={step} return={mean:.3f}", flush=True)
    env.close()
    evaluation.close()
    json_write(run / "policies.json", evaluation_records)
    return run / "best.pt"

