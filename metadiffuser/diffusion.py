"""DDPM noise loss, Eq. (6) CFG, and Eq. (8) noise-space dual guidance."""
import math
import torch
from torch import nn


def beta_schedule(steps, kind):
    if kind == "linear":
        return torch.linspace(1e-4, 0.02, steps, dtype=torch.float64)
    if kind != "cosine":
        raise ValueError(f"Unknown schedule: {kind}")
    x = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    abar = torch.cos(((x / steps + 0.008) / 1.008) * math.pi / 2).square()
    abar = abar / abar[0]
    return (1 - abar[1:] / abar[:-1]).clamp(0, 0.999)


class DualGuide:
    def __init__(self, world, cfg):
        self.world, self.cfg = world, cfg

    def components(self, x, z):
        s, a = x[..., :self.world.state_dim], x[..., self.world.state_dim:]
        rewards = self.world.reward(s, a, z).squeeze(-1).sum(-1)
        next_states = self.world.dynamics(s[:, :-1], a[:, :-1], z)
        error = (s[:, 1:] - next_states).square().sum(dim=(-1, -2))
        return rewards, error

    def score(self, x, z):
        r, error = self.components(x, z)
        sign = 1 if self.cfg.dynamics_sign == "paper_plus" else -1
        return self.cfg.reward_weight * r + sign * self.cfg.dynamics_weight * error

    def gradient(self, x, z):
        with torch.enable_grad():
            leaf = x.detach().requires_grad_(True)
            grad, = torch.autograd.grad(self.score(leaf, z.detach()).sum(), leaf)
        if self.cfg.hard_initial_state:
            grad[:, 0, :self.world.state_dim] = 0
        if self.cfg.guide_grad_clip is not None:
            norm = grad.flatten(1).norm(dim=1).clamp_min(1e-12)
            grad *= (self.cfg.guide_grad_clip / norm).clamp(max=1)[:, None, None]
        return grad


class GaussianDiffusion(nn.Module):
    def __init__(self, model, state_dim, cfg):
        super().__init__()
        self.model, self.state_dim, self.cfg = model, state_dim, cfg
        beta = beta_schedule(cfg.diffusion_steps, cfg.diffusion_beta_schedule)
        alpha = 1 - beta
        abar = alpha.cumprod(0)
        previous = torch.cat([torch.ones(1, dtype=torch.float64), abar[:-1]])
        values = dict(beta=beta, alpha=alpha, abar=abar,
                      posterior_variance=beta * (1 - previous) / (1 - abar),
                      posterior_x0=beta * previous.sqrt() / (1 - abar),
                      posterior_xt=(1 - previous) * alpha.sqrt() / (1 - abar))
        for name, value in values.items():
            self.register_buffer(name, value.float())

    def extract(self, name, t):
        return getattr(self, name)[t][:, None, None]

    def condition(self, x, initial_state):
        if self.cfg.hard_initial_state:
            x = x.clone()
            x[:, 0, :self.state_dim] = initial_state
        return x

    def q_sample(self, x, t, noise):
        abar = self.extract("abar", t)
        return abar.sqrt() * x + (1 - abar).sqrt() * noise

    def loss(self, x, z, *, t=None, noise=None, drop_mask=None):
        b = x.shape[0]
        t = torch.randint(self.cfg.diffusion_steps, (b,), device=x.device) if t is None else t
        noise = torch.randn_like(x) if noise is None else noise
        if drop_mask is None:
            drop_mask = torch.rand(b, device=x.device) < self.cfg.context_drop_prob
        noisy = self.condition(self.q_sample(x, t, noise), x[:, 0, :self.state_dim])
        predicted = self.model(noisy, t, z, drop_mask)
        mask = torch.ones_like(x)
        if self.cfg.hard_initial_state:
            mask[:, 0, :self.state_dim] = 0
        return ((predicted - noise).square() * mask).sum() / mask.sum()

    def predict_noise(self, x, t, z):
        b = x.shape[0]
        conditional = self.model(x, t, z, torch.zeros(b, dtype=torch.bool, device=x.device))
        unconditional = self.model(x, t, z, torch.ones(b, dtype=torch.bool, device=x.device))
        return unconditional + self.cfg.cfg_scale * (conditional - unconditional)

    def posterior(self, x, t, noise):
        abar = self.extract("abar", t)
        x0 = (x - (1 - abar).sqrt() * noise) / abar.sqrt()
        if self.cfg.clip_denoised is not None:
            x0 = x0.clamp(-self.cfg.clip_denoised, self.cfg.clip_denoised)
        mean = self.extract("posterior_x0", t) * x0 + self.extract("posterior_xt", t) * x
        return mean, self.extract("posterior_variance", t)

    @torch.no_grad()
    def sample(self, initial_state, z, action_dim, guide=None, generator=None, return_chain=False):
        shape = (len(z), self.cfg.horizon, self.state_dim + action_dim)
        x = torch.randn(shape, device=z.device, generator=generator) * self.cfg.temperature ** 0.5
        x = self.condition(x, initial_state)
        chain = [x.clone()] if return_chain else None
        for step in reversed(range(self.cfg.diffusion_steps)):
            t = torch.full((len(z),), step, device=z.device, dtype=torch.long)
            noise = self.predict_noise(x, t, z)
            if guide is not None and self.cfg.guide_scale != 0:
                noise = noise - self.cfg.guide_scale * (1 - self.extract("abar", t)).sqrt() * guide.gradient(x, z)
            mean, variance = self.posterior(x, t, noise)
            if step:
                mean = mean + (self.cfg.temperature * variance).sqrt() * torch.randn(
                    shape, device=z.device, generator=generator)
            x = self.condition(mean.detach(), initial_state)
            if not torch.isfinite(x).all():
                raise FloatingPointError("Nonfinite sampling trajectory; record guide settings before tuning")
            if return_chain:
                chain.append(x.clone())
        return (x, torch.stack(chain, 1)) if return_chain else x
