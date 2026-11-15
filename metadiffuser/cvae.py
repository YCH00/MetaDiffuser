"""CVAE trajectory-planner comparison from Sec. 5.6.2 / Appendix G.

The paper does not publish the CVAE topology, latent size, or KL weight. This
explicit MLP implementation is a disclosed baseline choice, not author code.
"""
import torch
from torch import nn
from .models import mlp


class CVAEPlanner(nn.Module):
    def __init__(self, state_dim, action_dim, cfg):
        super().__init__()
        self.cfg, self.state_dim, self.action_dim = cfg, state_dim, action_dim
        self.flat_dim = cfg.horizon * (state_dim + action_dim)
        cond_dim = cfg.context_dim + state_dim
        self.encoder = mlp(self.flat_dim + cond_dim, 2 * cfg.cvae_latent_dim, cfg.cvae_width, 2)
        self.decoder = mlp(cfg.cvae_latent_dim + cond_dim, self.flat_dim, cfg.cvae_width, 2)

    def decode(self, latent, z, initial):
        x = self.decoder(torch.cat([latent, z, initial], -1))
        x = x.reshape(len(z), self.cfg.horizon, self.state_dim + self.action_dim)
        if self.cfg.hard_initial_state:
            x = x.clone()
            x[:, 0, :self.state_dim] = initial
        return x

    @staticmethod
    def gaussian_kl(mean, log_variance):
        return .5 * (mean.square() + log_variance.exp() - 1 - log_variance).sum(-1)

    def loss(self, trajectory, z):
        initial = trajectory[:, 0, :self.state_dim]
        mean, log_var = self.encoder(torch.cat([trajectory.flatten(1), z, initial], -1)).chunk(2, -1)
        log_var = log_var.clamp(-20, 10)
        latent = mean + (.5 * log_var).exp() * torch.randn_like(mean)
        reconstruction = self.decode(latent, z, initial)
        mask = torch.ones_like(trajectory)
        if self.cfg.hard_initial_state:
            mask[:, 0, :self.state_dim] = 0
        reconstruction_nll = .5 * ((reconstruction - trajectory).square() * mask).flatten(1).sum(-1).mean()
        kl = self.gaussian_kl(mean, log_var).mean()
        return reconstruction_nll + self.cfg.cvae_kl_weight * kl, {
            "reconstruction_nll_without_constant": reconstruction_nll.item(), "kl": kl.item()}

    @torch.no_grad()
    def sample(self, initial_state, z, action_dim, guide=None, generator=None, return_chain=False):
        if guide is not None:
            raise ValueError("No diffusion noise-space guidance is defined for the CVAE baseline")
        if action_dim != self.action_dim or return_chain:
            raise ValueError("Invalid CVAE sampling request")
        latent = torch.randn((len(z), self.cfg.cvae_latent_dim), device=z.device, generator=generator)
        return self.decode(latent * self.cfg.temperature ** .5, z, initial_state)

