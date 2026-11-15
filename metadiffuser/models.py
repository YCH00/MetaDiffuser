"""Paper Sec. 4.1 and Appendix H. Unspecified topology is documented."""
import math
import torch
from torch import nn


def mlp(inp, out, width, layers=3):
    modules = []
    for _ in range(layers):
        modules += [nn.Linear(inp, width), nn.Mish()]
        inp = width
    return nn.Sequential(*modules, nn.Linear(inp, out))


class ContextEncoder(nn.Module):
    def __init__(self, state_dim, action_dim, cfg):
        super().__init__()
        self.net = mlp(cfg.history * (2 * state_dim + action_dim + 1),
                       cfg.context_dim, cfg.encoder_width, layers=3)

    def forward(self, context):
        return self.net(context.flatten(1))


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if dim < 4 or dim % 2:
            raise ValueError("Embedding dimension must be even and >= 4")
        self.register_buffer("frequencies", torch.exp(
            -math.log(10000) * torch.arange(dim // 2) / (dim // 2 - 1)))

    def forward(self, t):
        phase = t[:, None] * self.frequencies[None]
        return torch.cat([phase.sin(), phase.cos()], -1)


def conv_block(inp, out):
    return nn.Sequential(nn.Conv1d(inp, out, 5, padding=2),
                         nn.GroupNorm(math.gcd(8, out), out), nn.Mish())


class ResidualTemporalBlock(nn.Module):
    def __init__(self, inp, out, embedding_dim):
        super().__init__()
        self.first = conv_block(inp, out)
        self.second = conv_block(out, out)
        self.embedding = nn.Sequential(nn.Mish(), nn.Linear(embedding_dim, out))
        self.skip = nn.Conv1d(inp, out, 1) if inp != out else nn.Identity()

    def forward(self, x, embedding):
        y = self.first(x) + self.embedding(embedding).unsqueeze(-1)
        return self.second(y) + self.skip(x)


class ConditionalUNet(nn.Module):
    """Six residual blocks, two convs each; [B,H,Ds+Da] in and out."""
    def __init__(self, transition_dim, cfg):
        super().__init__()
        c, e = cfg.channels, cfg.embedding_dim
        self.time = nn.Sequential(SinusoidalEmbedding(e), nn.Linear(e, cfg.embedding_hidden),
                                  nn.Mish(), nn.Linear(cfg.embedding_hidden, e))
        self.context = nn.Sequential(nn.Linear(cfg.context_dim, cfg.embedding_hidden),
                                     nn.Mish(), nn.Linear(cfg.embedding_hidden, e))
        self.down1 = ResidualTemporalBlock(transition_dim, c, 2 * e)
        self.downsample = nn.Conv1d(c, c, 3, stride=2, padding=1)
        self.down2 = ResidualTemporalBlock(c, 2 * c, 2 * e)
        self.mid1 = ResidualTemporalBlock(2 * c, 2 * c, 2 * e)
        self.mid2 = ResidualTemporalBlock(2 * c, 2 * c, 2 * e)
        self.upsample = nn.ConvTranspose1d(2 * c, c, 4, stride=2, padding=1)
        self.up1 = ResidualTemporalBlock(2 * c, c, 2 * e)
        self.up2 = ResidualTemporalBlock(c, c, 2 * e)
        self.output = nn.Conv1d(c, transition_dim, 1)

    def forward(self, x, t, z, drop_mask=None):
        context = self.context(z)
        if drop_mask is not None:
            # Mask AFTER the embedding MLP: its biases cannot leak the label.
            context = context * (~drop_mask.bool()).to(context.dtype)[:, None]
        embedding = torch.cat([self.time(t), context], -1)
        skip = self.down1(x.transpose(1, 2), embedding)
        y = self.down2(self.downsample(skip), embedding)
        y = self.mid2(self.mid1(y, embedding), embedding)
        y = self.upsample(y)
        y = self.up2(self.up1(torch.cat([y, skip], 1), embedding), embedding)
        return self.output(y).transpose(1, 2)


class TransitionPredictor(nn.Module):
    """P(s'|s,a,z) or R(r|s,a,z), without access to the target/future.

    Uses the U-Net's contracting blocks on a length-one (s,a) sequence.
    Dynamics vector head resolves the scalar-output inconsistency in App. H.
    """
    def __init__(self, state_dim, action_dim, out_dim, cfg):
        super().__init__()
        self.kind = cfg.predictor
        if self.kind == "mlp":
            self.net = mlp(state_dim + action_dim + cfg.context_dim, out_dim, cfg.encoder_width)
        else:
            c, e = cfg.channels, cfg.embedding_dim
            self.context = nn.Sequential(nn.Linear(cfg.context_dim, cfg.embedding_hidden),
                                         nn.Mish(), nn.Linear(cfg.embedding_hidden, e))
            self.first = ResidualTemporalBlock(state_dim + action_dim, c, e)
            self.down = nn.Conv1d(c, c, 3, stride=2, padding=1)
            self.second = ResidualTemporalBlock(c, 2 * c, e)
            self.output = nn.Linear(2 * c, out_dim)

    def forward(self, s, a, z):
        shape = s.shape[:-1]
        while z.ndim < s.ndim:
            z = z.unsqueeze(-2)
        z = z.expand(*shape, z.shape[-1])
        x = torch.cat([s, a], -1)
        if self.kind == "mlp":
            return self.net(torch.cat([x, z], -1))
        e = self.context(z.reshape(-1, z.shape[-1]))
        x = x.reshape(-1, x.shape[-1], 1)
        x = self.second(self.down(self.first(x, e)), e).squeeze(-1)
        return self.output(x).reshape(*shape, -1)


class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, cfg):
        super().__init__()
        self.encoder = ContextEncoder(state_dim, action_dim, cfg)
        self.reward = TransitionPredictor(state_dim, action_dim, 1, cfg)
        self.dynamics = TransitionPredictor(state_dim, action_dim, state_dim, cfg)
        self.state_dim = state_dim
        self.action_dim = action_dim

    def encode(self, contexts):
        """Mean aggregate independent historical windows (Appendix I)."""
        if contexts.ndim == 3:
            return self.encoder(contexts)
        if contexts.ndim != 4:
            raise ValueError("Expected [B,h,F] or [B,W,h,F] contexts")
        b, w, h, f = contexts.shape
        return self.encoder(contexts.reshape(b * w, h, f)).reshape(b, w, -1).mean(1)

    def freeze(self):
        self.eval()
        self.requires_grad_(False)
        return self

