import copy
import pytest
import torch
from torch import nn
from metadiffuser.diffusion import DualGuide, GaussianDiffusion
from metadiffuser.models import ConditionalUNet, ContextEncoder, ResidualTemporalBlock, WorldModel


@pytest.mark.parametrize("horizon", [4, 16, 32])
def test_unet_shapes_and_gradients(tiny, horizon):
    tiny.horizon = horizon
    net = ConditionalUNet(7, tiny)
    x = torch.randn(3, horizon, 7, requires_grad=True)
    out = net(x, torch.arange(3), torch.randn(3, tiny.context_dim))
    assert out.shape == x.shape
    out.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert sum(isinstance(m, ResidualTemporalBlock) for m in net.modules()) == 6


def test_context_three_hidden_layers(tiny):
    encoder = ContextEncoder(2, 2, tiny)
    assert sum(isinstance(m, nn.Linear) for m in encoder.modules()) == 4


def test_null_label_is_independent_of_context_including_bias(tiny):
    net = ConditionalUNet(4, tiny)
    x, t = torch.randn(2, 4, 4), torch.ones(2)
    mask = torch.ones(2, dtype=torch.bool)
    a = net(x, t, torch.randn(2, tiny.context_dim), mask)
    b = net(x, t, 100 * torch.randn(2, tiny.context_dim), mask)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    conditional = net(x, t, torch.randn(2, tiny.context_dim), ~mask)
    assert not torch.allclose(a, conditional)


class KnownNoise(nn.Module):
    def forward(self, x, t, z, drop_mask):
        return torch.where(drop_mask[:, None, None], 2., 5.).expand_as(x)


def test_cfg_equation_six(tiny):
    diffusion = GaussianDiffusion(KnownNoise(), 2, tiny)
    x = torch.zeros(2, 4, 4)
    for w in (0., 1., 1.4, 2.):
        tiny.cfg_scale = w
        actual = diffusion.predict_noise(x, torch.ones(2, dtype=torch.long), torch.zeros(2, 8))
        torch.testing.assert_close(actual, torch.full_like(x, 2 + 3 * w))


def test_ddpm_closed_form_and_posterior(tiny):
    d = GaussianDiffusion(KnownNoise(), 2, tiny)
    x0, noise = torch.randn(2, 4, 4), torch.randn(2, 4, 4)
    t = torch.tensor([0, 3])
    xt = d.q_sample(x0, t, noise)
    mean, variance = d.posterior(xt, t, noise)
    direct = (xt - d.extract("beta", t) * noise / (1 - d.extract("abar", t)).sqrt()) / d.extract("alpha", t).sqrt()
    torch.testing.assert_close(mean, direct, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(mean[0], x0[0], atol=2e-6, rtol=2e-6)
    assert variance[0] == 0


def test_equation_eight_noise_correction_not_variance_scaled_mean(tiny):
    d = GaussianDiffusion(KnownNoise(), 2, tiny)
    x, eps, g = torch.randn(2, 4, 4), torch.randn(2, 4, 4), torch.randn(2, 4, 4)
    t = torch.tensor([0, 2])
    mean, _ = d.posterior(x, t, eps)
    guided_mean, _ = d.posterior(x, t, eps - (1 - d.extract("abar", t)).sqrt() * g)
    expected_shift = d.extract("beta", t) / d.extract("alpha", t).sqrt() * g
    torch.testing.assert_close(guided_mean - mean, expected_shift, atol=2e-6, rtol=2e-5)


def test_dropout_probability_means_drop_not_keep(tiny):
    class Capture(nn.Module):
        def forward(self, x, t, z, drop_mask):
            self.mask = drop_mask
            return torch.zeros_like(x)
    capture = Capture()
    d = GaussianDiffusion(capture, 2, tiny)
    tiny.context_drop_prob = .2
    d.loss(torch.zeros(10000, 4, 4), torch.zeros(10000, 8))
    assert abs(capture.mask.float().mean().item() - .2) < .02


def test_multi_window_context_aggregation(tiny):
    world = WorldModel(2, 2, tiny)
    contexts = torch.randn(3, 2, 4, 7)
    expected = torch.stack([world.encoder(contexts[:, i]) for i in range(2)]).mean(0)
    torch.testing.assert_close(world.encode(contexts), expected)


def test_loss_targets_noise_and_masks_initial_state(tiny):
    class Oracle(nn.Module):
        def forward(self, x, t, z, drop_mask):
            prediction = noise.clone()
            prediction[:, 0, :2] = 12345
            return prediction
    x, noise = torch.randn(2, 4, 4), torch.randn(2, 4, 4)
    d = GaussianDiffusion(Oracle(), 2, tiny)
    loss = d.loss(x, torch.zeros(2, 8), t=torch.tensor([1, 2]), noise=noise)
    assert loss.item() == 0


class ZeroDynamics(nn.Module):
    def forward(self, s, a, z):
        return s * 0


class ActionReward(nn.Module):
    def forward(self, s, a, z):
        return a.sum(-1, keepdim=True)


class AnalyticWorld(nn.Module):
    state_dim = 1
    def __init__(self):
        super().__init__()
        self.reward, self.dynamics = ActionReward(), ZeroDynamics()


@pytest.mark.parametrize("sign,direction", [("paper_plus", 1), ("consistency_minus", -1)])
def test_guide_sign_and_h_minus_one_transitions(tiny, sign, direction):
    tiny.dynamics_sign, tiny.dynamics_weight = sign, 1.
    guide = DualGuide(AnalyticWorld(), tiny)
    x = torch.ones(2, 4, 2)
    z = torch.zeros(2, 8)
    reward, error = guide.components(x, z)
    torch.testing.assert_close(reward, torch.tensor([4., 4.]))
    torch.testing.assert_close(error, torch.tensor([3., 3.]))
    grad = guide.gradient(x, z)
    torch.testing.assert_close(grad[:, 1:, 0], torch.full((2, 3), 2. * direction))
    assert (grad[:, 0, 0] == 0).all()
    assert (grad[..., 1] == 1).all()
    after_error = guide.components(x + 0.01 * grad, z)[1]
    assert ((after_error - error) * direction > 0).all()


def test_freeze_keeps_input_grad_and_inpainting(tiny):
    world = WorldModel(2, 2, tiny).freeze()
    guide = DualGuide(world, tiny)
    x, z = torch.randn(2, 4, 4), torch.randn(2, 8)
    assert torch.isfinite(guide.gradient(x, z)).all()
    assert all(p.grad is None for p in world.parameters())
    tiny.guide_scale = .001
    net = ConditionalUNet(4, tiny)
    diffusion = GaussianDiffusion(net, 2, tiny)
    initial = torch.tensor([[.1, .3], [-.2, .4]])
    result, chain = diffusion.sample(initial, z, 2, guide, return_chain=True)
    torch.testing.assert_close(chain[:, :, 0, :2], initial[:, None].expand(-1, 5, -1))
    assert not result.requires_grad
    assert all(p.grad is None for p in world.parameters())


def test_temporal_predictors_do_not_access_future(tiny):
    world = WorldModel(2, 2, tiny)
    s, a, z = torch.randn(2, 4, 2), torch.randn(2, 4, 2), torch.randn(2, 8)
    baseline = world.dynamics(s, a, z)
    s[:, 2:] += 100
    changed = world.dynamics(s, a, z)
    torch.testing.assert_close(baseline[:, :2], changed[:, :2])

