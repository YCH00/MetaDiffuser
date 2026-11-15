import torch
from metadiffuser.diffusion import GaussianDiffusion
from metadiffuser.models import ConditionalUNet


def test_fixed_noise_small_batch_can_overfit(tiny):
    """Detect detached outputs, wrong optimizer wiring, or ineffective loss."""
    model = ConditionalUNet(4, tiny)
    diffusion = GaussianDiffusion(model, 2, tiny)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    x, z = torch.randn(4, 4, 4), torch.randn(4, 8)
    t, noise = torch.tensor([0, 1, 2, 3]), torch.randn_like(x)
    mask = torch.tensor([False, True, False, True])
    initial = diffusion.loss(x, z, t=t, noise=noise, drop_mask=mask).item()
    for _ in range(80):
        loss = diffusion.loss(x, z, t=t, noise=noise, drop_mask=mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    final = diffusion.loss(x, z, t=t, noise=noise, drop_mask=mask).item()
    assert final < initial * .1
