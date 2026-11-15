import pytest
import torch
from metadiffuser.collect import collect_dataset
from metadiffuser.cvae import CVAEPlanner
from metadiffuser.evaluate import evaluate
from metadiffuser.train import pretrain
from metadiffuser.train_cvae import train_cvae


def test_cvae_kl_and_gradient(tiny):
    tiny.cvae_width, tiny.cvae_latent_dim = 16, 4
    model = CVAEPlanner(2, 2, tiny)
    assert model.gaussian_kl(torch.zeros(2, 4), torch.zeros(2, 4)).sum() == 0
    torch.testing.assert_close(model.gaussian_kl(torch.ones(2, 4), torch.zeros(2, 4)), torch.full((2,), 2.))
    x, z = torch.randn(3, 4, 4), torch.randn(3, 8)
    loss, metrics = model.loss(x, z)
    loss.backward()
    assert metrics["kl"] >= 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    sampled = model.sample(x[:, 0, :2], z, 2)
    torch.testing.assert_close(sampled[:, 0, :2], x[:, 0, :2])


def test_cvae_training_and_closed_loop(tiny, tmp_path):
    tiny.cvae_width, tiny.cvae_latent_dim = 16, 4
    dataset = tmp_path / "data"
    collect_dataset(tiny, dataset, controller_smoke=True)
    world = pretrain(tiny, dataset, tmp_path / "world", "cpu", max_steps=2)
    model = train_cvae(tiny, dataset, world, tmp_path / "cvae", "cpu", max_steps=2)
    report = evaluate(model, dataset, tmp_path / "eval", "cpu", episodes=1)
    assert report["method"] == "cvae"
    assert report["paper_table_1_literature_value"] == -6.11
    with pytest.raises(ValueError, match="guidance overrides"):
        evaluate(model, dataset, tmp_path / "invalid_eval", "cpu", overrides={"guide_scale": 0.})
