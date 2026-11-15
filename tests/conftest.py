import pytest
import torch
from metadiffuser.config import Config


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(42)


@pytest.fixture
def tiny():
    return Config(channels=16, encoder_width=32, embedding_hidden=32,
                  embedding_dim=8, context_dim=8, diffusion_steps=4,
                  pretrain_batch=4, diffusion_batch=4, train_tasks=2, test_tasks=1,
                  episodes_per_task=3, eval_episodes=1, checkpoint_every=2)

