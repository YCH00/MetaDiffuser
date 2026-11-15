import numpy as np
import torch
from metadiffuser.collect import load_actor, rollout
from metadiffuser.corro_collector import train_corro_expert
from metadiffuser.envs import make_tasks, make_env
from metadiffuser.runtime import load_checkpoint


def test_pinned_corro_sac_training_and_checkpoint_load(tiny, tmp_path):
    task = make_tasks(tiny)[0]
    checkpoint = train_corro_expert(tiny, task, tmp_path / "corro", "cpu", iterations=2)
    ck = load_checkpoint(checkpoint)
    assert ck["source"] == "pinned_corro_sac"
    assert ck["sac"]["alpha"] == .01
    assert ck["sac"]["gamma"] == .9
    actor = load_actor(checkpoint, "cpu", task)
    a, _ = actor(torch.zeros(2, 2), deterministic=True)
    assert (a.abs() <= .1).all()
    assert np.isfinite(ck["returns"]).all()
    # Stochastic expert collection must be reproducible by collection seed,
    # while different seeds must add coverage even from a fixed initial state.
    env = make_env(tiny, task["parameters"])
    def sample(seed):
        torch.manual_seed(seed)
        return rollout(env, lambda s: actor(torch.as_tensor(s)[None], False)[0][0].detach().numpy(), seed)
    first, repeated, other = sample(91), sample(91), sample(92)
    np.testing.assert_array_equal(first["actions"], repeated["actions"])
    assert not np.array_equal(first["actions"], other["actions"])
