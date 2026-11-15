import numpy as np
import torch
from metadiffuser.runtime import load_checkpoint, rng_state, restore_rng, seed_all


def test_numpy2_checkpoint_array_and_portable_rng(tmp_path):
    seed_all(31)
    state = rng_state()
    assert isinstance(state["numpy"][1], list)
    expected = np.random.randn(20)
    restore_rng(state)
    np.testing.assert_array_equal(np.random.randn(20), expected)
    # Include an ndarray as in checkpoints written before portable RNG storage.
    path = tmp_path / "legacy_numpy_array.pt"
    torch.save({"numpy": np.arange(13, dtype=np.uint32), "rng": state}, path)
    restored = load_checkpoint(path)
    np.testing.assert_array_equal(restored["numpy"], np.arange(13, dtype=np.uint32))
    restore_rng(restored["rng"])
    np.testing.assert_array_equal(np.random.randn(20), expected)
