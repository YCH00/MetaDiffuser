import json
from dataclasses import asdict
import numpy as np
import pytest
import torch
from metadiffuser.collect import collect_dataset, rollout
from metadiffuser.data import Normalizer, WindowSampler, load_episode, load_split
from metadiffuser.envs import PointRobot, make_tasks
from metadiffuser.evaluate import evaluate
from metadiffuser.runtime import load_checkpoint
from metadiffuser.sac import SACConfig, train_expert
from metadiffuser.train import pretrain, train_diffusion


def test_point_robot_reward_boundary_and_determinism(tiny):
    env = PointRobot({"goal": [.5, .5]}, tiny)
    a = np.array([.1, .1], dtype=np.float32)
    s, _ = env.reset(seed=7)
    assert np.all(s == 0)
    for step in range(20):
        s, r, term, trunc, _ = env.step(a)
        assert r == pytest.approx(-np.linalg.norm(s - [.5, .5]))
        assert trunc == (step == 19)
        assert not term
    tiny.point_initial_state = "square"
    s1, _ = env.reset(seed=9)
    s2, _ = env.reset(seed=9)
    np.testing.assert_array_equal(s1, s2)


def test_task_splits_and_square(tiny):
    tasks = make_tasks(tiny)
    assert len(tasks) == 3
    assert tasks == make_tasks(tiny)
    assert {t["task_id"] for t in tasks if t["split"] == "train"}.isdisjoint(
        {t["task_id"] for t in tasks if t["split"] == "test"})
    # Square distribution, not a fixed-radius semicircle.
    assert any(abs(np.linalg.norm(t["parameters"]["goal"]) - 1) > 1e-3 for t in tasks)


def test_data_roundtrip_no_test_statistics(tiny, tmp_path):
    root = tmp_path / "data"
    collect_dataset(tiny, root, controller_smoke=True)
    train, m = load_split(root, "train")
    warm, _ = load_split(root, "warm_start")
    assert len(train) == 4
    with pytest.raises(ValueError, match="training episodes only"):
        Normalizer.fit(train + warm, [-.1, -.1], [.1, .1])
    n = Normalizer.fit(train, m["action_low"], m["action_high"])
    action = np.array([[-.1, .04]])
    np.testing.assert_allclose(n.inverse_action(n.action(action)), action)
    sampler = WindowSampler(train, n, tiny)
    b = sampler.batch(4, "cpu")
    assert b["trajectory"].shape == (4, 4, 4)
    assert b["context"].shape == (4, 4, 7)
    # Tampering with an episode is rejected by the manifest checksum.
    path = root / m["episodes"][1]["path"]
    with path.open("ab") as f:
        f.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_split(root, "train")


def test_end_to_end_and_exact_resume(tiny, tmp_path):
    data = tmp_path / "data"
    collect_dataset(tiny, data, controller_smoke=True)
    pre = pretrain(tiny, data, tmp_path / "pre", "cpu", max_steps=2)
    uninterrupted = train_diffusion(tiny, data, pre, tmp_path / "full", "cpu", max_steps=4)
    partial = train_diffusion(tiny, data, pre, tmp_path / "resume", "cpu", max_steps=2)
    resumed = train_diffusion(tiny, data, pre, tmp_path / "resume", "cpu", max_steps=4, resume=partial)
    a, b = load_checkpoint(uninterrupted), load_checkpoint(resumed)
    original_world = load_checkpoint(pre)["world"]
    for name in original_world:
        torch.testing.assert_close(a["world"][name], original_world[name], rtol=0, atol=0)
    for name in a["diffusion"]:
        torch.testing.assert_close(a["diffusion"][name], b["diffusion"][name], rtol=0, atol=0)
    result = evaluate(resumed, data, tmp_path / "eval", "cpu", episodes=1, overrides={"guide_scale": 0.})
    assert np.isfinite(result["mean_return"])
    assert not result["comparable_to_paper"]
    record = json.loads((tmp_path / "eval" / "episodes.jsonl").read_text().splitlines()[0])
    assert record["length"] == 20
    assert record["warm_start_indices"]


def test_sac_short_training_produces_task_policy(tiny, tmp_path):
    task = make_tasks(tiny)[0]
    sac = SACConfig(steps=12, batch=4, width=16, start_steps=2, update_after=2, eval_every=6, eval_episodes=1)
    path = train_expert(tiny, task, tmp_path / "sac", "cpu", sac)
    checkpoint = load_checkpoint(path)
    assert checkpoint["task"] == task
    assert np.isfinite(checkpoint["returns"]).all()
