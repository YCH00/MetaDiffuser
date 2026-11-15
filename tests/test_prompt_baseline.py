import numpy as np
import pytest
import torch
pytest.importorskip("transformers", reason="Optional pinned Prompt-DT runtime is not installed")
from metadiffuser.collect import collect_dataset
from metadiffuser.data import load_split, Normalizer
from metadiffuser.prompt_baseline import PromptSettings, PromptSampler, make_model, sequence, train, evaluate
from metadiffuser.runtime import load_checkpoint


def test_prompt_batch_isolation_causality_and_padding(tiny, tmp_path):
    root = tmp_path / "data"
    collect_dataset(tiny, root, controller_smoke=True)
    eps, manifest = load_split(root, "train")
    n = Normalizer.fit(eps, manifest["action_low"], manifest["action_high"])
    settings = PromptSettings(hidden_size=16, layers=1, heads=1, dropout=0, batch=2)
    model = make_model(2, 2, tiny, settings).eval()
    sampler = PromptSampler(eps, n, tiny, settings)
    prompt, batch = sampler.batch("cpu")
    s, a, r, _, rtg, t, mask = batch
    original = model(s, a, r, rtg, t, attention_mask=mask, prompt=prompt)[1]
    altered = tuple(x.clone() for x in prompt)
    altered[0][1] += 10
    changed = model(s, a, r, rtg, t, attention_mask=mask, prompt=altered)[1]
    torch.testing.assert_close(original[0], changed[0], atol=0, rtol=0)
    assert not torch.equal(original[1], changed[1])
    # Current action is downstream of the current state token, never a label leak.
    changed_actions = a.clone()
    changed_actions[:, -1] += 100
    changed = model(s, changed_actions, r, rtg, t, attention_mask=mask, prompt=prompt)[1]
    torch.testing.assert_close(original[:, -1], changed[:, -1], atol=0, rtol=0)
    for h in (4, 10, 20):
        context = sequence(eps[0], 0, h, n, 1., True)
        assert context[0].shape[0] == h + 2
        assert context[-1][:2].sum() == 0
        assert context[4].shape[0] == context[0].shape[0]


def test_prompt_training_resume_and_shared_warm_indices(tiny, tmp_path):
    root = tmp_path / "data"
    collect_dataset(tiny, root, controller_smoke=True)
    settings = PromptSettings(hidden_size=16, layers=1, heads=1, dropout=.1, batch=2, train_steps=4)
    full = train(tiny, settings, root, tmp_path / "full", "cpu")
    partial = train(tiny, settings, root, tmp_path / "resume", "cpu", max_steps=2)
    resumed = train(tiny, settings, root, tmp_path / "resume", "cpu", resume=partial)
    x, y = load_checkpoint(full), load_checkpoint(resumed)
    for k in x["model"]:
        torch.testing.assert_close(x["model"][k], y["model"][k], atol=0, rtol=0)
    result = evaluate(resumed, root, tmp_path / "eval", "cpu", episodes=1)
    assert np.isfinite(result["mean_return"])
    assert result["training_budget_complete"]
