"""Exercise the actual pinned simulators; no mock or alternative physics backend."""
import argparse
import json
import sys
import traceback
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metadiffuser.config import Config, ENV_SPECS
from metadiffuser.envs import make_env, make_tasks


def verify(name):
    cfg = Config.for_env(name)
    tasks = make_tasks(cfg)
    assert tasks == make_tasks(cfg), "Task manifest is not reproducible"
    env = make_env(cfg, tasks[0]["parameters"])
    expected = {"point_robot": (2, 2), "ant_dir": (27, 8), "cheetah_dir": (20, 6),
                "cheetah_vel": (20, 6), "walker_param": (18, 6), "hopper_param": (12, 3)}
    try:
        first, _ = env.reset(seed=713)
        assert (len(first), env.action_space.shape[0]) == expected[name]
        # Independently reconstruct reward from physics displacement and the
        # prescribed task, including nonzero action cost where applicable.
        nonzero = (env.action_space.low + .63 * (env.action_space.high - env.action_space.low)).astype(np.float32)
        if name == "ant_dir":
            before = np.array(env.base.get_body_com("torso"))[:2]
        elif name.startswith("cheetah"):
            before = float(env.base.sim.data.qpos[0])
        elif name.endswith("param"):
            before = float(env.base.model.data.qpos[0, 0])
        observed, actual_reward, _, _, _ = env.step(nonzero)
        if name == "point_robot":
            expected_reward = -np.linalg.norm(observed - np.asarray(tasks[0]["parameters"]["goal"], np.float32))
        elif name == "ant_dir":
            velocity = (np.array(env.base.get_body_com("torso"))[:2] - before) / env.base.dt
            goal = tasks[0]["parameters"]["goal"]
            expected_reward = np.dot(velocity, [np.cos(goal), np.sin(goal)])
        elif name.startswith("cheetah"):
            velocity = (float(env.base.sim.data.qpos[0]) - before) / env.base.dt
            task = tasks[0]["parameters"]
            forward = velocity * task["direction"] if name == "cheetah_dir" else -abs(velocity - task["velocity"])
            expected_reward = forward - .05 * np.square(nonzero).sum()
        else:
            velocity = (float(env.base.model.data.qpos[0, 0]) - before) / env.base.dt
            expected_reward = velocity + (1. if name == "walker_param" else 0.) - .001 * np.square(nonzero).sum()
        np.testing.assert_allclose(actual_reward, expected_reward, rtol=1e-6, atol=1e-6)
        env.reset(seed=713)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        second, reward, _, _, _ = env.step(action)
        again, _ = env.reset(seed=713)
        np.testing.assert_array_equal(first, again)
        repeated, repeated_reward, _, _, _ = env.step(action)
        np.testing.assert_array_equal(second, repeated)
        assert reward == repeated_reward
        parameter_checks = {}
        if name.endswith("param"):
            assert set(tasks[0]["parameters"]) == {"body_mass", "body_inertia", "dof_damping", "geom_friction"}
            for key, value in tasks[0]["parameters"].items():
                np.testing.assert_allclose(getattr(env.base.model, key), value, rtol=0, atol=0)
                parameter_checks[key] = list(np.asarray(value).shape)
        env.reset(seed=713)
        total, length = 0., 0
        for _ in range(cfg.episode_limit):
            obs, reward, term, trunc, _ = env.step(action)
            assert np.isfinite(obs).all() and np.isfinite(reward)
            total += reward
            length += 1
            if term or trunc:
                break
        assert term or trunc, "Episode limit is not exposed as truncation"
        if name != "walker_param":
            assert length == cfg.episode_limit and trunc and not term
        return {"passed": True, "observation_dim": len(first), "action_dim": len(action),
                "zero_action_return": total, "episode_length": length, "terminated": term,
                "truncated": trunc, "parameter_shapes_verified": parameter_checks,
                "nonzero_action_reward_formula_verified": True,
                "seeded_reset_and_step_exactly_reproducible": True}
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=list(ENV_SPECS), nargs="+", default=list(ENV_SPECS))
    parser.add_argument("--output", default="runs/verification/environments.json")
    args = parser.parse_args()
    results = {}
    for name in args.env:
        try:
            results[name] = verify(name)
        except Exception:
            results[name] = {"passed": False, "traceback": traceback.format_exc()}
        print(json.dumps({name: results[name]}), flush=True)
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(results, indent=2), encoding="utf-8")
    sys.exit(0 if all(r["passed"] for r in results.values()) else 1)
