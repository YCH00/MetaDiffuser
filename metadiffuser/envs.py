"""Explicit task manifests and adapters. No silent modern-MuJoCo substitution."""
import importlib
import random
import sys
from pathlib import Path
import numpy as np


class Box:
    def __init__(self, low, high, shape):
        self.low = np.full(shape, low, dtype=np.float32)
        self.high = np.full(shape, high, dtype=np.float32)
        self.shape = shape


class PointRobot:
    """Square goals per App. E; transition/reward/action bounds follow CORRO.

    Initial state distribution is not specified in the paper: configurable.
    """
    def __init__(self, task, cfg):
        self.goal = np.asarray(task["goal"], dtype=np.float32)
        self.cfg = cfg
        self.action_space = Box(-0.1, 0.1, (2,))
        self.observation_space = Box(-np.inf, np.inf, (2,))
        self.rng = np.random.default_rng()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = (self.rng.uniform(-1, 1, 2).astype(np.float32)
                      if self.cfg.point_initial_state == "square" else np.zeros(2, np.float32))
        self.steps = 0
        return self.state.copy(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Invalid Point-Robot action")
        if np.any(action < self.action_space.low - 1e-6) or np.any(action > self.action_space.high + 1e-6):
            raise ValueError("Action outside environment bounds")
        self.state = self.state + action
        self.steps += 1
        reward = -float(np.linalg.norm(self.state - self.goal))
        return self.state.copy(), reward, False, self.steps >= self.cfg.episode_limit, {}

    def close(self):
        pass


LEGACY_CLASSES = {
    "cheetah_vel": ("half_cheetah_vel", "HalfCheetahVelEnv"),
    "cheetah_dir": ("half_cheetah_dir", "HalfCheetahDirEnv"),
    "ant_dir": ("ant_dir", "AntDirEnv"),
    "hopper_param": ("rand_param_envs.hopper_rand_params", "HopperRandParamsEnv"),
    "walker_param": ("rand_param_envs.walker2d_rand_params", "Walker2DRandParamsEnv"),
}


def legacy_base(cfg):
    root = Path(__file__).resolve().parents[1] / "third_party" / "CORRO"
    if not root.exists():
        raise RuntimeError("Fetch pinned third_party/CORRO first")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module, name = LEGACY_CLASSES[cfg.env]
    try:
        cls = getattr(importlib.import_module("environments.mujoco." + module), name)
        kwargs = {"max_episode_steps": cfg.episode_limit}
        if cfg.env != "cheetah_dir":
            kwargs["n_tasks"] = 1
        if cfg.env.endswith("param"):
            kwargs["log_scale_limit"] = cfg.random_param_log_limit
        return cls(**kwargs)
    except (ImportError, OSError) as exc:
        raise RuntimeError("Legacy CORRO MuJoCo backend unavailable. Use the isolated Linux legacy "
                           "environment described in docs/ENVIRONMENTS.md; do not substitute Gymnasium rewards.") from exc


class LegacyAdapter:
    def __init__(self, base, task, cfg):
        self.base, self.cfg = base, cfg
        if cfg.env == "cheetah_dir":
            base.set_task(task["direction"])
        elif cfg.env.endswith("param"):
            base.set_task({k: np.asarray(v) for k, v in task.items()})
        elif cfg.env == "cheetah_vel":
            base.set_goal(task["velocity"])
        else:
            base.set_goal(task["goal"])
        self.action_space = base.action_space
        self.observation_space = base.observation_space

    def reset(self, seed=None):
        if seed is not None:
            self.base.seed(seed)
        self.steps = 0
        return np.asarray(self.base.reset(), np.float32).reshape(-1), {}

    def step(self, action):
        obs, reward, done, info = self.base.step(action)
        self.steps += 1
        timeout = bool(info.get("TimeLimit.truncated", False) or info.get("bad_transition", False)
                       or self.steps >= self.cfg.episode_limit)
        return np.asarray(obs, np.float32).reshape(-1), float(reward), bool(done and not timeout), timeout, info

    def close(self):
        if hasattr(self.base, "close"):
            self.base.close()


def make_env(cfg, task):
    if cfg.env == "point_robot":
        return PointRobot(task, cfg)
    if cfg.backend != "corro_legacy":
        raise ValueError("Only audited CORRO legacy backend is supported")
    return LegacyAdapter(legacy_base(cfg), task, cfg)


def make_tasks(cfg):
    rng = np.random.default_rng(cfg.task_seed)
    n = cfg.train_tasks + cfg.test_tasks
    if cfg.env == "point_robot":
        params = [{"goal": g.tolist()} for g in rng.uniform(-cfg.point_goal_extent, cfg.point_goal_extent, (n, 2))]
    elif cfg.env == "cheetah_dir":
        if cfg.train_tasks != 2 or cfg.test_tasks != 2:
            raise ValueError("Cheetah-Dir must contain both directions in both splits")
        params = [{"direction": d} for d in (-1., 1., -1., 1.)]
    elif cfg.env == "cheetah_vel":
        params = [{"velocity": float(v)} for v in rng.uniform(0, cfg.cheetah_max_velocity, n)]
    elif cfg.env == "ant_dir":
        params = [{"goal": float(v)} for v in rng.uniform(0, 2 * np.pi, n)]
    else:
        # CORRO uses BOTH Python and NumPy global RNGs for task generation.
        py_state, np_state = random.getstate(), np.random.get_state()
        try:
            random.seed(cfg.task_seed)
            np.random.seed(cfg.task_seed)
            env = legacy_base(cfg)
            params = [{k: np.asarray(v).tolist() for k, v in p.items()} for p in env.sample_tasks(n)]
            env.close()
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
    return [{"task_id": f"{cfg.env}_{i:03d}", "parameters": p, "task_seed": cfg.task_seed,
             "split": "train" if i < cfg.train_tasks else "test"} for i, p in enumerate(params)]

