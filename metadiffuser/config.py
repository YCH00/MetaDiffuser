from dataclasses import asdict, dataclass
import json
from pathlib import Path


ENV_SPECS = {
    "point_robot": (4, 4, 20, 30, 10),
    "ant_dir": (10, 32, 200, 30, 10),
    "cheetah_dir": (10, 16, 200, 2, 2),
    "cheetah_vel": (10, 16, 200, 30, 10),
    "walker_param": (20, 32, 200, 30, 10),
    "hopper_param": (20, 32, 200, 30, 10),
}


@dataclass
class Config:
    env: str = "point_robot"
    history: int = 4
    horizon: int = 4
    episode_limit: int = 20
    train_tasks: int = 30
    test_tasks: int = 10
    task_seed: int = 2026
    seed: int = 0
    context_dim: int = 64
    encoder_width: int = 256
    channels: int = 64
    embedding_dim: int = 64
    embedding_hidden: int = 256
    predictor: str = "temporal"
    pretrain_lr: float = 2e-4
    pretrain_batch: int = 64
    pretrain_epochs: int = 1000
    diffusion_lr: float = 2e-4
    diffusion_batch: int = 32
    diffusion_train_steps: int = 1_000_000
    diffusion_steps: int = 100
    diffusion_beta_schedule: str = "cosine"
    context_drop_prob: float = 0.2
    cfg_scale: float = 1.4
    dynamics_weight: float = 1.0
    reward_weight: float = 1.0
    dynamics_sign: str = "paper_plus"
    guide_scale: float = 1.0
    guide_grad_clip: float | None = None
    temperature: float = 1.0
    hard_initial_state: bool = True
    clip_denoised: float | None = None
    ema_decay: float | None = None
    train_grad_clip: float | None = None
    reward_scale: float = 1.0
    reward_variance: float = 1.0
    dynamics_variance: float = 1.0
    context_windows: int = 1
    episodes_per_task: int = 100
    collection_deterministic: bool = False
    validation_fraction: float = 0.1
    eval_episodes: int = 10
    point_initial_state: str = "origin"
    point_goal_extent: float = 1.0
    cheetah_max_velocity: float = 3.0
    random_param_log_limit: float = 3.0
    backend: str = "corro_legacy"
    checkpoint_every: int = 1000
    cvae_latent_dim: int = 64
    cvae_width: int = 256
    cvae_kl_weight: float = 1.0
    provenance: str = "paper_constraints_with_disclosed_choices"

    def validate(self):
        if self.env not in ENV_SPECS:
            raise ValueError(f"Unknown environment: {self.env}")
        if self.horizon < 2 or self.horizon % 2:
            raise ValueError("The six-block U-Net requires an even horizon >= 2")
        if not 0 <= self.context_drop_prob <= 1:
            raise ValueError("Invalid context dropout probability")
        if self.dynamics_sign not in ("paper_plus", "consistency_minus"):
            raise ValueError("Select an explicit dynamics-sign interpretation")
        if self.predictor not in ("temporal", "mlp"):
            raise ValueError("Invalid predictor")
        if self.diffusion_steps < 2 or min(self.reward_variance, self.dynamics_variance) <= 0:
            raise ValueError("Invalid diffusion steps or Gaussian variances")
        if self.temperature < 0 or self.reward_scale <= 0:
            raise ValueError("Invalid temperature or reward scale")
        if min(self.history, self.pretrain_batch, self.diffusion_batch, self.context_windows,
               self.checkpoint_every, self.train_tasks, self.test_tasks) < 1:
            raise ValueError("Lengths, batch sizes, task counts and intervals must be positive")
        if self.ema_decay is not None and not 0 <= self.ema_decay < 1:
            raise ValueError("EMA decay must be in [0,1)")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("Validation fraction must be in (0,1)")
        if self.point_initial_state not in ("origin", "square"):
            raise ValueError("Unknown initial state distribution")
        return self

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8"))).validate()

    @classmethod
    def for_env(cls, env):
        h, H, limit, train, test = ENV_SPECS[env]
        # Table 7 selected dropout; Table 4 selected lambda. The joint winning
        # configuration is not released: these choices are explicitly inferred.
        drop = dict(point_robot=.3, ant_dir=.2, cheetah_dir=0., cheetah_vel=.3, walker_param=.2, hopper_param=.1)
        weight = dict(point_robot=.5, ant_dir=1., cheetah_dir=.5, cheetah_vel=1., walker_param=2., hopper_param=2.)
        cfg_scale = dict(point_robot=1.2, ant_dir=1.4, cheetah_dir=2., cheetah_vel=1.6, walker_param=1.8, hopper_param=2.)
        return cls(env=env, history=h, horizon=H, episode_limit=limit,
                   train_tasks=train, test_tasks=test, context_drop_prob=drop[env],
                   dynamics_weight=weight[env], cfg_scale=cfg_scale[env],
                   diffusion_steps=50 if env == "hopper_param" else 100)

