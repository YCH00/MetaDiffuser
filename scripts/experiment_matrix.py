"""Generate reviewable five-seed commands; execute sequentially with --execute.

Data collection is deliberately a prerequisite, not silently substituted by
controller data. Model variants change only one factor at a time.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metadiffuser.config import Config, ENV_SPECS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=ENV_SPECS, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--training-ablations", action="store_true")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    variants = [("main", {})]
    if args.training_ablations:
        variants += [(f"K_{k}", {"diffusion_steps": k}) for k in (20, 50, 100)]
        variants += [(f"drop_{p}", {"context_drop_prob": p}) for p in (0., .1, .2, .3, 1.)]
    sampling = [("main", {}), ("sign_minus", {"dynamics_sign": "consistency_minus"}),
                ("no_dynamics", {"dynamics_weight": 0.}), ("no_reward", {"reward_weight": 0.}),
                ("no_dual_guide", {"guide_scale": 0.}), ("zero_context", {"context_mode": "zero"}),
                ("shuffle_context", {"context_mode": "shuffled"})]
    commands = []
    base = [sys.executable, "-m", "metadiffuser.cli", "--device", args.device]
    for seed in args.seeds:
        for name, changes in variants:
            cfg = Config.for_env(args.env)
            cfg.seed = seed
            for key, value in changes.items():
                setattr(cfg, key, value)
            run = root / f"seed_{seed}" / name
            config = root / "configs" / f"seed_{seed}_{name}.json"
            cfg.save(config)
            # Pretraining is independent of diffusion K/dropout; the trainer checks
            # architecture compatibility but does not require matching these fields.
            pre = root / f"seed_{seed}" / "pretrain"
            if name == "main":
                commands.append(base + ["pretrain", "--config", str(config), "--dataset", args.dataset, "--output", str(pre)])
            diffusion = run / "diffusion"
            commands.append(base + ["train", "--config", str(config), "--dataset", args.dataset,
                                     "--pretrained", str(pre / "latest.pt"), "--output", str(diffusion)])
            for quality in ("expert", "medium", "random") if name == "main" else ("expert",):
                for sample_name, overrides in sampling if name == "main" else [("main", {})]:
                    commands.append(base + ["evaluate", "--checkpoint", str(diffusion / "latest.pt"),
                                             "--dataset", args.dataset, "--output", str(run / f"eval_{quality}_{sample_name}"),
                                             "--quality", quality, "--overrides", json.dumps(overrides)])
    (root / "commands.json").write_text(json.dumps(commands, indent=2), encoding="utf-8")
    print(f"Wrote {len(commands)} commands to {root / 'commands.json'}")
    if args.execute:
        manifest = json.loads((Path(args.dataset) / "manifest.json").read_text())
        if manifest["source"] == "controller_smoke_only":
            raise ValueError("Refusing to label controller smoke data as the formal experiment matrix")
        for command in commands:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
