import argparse
import json
from pathlib import Path
import torch
from .collect import collect_dataset
from .config import Config, ENV_SPECS
from .envs import make_tasks
from .evaluate import aggregate, evaluate
from .runtime import json_write, system_info
from .sac import SACConfig, train_expert
from .train import pretrain, train_diffusion


def main():
    parser = argparse.ArgumentParser(description="Auditable MetaDiffuser reproduction")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=4)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    p = sub.add_parser("init-configs")
    p.add_argument("--output", default="configs")
    p = sub.add_parser("tasks")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("expert")
    p.add_argument("--config", required=True)
    p.add_argument("--tasks", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--task-id")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--collector", choices=("corro", "independent"), default="corro")
    p.add_argument("--iterations", type=int, help="CORRO iteration budget; default follows its released config")
    p = sub.add_parser("collect")
    p.add_argument("--config", required=True)
    p.add_argument("--tasks")
    p.add_argument("--output", required=True)
    p.add_argument("--policy-root")
    p.add_argument("--controller-smoke", action="store_true")
    for command in ("pretrain", "train", "train-cvae"):
        p = sub.add_parser(command)
        p.add_argument("--config", required=True)
        p.add_argument("--dataset", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--max-steps", type=int)
        p.add_argument("--resume")
        if command in ("train", "train-cvae"):
            p.add_argument("--pretrained", required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--quality", choices=("expert", "medium", "random"), default="expert")
    p.add_argument("--episodes", type=int)
    p.add_argument("--overrides", default="{}", help="JSON object of sampling ablations")
    p = sub.add_parser("aggregate")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--output", required=True)
    p = sub.add_parser("import-corro")
    p.add_argument("--index", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.command == "doctor":
        print(json.dumps(system_info(), indent=2))
        return
    if args.command == "init-configs":
        for env in ENV_SPECS:
            Config.for_env(env).save(Path(args.output) / f"{env}.json")
        return
    cfg = Config.load(args.config) if hasattr(args, "config") else None
    if args.command == "tasks":
        json_write(args.output, make_tasks(cfg))
    elif args.command == "expert":
        tasks = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
        chosen = [t for t in tasks if args.task_id is None or t["task_id"] == args.task_id]
        if not chosen:
            raise ValueError("Unknown task ID")
        for task in chosen:
            if args.collector == "corro":
                from .corro_collector import train_corro_expert
                train_corro_expert(cfg, task, Path(args.output) / task["task_id"], args.device, args.iterations)
            else:
                train_expert(cfg, task, Path(args.output) / task["task_id"], args.device,
                             SACConfig(steps=args.steps, eval_every=args.eval_every))
    elif args.command == "collect":
        tasks = json.loads(Path(args.tasks).read_text(encoding="utf-8")) if args.tasks else None
        collect_dataset(cfg, args.output, tasks, args.policy_root, args.controller_smoke, args.device)
    elif args.command == "pretrain":
        pretrain(cfg, args.dataset, args.output, args.device, args.max_steps, args.resume)
    elif args.command == "train":
        train_diffusion(cfg, args.dataset, args.pretrained, args.output, args.device, args.max_steps, args.resume)
    elif args.command == "train-cvae":
        from .train_cvae import train_cvae
        train_cvae(cfg, args.dataset, args.pretrained, args.output, args.device, args.max_steps, args.resume)
    elif args.command == "evaluate":
        evaluate(args.checkpoint, args.dataset, args.output, args.device, args.quality,
                 args.episodes, json.loads(args.overrides))
    elif args.command == "aggregate":
        print(aggregate(args.inputs, args.output))
    elif args.command == "import-corro":
        from .import_data import import_corro
        import_corro(args.index, args.output)


if __name__ == "__main__":
    main()
