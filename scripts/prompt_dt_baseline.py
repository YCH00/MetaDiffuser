import argparse
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metadiffuser.config import Config
from metadiffuser.prompt_baseline import PromptSettings, train, evaluate


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("train", "evaluate"))
    p.add_argument("--config")
    p.add_argument("--settings", help="JSON file overriding original Prompt-DT defaults")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--resume")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--checkpoint")
    p.add_argument("--quality", default="expert", choices=("expert", "medium", "random"))
    p.add_argument("--episodes", type=int)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    if args.command == "train":
        if not args.config:
            p.error("Training requires --config")
        settings = PromptSettings(**json.loads(Path(args.settings).read_text())) if args.settings else PromptSettings()
        train(Config.load(args.config), settings, args.dataset, args.output, args.device, args.resume, args.max_steps)
    else:
        if not args.checkpoint:
            p.error("Evaluation requires --checkpoint")
        evaluate(args.checkpoint, args.dataset, args.output, args.device, args.quality, args.episodes)
