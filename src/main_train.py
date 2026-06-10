import json
import os

import dist
from config import parse_config, set_seed
from data.collate import build_dataloader
from data.dataset import build_dataset
from data.templates import build_tokenizer
from models.elm import build_elm
from train.trainer import Trainer


def main():
    cfg = parse_config(mode="train")
    dist.set_device(cfg.device)
    dist.setup(cfg)
    try:
        import torch
        set_seed(cfg.seed)
        torch.set_float32_matmul_precision("high")
        cfg.run_dir = dist.broadcast_object(next_run_dir(cfg))
        if dist.is_main():
            os.makedirs(cfg.run_dir, exist_ok=True)
            json.dump(vars(cfg), open(os.path.join(cfg.run_dir, "config.json"), "w"), indent=1, default=str)
            print(f"run dir: {cfg.run_dir}")
        tokenizer = build_tokenizer(cfg)
        dataset = build_dataset(cfg, tokenizer)
        dataloader = build_dataloader(cfg, dataset)
        model = dist.parallelize(build_elm(cfg, tokenizer), cfg)
        if cfg.train_phase == "rl":
            from train.rl import rl_step
            trainer = Trainer(cfg, model, dataloader, step_fn=rl_step(cfg, tokenizer))
        else:
            trainer = Trainer(cfg, model, dataloader)
        state = trainer.fit()
        if dist.is_main():
            open(os.path.join(cfg.run_dir, "DONE"), "w").close()
            print(f"done: {state['epoch']} epochs, {state['step']} steps, best {state['best_loss']:.4f}")
    finally:
        teardown_wandb(cfg)
        dist.teardown()


def next_run_dir(cfg):
    """runs/<elm>_<llm>_<encoder>/<data>/<n>, n auto-incremented (rank 0 picks).
    An explicitly passed --run_dir is used as-is."""
    if cfg.run_dir != "runs":
        return cfg.run_dir if dist.is_main() else None
    base = os.path.join(cfg.run_dir, f"{cfg.elm}_{cfg.llm}" + (f"_{cfg.encoder}" if cfg.encoder else ""),
                        "_".join(cfg.data))
    if not dist.is_main():
        return None
    os.makedirs(base, exist_ok=True)
    taken = [int(d) for d in os.listdir(base) if d.isdigit()]
    return os.path.join(base, str(max(taken) + 1 if taken else 0))


def teardown_wandb(cfg):
    if cfg.wandb and dist.is_main():
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
