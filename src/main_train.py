import dist
from config import parse_config
from data.collate import build_dataloader
from data.dataset import build_dataset
from data.templates import build_tokenizer
from models.elm import build_elm
from train.trainer import Trainer


def main():
    cfg = parse_config(mode="train")
    dist.setup(cfg)
    try:
        import torch
        torch.manual_seed(cfg.seed)
        torch.set_float32_matmul_precision("high")
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
            print(f"done: {state['epoch']} epochs, {state['step']} steps, best {state['best_loss']:.4f}")
    finally:
        teardown_wandb(cfg)
        dist.teardown()


def teardown_wandb(cfg):
    if cfg.wandb and dist.is_main():
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
