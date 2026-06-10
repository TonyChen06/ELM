"""The training loop: sft and pretrain share the loss step; RL plugs in its own
step function (train/rl.py). Handles gradient accumulation, clipping, LR
scheduling in optimizer steps, wandb, tokens/sec + MFU telemetry, epoch/step/
best checkpoints, early stopping, and resume."""
import math
import os
import time

import torch
from tqdm import tqdm

import dist
from train.optim import build_optimizer, build_scheduler


class Trainer:
    def __init__(self, cfg, model, dataloader, step_fn=None):
        self.cfg = cfg
        self.model = model
        self.dataloader = dataloader
        self.step_fn = step_fn or self.loss_step
        steps_per_epoch = math.ceil(len(dataloader) / cfg.grad_accum_steps)
        cfg.max_steps = steps_per_epoch * cfg.epochs
        self.optimizers = build_optimizer(cfg, model)
        self.schedulers = build_scheduler(cfg, self.optimizers)
        self.flops_per_token = 6 * sum(p.numel() for p in dist.unwrap(model).parameters())
        self.state = {"epoch": 0, "step": 0, "best_loss": float("inf"), "epoch_losses": []}
        if cfg.wandb and dist.is_main():
            import wandb
            wandb.init(project="elm", config=vars(cfg))
        if cfg.resume:
            self.load(cfg.resume)

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

    @staticmethod
    def loss_step(model, batch):
        return model(**batch).loss, {}

    def fit(self):
        cfg = self.cfg
        device = dist.device()
        for epoch in range(self.state["epoch"], cfg.epochs):
            if cfg.distributed and hasattr(self.dataloader.sampler, "set_epoch"):
                self.dataloader.sampler.set_epoch(epoch)
            self.model.train()
            running, n_batches, tokens, t0 = 0.0, 0, 0, time.perf_counter()
            progress = tqdm(self.dataloader, desc=f"epoch {epoch}", disable=not dist.is_main(), leave=False)
            self.zero_grad()
            for i, batch in enumerate(progress):
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                loss, metrics = self.step_fn(self.model, batch)
                (loss / cfg.grad_accum_steps).backward()
                running += loss.item()
                n_batches += 1
                tokens += int(batch["attention_mask"].sum().item())
                if (i + 1) % cfg.grad_accum_steps == 0 or (i + 1) == len(self.dataloader):
                    if cfg.grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                    for opt in self.optimizers:
                        opt.step()
                    for sched in self.schedulers:
                        sched.step()
                    self.zero_grad()
                    self.state["step"] += 1
                    self.log(loss=running / n_batches, lr=self.schedulers[0].get_last_lr()[0],
                             tokens_per_s=tokens / (time.perf_counter() - t0),
                             mfu=self.mfu(tokens, time.perf_counter() - t0), **metrics)
                    if cfg.save_steps and self.state["step"] % cfg.save_steps == 0:
                        self.save("step")
                if cfg.dev and i >= 1:
                    break
            epoch_loss = running / max(n_batches, 1)
            self.state["epoch"] = epoch + 1
            self.state["epoch_losses"].append(epoch_loss)
            if dist.is_main():
                if epoch_loss < self.state["best_loss"]:
                    self.state["best_loss"] = epoch_loss
                    self.save("best")
                self.save("last")
            if self.should_stop():
                if dist.is_main():
                    print(f"early stop at epoch {epoch} (loss {epoch_loss:.4f})")
                break
        return self.state

    def mfu(self, tokens, seconds):
        peak = 312e12 if "A100" in torch.cuda.get_device_name() else 165e12  # bf16 peak, default A6000-class
        return self.flops_per_token * tokens / seconds / (peak * dist.world_size()) if seconds else 0.0

    def should_stop(self):
        cfg, losses = self.cfg, self.state["epoch_losses"]
        if not cfg.early_stopping or len(losses) < cfg.patience + 1:
            return self._bcast(False)
        stop = min(losses[-cfg.patience:]) > min(losses[:-cfg.patience]) - cfg.patience_delta
        return self._bcast(stop)

    def _bcast(self, flag):
        if not self.cfg.distributed:
            return flag
        t = torch.tensor(int(flag), device=dist.device())
        torch.distributed.broadcast(t, src=0)
        return bool(t.item())

    def log(self, **metrics):
        if self.cfg.wandb and dist.is_main():
            import wandb
            wandb.log({f"train/{k}": v for k, v in metrics.items()}, step=self.state["step"])

    def save(self, tag):
        path = os.path.join(self.cfg.run_dir, tag)
        os.makedirs(path, exist_ok=True)
        torch.save({"model": dist.unwrap(self.model).state_dict(),
                    "optimizers": [o.state_dict() for o in self.optimizers],
                    "schedulers": [s.state_dict() for s in self.schedulers],
                    "state": self.state, "config": vars(self.cfg)},
                   os.path.join(path, "checkpoint.pt"))

    def load(self, path):
        payload = torch.load(os.path.join(path, "checkpoint.pt"), map_location="cpu", weights_only=False)
        dist.unwrap(self.model).load_state_dict(payload["model"])
        for opt, st in zip(self.optimizers, payload["optimizers"]):
            opt.load_state_dict(st)
        for sched, st in zip(self.schedulers, payload["schedulers"]):
            sched.load_state_dict(st)
        self.state = payload["state"]
        if dist.is_main():
            print(f"resumed from {path} at epoch {self.state['epoch']} step {self.state['step']}")
