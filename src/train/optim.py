"""Optimizers and LR schedules.

The schedule is counted in optimizer steps (the trainer sets cfg.max_steps =
ceil(batches / grad_accum) * epochs), so warmup_ratio and cosine endpoints mean
what they say regardless of gradient accumulation. Muon takes 2-D non-embedding
weights; everything else falls to AdamW at lr * muon_adamw_lr_ratio.
"""
import math

import torch
from torch.optim.lr_scheduler import LambdaLR

import dist


def _lr_scale(cfg):
    if not cfg.ref_global_bs:
        return 1.0
    effective = cfg.batch_size * cfg.grad_accum_steps * dist.world_size()
    return effective / cfg.ref_global_bs


def _wd(cfg, scale):
    return {"none": cfg.weight_decay,
            "inv_sqrt": cfg.weight_decay / math.sqrt(scale),
            "inv_linear": cfg.weight_decay / scale}[cfg.scale_wd]


def _is_muon_param(name, param):
    flat = name.lower()
    return param.ndim == 2 and not any(s in flat for s in ("emb", "head", "norm", "ln"))


def build_optimizer(cfg, model):
    scale = max(_lr_scale(cfg), 1e-8)
    lr, wd = cfg.lr * scale, _wd(cfg, scale)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    def adamw(params, alr):
        return torch.optim.AdamW(params, lr=alr, betas=tuple(cfg.betas), eps=cfg.eps,
                                 weight_decay=wd, fused=torch.cuda.is_available())

    if cfg.optimizer == "muon":
        from torch.optim import Muon
        muon = [p for n, p in trainable if _is_muon_param(n, p)]
        rest = [p for n, p in trainable if not _is_muon_param(n, p)]
        if dist.is_main():
            print(f"[optim] muon({len(muon)}p lr={lr:.2e}) + adamw({len(rest)}p "
                  f"lr={lr * cfg.muon_adamw_lr_ratio:.2e}) wd={wd:.2e} scale={scale:.3g}")
        return [Muon(muon, lr=lr, momentum=cfg.muon_momentum, nesterov=cfg.muon_nesterov,
                     ns_steps=cfg.muon_ns_steps, weight_decay=wd),
                adamw(rest, lr * cfg.muon_adamw_lr_ratio)]
    if dist.is_main():
        print(f"[optim] adamw({len(trainable)}p) lr={lr:.2e} wd={wd:.2e} scale={scale:.3g}")
    return [adamw([p for _, p in trainable], lr)]


def build_scheduler(cfg, optimizers):
    warmup = int(cfg.max_steps * cfg.warmup_ratio)

    def multiplier(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        if cfg.lr_schedule == "inv_sqrt":
            return 1.0 / math.sqrt(max(1, step - warmup + 1))
        if cfg.lr_schedule == "cosine":
            progress = min((step - warmup) / max(1, cfg.max_steps - warmup), 1.0)
            return cfg.min_lr_ratio + 0.5 * (1 - cfg.min_lr_ratio) * (1 + math.cos(math.pi * progress))
        return 1.0

    return [LambdaLR(opt, multiplier) for opt in optimizers]
