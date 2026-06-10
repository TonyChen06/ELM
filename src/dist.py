"""Distributed runtime: env-driven init, the parallelize() seam, rank helpers.

Single-process runs need no setup; torchrun presence is auto-detected.
parallelize() is the one place a model meets a parallelism strategy — DDP now,
FSDP2 here too (M9), tensor parallel later without touching callers.
"""
import os
from datetime import timedelta

import torch
import torch.distributed as dist


def setup(cfg):
    if cfg.distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
        torch.cuda.set_device(local_rank())


def teardown():
    if dist.is_initialized():
        dist.destroy_process_group()


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main() -> bool:
    return rank() == 0


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank()}")
    return torch.device("cpu")


def all_gather_object(obj):
    if world_size() == 1:
        return [obj]
    out = [None] * world_size()
    dist.all_gather_object(out, obj)
    return out


def parallelize(model, cfg):
    model = model.to(device())
    if cfg.compile:
        model = torch.compile(model)
    if not cfg.distributed:
        return model
    if cfg.parallel == "fsdp":
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
        policy = MixedPrecisionPolicy(param_dtype=getattr(torch, cfg.param_dtype),
                                      reduce_dtype=getattr(torch, cfg.reduce_dtype))
        for layer in model.llm.model.layers:
            fully_shard(layer, mp_policy=policy)
        fully_shard(model, mp_policy=policy)
        return model
    return torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank()],
        find_unused_parameters=(model.encoder is not None and "encoder" not in model.update))


def unwrap(model):
    model = getattr(model, "_orig_mod", model)
    return getattr(model, "module", model)
