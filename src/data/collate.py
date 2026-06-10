"""Batch collation: left-pad unpadded items to the batch max.

Padding lives only here. The attention mask is derived (0 over padding),
labels pad with IGNORE, and signal positions shift by each sample's own pad.
Train batches round the target length to a multiple of 8 (tensor-core-friendly
shapes), capped at max_seq_len; eval batches keep exact length.
"""
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.templates import IGNORE


class Collator:
    def __init__(self, cfg, pad_token_id):
        self.cfg = cfg
        self.pad_id = pad_token_id

    def __call__(self, items):
        lengths = [it["input_ids"].shape[0] for it in items]
        target = max(lengths)
        if self.cfg.mode == "train":
            target = min(-(-target // 8) * 8, self.cfg.max_seq_len)
        batch = {
            "input_ids": torch.full((len(items), target), self.pad_id, dtype=torch.int64),
            "attention_mask": torch.zeros(len(items), target, dtype=torch.float32),
        }
        if "labels" in items[0]:
            batch["labels"] = torch.full((len(items), target), IGNORE, dtype=torch.int64)
        for i, (item, n) in enumerate(zip(items, lengths)):
            batch["input_ids"][i, target - n:] = item["input_ids"]
            batch["attention_mask"][i, target - n:] = 1.0
            if "labels" in batch:
                batch["labels"][i, target - n:] = item["labels"]
        if "signal_pos" in items[0]:
            pos = torch.stack([it["signal_pos"] for it in items])
            pads = torch.tensor([target - n for n in lengths]).unsqueeze(1)
            batch["signal_pos"] = torch.where(pos >= 0, pos + pads, pos)
        for key in items[0]:
            if key not in ("input_ids", "labels", "signal_pos"):
                batch[key] = torch.stack([it[key] for it in items])
        return batch


def build_dataloader(cfg, dataset):
    sampler = DistributedSampler(dataset, seed=cfg.seed, shuffle=True) if cfg.distributed and cfg.mode == "train" else None
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size if cfg.mode == "train" else 1,
        shuffle=(cfg.mode == "train" and sampler is None),
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=Collator(cfg, dataset.tok.pad_token_id),
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=4 if cfg.num_workers else None,
    )
