"""Dataset: lazy multi-source HF rows -> representation strategy -> rendered item.

Items are unpadded; the collator pads to batch max. An item is:
  input_ids  int64 [L]
  labels     int64 [L]            (train phases; IGNORE outside assistant spans)
  signal_pos int64 [num_encoder_tokens]   (where to inject embeddings; absent for ecg_byte)
  ecg        float32 [leads, segment_len] (signal-derived reps; vision reps emit processor tensors)
"""
import bisect
import json
import random

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from data import templates
from registry import ECG_TOKEN_PREFIX, ELMS, HF_DATASET_ORG, SIGNAL_TOKEN


def load_rows(cfg):
    """The mixed datasets, kept lazy (no materialization)."""
    split = f"fold{cfg.fold}_{'train' if cfg.mode == 'train' else 'test'}"
    shards = []
    for name in cfg.data:
        ds = load_dataset(f"{HF_DATASET_ORG}/{name}", split=split)
        if cfg.data_subset:
            ds = ds.shuffle(seed=cfg.seed).select(range(int(len(ds) * cfg.data_subset)))
        shards.append(ds)
    return shards


def normalize_signal(x, eps):
    lo, hi = x.min(), x.max()
    return np.clip((x - lo) / (hi - lo + eps), 0.0, 1.0)


class ElmDataset(Dataset):
    def __init__(self, cfg, tokenizer, representation):
        self.cfg = cfg
        self.tok = tokenizer
        self.rep = representation
        self.shards = load_rows(cfg)
        self.offsets = np.cumsum([len(s) for s in self.shards]).tolist()
        self.system_prompt = open(cfg.system_prompt, encoding="utf-8").read() if cfg.system_prompt else None
        self.signal_id = tokenizer.convert_tokens_to_ids(SIGNAL_TOKEN)

    def __len__(self):
        return self.offsets[-1] if self.offsets else 0

    def _row(self, index):
        shard = bisect.bisect_right(self.offsets, index)
        row = self.shards[shard][index - (self.offsets[shard - 1] if shard else 0)]
        text = row["text"]
        if isinstance(text, str):
            try:
                text = json.loads(text)
            except ValueError:
                text = [{"role": "user", "content": text}]
        return row, text

    def load_signal(self, row):
        cfg = self.cfg
        if row["ecg_path"] == "noise" or cfg.perturb == "noise":
            ecg = np.random.randn(len(cfg.leads), cfg.segment_len)
        elif row["ecg_path"] == "flatline" or cfg.perturb == "zeros":
            ecg = np.full((len(cfg.leads), cfg.segment_len), float(np.random.choice(10)))
        else:
            payload = np.load(row["ecg_path"], allow_pickle=True).item()
            ecg = payload["ecg"][cfg.leads]
            if cfg.augment_ecg and cfg.mode == "train":
                ecg = self._augment(ecg)
        return normalize_signal(ecg.astype(np.float32), cfg.norm_eps)

    def _augment(self, signal):
        if random.random() < 0.5:
            signal = signal + np.random.normal(0, 0.05 * signal.std(), signal.shape)
            if random.random() < 0.5:
                wander = 0.07 * np.abs(signal).max() * np.sin(
                    np.linspace(0, random.randint(1, 5) * np.pi, signal.shape[1]))
                signal = signal + wander
        return signal

    def __getitem__(self, index):
        cfg = self.cfg
        row, text = self._row(index)
        messages = templates.to_messages(text, cfg, self.system_prompt)
        ids, labels = templates.render_train(self.tok, messages, cfg)
        ids, labels = self.rep.splice(self, row, ids, labels)
        ids, labels = self._truncate(ids, labels)
        item = {"input_ids": torch.tensor(ids, dtype=torch.int64),
                "labels": torch.tensor(labels, dtype=torch.int64)}
        if ELMS[cfg.elm].uses_signal_tokens:
            pos = [i for i, t in enumerate(ids) if t == self.signal_id]
            item["signal_pos"] = torch.tensor(pos or [-1] * cfg.num_encoder_tokens, dtype=torch.int64)
        item.update(self.rep.features(self, row))
        return item

    def _truncate(self, ids, labels):
        """Cap at max_seq_len, dropping unlabeled tokens first, then labeled
        ones, never the pre-signal prompt, signal tokens, or bos."""
        overflow = len(ids) - self.cfg.max_seq_len
        if overflow <= 0:
            return ids, labels
        protected = {self.signal_id, self.tok.bos_token_id}
        first_signal = next((i for i, t in enumerate(ids) if t == self.signal_id), len(ids))

        def priority(i):
            if i < first_signal:
                return 2
            return 1 if labels[i] != templates.IGNORE else 0

        droppable = sorted((i for i, t in enumerate(ids) if t not in protected), key=priority)
        drop = set(droppable[:overflow])
        kept = [(t, lab) for i, (t, lab) in enumerate(zip(ids, labels)) if i not in drop]
        kept = kept[-self.cfg.max_seq_len:]
        return [t for t, _ in kept], [lab for _, lab in kept]


# ---------------------------------------------------------------------------
# Representation strategies. splice() may rewrite token ids (symbolic);
# features() supplies the model-side tensors.
# ---------------------------------------------------------------------------
class Signal:
    def splice(self, ds, row, ids, labels):
        return ids, labels

    def features(self, ds, row):
        ecg = ds.load_signal(row)
        if ds.cfg.elm == "base_elf":
            ecg = ecg.reshape(-1)
        return {"ecg": torch.from_numpy(ecg)}


class Symbolic:
    """ECG-Byte: BPE-compressed signal tokens replace the <signal> placeholder."""

    def __init__(self):
        self._bpe = None

    def splice(self, ds, row, ids, labels):
        from data.bpe_tokenizer import EcgByte
        if ds.signal_id not in ids:  # perturb=only_text
            return ids, labels
        if self._bpe is None:
            self._bpe = EcgByte(ds.cfg.ecg_tokenizer)
        ecg_tokens = self._bpe.encode(ds.load_signal(row))
        ecg_ids = ds.tok.convert_tokens_to_ids([f"{ECG_TOKEN_PREFIX}{t}" for t in ecg_tokens])
        cap, floor = ds.cfg.max_seq_len, ds.cfg.min_ecg_tokens
        budget = cap - (len(ids) - 1)
        if len(ecg_ids) > max(budget, floor):
            ecg_ids = ecg_ids[: max(budget, floor)]
        at = ids.index(ds.signal_id)
        out_ids = ids[:at] + ecg_ids + ids[at + 1:]
        out_labels = labels[:at] + [templates.IGNORE] * len(ecg_ids) + labels[at + 1:]
        keep_through = at + len(ecg_ids)  # the min-ECG floor may overflow the cap: trim the tail text
        if len(out_ids) > cap:
            if keep_through > cap:
                raise ValueError("prompt prefix + min_ecg_tokens exceeds max_seq_len")
            out_ids, out_labels = out_ids[:cap], out_labels[:cap]
        return out_ids, out_labels

    def features(self, ds, row):
        return {}


def build_dataset(cfg, tokenizer):
    reps = {"signal": Signal, "symbolic": Symbolic}
    if cfg.representation in ("rgb", "stacked_signal"):
        from data.images import RGB, StackedSignal
        reps |= {"rgb": RGB, "stacked_signal": StackedSignal}
    return ElmDataset(cfg, tokenizer, reps[cfg.representation]())
