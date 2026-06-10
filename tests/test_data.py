"""M2 gate: dataset items and collated batches, real data, signal representation."""
import sys

sys.path.insert(0, "src")
import pytest
import torch

from config import Config
from data.collate import Collator, build_dataloader
from data.dataset import build_dataset
from data.templates import IGNORE, build_tokenizer

CFG = dict(data=["ecg-qa-ptbxl-250-2500"], representation="signal", elm="patch_elf",
           llm="qwen2.5-0.5b-instruct", num_encoder_tokens=100, data_subset=0.002,
           system_prompt="prompts/system_prompt.txt")


@pytest.fixture(scope="module")
def ds():
    cfg = Config(**CFG)
    cfg.mode = "train"
    return build_dataset(cfg, build_tokenizer(cfg))


def test_items(ds):
    for i in range(4):
        item = ds[i]
        n = item["input_ids"].shape[0]
        assert item["labels"].shape[0] == n and n <= ds.cfg.max_seq_len
        assert item["ecg"].dtype == torch.float32
        assert item["ecg"].shape == (12, 2500)
        assert 0.0 <= item["ecg"].min() and item["ecg"].max() <= 1.0
        pos = item["signal_pos"]
        assert pos.shape[0] == 100
        assert (item["input_ids"][pos] == ds.signal_id).all()
        assert (item["labels"][pos] == IGNORE).all()


def test_mixing_is_lazy_and_additive():
    cfg = Config(**{**CFG, "data": ["ecg-qa-ptbxl-250-2500", "ecg-qa-ptbxl-250-2500"]})
    cfg.mode = "train"
    twice = build_dataset(cfg, build_tokenizer(cfg))
    one = Config(**CFG); one.mode = "train"
    single = build_dataset(one, build_tokenizer(one))
    assert len(twice) == 2 * len(single)
    assert torch.equal(twice[len(single)]["input_ids"], single[0]["input_ids"])


def test_collate(ds):
    items = [ds[i] for i in range(4)]
    batch = Collator(ds.cfg, ds.tok.pad_token_id)(items)
    n = batch["input_ids"].shape[1]
    assert n % 8 == 0 and n <= ds.cfg.max_seq_len
    for i, item in enumerate(items):
        pad = n - item["input_ids"].shape[0]
        assert (batch["input_ids"][i, :pad] == ds.tok.pad_token_id).all()
        assert (batch["attention_mask"][i, :pad] == 0).all() and batch["attention_mask"][i, pad:].all()
        assert (batch["labels"][i, :pad] == IGNORE).all()
        assert torch.equal(batch["labels"][i, pad:], item["labels"])
        assert torch.equal(batch["signal_pos"][i], item["signal_pos"] + pad)
    assert batch["ecg"].shape == (4, 12, 2500)


def test_truncation_preserves_signal_tokens():
    cfg = Config(**{**CFG, "max_seq_len": 256})
    cfg.mode = "train"
    small = build_dataset(cfg, build_tokenizer(cfg))
    item = small[0]
    assert item["input_ids"].shape[0] <= 256
    assert (item["input_ids"] == small.signal_id).sum() == 100


def test_perturb_only_text():
    cfg = Config(**{**CFG, "perturb": "only_text"})
    cfg.mode = "train"
    ds2 = build_dataset(cfg, build_tokenizer(cfg))
    item = ds2[0]
    assert (item["input_ids"] == ds2.signal_id).sum() == 0
    assert (item["signal_pos"] == -1).all()


def test_dataloader_smoke(ds):
    dl = build_dataloader(ds.cfg, ds)
    batch = next(iter(dl))
    assert batch["input_ids"].shape[0] == 1
