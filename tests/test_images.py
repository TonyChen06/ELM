"""M5 gate: rgb and stacked_signal items + full forward through a vision ELM."""
import sys

sys.path.insert(0, "src")
import pytest
import torch

from config import Config
from data.collate import Collator
from data.dataset import build_dataset
from data.templates import build_tokenizer
from models.elm import build_elm

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
BASE = dict(data=["ecg-qa-ptbxl-250-2500"], llm="qwen2.5-0.5b-instruct", elm="mlp_llava",
            encoder="clip", num_encoder_tokens=1, data_subset=0.001,
            system_prompt="prompts/system_prompt.txt")


@pytest.mark.parametrize("rep", ["rgb", "stacked_signal"])
def test_items_and_forward(rep):
    cfg = Config(**BASE, representation=rep)
    cfg.mode = "train"
    tok = build_tokenizer(cfg)
    ds = build_dataset(cfg, tok)
    item = ds[0]
    assert "pixel_values" in item and item["pixel_values"].ndim == 3
    assert "ecg" not in item
    batch = Collator(cfg, tok.pad_token_id)([ds[0], ds[1]])
    assert batch["pixel_values"].shape[0] == 2
    model = build_elm(cfg, tok).to(DEV)
    out = model(**{k: v.to(DEV) for k, v in batch.items()})
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert any(p.grad is not None for p in model.connector.parameters())


def test_vit_encoder_forward():
    cfg = Config(**{**BASE, "encoder": "vit", "elm": "linear_llava"}, representation="stacked_signal")
    cfg.mode = "train"
    tok = build_tokenizer(cfg)
    ds = build_dataset(cfg, tok)
    batch = Collator(cfg, tok.pad_token_id)([ds[0]])
    model = build_elm(cfg, tok).to(DEV)
    out = model(**{k: v.to(DEV) for k, v in batch.items()})
    assert torch.isfinite(out.loss)
