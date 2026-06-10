"""M3 gate: every ELM composition — forward loss, backward grads honoring
--update, embedding injection at signal positions, generation. Real data,
GPU if available."""
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
BASE = dict(data=["ecg-qa-ptbxl-250-2500"], representation="signal", llm="qwen2.5-0.5b-instruct",
            data_subset=0.002, system_prompt="prompts/system_prompt.txt")

COMPOSITIONS = [
    ("patch_elf", "", 100),
    ("base_elf", "", 1),
    ("conv_elf", "", 100),
    ("mlp_llava", "st_mem", 50),
    ("linear_llava", "mtae", 50),
    ("mlp_llava", "mlae", 50),
    ("linear_llava", "merl", 4),
]


def make(elm, encoder, n_tokens, **over):
    cfg = Config(**BASE, elm=elm, encoder=encoder, num_encoder_tokens=n_tokens, **over)
    cfg.mode = "train"
    tok = build_tokenizer(cfg)
    ds = build_dataset(cfg, tok)
    batch = Collator(cfg, tok.pad_token_id)([ds[0], ds[1]])
    model = build_elm(cfg, tok).to(DEV)
    return cfg, model, {k: v.to(DEV) for k, v in batch.items()}


@pytest.mark.parametrize("elm,encoder,n", COMPOSITIONS)
def test_forward_backward_update_and_injection(elm, encoder, n):
    cfg, model, batch = make(elm, encoder, n)
    model.train()
    out = model(**batch)
    assert torch.isfinite(out.loss)

    # injection: embeddings at signal positions equal connector output
    feats = {k: v for k, v in batch.items() if k not in ("input_ids", "attention_mask", "labels", "signal_pos")}
    embeds = model.embeddings(batch["input_ids"], batch["signal_pos"], **feats)
    base = model.llm.get_input_embeddings()(batch["input_ids"])
    pos = batch["signal_pos"]
    assert not torch.equal(embeds[0, pos[0]], base[0, pos[0]])
    off_pos = torch.ones(embeds.shape[1], dtype=torch.bool, device=DEV)
    off_pos[pos[0]] = False
    assert torch.equal(embeds[0, off_pos], base[0, off_pos])

    out.loss.backward()
    for name, module in model.parts():
        wants_grad = name in model.update
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
        params_trainable = any(p.requires_grad for p in module.parameters())
        assert params_trainable == wants_grad, f"{name}: requires_grad mismatch"
        if wants_grad:
            assert has_grad, f"{name}: expected gradients"


def test_update_encoder_gets_grads():
    cfg, model, batch = make("mlp_llava", "st_mem", 50, update=["encoder", "connector"])
    model.train()
    model(**batch).loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    assert all(not p.requires_grad for p in model.llm.parameters())


def test_ecg_byte_plain_token_path():
    cfg = Config(**{**BASE, "representation": "symbolic"}, elm="ecg_byte", num_encoder_tokens=1)
    cfg.mode = "train"
    tok = build_tokenizer(cfg)
    ds = build_dataset(cfg, tok)
    batch = Collator(cfg, tok.pad_token_id)([ds[0]])
    model = build_elm(cfg, tok).to(DEV)
    out = model(**{k: v.to(DEV) for k, v in batch.items()})
    assert torch.isfinite(out.loss)


def test_generate():
    cfg, model, batch = make("patch_elf", "", 100)
    model.eval()
    out = model.generate(batch["input_ids"], batch["attention_mask"],
                         signal_pos=batch["signal_pos"], ecg=batch["ecg"], max_new_tokens=8)
    assert out.shape[0] == 2 and out.shape[1] <= 8


def test_peft_and_grad_ckpt_build():
    cfg, model, batch = make("patch_elf", "", 100, peft=True, gradient_checkpointing=True)
    model.train()
    out = model(**batch)
    out.loss.backward()
    assert torch.isfinite(out.loss)
