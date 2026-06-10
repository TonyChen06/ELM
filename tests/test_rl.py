"""M7 gate: SAPO step on real data — finite loss, gradients flow, rewards sane."""
import sys

sys.path.insert(0, "src")
import torch

from config import Config
from data.collate import Collator
from data.dataset import build_dataset
from data.templates import build_tokenizer
from models.elm import build_elm
from train.rl import reward, rl_step

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


def test_rewards():
    good = "<think>\nbecause</think><answer>afib</answer>"
    assert reward(good, "<answer>afib</answer>", explicit_thinking=False) == 3.0
    assert reward("nonsense", "<answer>afib</answer>", explicit_thinking=False) < 1.0


def test_sapo_step_real():
    cfg = Config(data=["ecg-qa-ptbxl-250-2500"], representation="signal", elm="patch_elf",
                 llm="qwen2.5-0.5b-instruct", num_encoder_tokens=100, data_subset=0.001,
                 system_prompt="prompts/system_prompt.txt", train_phase="rl",
                 rl_group_size=2, rl_max_new_tokens=16)
    cfg.mode = "train"
    tok = build_tokenizer(cfg)
    ds = build_dataset(cfg, tok)
    model = build_elm(cfg, tok).to(DEV)
    batch = {k: v.to(DEV) for k, v in Collator(cfg, tok.pad_token_id)([ds[0]]).items()}
    model.train()
    step = rl_step(cfg, tok)
    loss, metrics = step(model, batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.connector.parameters())
    assert 0.0 <= metrics["reward"] <= 3.0
