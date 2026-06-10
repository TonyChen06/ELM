"""M4 gate: loss decreases on real data; schedule exact under accumulation;
resume restores step/epoch/optimizer; muon builds and steps."""
import sys

sys.path.insert(0, "src")
import pytest
import torch

from config import Config
from data.collate import build_dataloader
from data.dataset import build_dataset
from data.templates import build_tokenizer
from models.elm import build_elm
from train.optim import build_optimizer, build_scheduler
from train.trainer import Trainer

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


def make(tmp_path, **over):
    cfg = Config(data=["ecg-qa-ptbxl-250-2500"], representation="signal", elm="patch_elf",
                 llm="qwen2.5-0.5b-instruct", num_encoder_tokens=100, data_subset=0.001,
                 system_prompt="prompts/system_prompt.txt", batch_size=2, lr=2e-4,
                 run_dir=str(tmp_path), **over)
    cfg.mode = "train"
    tok = build_tokenizer(cfg)
    ds = build_dataset(cfg, tok)
    model = build_elm(cfg, tok).to(DEV)
    return cfg, model, build_dataloader(cfg, ds)


def test_loss_decreases_and_checkpoints(tmp_path):
    cfg, model, dl = make(tmp_path, epochs=3)
    trainer = Trainer(cfg, model, dl)
    state = trainer.fit()
    losses = state["epoch_losses"]
    assert losses[-1] < losses[0], f"no learning: {losses}"
    assert (tmp_path / "best" / "checkpoint.pt").exists()
    assert (tmp_path / "last" / "checkpoint.pt").exists()


def test_resume_restores_progress(tmp_path):
    cfg, model, dl = make(tmp_path, epochs=2)
    t1 = Trainer(cfg, model, dl)
    t1.fit()
    saved_step = t1.state["step"]
    cfg2, model2, dl2 = make(tmp_path, epochs=2)
    cfg2.resume = str(tmp_path / "last")
    t2 = Trainer(cfg2, model2, dl2)
    assert t2.state["step"] == saved_step and t2.state["epoch"] == 2


def test_schedule_counts_optimizer_steps(tmp_path):
    cfg, model, dl = make(tmp_path, epochs=1, grad_accum_steps=4,
                          lr_schedule="cosine", warmup_ratio=0.25, min_lr_ratio=0.1)
    trainer = Trainer(cfg, model, dl)
    import math
    expected_steps = math.ceil(len(dl) / 4)
    assert cfg.max_steps == expected_steps
    mults = []
    for _ in range(cfg.max_steps):
        for opt in trainer.optimizers:
            opt.step()
        for sched in trainer.schedulers:
            sched.step()
        mults.append(trainer.schedulers[0].get_last_lr()[0] / (cfg.lr))
    warmup = int(cfg.max_steps * 0.25)
    assert mults[warmup - 1] == pytest.approx(1.0, abs=1e-6)
    assert mults[-1] == pytest.approx(0.1, abs=0.02)


def test_muon_builds_and_steps(tmp_path):
    cfg, model, dl = make(tmp_path, optimizer="muon", update=["connector"])
    opts = build_optimizer(cfg, model)
    assert len(opts) == 2  # Muon for 2-D weights, AdamW for the rest
    cfg.max_steps = 10
    scheds = build_scheduler(cfg, opts)
    batch = {k: v.to(DEV) for k, v in next(iter(dl)).items()}
    model.train()
    model(**batch).loss.backward()
    for opt in opts:
        opt.step()
    for sched in scheds:
        sched.step()
