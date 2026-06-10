# ELM — training and evaluation for ECG-language models

A ground-up rewrite of the ELM framework: every feature of the original in
~1/3 of the lines, plus FSDP2, collective checkpointing, mixed-precision
policy, MFU telemetry, and first-class `torch.compile`. See `PLAN.md` for the
architecture and milestone gates; `tests/` holds the gate suite.

## Setup

```bash
uv sync
# symbolic representation needs the Rust tokenizer once:
cd src/data/bpe && maturin develop --release
```

Data: preprocessed `.npy` files live in `../data/` next to this repo
(`ecg_path` fields in the HF datasets are relative to the repo root).

## Train

```bash
# single GPU
uv run src/main_train.py \
  --data ecg-qa-ptbxl-250-2500 --representation signal \
  --llm qwen2.5-1.5b-instruct --elm mlp_llava --encoder st_mem \
  --num_encoder_tokens 50 --update connector llm --lr 1e-4

# multi-GPU (DDP or FSDP2) — distributed mode is auto-detected
torchrun --standalone --nproc_per_node=4 src/main_train.py ... --parallel fsdp

# phases: --train_phase pretrain | sft | rl   (rl uses SAPO; see --rl_* flags)
```

Every CLI flag is a field of `Config` in `src/config.py` — that file is the
reference. Components (LLMs, encoders, ELM compositions, representations) are
declared in `src/registry.py`; adding one is one entry.

## Evaluate / chat

```bash
[torchrun ...] uv run src/main_eval.py --data ... --ckpt runs/best \
  --eval_batch_size 8                       # batched + distributed N-turn eval
uv run src/main_chat.py --ckpt runs/best --elm patch_elf --num_encoder_tokens 100
```

## Layout

```
src/config.py      one typed config; CLI generated from its fields
src/registry.py    all pluggable components
src/data/          templates (HF chat templates + masking), dataset (lazy,
                   strategy-per-representation), collate (all padding lives here)
src/models/        elm.py (one composed module), encoders/, connectors.py
src/train/         trainer.py, optim.py (AdamW | Muon+AdamW), rl.py (SAPO)
src/dist.py        mesh/DDP/FSDP2 seam, rank helpers
src/evaluate.py    batched + distributed eval, all metrics
```
