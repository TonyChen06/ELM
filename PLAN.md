# tonytry — ELM rebuilt from scratch

Ground-up rewrite of the ELM training/eval framework. Goals, in priority order:

1. **Every feature of the existing framework works** (parity matrix below).
2. **Plus a Megatron-class feature subset**: FSDP2 (DeviceMesh + `fully_shard`),
   distributed checkpointing (`torch.distributed.checkpoint`), explicit
   mixed-precision policy, tokens/sec + MFU telemetry, first-class
   `torch.compile`. Tensor parallel and sequence packing are deferred but the
   `parallelize(model, mesh)` seam is designed for them.
3. **Fewer total lines than the old framework** (old `src/`: 6,098 Python lines;
   target ≤ 4,200, measured the same way, `tests/` excluded on both sides).

No backward compatibility is kept (prompts, checkpoints, CLI) — every choice is
free to be the best one.

## Architecture

```
src/
  config.py        one typed dataclass; CLI auto-generated from its fields
  registry.py      LLMS / ENCODERS / CONNECTORS / ELMS / REPRESENTATIONS
  data/
    templates.py   HF chat templates + incremental masking (sft/pretrain/rl)
    dataset.py     one Dataset; representations are small strategy objects
    collate.py     dynamic padding; eval turn-flattening
    bpe/           ECG-Byte Rust tokenizer (unchanged crate)
  models/
    elm.py         one composed module: [encoder] -> [connector] -> HF LM
    encoders/      st_mem / mtae / mlae (shared ViT), merl, hf vision wrapper
    connectors.py  linear / mlp / patch / cnn-patch
  train/
    trainer.py     one loop; sft & pretrain are a loss step, RL is a step strategy
    optim.py       AdamW / Muon+AdamW; warmup + constant/cosine/inv_sqrt
    rl.py          SAPO loss, group rollouts, rewards
  dist.py          mesh, parallelize() (ddp|fsdp), DCP save/load
  evaluate.py      batched + distributed N-turn eval; all metrics
  main_train.py / main_eval.py / main_chat.py
```

## Parity matrix

- [ ] Data: HF datasets + mixing, folds, subsets; signal / symbolic / stacked_signal / rgb;
      noise/flatline synthetics; perturb noise/zeros/only_text; ECG + image augmentation;
      truncation preserving signal tokens
- [ ] Text: chat templates (llama3/qwen2.5/gemma2), label masking for sft/pretrain/rl,
      explicit_thinking, multi-turn
- [ ] Models: 3 LLM families (+LoRA, scratch, grad ckpt); merl/st_mem/mtae/mlae +
      clip/siglip/vit encoders; 4 connectors; llava×2 / elf×3 / ecg_byte; component freezing
- [ ] Train: sft/pretrain loop, grad accum, clip, early stopping, wandb,
      epoch/step/best/resume ckpts; AdamW & Muon; LR schedules in optimizer steps
- [ ] RL: SAPO, rollouts, format/tag/answer rewards, degenerate-group DDP safety
- [ ] Eval: batched + distributed; ACC/F1/BLEU/ROUGE/METEOR; thinking split;
      classification + confusion plots; pretrain breakdown; multi-seed stats
- [ ] Chat CLI
- [ ] Megatron subset: FSDP2, DCP, mixed precision, MFU logging, compile

## Milestones (gate each before the next)

| M | Deliverable | Gate |
|---|---|---|
| 0 | skeleton: branch, pyproject, config, registry | imports; config round-trips |
| 1 | templates + masking | masks cover exactly assistant spans, all families/phases, real data |
| 2 | data pipeline (signal) + collator | item/batch invariants; padding exactness |
| 3 | models + forward | every ELM type forward/backward on GPU |
| 4 | trainer + optim + ckpt | loss decreases on real data; schedule exact; resume works |
| 5 | rgb / stacked / symbolic | per-rep item checks |
| 6 | eval | end-to-end run; metrics computed; batched=unbatched (greedy, bf16 envelope) |
| 7 | RL | SAPO step runs, finite loss, DDP-safe |
| 8 | chat | works, incl. no-ECG and all ELM types |
| 9 | FSDP2 + DCP + MFU + compile | 2/4-GPU runs; FSDP≈DDP loss; 7B fits |
| 10 | line audit + README | receipts |
