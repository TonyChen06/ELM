"""Single typed configuration for train / eval / chat.

The CLI is generated from the dataclass fields: every field becomes
``--<name>``; bools become ``--<name>/--no-<name>``; list fields take
``nargs="+"``. Defaults live here and nowhere else.
"""
import argparse
import os
from dataclasses import MISSING, dataclass, field, fields


@dataclass
class Config:
    # run
    seed: int = 0
    dev: bool = False                     # tiny verbose runs: break after first batch
    wandb: bool = False
    run_dir: str = "runs"

    # data
    data: list[str] = field(default_factory=list)       # HF dataset names (mixable)
    representation: str = "signal"                      # signal | symbolic | stacked_signal | rgb
    fold: str = "1"
    data_subset: float = 0.0                            # 0 = all; else fraction of split
    num_workers: int = 0
    segment_len: int = 2500
    leads: list[int] = field(default_factory=lambda: list(range(12)))
    augment_ecg: bool = False
    augment_rgb: bool = False
    perturb: str = ""                                   # "" | noise | zeros | only_text
    ecg_tokenizer: str = "src/data/bpe/ecg_byte_tokenizer_10000.pkl"
    min_ecg_tokens: int = 512                           # symbolic: ECG tokens kept under truncation
    norm_eps: float = 1e-6

    # text
    llm: str = "qwen2.5-1.5b-instruct"
    system_prompt: str = "prompts/system_prompt.txt"
    max_seq_len: int = 2048                             # truncation cap (not a padded shape)
    max_new_tokens: int = 128
    train_phase: str = "sft"                            # pretrain | sft | rl
    explicit_thinking: bool = False                     # mask loss through "<think>\n" prefix

    # model
    elm: str = "mlp_llava"                              # see registry.ELMS
    encoder: str = ""                                   # see registry.ENCODERS ("" = encoder-free)
    num_encoder_tokens: int = 1
    encoder_ckpt: str = ""
    ckpt: str = ""                                      # ELM checkpoint to load
    update: list[str] = field(default_factory=lambda: ["connector", "llm"])
    scratch: bool = False                               # random-init LLM weights
    attention: str = "sdpa"
    gradient_checkpointing: bool = False
    peft: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # optimization
    optimizer: str = "adamw"                            # adamw | muon
    lr: float = 1e-4
    weight_decay: float = 1e-2
    betas: list[float] = field(default_factory=lambda: [0.9, 0.99])
    eps: float = 1e-8
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_adamw_lr_ratio: float = 0.015
    lr_schedule: str = "constant"                       # constant | cosine | inv_sqrt
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.1
    ref_global_bs: int = 0                              # 0 = no LR scaling
    scale_wd: str = "none"                              # none | inv_sqrt | inv_linear

    # training loop
    batch_size: int = 1
    grad_accum_steps: int = 1
    epochs: int = 1
    grad_clip: float = 0.0
    early_stopping: bool = False
    patience: int = 5
    patience_delta: float = 0.1
    save_steps: int = 0                                 # 0 = epoch checkpoints only
    resume: str = ""                                    # checkpoint dir to resume from

    # parallelism / performance
    parallel: str = "ddp"                               # ddp | fsdp (when launched via torchrun)
    compile: bool = False
    param_dtype: str = "bfloat16"
    reduce_dtype: str = "float32"                       # FSDP gradient-reduce dtype

    # eval
    eval_batch_size: int = 1
    eval_seeds: list[int] = field(default_factory=lambda: [1337, 1338])

    # rl
    rl_algo: str = "sapo"
    rl_group_size: int = 4
    rl_max_new_tokens: int = 512
    rl_temperature: float = 1.0
    rl_top_p: float = 1.0
    rl_tau_pos: float = 1.0
    rl_tau_neg: float = 1.05
    rl_loss_agg: str = "seq-mean-token-mean"

    # set at runtime, not CLI
    mode: str = "train"                                 # train | eval | chat
    max_steps: int = 0                                  # optimizer steps; set by the trainer

    @property
    def distributed(self) -> bool:
        return "RANK" in os.environ

    def validate(self):
        from registry import ELMS, ENCODERS, LLMS, REPRESENTATIONS
        checks = {
            "llm": (self.llm, LLMS), "elm": (self.elm, ELMS),
            "representation": (self.representation, REPRESENTATIONS),
        }
        for name, (value, table) in checks.items():
            if value not in table:
                raise ValueError(f"Unknown {name} '{value}'. Known: {sorted(table)}")
        if self.encoder and self.encoder not in ENCODERS:
            raise ValueError(f"Unknown encoder '{self.encoder}'. Known: {sorted(ENCODERS)}")
        if ELMS[self.elm].needs_encoder and not self.encoder:
            raise ValueError(f"elm '{self.elm}' requires --encoder")
        if self.train_phase not in ("pretrain", "sft", "rl"):
            raise ValueError(f"Unknown train_phase '{self.train_phase}'")
        if self.perturb not in ("", "noise", "zeros", "only_text"):
            raise ValueError(f"Unknown perturb '{self.perturb}'")
        return self


_RUNTIME_ONLY = {"mode", "max_steps"}


def parse_config(argv=None, mode="train") -> Config:
    parser = argparse.ArgumentParser(description="ELM training/eval framework")
    for f in fields(Config):
        if f.name in _RUNTIME_ONLY:
            continue
        flag = f"--{f.name}"
        default = f.default_factory() if f.default_factory is not MISSING else f.default
        if f.type == "bool" or isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        elif isinstance(default, list):
            item_type = type(default[0]) if default else (float if f.name == "betas" else (int if f.name in ("leads", "eval_seeds") else str))
            parser.add_argument(flag, nargs="+", type=item_type, default=default)
        else:
            parser.add_argument(flag, type=type(default), default=default)
    args = parser.parse_args(argv)
    cfg = Config(**vars(args))
    cfg.mode = mode
    return cfg.validate()
