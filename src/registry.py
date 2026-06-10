"""Single source of truth for every pluggable component.

Adding a model/encoder/connector/representation = one entry here. Builders are
lazy (import inside the lambda/function) so importing the registry stays cheap.
"""
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# LLMs: any HF causal LM with a chat template works. dtype is the load dtype.
# ---------------------------------------------------------------------------
LLMS = {
    "llama-3.2-3b-instruct":  {"hf_id": "meta-llama/Llama-3.2-3B-Instruct"},
    "llama-3.2-1b-instruct":  {"hf_id": "meta-llama/Llama-3.2-1B-Instruct"},
    "qwen2.5-7b-instruct":    {"hf_id": "Qwen/Qwen2.5-7B-Instruct"},
    "qwen2.5-3b-instruct":    {"hf_id": "Qwen/Qwen2.5-3B-Instruct"},
    "qwen2.5-1.5b-instruct":  {"hf_id": "Qwen/Qwen2.5-1.5B-Instruct"},
    "qwen2.5-0.5b-instruct":  {"hf_id": "Qwen/Qwen2.5-0.5B-Instruct"},
    "gemma-2-2b-it":          {"hf_id": "google/gemma-2-2b-it", "system_role": False},
}

# ---------------------------------------------------------------------------
# Encoders. embed_dim = per-token feature size the connector receives.
# kind: "ecg" encoders consume the raw signal; "vision" consume processor output.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EncoderSpec:
    kind: str                 # ecg | vision
    embed_dim: int | None     # None = read from the HF config at build time
    build: "callable"
    hf_processor: str | None = None


def _build_st_mem(cfg):
    from models.encoders.vit1d import MaskedViT1d
    return MaskedViT1d(variant="st_mem", seq_len=cfg.segment_len, num_tokens=cfg.num_encoder_tokens)


def _build_mtae(cfg):
    from models.encoders.vit1d import MaskedViT1d
    return MaskedViT1d(variant="mtae", seq_len=cfg.segment_len, num_tokens=cfg.num_encoder_tokens)


def _build_mlae(cfg):
    from models.encoders.vit1d import MaskedViT1d
    return MaskedViT1d(variant="mlae", seq_len=cfg.segment_len, num_tokens=cfg.num_encoder_tokens)


def _build_merl(cfg):
    from models.encoders.merl import Merl
    return Merl(num_tokens=cfg.num_encoder_tokens)


def _build_hf_vision(cfg):
    from models.encoders.hf_vision import HFVision
    return HFVision(ENCODERS[cfg.encoder].hf_processor)


ENCODERS = {
    "st_mem": EncoderSpec("ecg", 768, _build_st_mem),
    "mtae":   EncoderSpec("ecg", 256, _build_mtae),
    "mlae":   EncoderSpec("ecg", 256, _build_mlae),
    "merl":   EncoderSpec("ecg", 2048, _build_merl),
    "clip":   EncoderSpec("vision", None, _build_hf_vision, "openai/clip-vit-base-patch32"),
    "siglip2": EncoderSpec("vision", None, _build_hf_vision, "google/siglip2-so400m-patch16-naflex"),
    "vit":    EncoderSpec("vision", None, _build_hf_vision, "google/vit-base-patch16-224-in21k"),
}

# ---------------------------------------------------------------------------
# ELM compositions. connector receives (in_dim, llm_hidden, cfg).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ElmSpec:
    needs_encoder: bool
    uses_signal_tokens: bool = True       # <signal> placeholders + embedding injection
    connector: str | None = None          # key into CONNECTORS; None = no connector (ecg_byte)
    field_notes: str = ""


ELMS = {
    "linear_llava": ElmSpec(needs_encoder=True, connector="linear"),
    "mlp_llava":    ElmSpec(needs_encoder=True, connector="mlp"),
    "base_elf":     ElmSpec(needs_encoder=False, connector="flat_linear"),
    "patch_elf":    ElmSpec(needs_encoder=False, connector="patch"),
    "conv_elf":     ElmSpec(needs_encoder=False, connector="cnn_patch"),
    "ecg_byte":     ElmSpec(needs_encoder=False, uses_signal_tokens=False, connector=None),
}


def build_connector(name, in_dim, llm_hidden, cfg):
    import models.connectors as c
    table = {
        "linear": lambda: c.Linear(in_dim, llm_hidden),
        "mlp": lambda: c.Mlp(in_dim, llm_hidden),
        "flat_linear": lambda: c.Linear(len(cfg.leads) * cfg.segment_len, llm_hidden),
        "patch": lambda: c.Patch(cfg.num_encoder_tokens, len(cfg.leads), cfg.segment_len, llm_hidden),
        "cnn_patch": lambda: c.CnnPatch(cfg.num_encoder_tokens, len(cfg.leads), llm_hidden),
    }
    return table[name]()


# ---------------------------------------------------------------------------
# Data representations: strategy classes living in data/dataset.py.
# ---------------------------------------------------------------------------
REPRESENTATIONS = {"signal", "symbolic", "stacked_signal", "rgb"}

# Special tokens
SIGNAL_TOKEN = "<signal>"
ECG_TOKEN_PREFIX = "signal_"
RL_TOKENS = ["<think>", "</think>", "<answer>", "</answer>"]
HF_DATASET_ORG = "ELM-Research"
