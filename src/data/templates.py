"""Tokenization, conversation rendering, and label masking.

Rendering uses the tokenizer's own chat template (any HF chat model works).
Labels are derived by incremental rendering: the template applied to
messages[:k] must be a token-prefix of messages[:k+1]; the assistant-content
span of each turn is the difference between the render with
add_generation_prompt=True and the full render. Labels cover exactly that
span up to and including the turn's end-of-turn token.
"""
import re

from transformers import AutoTokenizer

from registry import LLMS, RL_TOKENS, SIGNAL_TOKEN

_TAG_RE = re.compile(r"<\s*(?:ecg|image)\s*>\s*\n?", flags=re.IGNORECASE)
_IMAGE_WORD_RE = re.compile(r"\b(image)\b", flags=re.IGNORECASE)
_LEADING_PREFIX_RE = re.compile(
    r"^\s*(?:[:：]\s*|(?:user|assistant|human|gpt|model|system|q|a)\s*:\s*)+",
    flags=re.IGNORECASE,
)
IGNORE = -100


def build_tokenizer(cfg):
    tok = AutoTokenizer.from_pretrained(LLMS[cfg.llm]["hf_id"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    extra = [SIGNAL_TOKEN] + (RL_TOKENS if cfg.train_phase in ("sft", "rl") else [])
    tok.add_special_tokens({"additional_special_tokens": extra})
    if cfg.representation == "symbolic":
        from data.bpe_tokenizer import ecg_vocab_tokens
        tok.add_tokens(ecg_vocab_tokens(cfg.ecg_tokenizer))
    return tok


def clean_text(text: str) -> str:
    text = _TAG_RE.sub("", text)
    text = _IMAGE_WORD_RE.sub(lambda m: "Signal" if m.group(1)[0].isupper() else "signal", text)
    return _LEADING_PREFIX_RE.sub("", text)


def to_messages(raw, cfg, system_prompt: str | None):
    """Dataset turn dicts ({from,value} or {role,content}) -> chat messages.

    The signal placeholder block is prepended to the first user message
    (unless perturb=only_text or the ELM consumes tokens directly).
    """
    signal_block = "" if cfg.perturb == "only_text" else SIGNAL_TOKEN * cfg.num_encoder_tokens + "\n"
    messages = []
    if system_prompt and LLMS[cfg.llm].get("system_role", True):
        messages.append({"role": "system", "content": system_prompt})
    elif system_prompt:  # models without a system role get it folded into the first user turn
        signal_block = system_prompt + "\n\n" + signal_block
    first_user = True
    for turn in raw:
        speaker = turn.get("from", turn.get("role", "")).lower()
        role = "user" if speaker in ("human", "user") else "assistant"
        content = clean_text(turn.get("value", turn.get("content", "")))
        if role == "user" and first_user:
            content = signal_block + content
            first_user = False
        messages.append({"role": role, "content": content})
    return messages


def _render(tok, messages, generation_prompt=False):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation_prompt)
    return tok.encode(text, add_special_tokens=False)


def render_train(tok, messages, cfg):
    """Full conversation -> (input_ids, labels). Labels cover assistant content
    plus its end-of-turn token; everything else (headers, user turns, padding
    added later) is IGNORE."""
    if cfg.train_phase == "pretrain":
        return _render_pretrain(tok, messages, cfg)
    think_ids = tok.encode("<think>\n", add_special_tokens=False) if cfg.explicit_thinking else []
    ids, labels, prev = [], [], []
    for k, msg in enumerate(messages):
        full = _render(tok, messages[: k + 1])
        assert full[: len(prev)] == prev, "chat template is not prefix-stable"
        if msg["role"] == "assistant":
            gen = _render(tok, messages[:k], generation_prompt=True)
            assert full[: len(gen)] == gen, "generation prompt diverges from full render"
            span_start = len(gen)
            if think_ids and full[span_start : span_start + len(think_ids)] == think_ids:
                span_start += len(think_ids)
            span_end = _end_of_turn(tok, full, span_start)
            turn_labels = [IGNORE] * len(full)
            turn_labels[span_start:span_end] = full[span_start:span_end]
            labels = labels + turn_labels[len(prev):]
        else:
            labels = labels + [IGNORE] * (len(full) - len(prev))
        ids, prev = full, full
    return ids, labels


_EOT_TOKENS = ["<|im_end|>", "<|eot_id|>", "<end_of_turn>"]  # extend via LLMS["eot_token"]


def end_of_turn_ids(tok, llm_name=None) -> set:
    """Token ids that terminate an assistant turn (never content tokens like <think>)."""
    names = list(_EOT_TOKENS)
    if llm_name and LLMS[llm_name].get("eot_token"):
        names.append(LLMS[llm_name]["eot_token"])
    ids = {tok.eos_token_id}
    for name in names:
        i = tok.convert_tokens_to_ids(name)
        if i is not None and i != tok.unk_token_id:
            ids.add(i)
    ids.discard(None)
    return ids


def _end_of_turn(tok, ids, start):
    """Index one past the first end-of-turn token at or after start."""
    stop = end_of_turn_ids(tok)
    for i in range(start, len(ids)):
        if ids[i] in stop:
            return i + 1
    return len(ids)


def _render_pretrain(tok, messages, cfg):
    text = " ".join(m["content"] for m in messages if m["role"] != "system")
    signal_block = "" if cfg.perturb == "only_text" else SIGNAL_TOKEN * cfg.num_encoder_tokens + "\n"
    body = tok.encode(signal_block + text, add_special_tokens=False)
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    ids = bos + body + [tok.eos_token_id]
    sig = tok.convert_tokens_to_ids(SIGNAL_TOKEN)
    prefix_end = max((i for i, t in enumerate(ids) if t == sig), default=len(bos) - 1) + 1
    if prefix_end < len(ids) and signal_block:
        prefix_end += 1  # the newline separating signals from text
    labels = [IGNORE] * prefix_end + ids[prefix_end:]
    return ids, labels


def render_inference(tok, messages, cfg):
    """Prompt ids for generation (assistant header appended, no content)."""
    ids = _render(tok, messages, generation_prompt=True)
    if cfg.explicit_thinking:
        ids = ids + tok.encode("<think>\n", add_special_tokens=False)
    return ids


def response_ranges(labels):
    """Contiguous labeled spans -> [(start, end)], one per assistant turn."""
    ranges, start = [], None
    for i, lab in enumerate(labels):
        if lab != IGNORE and start is None:
            start = i
        elif lab == IGNORE and start is not None:
            ranges.append((start, i))
            start = None
    if start is not None:
        ranges.append((start, len(labels)))
    return ranges
