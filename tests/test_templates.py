"""M1 gate: label masks cover exactly assistant content (+ end-of-turn).

Uses real conversations from the ECG-QA dataset (multi-turn) for every cached
LLM family, across sft / pretrain / explicit_thinking. Run: pytest tests/test_templates.py -q
"""
import json
import sys

sys.path.insert(0, "src")
import pytest
from datasets import load_dataset

from config import Config
from data.templates import (IGNORE, build_tokenizer, render_inference,
                            render_train, response_ranges, to_messages)

LLMS_TO_TEST = ["qwen2.5-0.5b-instruct", "llama-3.2-1b-instruct"]
SYSTEM = open("prompts/system_prompt.txt").read()


def conversations(n=4):
    ds = load_dataset("ELM-Research/ecg-qa-ptbxl-250-2500", split="fold1_test")
    out = []
    for i in range(0, n * 40, 40):
        out.append(json.loads(ds[i]["text"]))
    return out


@pytest.fixture(scope="module", params=LLMS_TO_TEST)
def setup(request):
    cfg = Config(llm=request.param, num_encoder_tokens=5, train_phase="sft")
    return cfg, build_tokenizer(cfg)


def test_masks_cover_exactly_assistant_content(setup):
    cfg, tok = setup
    for raw in conversations():
        messages = to_messages(raw, cfg, SYSTEM)
        ids, labels = render_train(tok, messages, cfg)
        assert len(ids) == len(labels)
        spans = response_ranges(labels)
        assistant_contents = [m["content"] for m in messages if m["role"] == "assistant"]
        assert len(spans) == len(assistant_contents)
        for (s, e), content in zip(spans, assistant_contents):
            assert labels[s:e] == ids[s:e]
            decoded = tok.decode(ids[s:e], skip_special_tokens=True).strip()
            assert decoded == content.strip(), f"span={decoded[:60]!r} vs {content[:60]!r}"
            assert ids[e - 1] in __import__("data.templates", fromlist=["end_of_turn_ids"]).end_of_turn_ids(tok)
        # nothing outside spans is labeled
        outside = [l for i, l in enumerate(labels) if not any(s <= i < e for s, e in spans)]
        assert all(l == IGNORE for l in outside)


def test_signal_placeholders_present_once(setup):
    cfg, tok = setup
    sig = tok.convert_tokens_to_ids("<signal>")
    messages = to_messages(conversations(1)[0], cfg, SYSTEM)
    ids, labels = render_train(tok, messages, cfg)
    positions = [i for i, t in enumerate(ids) if t == sig]
    assert len(positions) == cfg.num_encoder_tokens
    assert all(labels[i] == IGNORE for i in positions)


def test_pretrain_masks_prefix_only(setup):
    cfg, tok = setup
    cfg = Config(llm=cfg.llm, num_encoder_tokens=5, train_phase="pretrain")
    tok = build_tokenizer(cfg)
    messages = to_messages(conversations(1)[0], cfg, None)
    ids, labels = render_train(tok, messages, cfg)
    first = next(i for i, l in enumerate(labels) if l != IGNORE)
    assert all(l == IGNORE for l in labels[:first])
    assert labels[first:] == ids[first:]
    sig = tok.convert_tokens_to_ids("<signal>")
    assert all(labels[i] == IGNORE for i, t in enumerate(ids) if t == sig)
    assert ids[-1] == tok.eos_token_id and labels[-1] == ids[-1]


def test_explicit_thinking_prefix_unlabeled(setup):
    cfg, tok = setup
    cfg = Config(llm=cfg.llm, num_encoder_tokens=1, train_phase="rl", explicit_thinking=True)
    tok = build_tokenizer(cfg)
    raw = [{"from": "human", "value": "q"},
           {"from": "gpt", "value": "<think>\nbecause</think><answer>yes</answer>"}]
    ids, labels = render_train(tok, to_messages(raw, cfg, SYSTEM), cfg)
    (s, e), = response_ranges(labels)
    decoded = tok.decode(ids[s:e], skip_special_tokens=False)
    assert not decoded.startswith("<think>")        # opener consumed as prompt prefix
    assert "</think>" in decoded and "<answer>" in decoded  # RL tokens stay labeled content


def test_inference_prompt_is_train_prefix(setup):
    cfg, tok = setup
    messages = to_messages(conversations(1)[0], cfg, SYSTEM)
    first_assistant = next(i for i, m in enumerate(messages) if m["role"] == "assistant")
    prompt = render_inference(tok, messages[:first_assistant], cfg)
    full, _ = render_train(tok, messages, cfg)
    assert full[: len(prompt)] == prompt
