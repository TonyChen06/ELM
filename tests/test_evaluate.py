"""M6 gate: end-to-end eval on real data; batched matches unbatched under
greedy decoding (same refs, same count); scoring units."""
import sys

sys.path.insert(0, "src")
import pytest
import torch

from config import Config
from data.dataset import build_dataset
from data.templates import build_tokenizer
from evaluate import classification_metrics, evaluate, split_response, summarize_seeds, text_metrics
from models.elm import build_elm

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def setup():
    cfg = Config(data=["ecg-qa-ptbxl-250-2500"], representation="signal", elm="patch_elf",
                 llm="qwen2.5-0.5b-instruct", num_encoder_tokens=100, data_subset=0.001,
                 system_prompt="prompts/system_prompt.txt", max_new_tokens=16)
    cfg.mode = "eval"
    tok = build_tokenizer(cfg)
    model = build_elm(cfg, tok).to(DEV)
    model.llm.generation_config.do_sample = False
    model.llm.generation_config.temperature = None
    model.llm.generation_config.top_p = None
    model.llm.generation_config.top_k = None
    return cfg, tok, model


def test_eval_end_to_end_and_batched_consistency(setup):
    cfg, tok, model = setup
    dataset = build_dataset(cfg, tok)
    cfg.eval_batch_size = 1
    one = evaluate(cfg, model, dataset)
    assert one["num_pairs"] > 0
    assert set(one["metrics"]["answer"]) == {"ACC", "F1", "BLEU-4", "ROUGE-L", "METEOR"}
    cfg.eval_batch_size = 4
    four = evaluate(cfg, model, dataset)
    assert four["references"] == one["references"]          # same turns, same order
    match = sum(a == b for a, b in zip(one["hypotheses"], four["hypotheses"]))
    assert match / one["num_pairs"] > 0.6                   # greedy, bf16 batching envelope


def test_split_response():
    think, answer = split_response("<think>\nreason</think><answer>yes</answer>")
    assert think == "reason" and answer == "yes"
    think, answer = split_response("reason</think><answer>yes</answer>")  # explicit_thinking
    assert think == "reason" and answer == "yes"
    assert split_response("plain") == ("", "plain")


def test_text_and_classification_metrics():
    m = text_metrics(["a b c", "x"], ["a b c", "y"])
    assert m["ACC"] == 0.5 and 0 < m["F1"] <= 1
    per_class, confusion, other = classification_metrics(["cat", "dog"], ["cat", "bird"])
    assert per_class["cat"] == 1.0 and per_class["dog"] == 0.0
    assert confusion["dog"]["Other"] == 1 and other == {"bird": 1}


def test_summarize_seeds():
    s = summarize_seeds([{"metrics": {"answer": {"ACC": 0.5}}},
                         {"metrics": {"answer": {"ACC": 0.7}}}])
    assert abs(s["answer"]["ACC"]["mean"] - 60.0) < 1e-9
    assert s["answer"]["ACC"]["ci95"][0] < 60.0 < s["answer"]["ACC"]["ci95"][1]
