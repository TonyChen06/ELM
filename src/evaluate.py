"""N-turn evaluation: flatten (sample, turn) pairs, generate in batches of
eval_batch_size (sharded across ranks when distributed), score.

Metrics: exact match, token F1, BLEU-4, ROUGE-L, METEOR — computed separately
for <think> and answer segments when present — plus classification metrics
for label-style datasets."""
import re
import string
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm

import dist
from data.collate import Collator
from data.templates import end_of_turn_ids, response_ranges

_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


# --------------------------- generation ------------------------------------
def flatten_turns(dataset, dev=False):
    turns = []
    for index in range(len(dataset)):
        item = dataset[index]
        labels = item["labels"].tolist()
        ids = item["input_ids"]
        for start, end in response_ranges(labels):
            gt = dataset.tok.decode(ids[start:end].tolist(), skip_special_tokens=True).strip()
            turn = {"order": len(turns), "gt": gt,
                    "input_ids": ids[:start]}
            if "signal_pos" in item:
                pos = item["signal_pos"]
                turn["signal_pos"] = torch.where(pos < start, pos, torch.full_like(pos, -1))
            for key, value in item.items():
                if key not in ("input_ids", "labels", "signal_pos"):
                    turn[key] = value
            turns.append(turn)
        if dev and index >= 1:
            break
    return turns


def evaluate(cfg, model, dataset):
    model.eval()
    gen_model = dist.unwrap(model)
    turns = flatten_turns(dataset, cfg.dev)
    local = turns[dist.rank()::dist.world_size()]
    collate = Collator(cfg, dataset.tok.pad_token_id)
    stop = end_of_turn_ids(dataset.tok, cfg.llm)
    device = dist.device()

    results = []
    with torch.no_grad():
        for at in tqdm(range(0, len(local), cfg.eval_batch_size),
                       desc=f"eval bs={cfg.eval_batch_size}", disable=not dist.is_main(), leave=False):
            chunk = local[at:at + cfg.eval_batch_size]
            batch = collate([{k: v for k, v in t.items() if k not in ("order", "gt")} for t in chunk])
            batch = {k: v.to(device) for k, v in batch.items()}
            out = gen_model.generate(**batch, max_new_tokens=cfg.max_new_tokens)
            for turn, row in zip(chunk, out):
                ids = row.tolist()
                cut = next((i for i, t in enumerate(ids) if t in stop), len(ids))
                text = dataset.tok.decode(ids[:cut], skip_special_tokens=True).strip()
                prompt = dataset.tok.decode(turn["input_ids"].tolist(), skip_special_tokens=True).strip()
                if cfg.dev and dist.is_main():
                    print(f"\n[turn {turn['order']}]\nGT: {turn['gt']}\nGen: {text}\n" + "-" * 80)
                results.append((turn["order"], prompt, turn["gt"], text))

    results = sorted(r for shard in dist.all_gather_object(results) for r in shard)
    prompts, refs, hyps = [], [], []
    for _, prompt, gt, hyp in results:
        if gt and hyp:
            prompts.append(prompt)
            refs.append(gt)
            hyps.append(hyp)
    return score(cfg, prompts, refs, hyps)


# ----------------------------- scoring -------------------------------------
def split_response(text):
    if "</think>" in text and "<think>" not in text:
        text = "<think>\n" + text  # explicit_thinking: opener was the prompt suffix
    think, answer = _THINK.search(text), _ANSWER.search(text)
    body = answer.group(1) if answer else (text[think.end():] if think else text)
    return (think.group(1).strip() if think else ""), body.strip()


def _norm(text):
    return " ".join(text.lower().translate(str.maketrans("", "", string.punctuation)).split())


def _f1(ref, hyp):
    r, h = _norm(ref).split(), _norm(hyp).split()
    if not r or not h:
        return float(r == h)
    common = sum((Counter(r) & Counter(h)).values())
    return 2 * common / (len(r) + len(h)) if common else 0.0


def text_metrics(refs, hyps):
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
    from nltk.translate.meteor_score import meteor_score
    from rouge_score.rouge_scorer import RougeScorer
    pairs = [(r, h) for r, h in zip(refs, hyps) if r and h]
    if not pairs:
        return dict.fromkeys(("ACC", "F1", "BLEU-4", "ROUGE-L", "METEOR"), 0.0)
    refs, hyps = map(list, zip(*pairs))
    rouge = RougeScorer(["rougeL"], use_stemmer=True)
    return {
        "ACC": float(np.mean([r == h for r, h in pairs])),
        "F1": float(np.mean([_f1(r, h) for r, h in pairs])),
        "BLEU-4": corpus_bleu([[r.split()] for r in refs], [h.split() for h in hyps],
                              smoothing_function=SmoothingFunction().method1),
        "ROUGE-L": float(np.mean([rouge.score(r, h)["rougeL"].fmeasure for r, h in pairs])),
        "METEOR": float(np.mean([meteor_score([r.split()], h.split()) for r, h in pairs])),
    }


def classification_metrics(refs, hyps):
    classes = sorted(set(refs))
    other = Counter(h for h in hyps if h not in classes)
    mapped = [h if h in classes else "Other" for h in hyps]
    columns = classes + (["Other"] if other else [])
    confusion = {r: {c: 0 for c in columns} for r in classes}
    for r, h in zip(refs, mapped):
        confusion[r][h] += 1
    per_class = {c: confusion[c][c] / max(sum(confusion[c].values()), 1) for c in classes}
    return per_class, confusion, dict(other)


def pretrain_breakdown(refs, hyps):
    as_set = lambda s: {x.strip() for x in (s or "").split(";") if x.strip()}  # noqa: E731
    counts = Counter()
    missed, extra = Counter(), Counter()
    for r, h in zip(refs, hyps):
        a, b = as_set(r), as_set(h)
        if not b:
            counts["other"] += 1
        elif a == b:
            counts["matched"] += 1
        else:
            counts["not_matched"] += 1
            m, e = a - b, b - a
            missed.update(m)
            extra.update(e)
            counts["missed_inst"] += bool(m)
            counts["extra_inst"] += bool(e)
            counts["both" if (m and e) else ("only_missed" if m else "only_extra")] += 1
    return {"n": len(refs), **counts,
            "top_missed": missed.most_common(15), "top_extra": extra.most_common(15)}


def score(cfg, prompts, refs, hyps):
    ref_parts = [split_response(r) for r in refs]
    hyp_parts = [split_response(h) for h in hyps]
    metrics = {"answer": text_metrics([r[1] for r in ref_parts], [h[1] for h in hyp_parts])}
    think = [(r[0], h[0]) for r, h in zip(ref_parts, hyp_parts) if r[0] and h[0]]
    if think:
        metrics["thinking"] = text_metrics([t[0] for t in think], [t[1] for t in think])
    out = {"num_pairs": len(refs), "metrics": metrics,
           "prompts": prompts, "references": refs, "hypotheses": hyps}
    if cfg.train_phase == "pretrain" and refs:
        out["pretrain_breakdown"] = pretrain_breakdown([r[1] for r in ref_parts], [h[1] for h in hyp_parts])
    if any(d.startswith("ecg-comp") for d in cfg.data):
        per_class, confusion, other = classification_metrics(
            [r[1] for r in ref_parts], [h[1] for h in hyp_parts])
        metrics["per_class_acc"] = per_class
        out["confusion_matrix"] = confusion
        out["other_outputs"] = other
    if dist.is_main():
        print(f"pairs={len(refs)}")
        for group, values in metrics.items():
            print(f"  [{group}] " + " ".join(f"{k}={v:.4f}" for k, v in values.items()
                                             if isinstance(v, float)))
    return out


def summarize_seeds(per_seed):
    """Across-seed mean/std/95% CI for every numeric metric (recursive)."""
    import scipy.stats as st

    def merge(values):
        if isinstance(values[0], dict):
            return {k: merge([v[k] for v in values]) for k in values[0]}
        arr = np.array(values, dtype=float) * 100
        if len(arr) < 2:
            return {"mean": float(arr.mean()), "std": 0.0, "ci95": [float(arr.mean())] * 2}
        half = st.t.ppf(0.975, len(arr) - 1) * arr.std(ddof=1) / np.sqrt(len(arr))
        return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)),
                "ci95": [float(arr.mean() - half), float(arr.mean() + half)]}

    return merge([s["metrics"] for s in per_seed])
