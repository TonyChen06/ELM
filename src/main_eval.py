"""Multi-seed evaluation: per-(fold, seed) metrics + plots, across-seed stats.

Run: [torchrun ...] src/main_eval.py --data ... --ckpt runs/best
"""
import json
import os
from collections import Counter


import dist
from config import parse_config, set_seed
from data.dataset import build_dataset
from data.templates import build_tokenizer
from evaluate import evaluate, summarize_seeds
from models.elm import build_elm


def barh(pairs, path, title):
    if not pairs:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [str(k)[:60] for k, _ in pairs]
    fig, ax = plt.subplots(figsize=(10, max(3, 0.45 * len(pairs) + 1.5)))
    ax.barh(labels, [v for _, v in pairs])
    ax.invert_yaxis()
    ax.set_title(title)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def confusion_png(confusion, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    rows = list(confusion)
    cols = list(next(iter(confusion.values())))
    m = np.array([[confusion[r][c] for c in cols] for r in rows], dtype=float)
    normed = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(max(4, 1.5 * len(cols)), max(4, 1.5 * len(rows))))
    ax.imshow(normed, cmap="Blues", vmin=0, vmax=1)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            ax.text(j, i, f"{int(m[i, j])}\n{normed[i, j]:.0%}", ha="center", va="center",
                    color="white" if normed[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_artifacts(out, prefix):
    wrong = Counter(h for r, h in zip(out["references"], out["hypotheses"]) if r != h)
    barh(wrong.most_common(10), f"{prefix}_incorrect.png", "top incorrect predictions")
    if "confusion_matrix" in out:
        confusion_png(out["confusion_matrix"], f"{prefix}_confusion.png")
        barh(sorted(out["other_outputs"].items(), key=lambda kv: -kv[1])[:10],
             f"{prefix}_other.png", "top out-of-label outputs")
    if "pretrain_breakdown" in out:
        b = out["pretrain_breakdown"]
        barh([(k, b.get(k, 0)) for k in ("matched", "not_matched", "other")],
             f"{prefix}_pretrain_match.png", f"pretrain match (N={b['n']})")
        barh([(k, b.get(k, 0)) for k in ("only_missed", "only_extra", "both")],
             f"{prefix}_pretrain_partition.png", "mismatch partition")
        barh([(k, b.get(k, 0)) for k in ("missed_inst", "extra_inst")],
             f"{prefix}_pretrain_disagreement.png", "per-instance disagreement")
        barh(b["top_missed"], f"{prefix}_pretrain_missed.png", "top missed statements")
        barh(b["top_extra"], f"{prefix}_pretrain_extra.png", "top extra statements")


def main():
    cfg = parse_config(mode="eval")
    dist.set_device(cfg.device)
    dist.setup(cfg)
    try:
        tokenizer = build_tokenizer(cfg)
        model = dist.parallelize(build_elm(cfg, tokenizer), cfg)
        os.makedirs(cfg.run_dir, exist_ok=True)
        prompt_name = os.path.splitext(os.path.basename(cfg.system_prompt))[0] if cfg.system_prompt else "noprompt"
        prefix = os.path.join(cfg.run_dir,
                              f"eval_{'_'.join(cfg.data)}_{prompt_name}_{cfg.perturb or 'none'}_{cfg.max_new_tokens}")
        per_seed = []
        for seed in cfg.eval_seeds:
            if dist.is_main():
                print(f"fold {cfg.fold} seed {seed}")
            cfg.seed = seed
            set_seed(seed)
            dataset = build_dataset(cfg, tokenizer)
            out = evaluate(cfg, model, dataset)
            per_seed.append(out)
            if dist.is_main():
                save_artifacts(out, f"{prefix}_s{seed}")
                if len(per_seed) == 1:
                    examples = [{"prompt": p, "predicted": h, "ground_truth": r}
                                for p, h, r in zip(out["prompts"], out["hypotheses"], out["references"])]
                    json.dump(examples, open(f"{prefix}_examples.json", "w"), indent=1)
        if dist.is_main():
            json.dump(summarize_seeds(per_seed), open(f"{prefix}_stats.json", "w"), indent=1)
            print(f"wrote {prefix}_stats.json")
    finally:
        dist.teardown()


if __name__ == "__main__":
    main()
