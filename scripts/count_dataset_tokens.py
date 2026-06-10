"""Count tokenizer tokens per dataset (text content only, no template overhead).

Usage (repo root):
  uv run scripts/count_dataset_tokens.py --datasets ecg-qa-ptbxl-250-2500 \
      --llm qwen2.5-1.5b-instruct --split fold1_train [--limit 1000]
"""
import argparse
import json
import sys

sys.path.insert(0, "src")
from datasets import load_dataset
from transformers import AutoTokenizer

from registry import HF_DATASET_ORG, LLMS


def turns(text):
    try:
        text = json.loads(text)
    except (ValueError, TypeError):
        pass
    if isinstance(text, str):
        return [text]
    return [t.get("value") or t.get("content") or "" for t in text if isinstance(t, dict)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--llm", default="qwen2.5-1.5b-instruct", choices=sorted(LLMS))
    parser.add_argument("--split", default="fold1_train")
    parser.add_argument("--limit", type=int, default=0, help="rows per dataset (0 = all)")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(LLMS[args.llm]["hf_id"])
    grand = 0
    for name in args.datasets:
        rows = load_dataset(f"{HF_DATASET_ORG}/{name}", split=args.split)["text"]
        if args.limit:
            rows = rows[: args.limit]
        total = sum(len(tok.encode(t, add_special_tokens=False)) for row in rows for t in turns(row))
        grand += total
        print(f"{name}\t{total:,} tokens / {len(rows):,} rows")
    print(f"TOTAL\t{grand:,}")


if __name__ == "__main__":
    main()
