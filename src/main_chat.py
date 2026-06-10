"""Interactive chat with a trained ELM.

  /ecg <path>   load an ECG (.npy); conditions all later turns
  /clear        reset the conversation
  /quit         exit

Works for every ELM type; without an ECG loaded, signal positions are -1 and
the model runs text-only.
"""
import numpy as np
import torch

import dist
from config import parse_config
from data.dataset import normalize_signal
from data.templates import build_tokenizer, end_of_turn_ids, render_inference, to_messages
from models.elm import build_elm
from registry import ELMS, SIGNAL_TOKEN


def load_ecg(path, cfg):
    payload = np.load(path, allow_pickle=True)
    ecg = (payload.item()["ecg"] if payload.dtype == object else payload)[cfg.leads]
    ecg = normalize_signal(ecg.astype(np.float32), cfg.norm_eps)
    return torch.from_numpy(ecg.reshape(-1) if cfg.elm == "base_elf" else ecg)


def main():
    cfg = parse_config(mode="chat")
    tok = build_tokenizer(cfg)
    model = build_elm(cfg, tok).to(dist.device()).eval()
    stop = end_of_turn_ids(tok, cfg.llm)
    system = open(cfg.system_prompt, encoding="utf-8").read() if cfg.system_prompt else None
    uses_signal = ELMS[cfg.elm].uses_signal_tokens
    history, ecg = [], None
    print(__doc__)

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user == "/quit":
            break
        if user == "/clear":
            history, ecg = [], None
            print("cleared\n")
            continue
        if user.startswith("/ecg "):
            try:
                ecg = load_ecg(user[5:].strip(), cfg)
                print(f"ECG loaded {tuple(ecg.shape)}\n")
            except (OSError, KeyError, IndexError) as e:
                print(f"could not load ECG: {e}\n")
            continue

        history.append({"from": "human", "value": user})
        messages = to_messages(history, cfg, system)
        ids = render_inference(tok, messages, cfg)
        batch = {"input_ids": torch.tensor([ids]),
                 "attention_mask": torch.ones(1, len(ids))}
        if uses_signal:
            pos = [i for i, t in enumerate(ids) if t == tok.convert_tokens_to_ids(SIGNAL_TOKEN)]
            batch["signal_pos"] = torch.tensor([pos or [-1] * cfg.num_encoder_tokens])
            if ecg is None:  # text-only turn: zero signal, injection disabled
                shape = (len(cfg.leads) * cfg.segment_len,) if cfg.elm == "base_elf" \
                    else (len(cfg.leads), cfg.segment_len)
                batch["ecg"] = torch.zeros(*shape).unsqueeze(0)
                batch["signal_pos"] = torch.full_like(batch["signal_pos"], -1)
            else:
                batch["ecg"] = ecg.unsqueeze(0)
        batch = {k: v.to(dist.device()) for k, v in batch.items()}
        out = model.generate(**batch, max_new_tokens=cfg.max_new_tokens)[0].tolist()
        cut = next((i for i, t in enumerate(out) if t in stop), len(out))
        text = tok.decode(out[:cut], skip_special_tokens=True).strip()
        stored = f"<think>\n{text}" if cfg.explicit_thinking else text
        history.append({"from": "gpt", "value": stored})
        print(f"ELM: {text}\n")


if __name__ == "__main__":
    main()
