"""Group-relative RL (SAPO): per prompt, sample G rollouts, score with
format/tag/answer rewards, compute group-relative advantages, and apply the
smoothed-gate policy loss. Degenerate groups (zero reward variance) keep their
forward/backward graph but contribute zero gradient, so DDP collectives stay
aligned across ranks."""
import re

import torch

import dist
from data.templates import IGNORE, end_of_turn_ids

_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


# ------------------------------ rewards -------------------------------------
def _answer_body(text):
    m = _ANSWER.search(text)
    return (m.group(1) if m else text).strip().lower()


def _labels(text):
    return {x.strip() for x in _answer_body(text).split(";") if x.strip()}


def reward(text, gt, explicit_thinking):
    tags = ("</think>", "<answer>", "</answer>") if explicit_thinking \
        else ("<think>", "</think>", "<answer>", "</answer>")
    fmt = re.compile((r"^\s*[\s\S]*?" if explicit_thinking else r"^\s*<think>[\s\S]*?")
                     + r"</think>\s*<answer>[\s\S]*?</answer>\s*$")
    p, g = _labels(text), _labels(gt)
    f1 = 2 * len(p & g) / max(len(p) + len(g), 1)
    return (float(bool(fmt.fullmatch(text)))
            + sum(text.count(t) == 1 for t in tags) / len(tags)
            + 0.5 * f1 + 0.5 * float(_answer_body(text) == _answer_body(gt)))


# ------------------------------ rollout -------------------------------------
def _response_mask(tokens, stop_ids, pad_id):
    """1 through the first end-of-turn token, 0 after; pad positions zeroed."""
    is_stop = torch.isin(tokens, stop_ids)
    mask = ((is_stop.cumsum(1) - is_stop.int()) == 0).float()
    if pad_id not in stop_ids.tolist():
        mask = mask * (tokens != pad_id)
    return mask


def _log_prob(model, input_ids, attention_mask, prompt_len, features):
    logits = model(input_ids, attention_mask, **features).logits[:, prompt_len - 1:-1]
    targets = input_ids[:, prompt_len:]
    return torch.log_softmax(logits.float(), -1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def sapo_loss(old_lp, lp, advantages, mask, tau_pos, tau_neg, agg, batch_n):
    ratio = (lp - old_lp).clamp(-20, 20).exp()
    taus = torch.where(advantages > 0, tau_pos, tau_neg)
    losses = -torch.sigmoid(taus * (ratio - 1.0)) * (4.0 / taus) * advantages
    per_token = losses * mask
    if agg == "token-mean":
        return per_token.sum() / mask.sum().clamp(min=1)
    seq = per_token.sum(-1)
    if agg == "seq-mean-token-mean":
        seq = seq / mask.sum(-1).clamp(min=1)
    elif agg == "seq-mean-token-sum-norm":
        seq = seq / mask.shape[-1]
    return seq.sum() / batch_n


def rl_step(cfg, tokenizer):
    stop_list = None

    def step(model, batch):
        nonlocal stop_list
        base = dist.unwrap(model)
        device = batch["input_ids"].device
        if stop_list is None:
            stop_list = torch.tensor(sorted(end_of_turn_ids(tokenizer, cfg.llm)), device=device)
        pad_id = tokenizer.pad_token_id
        B, G = batch["input_ids"].shape[0], cfg.rl_group_size
        total, mean_reward, kl = 0.0, 0.0, 0.0
        for i in range(B):
            labels = batch["labels"][i]
            start = int((labels != IGNORE).nonzero()[0])
            gt = tokenizer.decode(labels[labels != IGNORE].tolist(), skip_special_tokens=True)
            prompt = {"input_ids": batch["input_ids"][i:i + 1, :start].expand(G, -1).contiguous(),
                      "attention_mask": batch["attention_mask"][i:i + 1, :start].expand(G, -1).contiguous()}
            features = {k: v[i:i + 1].expand(G, *v.shape[1:]).contiguous() for k, v in batch.items()
                        if k not in ("input_ids", "attention_mask", "labels", "signal_pos")}
            if "signal_pos" in batch:
                pos = batch["signal_pos"][i:i + 1].expand(G, -1).contiguous()
                prompt["signal_pos"] = torch.where(pos < start, pos, torch.full_like(pos, -1))

            was_training = base.training
            base.eval()
            new = base.generate(**prompt, max_new_tokens=cfg.rl_max_new_tokens, do_sample=True,
                                temperature=cfg.rl_temperature, top_p=cfg.rl_top_p)
            if was_training:
                base.train()
            if new.shape[1] == 0:
                new = torch.full((G, 1), pad_id, dtype=torch.long, device=device)
            mask = _response_mask(new, stop_list, pad_id)
            rewards = torch.tensor([reward(tokenizer.decode(row[m.bool()].tolist(), skip_special_tokens=True),
                                           gt, cfg.explicit_thinking)
                                    for row, m in zip(new, mask)], device=device)
            degenerate = rewards.std(unbiased=False) < 1e-6
            adv = ((rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)).unsqueeze(1).expand_as(mask)

            ids = torch.cat([prompt["input_ids"], new], dim=1)
            attn = torch.cat([prompt["attention_mask"], mask], dim=1)
            lp_features = dict(features)
            if "signal_pos" in prompt:
                lp_features["signal_pos"] = prompt["signal_pos"]
            with torch.no_grad():
                old_lp = _log_prob(base, ids, attn, start, lp_features)
            lp = _log_prob(model, ids, attn, start, lp_features)
            loss = sapo_loss(old_lp, lp, adv, mask, cfg.rl_tau_pos, cfg.rl_tau_neg,
                             cfg.rl_loss_agg, batch_n=B * dist.world_size()) * dist.world_size()
            total = total + loss * (0.0 if degenerate else 1.0)
            mean_reward += rewards.mean().item() / B
            kl += ((old_lp - lp) * mask).sum().item() / mask.sum().clamp(min=1).item() / B
        return total, {"reward": mean_reward, "kl": kl}

    return step
