"""Full composition matrix: build -> forward -> backward -> optimizer step ->
save -> strict reload into a fresh model -> forward. Real data, every viable
(representation, elm, encoder, llm) combination over the locally cached LLMs.

Slow (rebuilds the model per combo): run with -m matrix.
  uv run python -m pytest tests/test_full_matrix.py -q -m matrix
"""
import sys

sys.path.insert(0, "src")
import pytest
import torch

from config import Config, set_seed
from data.collate import Collator
from data.dataset import build_dataset
from data.templates import build_tokenizer
from models.elm import build_elm
from train.optim import build_optimizer, build_scheduler

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
QWEN, LLAMA = "qwen2.5-0.5b-instruct", "llama-3.2-1b-instruct"

ELFS = [("signal", "patch_elf", "", 100), ("signal", "base_elf", "", 1), ("signal", "conv_elf", "", 100)]
ECG_LLAVAS = [("signal", elm, enc, 50) for elm in ("mlp_llava", "linear_llava")
              for enc in ("st_mem", "mtae", "mlae", "merl")]
VISION = [(rep, elm, enc, 1) for rep in ("rgb", "stacked_signal")
          for elm in ("mlp_llava", "linear_llava") for enc in ("clip", "vit")]
SYMBOLIC = [("symbolic", "ecg_byte", "", 1)]
# full matrix on the small LLM; spot checks across each family on the larger one
MATRIX = [(QWEN, *c) for c in ELFS + ECG_LLAVAS + VISION + SYMBOLIC] + \
         [(LLAMA, *c) for c in [ELFS[0], ECG_LLAVAS[0], VISION[0], SYMBOLIC[0]]]
# not locally available: gemma-2-2b-it (LLM), siglip2 (vision encoder)


@pytest.mark.matrix
@pytest.mark.parametrize("llm,rep,elm,encoder,n_tokens", MATRIX)
def test_build_step_save_reload(llm, rep, elm, encoder, n_tokens, tmp_path):
    set_seed(0)
    cfg = Config(data=["ecg-qa-ptbxl-250-2500"], representation=rep, elm=elm, encoder=encoder,
                 llm=llm, num_encoder_tokens=n_tokens, data_subset=0.001,
                 system_prompt="prompts/system_prompt.txt", batch_size=2, lr=1e-4)
    cfg.mode = "train"
    cfg.max_steps = 4
    tok = build_tokenizer(cfg)
    dataset = build_dataset(cfg, tok)
    batch = Collator(cfg, tok.pad_token_id)([dataset[0], dataset[1]])
    batch = {k: v.to(DEV) for k, v in batch.items()}

    model = build_elm(cfg, tok).to(DEV)
    model.train()
    optimizers = build_optimizer(cfg, model)
    schedulers = build_scheduler(cfg, optimizers)

    out = model(**batch)
    assert torch.isfinite(out.loss), "forward loss not finite"
    out.loss.backward()
    for opt in optimizers:
        opt.step()
    for sched in schedulers:
        sched.step()

    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    del model, optimizers, schedulers
    torch.cuda.empty_cache()

    set_seed(0)
    fresh = build_elm(cfg, tok).to(DEV)
    fresh.load_state_dict(torch.load(path, map_location=DEV, weights_only=True), strict=True)
    fresh.eval()
    with torch.no_grad():
        out2 = fresh(**batch)
    assert torch.isfinite(out2.loss), "post-reload loss not finite"
