"""The composed ECG-language model: [encoder] -> [connector] -> HF causal LM.

One module covers every composition in the registry:
  llava-style   encoder + connector, embeddings injected at <signal> positions
  elf-style     connector only (consumes the raw signal directly)
  ecg_byte      neither (signal arrives as vocabulary tokens)
"""
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from registry import ELMS, ENCODERS, LLMS, build_connector


class Elm(nn.Module):
    def __init__(self, llm, encoder=None, connector=None, update=("connector", "llm")):
        super().__init__()
        self.llm = llm
        self.encoder = encoder
        self.connector = connector
        self.update = set(update)
        for name, module in self.parts():
            module.requires_grad_(name in self.update)

    def parts(self):
        return [(n, m) for n, m in
                (("encoder", self.encoder), ("connector", self.connector), ("llm", self.llm)) if m is not None]

    def train(self, mode=True):
        super().train(mode)
        for name, module in self.parts():
            module.train(mode and name in self.update)
        return self

    def embeddings(self, input_ids, signal_pos, **features):
        """Token embeddings with connector outputs injected at signal positions."""
        embeds = self.llm.get_input_embeddings()(input_ids)
        if self.connector is None or signal_pos is None or not features:
            return embeds
        if self.encoder is not None:
            with torch.set_grad_enabled(self.training and "encoder" in self.update):
                x = self.encoder(features["ecg"]) if "ecg" in features else self.encoder(**features)
        else:
            x = features["ecg"]  # ELF: the connector consumes the raw signal
        signal = self.connector(x).to(embeds.dtype)
        valid = signal_pos >= 0
        if valid.any():
            batch_idx = torch.arange(embeds.shape[0], device=embeds.device).unsqueeze(1).expand_as(signal_pos)
            embeds = embeds.clone()
            embeds[batch_idx[valid], signal_pos[valid]] = signal[valid]
        return embeds

    def forward(self, input_ids, attention_mask, labels=None, signal_pos=None, **features):
        embeds = self.embeddings(input_ids, signal_pos, **features)
        return self.llm(inputs_embeds=embeds, attention_mask=attention_mask, labels=labels)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, signal_pos=None, **kwargs):
        features = {k: kwargs.pop(k) for k in list(kwargs) if isinstance(kwargs[k], torch.Tensor)}
        embeds = self.embeddings(input_ids, signal_pos, **features)
        was_checkpointing = getattr(self.llm, "is_gradient_checkpointing", False)
        if was_checkpointing:
            self.llm.gradient_checkpointing_disable()
        self.llm.config.use_cache = True
        try:
            return self.llm.generate(inputs_embeds=embeds, attention_mask=attention_mask,
                                     pad_token_id=self.llm.config.pad_token_id, **kwargs)
        finally:
            if was_checkpointing:
                self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                self.llm.config.use_cache = False


def build_elm(cfg, tokenizer):
    spec = ELMS[cfg.elm]
    dtype = getattr(torch, cfg.param_dtype)
    hf_id = LLMS[cfg.llm]["hf_id"]
    if cfg.scratch:
        config = AutoConfig.from_pretrained(hf_id, attn_implementation=cfg.attention)
        llm = AutoModelForCausalLM.from_config(config).to(dtype)
    else:
        llm = AutoModelForCausalLM.from_pretrained(hf_id, dtype=dtype, attn_implementation=cfg.attention)
    llm.resize_token_embeddings(len(tokenizer))
    llm.config.pad_token_id = tokenizer.pad_token_id
    if cfg.gradient_checkpointing:
        llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        llm.config.use_cache = False
    if cfg.peft:
        from peft import LoraConfig, TaskType, get_peft_model
        llm = get_peft_model(llm, LoraConfig(r=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
                                             lora_dropout=cfg.lora_dropout, task_type=TaskType.CAUSAL_LM))

    encoder = connector = None
    if cfg.encoder:
        enc_spec = ENCODERS[cfg.encoder]
        encoder = enc_spec.build(cfg)
        if cfg.encoder_ckpt:
            state = torch.load(cfg.encoder_ckpt, map_location="cpu", weights_only=True)
            result = encoder.load_state_dict(state.get("model_state_dict", state), strict=False)
            import dist
            if dist.is_main():
                print(f"[encoder] loaded {cfg.encoder_ckpt}: "
                      f"{len(encoder.state_dict()) - len(result.missing_keys)} matched, "
                      f"{len(result.missing_keys)} missing, {len(result.unexpected_keys)} unexpected")
        in_dim = enc_spec.embed_dim or encoder.embed_dim
    else:
        in_dim = None
    if spec.connector:
        connector = build_connector(spec.connector, in_dim, llm.config.hidden_size, cfg).to(dtype)

    model = Elm(llm, encoder, connector, update=cfg.update)
    if cfg.ckpt:
        state = torch.load(cfg.ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("model", state))
    return model
