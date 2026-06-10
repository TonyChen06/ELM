"""Connectors: encoder features (or raw signal, for ELF) -> LLM token embeddings.

Every connector returns (B, N, llm_hidden) and casts its input to the live
weight dtype, so the model can be moved between dtypes freely.
"""
from torch import nn
from torch.nn import functional as F


class Linear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        out = self.proj(x.to(self.proj.weight.dtype))
        return out.unsqueeze(1) if out.ndim == 2 else out


class Mlp(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))

    def forward(self, x):
        out = self.proj(x.to(self.proj[0].weight.dtype))
        return out.unsqueeze(1) if out.ndim == 2 else out


class Patch(nn.Module):
    """Non-overlapping time patches, each linearly projected to one token."""

    def __init__(self, num_patches, num_leads, seq_len, out_dim):
        super().__init__()
        if seq_len % num_patches:
            raise ValueError(f"segment_len {seq_len} not divisible by num_encoder_tokens {num_patches}")
        self.n = num_patches
        self.proj = nn.Linear(num_leads * (seq_len // num_patches), out_dim)

    def forward(self, ecg):
        b, c, t = ecg.shape
        x = ecg.reshape(b, c, self.n, t // self.n).permute(0, 2, 1, 3).reshape(b, self.n, -1)
        return self.proj(x.to(self.proj.weight.dtype))


class CnnPatch(nn.Module):
    """Per-patch conv stack -> pooled -> linear, one token per patch."""

    def __init__(self, num_patches, num_leads, out_dim):
        super().__init__()
        self.n = num_patches
        self.conv1 = nn.Conv1d(num_leads, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, 7, padding=3)
        self.fc = nn.Linear(128, out_dim)

    def forward(self, ecg):
        b, c, t = ecg.shape
        x = ecg.reshape(b, c, self.n, t // self.n).permute(0, 2, 1, 3).reshape(b * self.n, c, -1)
        x = F.relu(self.conv2(F.relu(self.conv1(x.to(self.conv1.weight.dtype)))))
        return self.fc(x.mean(dim=-1)).view(b, self.n, -1)
