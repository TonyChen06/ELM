"""One masked-autoencoder ViT over 12-lead ECG, three patching variants.

  st_mem  patches each lead over time, adds lead + separator embeddings,
          and returns encoder tokens (no decoder).
  mtae    patches time across all leads jointly; embeddings come from the
          MAE decoder's latents.
  mlae    one patch per lead (whole-lead tokens); decoder latents likewise.

All variants share the patch-embed -> transformer -> norm trunk and pool to
cfg.num_encoder_tokens output tokens. Attention is SDPA.
"""
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4, head_dim=64):
        super().__init__()
        inner = heads * head_dim
        self.norm1 = nn.LayerNorm(dim)
        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner * 3, bias=True)
        self.proj = nn.Linear(inner, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))

    def forward(self, x):
        q, k, v = rearrange(self.to_qkv(self.norm1(x)), "b n (three h d) -> three b h n d",
                            three=3, h=self.heads)
        att = F.scaled_dot_product_attention(q, k, v)
        x = x + self.proj(rearrange(att, "b h n d -> b n (h d)"))
        return x + self.mlp(self.norm2(x))


def sincos_1d(dim, length, cls=True):
    pos = torch.arange(length, dtype=torch.float32)[:, None]
    omega = 1.0 / 10000 ** (torch.arange(dim // 2, dtype=torch.float32) / (dim // 2))
    table = torch.cat([(pos * omega).sin(), (pos * omega).cos()], dim=1)
    return torch.cat([torch.zeros(1, dim), table]) if cls else table


def pick_patch_size(seq_len, lo=16, hi=64, target=32):
    sizes = [p for p in range(1, seq_len + 1) if seq_len % p == 0 and lo <= seq_len // p <= hi]
    return min(sizes or [s for s in range(1, seq_len + 1) if seq_len % s == 0],
               key=lambda p: abs(seq_len // p - target))


class MaskedViT1d(nn.Module):
    def __init__(self, variant, seq_len, num_tokens, num_leads=12, dim=768, depth=12, heads=12,
                 dec_dim=256, dec_depth=4, dec_heads=4):
        super().__init__()
        self.variant = variant
        patch = pick_patch_size(seq_len)
        self.n_patches = {"st_mem": seq_len // patch, "mtae": seq_len // patch, "mlae": num_leads}[variant]
        patch_dim = {"st_mem": patch, "mtae": patch * num_leads, "mlae": seq_len}[variant]
        self.patch = patch
        self.embed = nn.Sequential(nn.LayerNorm(patch_dim), nn.Linear(patch_dim, dim), nn.LayerNorm(dim))
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.pool = nn.AdaptiveAvgPool1d(num_tokens)

        if variant == "st_mem":
            self.pos = nn.Parameter(torch.randn(1, self.n_patches + 2, dim))
            self.sep = nn.Parameter(torch.randn(dim))
            self.lead = nn.Parameter(torch.randn(num_leads, dim))
        else:  # MAE: fixed sincos positions, cls token, decoder whose latents are the features
            self.pos = nn.Parameter(sincos_1d(dim, self.n_patches).unsqueeze(0), requires_grad=False)
            self.cls = nn.Parameter(torch.zeros(dim))
            self.to_dec = nn.Linear(dim, dec_dim)
            self.dec_pos = nn.Parameter(sincos_1d(dec_dim, self.n_patches).unsqueeze(0), requires_grad=False)
            self.dec_blocks = nn.ModuleList(Block(dec_dim, dec_heads) for _ in range(dec_depth))

    def tokens(self, ecg):
        """(B, leads, seq) -> per-variant token features."""
        if self.variant == "st_mem":
            x = rearrange(ecg, "b c (n p) -> b c n p", p=self.patch)
            x = self.embed(x) + self.pos[:, 1:-1].unsqueeze(1)
            sep = self.sep[None, None, None, :] + torch.stack([self.pos[:, 0], self.pos[:, -1]])[:, :, None]
            left, right = sep[0].expand(*x.shape[:2], 1, -1), sep[1].expand(*x.shape[:2], 1, -1)
            x = torch.cat([left, x, right], dim=2) + self.lead[None, :, None, :]
            x = rearrange(x, "b c n d -> b (c n) d")
            for block in self.blocks:
                x = block(x)
            x = rearrange(self.norm(x), "b (c n) d -> b c n d", c=ecg.shape[1])[:, :, 1:-1]
            return rearrange(x, "b c n d -> b (c n) d")
        if self.variant == "mtae":
            x = rearrange(ecg, "b c (n p) -> b n (p c)", p=self.patch)
        else:  # mlae: one token per lead
            x = ecg
        x = self.embed(x) + self.pos[:, 1:]
        cls = (self.cls + self.pos[0, 0]).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.to_dec(self.norm(x)) + self.dec_pos[:, : x.shape[1]]
        for block in self.dec_blocks:
            x = block(x)
        return x[:, 1:]

    def forward(self, ecg):
        x = self.tokens(ecg.float())
        return self.pool(x.transpose(1, 2)).transpose(1, 2)
