from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(t: int, device) -> torch.Tensor:
    return torch.tril(torch.ones(t, t, dtype=torch.bool, device=device))


def local_mask(t: int, window: int, device) -> torch.Tensor:
    i = torch.arange(t, device=device)[:, None]
    j = torch.arange(t, device=device)[None, :]
    return (j <= i) & (j >= i - window + 1)


def sparse_block_mask(t: int, block: int, device) -> torch.Tensor:
    i = torch.arange(t, device=device)[:, None]
    j = torch.arange(t, device=device)[None, :]
    same = (i // block) == (j // block)
    prev = (i // block) == (j // block + 1)
    global_first = j < min(block, t)
    return (j <= i) & (same | prev | global_first)


def rotate_half(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(q, k):
    _, _, t, d = q.shape
    freqs = 1.0 / (10000 ** (torch.arange(0, d, 2, device=q.device).float() / d))
    pos = torch.arange(t, device=q.device).float()
    angles = torch.einsum("t,f->tf", pos, freqs)
    emb = torch.repeat_interleave(angles, 2, dim=-1)[None, None, :, :]
    return q * emb.cos() + rotate_half(q) * emb.sin(), k * emb.cos() + rotate_half(k) * emb.sin()


def alibi(heads: int, t: int, device) -> torch.Tensor:
    slopes = torch.tensor([2 ** (-8 * (i + 1) / heads) for i in range(heads)], device=device)
    dist = torch.arange(t, device=device)[None, :] - torch.arange(t, device=device)[:, None]
    return -slopes[:, None, None] * dist.clamp(max=0).abs().float()[None, :, :]


class RelativeBias(nn.Module):
    def __init__(self, heads: int, max_distance: int = 4096):
        super().__init__()
        self.max_distance = max_distance
        self.bias = nn.Embedding(2 * max_distance + 1, heads)

    def forward(self, t: int, device):
        pos = torch.arange(t, device=device)
        rel = (pos[:, None] - pos[None, :]).clamp(-self.max_distance, self.max_distance) + self.max_distance
        return self.bias(rel).permute(2, 0, 1).unsqueeze(0)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, attention_type: str, pos_encoding: str, dropout: float, window: int, block: int, kv_heads: Optional[int]):
        super().__init__()
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        self.attention_type = attention_type
        self.pos_encoding = pos_encoding
        self.window = window
        self.block = block
        self.kv_heads = kv_heads or heads
        if dim % heads or heads % self.kv_heads:
            raise ValueError("invalid head configuration")
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, self.kv_heads * self.head_dim)
        self.v = nn.Linear(dim, self.kv_heads * self.head_dim)
        self.o = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.rel = RelativeBias(heads) if pos_encoding == "relative" else None

    def shape_q(self, x):
        b, t, _ = x.shape
        return x.view(b, t, self.heads, self.head_dim).transpose(1, 2)

    def shape_kv(self, x):
        b, t, _ = x.shape
        x = x.view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        if self.kv_heads != self.heads:
            x = x.repeat_interleave(self.heads // self.kv_heads, dim=1)
        return x

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = self.shape_q(self.q(x)), self.shape_kv(self.k(x)), self.shape_kv(self.v(x))
        if self.pos_encoding == "rope":
            q, k = apply_rope(q, k)
        if self.attention_type == "linear":
            qf, kf = F.elu(q) + 1.0, F.elu(k) + 1.0
            kv = torch.einsum("bhtd,bhte->bhtde", kf, v).cumsum(dim=2)
            ks = kf.cumsum(dim=2)
            denom = torch.einsum("bhtd,bhtd->bht", qf, ks).clamp_min(1e-6)
            out = torch.einsum("bhtd,bhtde->bhte", qf, kv) / denom[..., None]
        elif self.attention_type == "relu":
            scores = F.relu(q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(~causal_mask(t, x.device)[None, None], 0.0)
            out = (scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)) @ v
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.pos_encoding == "alibi":
                scores = scores + alibi(self.heads, t, x.device).unsqueeze(0)
            if self.rel is not None:
                scores = scores + self.rel(t, x.device)
            mask = causal_mask(t, x.device)
            if self.attention_type == "local":
                mask = local_mask(t, self.window, x.device)
            elif self.attention_type == "sparse_block":
                mask = sparse_block_mask(t, self.block, x.device)
            scores = scores.masked_fill(~mask[None, None], torch.finfo(scores.dtype).min)
            out = self.drop(F.softmax(scores, dim=-1)) @ v
        return self.o(out.transpose(1, 2).contiguous().view(b, t, self.dim))


class Conv1D(nn.Module):
    def __init__(self, dim: int, kernel: int):
        super().__init__()
        self.depth = nn.Conv1d(dim, dim, kernel, padding=kernel - 1, groups=dim)
        self.point = nn.Conv1d(dim, dim, 1)

    def forward(self, x):
        t = x.size(1)
        y = self.depth(x.transpose(1, 2))[..., :t]
        return F.gelu(self.point(y).transpose(1, 2))


class Block(nn.Module):
    def __init__(self, cfg: "ModelConfig", idx: int):
        super().__init__()
        self.replace = cfg.conv_mode == "replace_every_other" and idx % 2 == 1
        self.pre_conv = Conv1D(cfg.dim, cfg.conv_kernel) if cfg.conv_mode == "pre_conv" else None
        self.inter_conv = Conv1D(cfg.dim, cfg.conv_kernel) if cfg.conv_mode == "interleaved" else None
        self.conv = Conv1D(cfg.dim, cfg.conv_kernel) if self.replace else None
        self.ln1 = nn.LayerNorm(cfg.dim)
        self.ln2 = nn.LayerNorm(cfg.dim)
        self.attn = CausalSelfAttention(cfg.dim, cfg.heads, cfg.attention_type, cfg.pos_encoding, cfg.dropout, cfg.window_size, cfg.sparse_block_size, cfg.kv_heads)
        hidden = cfg.ff_dim
        self.gated = cfg.conv_mode == "gated_ffn"
        self.fc1 = nn.Linear(cfg.dim, hidden)
        self.gate = nn.Linear(cfg.dim, hidden) if self.gated else None
        self.fc2 = nn.Linear(hidden, cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        if self.pre_conv is not None:
            x = x + self.pre_conv(self.ln1(x))
        if self.replace:
            x = x + self.conv(self.ln1(x))
        else:
            x = x + self.attn(self.ln1(x))
        if self.inter_conv is not None:
            x = x + self.inter_conv(self.ln2(x))
        h = F.gelu(self.fc1(self.ln2(x)))
        if self.gated:
            h = h * torch.sigmoid(self.gate(self.ln2(x)))
        return x + self.drop(self.fc2(h))


@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int = 128
    dim: int = 128
    layers: int = 2
    heads: int = 4
    ff_dim: int = 512
    dropout: float = 0.1
    attention_type: str = "standard"
    pos_encoding: str = "learned"
    window_size: int = 128
    sparse_block_size: int = 64
    kv_heads: Optional[int] = None
    conv_mode: str = "none"
    conv_kernel: int = 5


class LongContextLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos = nn.Embedding(cfg.context_length, cfg.dim) if cfg.pos_encoding == "learned" else None
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.layers)])
        self.ln = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

    def sinusoidal(self, t, device):
        pos = torch.arange(t, device=device).float()[:, None]
        div = torch.exp(torch.arange(0, self.cfg.dim, 2, device=device).float() * (-math.log(10000.0) / self.cfg.dim))
        pe = torch.zeros(t, self.cfg.dim, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe[None]

    def forward(self, idx, targets=None):
        _, t = idx.shape
        x = self.tok(idx)
        if self.pos is not None:
            x = x + self.pos(torch.arange(t, device=idx.device))[None]
        elif self.cfg.pos_encoding == "sinusoidal":
            x = x + self.sinusoidal(t, idx.device)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln(x))
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return {"logits": logits, "loss": loss}
