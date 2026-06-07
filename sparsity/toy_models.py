from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyClassifier(nn.Module):
    def __init__(self, vocab_size: int = 4096, hidden: int = 64, classes: int = 2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.classifier = nn.Linear(hidden, classes)

    def forward(self, input_ids, attention_mask=None, labels=None, **_):
        x = self.emb(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        mask = attention_mask.unsqueeze(-1)
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        h = F.relu(self.fc1(pooled))
        h = F.relu(self.fc2(h))
        logits = self.classifier(h)
        loss = None if labels is None else F.cross_entropy(logits, labels)
        return type("Output", (), {"logits": logits, "loss": loss})


class TinyXLSTMClassifier(nn.Module):
    """xLSTM classifier.

    Uses the installed xlstm package when available. The small PyTorch cell below
    is kept as a fallback so tests and CPU-only environments still work.
    """

    def __init__(self, vocab_size: int = 4096, hidden: int = 64, classes: int = 2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.uses_real_xlstm = False
        try:
            from xlstm import mLSTMBlockConfig, mLSTMLayerConfig, xLSTMBlockStack, xLSTMBlockStackConfig

            cfg = xLSTMBlockStackConfig(
                mlstm_block=mLSTMBlockConfig(mlstm=mLSTMLayerConfig(embedding_dim=hidden, num_heads=4, context_length=64)),
                slstm_at=[],
                context_length=64,
                num_blocks=1,
                embedding_dim=hidden,
                dropout=0.0,
            )
            self.sequence = xLSTMBlockStack(cfg)
            self.uses_real_xlstm = True
        except Exception:
            self.input_proj = nn.Linear(hidden, 4 * hidden)
            self.recurrent = nn.Linear(hidden, 4 * hidden)
        self.norm = nn.LayerNorm(hidden)
        self.classifier = nn.Linear(hidden, classes)
        self.hidden = hidden

    def forward(self, input_ids, attention_mask=None, labels=None, **_):
        x = self.emb(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if self.uses_real_xlstm:
            h = self.sequence(x)
            mask = attention_mask.unsqueeze(-1).to(dtype=h.dtype)
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            logits = self.classifier(self.norm(pooled))
            loss = None if labels is None else F.cross_entropy(logits, labels)
            return type("Output", (), {"logits": logits, "loss": loss})
        h = torch.zeros(x.size(0), self.hidden, device=x.device)
        c = torch.zeros_like(h)
        stabilizer = torch.zeros_like(h)
        for t in range(x.size(1)):
            prev_h, prev_c, prev_stabilizer = h, c, stabilizer
            i_logit, f_logit, z, o = (self.input_proj(x[:, t]) + self.recurrent(h)).chunk(4, dim=-1)
            f_log = -F.softplus(f_logit)
            new_stabilizer = torch.maximum(f_log + stabilizer, i_logit)
            i = torch.exp(i_logit - new_stabilizer)
            f = torch.exp(f_log + stabilizer - new_stabilizer)
            c = f * c + i * torch.tanh(z)
            h = torch.sigmoid(o) * torch.tanh(c)
            keep = attention_mask[:, t].unsqueeze(-1).to(dtype=h.dtype)
            h = keep * h + (1.0 - keep) * prev_h
            c = keep * c + (1.0 - keep) * prev_c
            stabilizer = keep * new_stabilizer + (1.0 - keep) * prev_stabilizer
        logits = self.classifier(self.norm(h))
        loss = None if labels is None else F.cross_entropy(logits, labels)
        return type("Output", (), {"logits": logits, "loss": loss})


class TinyMambaClassifier(nn.Module):
    """Mamba classifier.

    Uses mamba_ssm.Mamba when available. The selective-SSM block below is the
    fallback used only when the package cannot be imported.
    """

    def __init__(self, vocab_size: int = 4096, hidden: int = 64, classes: int = 2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.uses_real_mamba = False
        try:
            from mamba_ssm import Mamba

            self.sequence = Mamba(d_model=hidden, d_state=16, d_conv=4, expand=2, use_fast_path=False)
            self.uses_real_mamba = True
        except Exception:
            pass
        self.in_proj = nn.Linear(hidden, 2 * hidden)
        self.delta_proj = nn.Linear(hidden, hidden)
        self.b_proj = nn.Linear(hidden, hidden)
        self.c_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)
        self.a_log = nn.Parameter(torch.zeros(hidden))
        self.skip = nn.Parameter(torch.ones(hidden))
        self.classifier = nn.Linear(hidden, classes)
        self.hidden = hidden

    def forward(self, input_ids, attention_mask=None, labels=None, **_):
        x = self.emb(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if self.uses_real_mamba and x.is_cuda:
            h = self.sequence(x)
            mask = attention_mask.unsqueeze(-1).to(dtype=h.dtype)
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            logits = self.classifier(pooled)
            loss = None if labels is None else F.cross_entropy(logits, labels)
            return type("Output", (), {"logits": logits, "loss": loss})
        state = torch.zeros(x.size(0), self.hidden, device=x.device)
        pooled = torch.zeros_like(state)
        denom = torch.zeros(x.size(0), 1, device=x.device)
        a = -torch.exp(self.a_log).unsqueeze(0)
        for t in range(x.size(1)):
            u, gate = self.in_proj(x[:, t]).chunk(2, dim=-1)
            delta = F.softplus(self.delta_proj(u))
            b = torch.tanh(self.b_proj(u))
            c = torch.tanh(self.c_proj(u))
            decay = torch.exp(torch.clamp(delta * a, min=-30.0, max=0.0))
            proposed = decay * state + (1.0 - decay) * b
            y = c * proposed + self.skip * u
            y = torch.sigmoid(gate) * y
            keep = attention_mask[:, t].unsqueeze(-1).to(dtype=y.dtype)
            state = keep * proposed + (1.0 - keep) * state
            pooled = pooled + keep * y
            denom = denom + keep
        logits = self.classifier(torch.tanh(self.out_proj(pooled / denom.clamp_min(1.0))))
        loss = None if labels is None else F.cross_entropy(logits, labels)
        return type("Output", (), {"logits": logits, "loss": loss})
