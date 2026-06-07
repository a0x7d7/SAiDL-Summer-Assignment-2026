from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float = 16.0, method: str = "lora", l1_lambda: float = 0.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.method = method
        self.l1_lambda = l1_lambda
        self.A = nn.Parameter(torch.randn(rank, base.in_features) * 0.02)
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        self.gate = nn.Parameter(torch.ones(rank))
        self.register_buffer("rank_mask", torch.ones(rank))
        for p in self.base.parameters():
            p.requires_grad = False

    @property
    def scale(self):
        return self.alpha / max(self.rank, 1)

    def active_vector(self):
        if self.method in {"sora", "adalora"}:
            return self.gate * self.rank_mask
        return self.rank_mask

    @property
    def weight(self):
        weight = self.base.weight
        active = self.active_vector().to(device=weight.device, dtype=weight.dtype)
        update = (self.B.to(device=weight.device, dtype=weight.dtype) * active.unsqueeze(0)) @ self.A.to(
            device=weight.device, dtype=weight.dtype
        )
        return weight + self.scale * update

    @property
    def bias(self):
        if self.base.bias is None:
            return None
        return self.base.bias.to(device=self.base.weight.device, dtype=self.base.weight.dtype)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def effective_rank(self, tol: float = 1e-4):
        return int((self.active_vector().detach().abs() > tol).sum().item())

    def proximal_step(self, lr: float):
        if self.method != "sora" or self.l1_lambda <= 0:
            return
        with torch.no_grad():
            self.gate.copy_(torch.sign(self.gate) * torch.clamp(self.gate.abs() - lr * self.l1_lambda, min=0.0))

    def adalora_prune_to(self, target_rank: int):
        if self.method != "adalora" or target_rank >= self.rank:
            return
        with torch.no_grad():
            score = self.gate.abs() * self.B.norm(dim=0) * self.A.norm(dim=1)
            keep = torch.topk(score, k=max(target_rank, 1)).indices
            mask = torch.zeros_like(self.rank_mask)
            mask[keep] = 1.0
            self.rank_mask.copy_(mask)


@dataclass
class AdapterConfig:
    method: str = "lora"
    rank: int = 8
    alpha: float = 16.0
    target_modules: tuple[str, ...] = (
        "query",
        "key",
        "value",
        "dense",
        "out_proj",
        "fc",
        "classifier",
        "input_proj",
        "recurrent",
        "in_proj",
        "x_proj",
        "dt_proj",
        "state_proj",
        "delta_proj",
        "b_proj",
        "c_proj",
    )
    l1_lambda: float = 1e-3
    adalora_target_rank: int = 4


def should_wrap(name: str, module: nn.Module, targets: tuple[str, ...]):
    if not isinstance(module, nn.Linear):
        return False
    lname = name.lower()
    return any(t in lname for t in targets)


def inject_adapters(model: nn.Module, cfg: AdapterConfig):
    replacements = []
    for name, module in model.named_modules():
        if should_wrap(name, module, cfg.target_modules):
            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            replacements.append((parent_name, child_name, module))
    for parent_name, child_name, module in replacements:
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, LoRALinear(module, cfg.rank, cfg.alpha, cfg.method, cfg.l1_lambda))
    for name, p in model.named_parameters():
        p.requires_grad = name.endswith(".A") or name.endswith(".B") or name.endswith(".gate")
    return model


def adapter_layers(model):
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


def trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def total_parameters(model):
    return sum(p.numel() for p in model.parameters())


def effective_rank(model):
    layers = adapter_layers(model)
    if not layers:
        return 0.0
    return sum(layer.effective_rank() for layer in layers) / len(layers)


def proximal_step(model, lr: float):
    for layer in adapter_layers(model):
        layer.proximal_step(lr)


def adalora_prune(model, target_rank: int):
    for layer in adapter_layers(model):
        layer.adalora_prune_to(target_rank)
