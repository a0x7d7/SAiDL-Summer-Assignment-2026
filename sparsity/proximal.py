from __future__ import annotations

import numpy as np
import torch


def numpy_sgd_l1_update(g, grad, lr, lam, subgradient_at_zero=0.0):
    subgrad = np.sign(g)
    subgrad[g == 0] = subgradient_at_zero
    return g - lr * (grad + lam * subgrad)


def numpy_prox_l1_update(g, grad, lr, lam):
    z = g - lr * grad
    return np.sign(z) * np.maximum(np.abs(z) - lr * lam, 0.0)


def torch_sgd_l1_update(g, grad, lr, lam, subgradient_at_zero=0.0):
    subgrad = torch.sign(g)
    subgrad = torch.where(g == 0, torch.full_like(g, subgradient_at_zero), subgrad)
    return g - lr * (grad + lam * subgrad)


def torch_prox_l1_update(g, grad, lr, lam):
    z = g - lr * grad
    return torch.sign(z) * torch.clamp(torch.abs(z) - lr * lam, min=0.0)


def compare_updates():
    g_np = np.array([-0.05, 0.0, 0.03, 0.7], dtype=np.float64)
    grad_np = np.array([0.2, -0.1, 0.05, -0.3], dtype=np.float64)
    lr, lam = 0.1, 0.5
    g_t = torch.tensor(g_np, dtype=torch.float64)
    grad_t = torch.tensor(grad_np, dtype=torch.float64)
    return {
        "numpy_sgd": numpy_sgd_l1_update(g_np, grad_np, lr, lam).tolist(),
        "numpy_prox": numpy_prox_l1_update(g_np, grad_np, lr, lam).tolist(),
        "torch_sgd": torch_sgd_l1_update(g_t, grad_t, lr, lam).tolist(),
        "torch_prox": torch_prox_l1_update(g_t, grad_t, lr, lam).tolist(),
    }


if __name__ == "__main__":
    print(compare_updates())
