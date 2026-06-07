from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm import tqdm

try:
    from .adapters import AdapterConfig, adalora_prune, adapter_layers, effective_rank, inject_adapters, proximal_step, trainable_parameters, total_parameters
    from .data import ColaConfig, load_cola, matthews_corrcoef
    from .toy_models import TinyClassifier, TinyMambaClassifier, TinyXLSTMClassifier
except ImportError:
    from adapters import AdapterConfig, adalora_prune, adapter_layers, effective_rank, inject_adapters, proximal_step, trainable_parameters, total_parameters
    from data import ColaConfig, load_cola, matthews_corrcoef
    from toy_models import TinyClassifier, TinyMambaClassifier, TinyXLSTMClassifier


def device_auto():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(kind: str, model_name: str, tokenizer, use_tiny: bool):
    if kind == "deberta" and not use_tiny:
        try:
            from transformers import AutoModelForSequenceClassification

            return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        except Exception:
            pass
    vocab = int(getattr(tokenizer, "vocab_size", len(getattr(tokenizer, "vocab", {})) or 4096))
    if kind == "xlstm":
        return TinyXLSTMClassifier(vocab)
    if kind == "mamba":
        return TinyMambaClassifier(vocab)
    return TinyClassifier(vocab)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels, losses = [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        losses.append(float(out.loss))
        preds.extend(out.logits.argmax(dim=-1).cpu().tolist())
        labels.extend(batch["labels"].cpu().tolist())
    acc = sum(int(a == b) for a, b in zip(preds, labels)) / max(len(labels), 1)
    return {"eval_loss": sum(losses) / max(len(losses), 1), "accuracy": acc, "mcc": matthews_corrcoef(labels, preds)}


def run_experiment(cfg: dict[str, Any]):
    torch.manual_seed(int(cfg.get("seed", 0)))
    device = device_auto() if str(cfg.get("device", "auto")) == "auto" else torch.device(cfg["device"])
    data_cfg = ColaConfig(**cfg.get("data", {}))
    train_loader, val_loader, tokenizer = load_cola(data_cfg)
    method = cfg.get("method", "lora")
    model = build_model(cfg.get("backbone", "deberta"), data_cfg.model_name, tokenizer, data_cfg.use_tiny_fallback)
    model = inject_adapters(model, AdapterConfig(method=method, **cfg.get("adapter", {}))).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(cfg.get("lr", 3e-4)))
    grad_clip = float(cfg.get("grad_clip", 1.0))
    epochs, max_steps = int(cfg.get("epochs", 1)), cfg.get("max_steps")
    max_steps = int(max_steps) if max_steps is not None else None
    step, start = 0, time.perf_counter()
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"{method} epoch {epoch+1}", leave=False):
            if max_steps is not None and step >= max_steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            if method == "sora" and cfg.get("sora_mode", "proximal") == "sgd_l1":
                lam = float(cfg.get("adapter", {}).get("l1_lambda", 1e-3))
                l1 = sum(layer.gate.abs().sum() for layer in adapter_layers(model))
                loss = loss + lam * l1
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
            opt.step()
            if method == "sora" and cfg.get("sora_mode", "proximal") == "proximal":
                proximal_step(model, float(cfg.get("lr", 3e-4)))
            step += 1
            losses.append(float(loss.detach()))
        if method == "adalora":
            adalora_prune(model, int(cfg.get("adapter", {}).get("adalora_target_rank", 4)))
        history.append({"epoch": epoch + 1, "train_loss": sum(losses) / max(len(losses), 1), **evaluate(model, val_loader, device)})
        if max_steps is not None and step >= max_steps:
            break
    result = {
        "config": cfg,
        "history": history,
        "seconds": time.perf_counter() - start,
        "trainable_params": trainable_parameters(model),
        "total_params": total_parameters(model),
        "effective_rank": effective_rank(model),
    }
    out_dir = Path(cfg.get("output_dir", "results/sparsity"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{cfg.get('name', method)}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = yaml.safe_load(Path(parser.parse_args().config).read_text(encoding="utf-8"))
    print(json.dumps(run_experiment(cfg), indent=2))


if __name__ == "__main__":
    main()
