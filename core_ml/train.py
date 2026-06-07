from __future__ import annotations

import argparse
import json
import math
import time
import tracemalloc
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm import tqdm

try:
    from .data import DataConfig, SequenceDataset, load_language_data, make_loaders
    from .model import LongContextLM, ModelConfig
except ImportError:
    from data import DataConfig, SequenceDataset, load_language_data, make_loaders
    from model import LongContextLM, ModelConfig


def auto_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    model.eval()
    losses, tokens = [], 0
    start = time.perf_counter()
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        out = model(x, y)
        losses.append(float(out["loss"]))
        tokens += x.numel()
    elapsed = max(time.perf_counter() - start, 1e-9)
    loss = sum(losses) / max(len(losses), 1)
    return {"val_loss": loss, "val_ppl": math.exp(min(loss, 20)), "eval_tokens_per_sec": tokens / elapsed}


@torch.no_grad()
def infer_bench(model, vocab_size, context, device, steps):
    model.eval()
    x = torch.randint(0, vocab_size, (1, context), device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {"inference_tokens_per_sec": steps * context / max(time.perf_counter() - start, 1e-9)}


def memory_mb(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024**2)
    _, peak = tracemalloc.get_traced_memory()
    return peak / (1024**2)


def run_experiment(cfg: dict[str, Any]):
    torch.manual_seed(int(cfg.get("seed", 0)))
    device_name = str(cfg.get("device", "auto"))
    device = auto_device() if device_name == "auto" else torch.device(device_name)
    data_cfg = DataConfig(**cfg.get("data", {}))
    train_loader, val_loader, vocab = make_loaders(data_cfg)
    model_args = dict(cfg.get("model", {}))
    model_ctx = int(model_args.pop("context_length", data_cfg.context_length))
    model = LongContextLM(ModelConfig(vocab_size=vocab, context_length=model_ctx, **model_args)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 3e-4)), weight_decay=float(cfg.get("weight_decay", 0.01)))
    epochs, max_steps = int(cfg.get("epochs", 1)), cfg.get("max_steps")
    max_steps = int(max_steps) if max_steps is not None else None
    tracemalloc.start()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history, step = [], 0
    total_start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        losses, tokens = [], 0
        epoch_start = time.perf_counter()
        for x, y in tqdm(train_loader, desc=f"core epoch {epoch+1}", leave=False):
            if max_steps is not None and step >= max_steps:
                break
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = model(x, y)["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 1.0)))
            opt.step()
            losses.append(float(loss.detach()))
            tokens += x.numel()
            step += 1
        elapsed = max(time.perf_counter() - epoch_start, 1e-9)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": sum(losses) / max(len(losses), 1),
                "train_tokens_per_sec": tokens / elapsed,
                "epoch_seconds": elapsed,
                **evaluate(model, val_loader, device, cfg.get("max_eval_batches")),
            }
        )
        if max_steps is not None and step >= max_steps:
            break
    metrics = {
        "config": cfg,
        "model_config": asdict(model.cfg),
        "history": history,
        "total_seconds": time.perf_counter() - total_start,
        "peak_memory_mb": memory_mb(device),
        **infer_bench(model, vocab, data_cfg.context_length, device, int(cfg.get("benchmark_steps", 3))),
    }
    if cfg.get("eval_context_lengths"):
        data, _ = load_language_data(data_cfg.tokenizer, data_cfg.split_limit)
        metrics["context_evaluations"] = {}
        for length in cfg["eval_context_lengths"]:
            if model.pos is not None and int(length) > model.cfg.context_length:
                metrics["context_evaluations"][str(length)] = {"skipped": "position table too short"}
                continue
            ds = SequenceDataset(data["validation"], int(length), data_cfg.max_val_samples)
            loader = torch.utils.data.DataLoader(ds, batch_size=data_cfg.batch_size)
            metrics["context_evaluations"][str(length)] = evaluate(model, loader, device, cfg.get("max_eval_batches"))
    out_dir = Path(cfg.get("output_dir", "results/core_ml"))
    out_dir.mkdir(parents=True, exist_ok=True)
    name = cfg.get("name", "core_ml_run")
    (out_dir / f"{name}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = yaml.safe_load(Path(parser.parse_args().config).read_text(encoding="utf-8"))
    print(json.dumps(run_experiment(cfg)["history"][-1], indent=2))


if __name__ == "__main__":
    main()
