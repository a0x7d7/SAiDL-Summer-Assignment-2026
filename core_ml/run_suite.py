from __future__ import annotations

import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

try:
    from .train import run_experiment
except ImportError:
    from train import run_experiment


def run_or_load(cfg):
    out = Path(cfg["output_dir"]) / f"{cfg['name']}.json"
    if out.exists():
        return json.loads(out.read_text(encoding="utf-8"))
    return run_experiment(cfg)


def row(rec):
    h = rec["history"][-1]
    m = rec["model_config"]
    return {
        "name": rec["config"]["name"],
        "context": rec["config"]["data"]["context_length"],
        "attention": m["attention_type"] if not m["kv_heads"] else "gqa",
        "position": m["pos_encoding"],
        "conv": m["conv_mode"],
        "train_loss": h["train_loss"],
        "val_loss": h["val_loss"],
        "ppl": h["val_ppl"],
        "epoch_seconds": h["epoch_seconds"],
        "train_tps": h["train_tokens_per_sec"],
        "infer_tps": rec["inference_tokens_per_sec"],
        "peak_memory_mb": rec["peak_memory_mb"],
        "stable": bool(pd.notna(h["train_loss"]) and pd.notna(h["val_loss"])),
    }


def suite(config_path: str):
    base = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    records = []
    baseline = copy.deepcopy(base)
    baseline["name"] = "baseline_ctx1024"
    baseline["data"]["context_length"] = 1024
    baseline["model"]["context_length"] = 1024
    records.append(run_or_load(baseline))
    for ctx in base.get("suite_contexts", [512, 1024, 2048]):
        for attention, kv in [("standard", None), ("local", None), ("sparse_block", None), ("linear", None), ("standard", 1)]:
            cfg = copy.deepcopy(base)
            cfg["name"] = f"attention_{'gqa' if kv else attention}_ctx{ctx}"
            cfg["data"]["context_length"] = ctx
            cfg["model"]["context_length"] = ctx
            cfg["model"]["attention_type"] = attention
            cfg["model"]["kv_heads"] = kv
            cfg["model"]["pos_encoding"] = "alibi"
            records.append(run_or_load(cfg))
    for pos in ["learned", "rope", "alibi", "relative"]:
        cfg = copy.deepcopy(base)
        cfg["name"] = f"position_{pos}_train512"
        cfg["data"]["context_length"] = 512
        cfg["model"]["context_length"] = 2048 if pos == "learned" else 512
        cfg["model"]["pos_encoding"] = pos
        cfg["eval_context_lengths"] = [512, 1024, 2048]
        records.append(run_or_load(cfg))
    for conv in ["pre_conv", "interleaved", "replace_every_other", "gated_ffn"]:
        cfg = copy.deepcopy(base)
        cfg["name"] = f"hybrid_{conv}_ctx512"
        cfg["data"]["context_length"] = 512
        cfg["model"]["context_length"] = 512
        cfg["model"]["attention_type"] = "linear"
        cfg["model"]["pos_encoding"] = "alibi"
        cfg["model"]["conv_mode"] = conv
        records.append(run_or_load(cfg))
    rows = [row(r) for r in records]
    for rec in records:
        for length, metrics in rec.get("context_evaluations", {}).items():
            if "val_ppl" in metrics:
                base_row = row(rec)
                base_row.update({"name": f"{base_row['name']}_eval{length}", "context": int(length), "val_loss": metrics["val_loss"], "ppl": metrics["val_ppl"], "train_loss": None, "epoch_seconds": None, "train_tps": None, "infer_tps": metrics["eval_tokens_per_sec"]})
                rows.append(base_row)
    out_dir = Path(base["output_dir"])
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.md").write_text(df.to_markdown(index=False), encoding="utf-8")
    att = df[df["name"].str.startswith("attention_") & ~df["name"].str.contains("_eval")]
    fig, ax = plt.subplots(figsize=(8, 4))
    for attention, group in att.groupby("attention"):
        ax.plot(group["context"], group["infer_tps"], marker="o", label=attention)
    ax.set_xlabel("Context length")
    ax.set_ylabel("Inference tokens/sec")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "attention_throughput.png", dpi=160)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    print(suite(parser.parse_args().config).to_markdown(index=False))
