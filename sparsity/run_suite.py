from __future__ import annotations

import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

try:
    from .proximal import compare_updates
    from .train import run_experiment
except ImportError:
    from proximal import compare_updates
    from train import run_experiment


def run_or_load(cfg):
    out = Path(cfg["output_dir"]) / f"{cfg['name']}.json"
    if out.exists():
        return json.loads(out.read_text(encoding="utf-8"))
    return run_experiment(cfg)


def suite(config_path: str):
    base = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    records = []
    for method in ["lora", "adalora", "sora"]:
        cfg = copy.deepcopy(base)
        cfg["method"] = method
        cfg["name"] = method
        records.append(run_or_load(cfg))
    for mode in ["proximal", "sgd_l1"]:
        cfg = copy.deepcopy(base)
        cfg["method"] = "sora"
        cfg["sora_mode"] = mode
        cfg["name"] = f"sora_{mode}"
        records.append(run_or_load(cfg))
    for backbone in ["xlstm", "mamba"]:
        cfg = copy.deepcopy(base)
        cfg["backbone"] = backbone
        cfg["method"] = "sora"
        cfg["name"] = f"{backbone}_sora"
        cfg["epochs"] = int(base.get("sequence_epochs", cfg.get("epochs", 1)))
        cfg["max_steps"] = int(base.get("sequence_max_steps", cfg.get("max_steps", 100)))
        cfg["lr"] = float(base.get("sequence_lr", cfg.get("lr", 3e-4)))
        records.append(run_or_load(cfg))
    rows = []
    for rec in records:
        h = rec["history"][-1]
        rows.append(
            {
                "name": rec["config"]["name"],
                "method": rec["config"]["method"],
                "backbone": rec["config"].get("backbone", "deberta"),
                "mcc": h["mcc"],
                "accuracy": h["accuracy"],
                "eval_loss": h["eval_loss"],
                "trainable_params": rec["trainable_params"],
                "total_params": rec["total_params"],
                "effective_rank": rec["effective_rank"],
                "seconds": rec["seconds"],
            }
        )
    out_dir = Path(base["output_dir"])
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.md").write_text(df.to_markdown(index=False), encoding="utf-8")
    (out_dir / "proximal_vs_sgd.json").write_text(json.dumps(compare_updates(), indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["name"], df["effective_rank"])
    ax.set_ylabel("Mean effective rank")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out_dir / "effective_rank.png", dpi=160)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    print(suite(parser.parse_args().config).to_markdown(index=False))
