from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset


TINY_COLA = [
    ("The cat sat on the mat.", 1),
    ("The cat on the mat sat.", 1),
    ("Cat the sat mat on.", 0),
    ("She enjoys reading books.", 1),
    ("Enjoys she books reading.", 0),
    ("The dog chased the ball.", 1),
    ("Chased dog the ball the.", 0),
    ("A student solved the problem.", 1),
    ("Solved problem the student a.", 0),
] * 32


class SimpleTokenizer:
    def __init__(self, texts: List[str], vocab_size: int = 4096):
        vocab = {"[PAD]": 0, "[UNK]": 1}
        for text in texts:
            for tok in text.lower().replace(".", " .").split():
                if tok not in vocab and len(vocab) < vocab_size:
                    vocab[tok] = len(vocab)
        self.vocab = vocab
        self.vocab_size = len(vocab)

    def __call__(self, texts, padding=True, truncation=True, max_length=64, return_tensors=None):
        ids, mask = [], []
        for text in texts:
            row = [self.vocab.get(tok, 1) for tok in text.lower().replace(".", " .").split()]
            row = row[:max_length]
            attn = [1] * len(row)
            while len(row) < max_length:
                row.append(0)
                attn.append(0)
            ids.append(row)
            mask.append(attn)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer([self.texts[idx]], padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()} | {"labels": torch.tensor(self.labels[idx], dtype=torch.long)}


@dataclass
class ColaConfig:
    model_name: str = "microsoft/deberta-v3-base"
    data_dir: str = "glue_data/CoLA"
    max_length: int = 64
    batch_size: int = 8
    max_train_samples: int | None = 256
    max_eval_samples: int | None = 128
    use_tiny_fallback: bool = False


def _read_glue_cola_tsv(path: Path, max_samples: int | None = None):
    texts: list[str] = []
    labels: list[int] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            try:
                label = int(row[1])
            except ValueError:
                continue
            texts.append(row[-1])
            labels.append(label)
            if max_samples is not None and len(texts) >= max_samples:
                break
    return texts, labels


def _load_local_glue_cola(cfg: ColaConfig):
    data_dir = Path(cfg.data_dir)
    train_path = data_dir / "train.tsv"
    dev_path = data_dir / "dev.tsv"
    if not train_path.exists() or not dev_path.exists():
        raise FileNotFoundError(f"Expected {train_path} and {dev_path}")
    texts_train, labels_train = _read_glue_cola_tsv(train_path, cfg.max_train_samples)
    texts_val, labels_val = _read_glue_cola_tsv(dev_path, cfg.max_eval_samples)
    if not texts_train or not texts_val:
        raise ValueError(f"No CoLA rows found in {data_dir}")
    return texts_train, labels_train, texts_val, labels_val


def load_cola(cfg: ColaConfig):
    texts_train: list[str]
    labels_train: list[int]
    texts_val: list[str]
    labels_val: list[int]
    try:
        if cfg.use_tiny_fallback:
            raise RuntimeError("forced fallback")
        from transformers import AutoTokenizer

        try:
            texts_train, labels_train, texts_val, labels_val = _load_local_glue_cola(cfg)
            source = f"local_glue:{cfg.data_dir}"
        except Exception:
            from datasets import load_dataset

            ds = load_dataset("glue", "cola")
            train = ds["train"]
            val = ds["validation"]
            if cfg.max_train_samples:
                train = train.select(range(min(cfg.max_train_samples, len(train))))
            if cfg.max_eval_samples:
                val = val.select(range(min(cfg.max_eval_samples, len(val))))
            texts_train, labels_train = list(train["sentence"]), list(train["label"])
            texts_val, labels_val = list(val["sentence"]), list(val["label"])
            source = "hf_glue:glue/cola"
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    except Exception:
        split = int(0.8 * len(TINY_COLA))
        texts_train = [x for x, _ in TINY_COLA[:split]]
        labels_train = [y for _, y in TINY_COLA[:split]]
        texts_val = [x for x, _ in TINY_COLA[split:]]
        labels_val = [y for _, y in TINY_COLA[split:]]
        tokenizer = SimpleTokenizer(texts_train + texts_val)
        source = "tiny_fallback"
    train_ds = TextDataset(texts_train, labels_train, tokenizer, cfg.max_length)
    val_ds = TextDataset(texts_val, labels_val, tokenizer, cfg.max_length)
    print(
        f"[cola] source={source} train_samples={len(train_ds)} val_samples={len(val_ds)} "
        f"train_batches={len(DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True))} "
        f"val_batches={len(DataLoader(val_ds, batch_size=cfg.batch_size))}"
    )
    return DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True), DataLoader(val_ds, batch_size=cfg.batch_size), tokenizer


def matthews_corrcoef(y_true, y_pred):
    y_true, y_pred = torch.tensor(y_true), torch.tensor(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return 0.0 if denom == 0 else (tp * tn - fp * fn) / denom
