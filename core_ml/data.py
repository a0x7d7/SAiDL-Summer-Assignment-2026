from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
from torch.utils.data import DataLoader, Dataset


TINY_TEXT = """
Long-context modelling studies how sequence models use information far back in history.
Attention is expressive but expensive because every token can compare with every earlier token.
Efficient variants restrict the pattern, share key-value heads, or use kernel tricks.
Position encodings decide whether a short-trained model can be evaluated at longer lengths.
Convolutions add local n-gram bias that is useful for language modelling.
"""


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8", errors="ignore"))

    def decode(self, ids: Iterable[int]) -> str:
        return bytes(int(i) for i in ids).decode("utf-8", errors="ignore")


def get_tokenizer(name: str):
    if name == "byte":
        return ByteTokenizer()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def encode(tokenizer, text: str) -> List[int]:
    if isinstance(tokenizer, ByteTokenizer):
        return tokenizer.encode(text)
    return tokenizer.encode(text, add_special_tokens=False)


def load_language_data(tokenizer_name: str, split_limit: int | None = None):
    tokenizer = get_tokenizer(tokenizer_name)
    texts = {}
    try:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
        for split in ["train", "validation", "test"]:
            lines = [row["text"] for row in ds[split] if row["text"].strip()]
            if split_limit:
                lines = lines[:split_limit]
            texts[split] = "\n".join(lines)
    except Exception:
        texts = {"train": TINY_TEXT * 128, "validation": TINY_TEXT * 16, "test": TINY_TEXT * 16}
    tensors = {k: torch.tensor(encode(tokenizer, v), dtype=torch.long) for k, v in texts.items()}
    vocab_size = int(getattr(tokenizer, "vocab_size", 256))
    return tensors, vocab_size


class SequenceDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, context_length: int, max_samples: int | None = None):
        if tokens.numel() < context_length + 2:
            repeats = (context_length + 2 + max(tokens.numel(), 1) - 1) // max(tokens.numel(), 1)
            tokens = tokens.repeat(repeats)
        self.tokens = tokens
        self.context_length = context_length
        self.max_samples = max_samples

    def __len__(self):
        n = max(0, self.tokens.numel() - self.context_length - 1)
        return min(n, self.max_samples) if self.max_samples else n

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.context_length]
        y = self.tokens[idx + 1 : idx + self.context_length + 1]
        return x, y


@dataclass
class DataConfig:
    tokenizer: str = "byte"
    context_length: int = 128
    batch_size: int = 4
    split_limit: int | None = None
    max_train_samples: int | None = 512
    max_val_samples: int | None = 128


def make_loaders(cfg: DataConfig):
    data, vocab_size = load_language_data(cfg.tokenizer, cfg.split_limit)
    train_ds = SequenceDataset(data["train"], cfg.context_length, cfg.max_train_samples)
    val_ds = SequenceDataset(data["validation"], cfg.context_length, cfg.max_val_samples)
    train = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    return train, val, vocab_size
