"""The corpus of the experiments: English Wikipedia, Russian Wikipedia and Python code, in equal shares.

Three kinds of text that differ in alphabet, vocabulary and structure, so that it can be seen whether the experts of
a mixture-of-experts model divide the work between them along these lines. The texts come from pinned sources:

- the first articles of the 20231101 dumps of English and Russian Wikipedia (wikimedia/wikipedia on the Hugging
  Face Hub, at a fixed commit);
- the Python files of the standard library of CPython 3.12.7, tests included, from the release tarball.

A byte-level BPE tokenizer with 8,192 tokens is trained on equal samples of the three, and each text is cut to the
same number of tokens, its last 2% kept apart for validation. Everything goes to DATA_DIR (default
~/.cache/tfs-moe), outside the repository:

    uv run python articles/moe/prepare.py
"""

import json
import os
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

DATA_DIR = Path(os.environ.get("DATA_DIR", Path.home() / ".cache" / "tfs-moe"))
WIKIPEDIA = "wikimedia/wikipedia"
WIKIPEDIA_REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
CPYTHON = "https://www.python.org/ftp/python/3.12.7/Python-3.12.7.tgz"
VOCAB = 8192
TOKENS = 8_000_000  # per kind of text
CHARS = 45_000_000  # more than enough characters of each to reach TOKENS
VALIDATION = 0.02
DOMAINS = ["en", "ru", "code"]


def wikipedia(language):
    pages = load_dataset(WIKIPEDIA, f"20231101.{language}", split="train", streaming=True,
                         revision=WIKIPEDIA_REVISION)
    parts, size = [], 0
    for page in pages:
        text = page["title"] + "\n\n" + page["text"].strip() + "\n\n"
        parts.append(text)
        size += len(text)
        if size >= CHARS:
            break
    return "".join(parts)


def python_stdlib():
    archive = DATA_DIR / "Python-3.12.7.tgz"
    if not archive.exists():
        urllib.request.urlretrieve(CPYTHON, archive)
    parts = []
    with tarfile.open(archive) as tar:
        members = sorted(tar.getmembers(), key=lambda m: m.name)
        for member in members:
            path = member.name.split("/", 1)[-1]
            if not (path.startswith("Lib/") and path.endswith(".py") and member.isfile()):
                continue
            source = tar.extractfile(member).read().decode("utf-8", errors="replace")
            parts.append(f"# {path}\n{source}\n\n")
    return "".join(parts)


def train_tokenizer(texts):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  special_tokens=["<|doc|>"])
    sample = [text[:8_000_000] for text in texts]  # the same amount of each kind
    tokenizer.train_from_iterator((chunk[i:i + 100_000] for chunk in sample for i in range(0, len(chunk), 100_000)),
                                  trainer)
    return tokenizer


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    texts = {"en": wikipedia("en"), "ru": wikipedia("ru"), "code": python_stdlib()}
    for name, text in texts.items():
        print(f"{name}: {len(text):,} characters", flush=True)
    tokenizer = train_tokenizer([texts[name] for name in DOMAINS])
    tokenizer.save(str(DATA_DIR / "tokenizer.json"))
    meta = {"vocab": VOCAB, "wikipedia": f"{WIKIPEDIA}@{WIKIPEDIA_REVISION}", "code": CPYTHON, "domains": {}}
    for name in DOMAINS:
        text = texts[name]
        ids = []
        for i in range(0, len(text), 1_000_000):
            ids.extend(tokenizer.encode(text[i:i + 1_000_000]).ids)
            if len(ids) >= TOKENS:
                break
        if len(ids) < TOKENS:
            raise SystemExit(f"{name}: only {len(ids):,} tokens")
        ids = np.array(ids[:TOKENS], dtype=np.uint16)
        split = int(TOKENS * (1 - VALIDATION))
        ids[:split].tofile(DATA_DIR / f"{name}_train.bin")
        ids[split:].tofile(DATA_DIR / f"{name}_val.bin")
        used = len(tokenizer.decode(ids.tolist()))
        meta["domains"][name] = {"tokens": TOKENS, "characters": used, "characters_per_token": used / TOKENS}
        print(name, meta["domains"][name], flush=True)
    (DATA_DIR / "meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
