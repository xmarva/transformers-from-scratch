"""Lesson 5's recipes trained again with other seeds: how much the scores move by chance alone.

Runs as a Kaggle script kernel on a T4, one part per kernel (two GPU sessions in parallel): set PART to "a" or "b".
Part a: the 2017 design and the last rung (+ RoPE) for 5 epochs with seeds 43-46, and the 2017 design for 20
epochs with seed 43 (about 65 minutes). Part b: + RoPE for 20 epochs with seeds 43 and 44, the 2017 design with
seed 44 (about 80 minutes). The results are published as the dataset xmarva/tfs-seed-runs-en-ru, which lessons 5
and 6 read.
"""
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "sacrebleu==2.6.0",
                       "git+https://github.com/xmarva/transformers-from-scratch@lesson-05"])

import json
import math
import os
import random
import time
import warnings
from pathlib import Path

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import pandas as pd
import sacrebleu
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer

from tfs import decoding, env, training
from tfs.comet import Comet
from tfs.modern import Seq2Seq

PART = "a"
PLANS = {  # (name, setup, seed, epochs)
    "a": [("2017 s44", "2017", 44, 5), ("2017 s45", "2017", 45, 5), ("2017 s46", "2017", 46, 5),
          ("+ RoPE s43", "+ RoPE", 43, 5), ("+ RoPE s44", "+ RoPE", 44, 5), ("+ RoPE s45", "+ RoPE", 45, 5),
          ("+ RoPE s46", "+ RoPE", 46, 5), ("2017 s43, 20 epochs", "2017", 43, 20)],
    "b": [("+ RoPE s43, 20 epochs", "+ RoPE", 43, 20), ("+ RoPE s44, 20 epochs", "+ RoPE", 44, 20),
          ("2017 s44, 20 epochs", "2017", 44, 20)],
}
PLAN = PLANS[PART]
OUT = Path("/kaggle/working/seeds")
OUT.mkdir(parents=True, exist_ok=True)
device = "cuda"
SEED = 42
env.seed_everything(SEED)
T0 = time.time()
R = {"part": PART, "runs": {}}


def log(*args):
    print(f"[{(time.time() - T0) / 60:5.1f} min]", *args, flush=True)


def save():
    (OUT / f"results-{PART}.json").write_text(json.dumps(R, indent=1, ensure_ascii=False))


log("GPU", torch.cuda.get_device_name(0))
tokenizer = Tokenizer.from_file(hf_hub_download("xmarva/tfs-bpe-en-ru-16k", "tokenizer.json"))
splits = {}
for name in ["train", "validation", "test"]:
    path = hf_hub_download("xmarva/tfs-tatoeba-en-ru", f"data/{name}.parquet", repo_type="dataset")
    splits[name] = pd.read_parquet(path)
train, valid, test = splits["train"], splits["validation"], splits["test"]
VOCAB_SIZE = tokenizer.get_vocab_size()
PAD, BOS, EOS = (tokenizer.token_to_id(t) for t in ["<pad>", "<bos>", "<eos>"])


def encode_all(sentences):
    return [encoding.ids for encoding in tokenizer.encode_batch(list(sentences))]


train_src, train_tgt = encode_all(train.en), encode_all(train.ru)
valid_src, valid_tgt = encode_all(valid.en), encode_all(valid.ru)
keep = [i for i in range(len(train_src)) if len(train_src[i]) <= 64 and len(train_tgt[i]) + 1 <= 64]
special = {"pad_id": PAD, "bos_id": BOS, "eos_id": EOS}
train_batches = training.make_batches(train_src, train_tgt, keep, 6_000, **special)
valid_batches = training.make_batches(valid_src, valid_tgt, range(len(valid_src)), 6_000, **special)

VARIANTS = {
    "2017": {"norm": "post", "ffn": "relu", "pos": "sinusoidal"},
    "+ RoPE": {"norm": "rms", "ffn": "swiglu", "pos": "rope"},
}


def ablation_schedule(step):
    return 1e-3 * min(step / 1_000, math.sqrt(1_000 / step))


def new_model(setup, seed):
    torch.manual_seed(seed)
    return Seq2Seq(setup, VOCAB_SIZE, pad_id=PAD).to(device)


def train_model(setup, epochs, schedule, seed, checkpoint=None):  # lesson 5's train_model
    random.seed(seed)
    model = new_model(setup, seed)
    optimizer = training.adam(model)
    history, done, best = [], 0, float("inf")
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        losses = training.train_steps(model, optimizer, train_batches, schedule, len(train_batches), pad_id=PAD,
                                      first_step=done + 1, device=device)
        done += len(losses)
        loss = training.validation_loss(model, valid_batches, pad_id=PAD, device=device)
        history.append({"epoch": epoch, "validation loss": loss, "minutes": (time.perf_counter() - start) / 60})
        if checkpoint is not None and loss < best:
            best = loss
            save_file(model.state_dict(), checkpoint)
    if checkpoint is not None:
        model.load_state_dict(load_file(checkpoint, device=device))
    return model.eval(), history


comet = Comet(device)
sample = test.sample(5_000, random_state=SEED).reset_index(drop=True)
sentences = list(dict.fromkeys(test.en))
for name, key, seed, epochs in PLAN:
    if epochs == 5:  # lesson 5's ladder: greedy on the 5,000-pair sample
        model, history = train_model(VARIANTS[key], 5, ablation_schedule, seed)
        frame, hyps = sample, decoding.greedy(model, tokenizer, list(sample.en))
    else:  # lesson 5's final recipe: 20 epochs, the 2017 schedule, the best epoch; beam on the whole test set
        checkpoint = str(OUT / f"{key.strip('+ ')}-{seed}.safetensors")
        model, history = train_model(VARIANTS[key], epochs, training.paper_schedule, seed, checkpoint)
        out = dict(zip(sentences, decoding.beam_search(model, tokenizer, sentences)))
        frame, hyps = test, list(test.en.map(out))
    frame[name] = hyps
    frame[f"COMET {name}"] = comet.score(frame.en, hyps, frame.ru)
    R["runs"][name] = {"setup": key, "seed": seed, "epochs": epochs, "history": history,
                       "validation loss": min(h["validation loss"] for h in history),
                       "BLEU": sacrebleu.corpus_bleu(hyps, [list(frame.ru)]).score,
                       "chrF": sacrebleu.corpus_chrf(hyps, [list(frame.ru)]).score,
                       "COMET": float(frame[f"COMET {name}"].mean())}
    log(name, {k: v for k, v in R["runs"][name].items() if k != "history"})
    save()
    if epochs == 5:
        sample.to_parquet(OUT / f"sample-{PART}.parquet")
    else:
        test.to_parquet(OUT / f"test-{PART}.parquet")
R["minutes"] = (time.time() - T0) / 60
save()
log("done")
