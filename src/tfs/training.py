"""The training loop of lesson 3, so that later lessons can import it.

The code is the lesson's: batches built from tokens with similar lengths side by side, cross-entropy with label
smoothing, Adam with a learning rate set before every step, gradient clipping and mixed precision. Only the
settings that lesson 3 kept in global variables (the special tokens, the vocabulary size) became arguments.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F


def pad_batch(sequences, pad_id):
    """Token id lists -> one tensor (B, T), shorter rows padded with `pad_id`."""
    longest = max(len(ids) for ids in sequences)
    batch = torch.full((len(sequences), longest), pad_id)
    for row, ids in enumerate(sequences):
        batch[row, : len(ids)] = torch.tensor(ids)
    return batch


def make_batches(src, tgt, rows, max_tokens, pad_id, bos_id, eos_id, sort=True):
    """Batches of at most `max_tokens` positions per side, padding included, as (src, tgt_in, tgt_out) tensors."""
    if sort:
        rows = sorted(rows, key=lambda i: (len(tgt[i]), len(src[i])))
    batches, current, longest = [], [], 0
    for i in rows:
        new_longest = max(longest, len(src[i]), len(tgt[i]) + 1)
        if current and new_longest * (len(current) + 1) > max_tokens:
            batches.append(current)
            current, new_longest = [], max(len(src[i]), len(tgt[i]) + 1)
        current.append(i)
        longest = new_longest
    batches.append(current)

    tensors = []
    for batch in batches:
        src_ids = pad_batch([src[i] for i in batch], pad_id)
        tgt_in = pad_batch([[bos_id] + tgt[i] for i in batch], pad_id)
        tgt_out = pad_batch([tgt[i] + [eos_id] for i in batch], pad_id)
        tensors.append((src_ids, tgt_in, tgt_out))
    return tensors


def batch_loss(model, batch, pad_id, label_smoothing=0.1, device="cuda"):
    src_ids, tgt_in, tgt_out = (t.to(device) for t in batch)
    logits = model(src_ids, tgt_in).float()  # the loss in float32
    flat_logits = logits.reshape(-1, logits.shape[-1])
    return F.cross_entropy(flat_logits, tgt_out.reshape(-1), ignore_index=pad_id, label_smoothing=label_smoothing)


def adam(model):
    """Adam with the 2017 paper's settings; the learning rate is set before every step, from a schedule."""
    return torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)


def paper_schedule(step, d_model=256, warmup=4000):
    """Vaswani et al. (2017): linear warm-up, then decay like 1/sqrt(step)."""
    step = max(step, 1)
    return d_model**-0.5 * min(step**-0.5, step * warmup**-1.5)


def train_steps(
    model, optimizer, batches, schedule, n_steps, pad_id, first_step=1, use_amp=True, clip=1.0, device="cuda"
):
    """`n_steps` training steps over the batches in shuffled order; returns the loss of every step."""
    use_amp = use_amp and device == "cuda"
    scaler = torch.amp.GradScaler(device, enabled=use_amp)
    model.train()
    losses = []
    while len(losses) < n_steps:
        order = list(range(len(batches)))
        random.shuffle(order)  # a new order on every pass over the data
        for index in order:
            for group in optimizer.param_groups:
                group["lr"] = schedule(first_step + len(losses))
            with torch.autocast(device, dtype=torch.float16, enabled=use_amp):
                loss = batch_loss(model, batches[index], pad_id, device=device)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
            if len(losses) == n_steps:
                break
    return losses


@torch.no_grad()
def validation_loss(model, batches, pad_id, device="cuda"):
    """Plain cross-entropy per target token, without label smoothing."""
    model.eval()
    total, count = 0.0, 0
    for src_ids, tgt_in, tgt_out in batches:
        src_ids, tgt_in, tgt_out = src_ids.to(device), tgt_in.to(device), tgt_out.to(device)
        with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
            logits = model(src_ids, tgt_in)
        flat = logits.reshape(-1, logits.shape[-1]).float()
        loss = F.cross_entropy(flat, tgt_out.reshape(-1), ignore_index=pad_id, reduction="sum")
        total += loss.item()
        count += int((tgt_out != pad_id).sum())
    return total / count
