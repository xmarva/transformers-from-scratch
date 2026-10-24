"""Greedy decoding and beam search of lesson 3, run on many sentences at once, so that later lessons can import them.

The search is the lesson's, step for step: the same candidates, the same order, the same length penalty. Only the
calls to the model are batched: at every step the live beams of all sentences in a batch go through the decoder
together. Lesson 4 checks that a batch gives the translations that one sentence at a time does.
"""

from __future__ import annotations

import torch


def _special_ids(tokenizer):
    return tokenizer.token_to_id("<pad>"), tokenizer.token_to_id("<bos>"), tokenizer.token_to_id("<eos>")


def _pad(sequences, pad_id, device):
    longest = max(len(ids) for ids in sequences)
    batch = torch.full((len(sequences), longest), pad_id, device=device)
    for row, ids in enumerate(sequences):
        batch[row, : len(ids)] = torch.tensor(ids, device=device)
    return batch


def _batches(sentences, tokenizer, batch_size):
    """Indices of the sentences in batches of similar length, so that little of a batch is padding."""
    lengths = [len(tokenizer.encode(sentence).ids) for sentence in sentences]
    order = sorted(range(len(sentences)), key=lambda i: lengths[i])
    return [order[start : start + batch_size] for start in range(0, len(order), batch_size)]


def length_penalty(length, alpha=0.6):
    """Wu et al. (2016): ((5 + |y|) / 6) ** alpha."""
    return ((5 + length) / 6) ** alpha


@torch.no_grad()
def greedy(model, tokenizer, sentences, max_tokens=60, batch_size=256):
    """Translate every sentence by taking the most probable token at every step."""
    model.eval()
    device = next(model.parameters()).device
    pad, bos, eos = _special_ids(tokenizer)
    translations = [None] * len(sentences)
    for indices in _batches(sentences, tokenizer, batch_size):
        src = _pad([tokenizer.encode(sentences[i]).ids for i in indices], pad, device)
        memory = model.encode(src)
        written = torch.full((len(indices), 1), bos, device=device)
        done = torch.zeros(len(indices), dtype=torch.bool, device=device)
        for _ in range(max_tokens):
            next_ids = model.decode(written, memory, src)[:, -1].argmax(dim=-1)
            next_ids[done] = pad  # a finished sentence only collects padding
            written = torch.cat([written, next_ids[:, None]], dim=1)
            done |= next_ids == eos
            if done.all():
                break
        for row, i in enumerate(indices):
            ids = []
            for token_id in written[row, 1:].tolist():
                if token_id in (eos, pad):
                    break
                ids.append(token_id)
            translations[i] = tokenizer.decode(ids)
    return translations


@torch.no_grad()
def beam_search(model, tokenizer, sentences, beam_size=4, alpha=0.6, max_tokens=60, batch_size=64):
    """Translate every sentence with the beam search of lesson 3 and its length penalty."""
    model.eval()
    device = next(model.parameters()).device
    pad, bos, eos = _special_ids(tokenizer)
    translations = [None] * len(sentences)
    for indices in _batches(sentences, tokenizer, batch_size):
        src = _pad([tokenizer.encode(sentences[i]).ids for i in indices], pad, device)
        memory = model.encode(src)
        beams = [[([bos], 0.0)] for _ in indices]  # per sentence: (tokens so far, total log-probability)
        finished = [[] for _ in indices]
        active = [True for _ in indices]

        for _ in range(max_tokens):
            rows, owners = [], []
            for k in range(len(indices)):
                if active[k]:
                    for tokens, _ in beams[k]:
                        rows.append(tokens)
                        owners.append(k)
            if not rows:
                break
            owner = torch.tensor(owners, device=device)
            tgt = torch.tensor(rows, device=device)  # all live beams have the same length
            scores = model.decode(tgt, memory[owner], src[owner])[:, -1]
            top = scores.float().log_softmax(dim=-1).topk(beam_size)
            values, ids = top.values.tolist(), top.indices.tolist()

            row = 0
            for k in range(len(indices)):
                if not active[k]:
                    continue
                candidates = []
                for tokens, total in beams[k]:
                    for log_p, token_id in zip(values[row], ids[row], strict=True):
                        candidates.append((tokens + [token_id], total + log_p))
                    row += 1
                candidates.sort(key=lambda candidate: candidate[1], reverse=True)
                beams[k] = []
                for tokens, total in candidates:
                    if tokens[-1] == eos:
                        finished[k].append((tokens, total))
                    else:
                        beams[k].append((tokens, total))
                    if len(beams[k]) == beam_size:
                        break
                if len(finished[k]) >= beam_size:
                    active[k] = False

        for k, i in enumerate(indices):

            def normalized(candidate):
                tokens, total = candidate
                return total / length_penalty(len(tokens) - 1, alpha)  # length without <bos>

            best_tokens, _ = max(finished[k] or beams[k], key=normalized)
            translations[i] = tokenizer.decode(best_tokens[1:])  # decode drops <eos>
    return translations
