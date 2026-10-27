# Greedy decoding and beam search of lesson 3, batched as in lesson 4, with lesson 5's KV
# cache: every step feeds the decoder only the newest token. A source longer than the
# model's positions is cut to fit, instead of failing inside the model.

import torch


def special_ids(tokenizer):
    pad_id = tokenizer.token_to_id("<pad>")
    return pad_id, tokenizer.token_to_id("<bos>"), tokenizer.token_to_id("<eos>")


def source_ids(tokenizer, sentence, limit):
    return tokenizer.encode(sentence).ids[:limit]


def pad(sequences, pad_id, device):
    longest = max(len(ids) for ids in sequences)
    batch = torch.full((len(sequences), longest), pad_id, device=device)
    for row, ids in enumerate(sequences):
        batch[row, : len(ids)] = torch.tensor(ids, device=device)
    return batch


def batches(sources, batch_size):
    # sentences in batches of similar length, so that little of a batch is padding
    order = sorted(range(len(sources)), key=lambda i: len(sources[i]))
    return [order[start : start + batch_size] for start in range(0, len(order), batch_size)]


def length_penalty(length, alpha=0.6):
    # Wu et al. (2016): ((5 + |y|) / 6) ** alpha
    return ((5 + length) / 6) ** alpha


def reorder(caches, rows):
    # the cached keys and values follow their hypotheses: row r of the new caches is row
    # rows[r] of the old ones
    for cache in caches:
        for part in cache.values():
            for key in ("k", "v"):
                if key in part:
                    part[key] = part[key][rows]


@torch.inference_mode()
def greedy(model, tokenizer, sentences, max_tokens=128, batch_size=256):
    model.eval()
    device = next(model.parameters()).device
    pad_id, bos, eos = special_ids(tokenizer)
    sources = []
    for sentence in sentences:
        sources.append(source_ids(tokenizer, sentence, model.config.max_len))
    max_tokens = min(max_tokens, model.config.max_len - 1)
    translations = [None] * len(sentences)
    for indices in batches(sources, batch_size):
        src = pad([sources[i] for i in indices], pad_id, device)
        memory_mask = (src != pad_id)[:, None, None, :]
        caches = model.new_caches(model.encode(src))
        token = torch.full((len(indices), 1), bos, device=device)
        written = []
        done = torch.zeros(len(indices), dtype=torch.bool, device=device)
        for position in range(max_tokens):
            scores = model.decode_step(token, memory_mask, caches, position)
            next_ids = scores.argmax(dim=-1)
            next_ids[done] = pad_id  # a finished sentence only collects padding
            written.append(next_ids)
            done |= next_ids == eos
            token = next_ids[:, None]
            if done.all():
                break
        written = torch.stack(written, dim=1)
        for row, i in enumerate(indices):
            ids = []
            for token_id in written[row].tolist():
                if token_id in (eos, pad_id):
                    break
                ids.append(token_id)
            translations[i] = tokenizer.decode(ids)
    return translations


@torch.inference_mode()
def beam_search(model, tokenizer, sentences, beam_size=4, alpha=0.6, max_tokens=128,
                batch_size=64):
    model.eval()
    device = next(model.parameters()).device
    pad_id, bos, eos = special_ids(tokenizer)
    sources = []
    for sentence in sentences:
        sources.append(source_ids(tokenizer, sentence, model.config.max_len))
    max_tokens = min(max_tokens, model.config.max_len - 1)
    translations = [None] * len(sentences)
    for indices in batches(sources, batch_size):
        src = pad([sources[i] for i in indices], pad_id, device)
        memory_mask = (src != pad_id)[:, None, None, :]
        caches = model.new_caches(model.encode(src))
        # per sentence: (tokens so far, total log-probability, its row in the caches)
        beams = []
        for k in range(len(indices)):
            beams.append([([bos], 0.0, k)])
        finished = [[] for _ in indices]
        active = [True for _ in indices]

        for position in range(max_tokens):
            rows, owners, last = [], [], []
            for k in range(len(indices)):
                if active[k]:
                    for tokens, _, row in beams[k]:
                        rows.append(row)
                        owners.append(k)
                        last.append(tokens[-1])
            if not rows:
                break
            reorder(caches, torch.tensor(rows, device=device))
            token = torch.tensor(last, device=device)[:, None]
            owner = torch.tensor(owners, device=device)
            scores = model.decode_step(token, memory_mask[owner], caches, position)
            top = scores.float().log_softmax(dim=-1).topk(beam_size)
            values, ids = top.values.tolist(), top.indices.tolist()

            row = 0
            for k in range(len(indices)):
                if not active[k]:
                    continue
                candidates = []
                for tokens, total, _ in beams[k]:
                    for log_p, token_id in zip(values[row], ids[row], strict=True):
                        candidates.append((tokens + [token_id], total + log_p, row))
                    row += 1
                candidates.sort(key=lambda candidate: candidate[1], reverse=True)
                beams[k] = []
                for tokens, total, parent in candidates:
                    if tokens[-1] == eos:
                        finished[k].append((tokens, total))
                    else:
                        beams[k].append((tokens, total, parent))
                    if len(beams[k]) == beam_size:
                        break
                if len(finished[k]) >= beam_size:
                    active[k] = False

        for k, i in enumerate(indices):

            def normalized(candidate):
                tokens, total = candidate[0], candidate[1]
                return total / length_penalty(len(tokens) - 1, alpha)  # without <bos>

            best = max(finished[k] or beams[k], key=normalized)
            translations[i] = tokenizer.decode(best[0][1:])  # decode drops <eos>
    return translations
