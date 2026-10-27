# Greedy decoding and beam search of lesson 3, batched as in lesson 4, moved here from the
# course's notebooks. One change: a source longer than the model's positions is cut to fit,
# instead of failing inside the model.

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
        memory = model.encode(src)
        written = torch.full((len(indices), 1), bos, device=device)
        done = torch.zeros(len(indices), dtype=torch.bool, device=device)
        for _ in range(max_tokens):
            next_ids = model.decode(written, memory, src)[:, -1].argmax(dim=-1)
            next_ids[done] = pad_id  # a finished sentence only collects padding
            written = torch.cat([written, next_ids[:, None]], dim=1)
            done |= next_ids == eos
            if done.all():
                break
        for row, i in enumerate(indices):
            ids = []
            for token_id in written[row, 1:].tolist():
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
        memory = model.encode(src)
        # per sentence: a list of (tokens so far, total log-probability)
        beams = [[([bos], 0.0)] for _ in indices]
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
                return total / length_penalty(len(tokens) - 1, alpha)  # without <bos>

            best_tokens, _ = max(finished[k] or beams[k], key=normalized)
            translations[i] = tokenizer.decode(best_tokens[1:])  # decode drops <eos>
    return translations
