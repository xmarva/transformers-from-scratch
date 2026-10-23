"""Measurements for the post "LLMs for Those Who Missed Out" on xmarva.github.io, run on a laptop CPU.

Every model here was public before the post's date (April 2024): GPT-2 (OpenAI, 2019), five Pythia models
(EleutherAI, 2023, all trained on the same data, the Pile, in the same order) and TinyLlama 1.1B (December 2023,
Llama 2's architecture and tokenizer, 3 trillion tokens), with its chat version.

    next_token   TinyLlama's probabilities for the token after "The capital of Australia is", and a greedy
                 continuation with the runners-up at every step
    made_up      what TinyLlama says about Tom Cruise's mother, and what its chat version says about her and her son
    compression  how many bits per character each model needs to encode a text it has never seen (the first version
                 of this post, written in April 2024), against gzip, bzip2 and xz
    chat         the same question to the base model and to the chat model

Writes data/measurements.json.

    uv run python articles/llms_intro/measure.py articles/llms_intro/data/first-version.md
"""

import bz2
import gzip
import json
import lzma
import math
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
CHAT = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
COMPRESSORS = ["openai-community/gpt2", "EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
               "EleutherAI/pythia-1.4b", BASE]
PROMPT = "The capital of Australia is"
QUESTIONS = ["What is the capital of Australia?", "Write a haiku about a cat."]
OUT = Path(__file__).parent / "data" / "measurements.json"


def load(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    return tokenizer, model


def piece(tokenizer, i):
    """A token as it appears in text, with its leading space (SentencePiece writes it as ▁)."""
    return tokenizer.convert_ids_to_tokens(int(i)).replace("▁", " ").replace("<0x0A>", "\n")


def plain_text(markdown):
    """The prose of a Jekyll post: no front matter, no HTML, no Liquid, no Markdown markup, no links' targets."""
    body = markdown.split("---", 2)[2]
    body = re.sub(r"\{%.*?%\}", "", body)
    body = re.sub(r"<[^>]+>", "", body)
    body = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", body)
    body = re.sub(r"^#+\s*|^\s*-\s+", "", body, flags=re.M)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


# ---------------------------------------------------------------- the next word

tokenizer, model = load(BASE)
ids = tokenizer(PROMPT, return_tensors="pt").input_ids
with torch.no_grad():
    probs = model(ids).logits[0, -1].softmax(-1)
top = probs.topk(12)
next_token = {
    "model": BASE,
    "prompt": PROMPT,
    "top": [[piece(tokenizer, i), p] for p, i in zip(top.values.tolist(), top.indices.tolist(), strict=True)],
}

steps = []  # greedy: append the most likely token and ask again
for _ in range(14):
    with torch.no_grad():
        p = model(ids).logits[0, -1].softmax(-1)
    best = p.topk(3)
    steps.append([[piece(tokenizer, i), q] for q, i in zip(best.values.tolist(), best.indices.tolist(),
                                                            strict=True)])
    ids = torch.cat([ids, best.indices[:1].view(1, 1)], dim=1)
next_token["greedy"] = steps
word = steps[:3]  # " Can", "ber", "ra": the greedy path spells the whole word
assert "".join(token for (token, _), *_ in word) == " Canberra"
next_token["canberra"] = word[0][0][1] * word[1][0][1] * word[2][0][1]
print(next_token["top"][:5], repr("".join(s[0][0] for s in steps)), next_token["canberra"])

made_up = {"base_prompt": "Tom Cruise's mother is"}
ids = tokenizer(made_up["base_prompt"], return_tensors="pt").input_ids
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=24, do_sample=False, max_length=None)
made_up["base"] = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
print(made_up)

# ---------------------------------------------------------------- prediction is compression

text = plain_text(Path(sys.argv[1]).read_text(encoding="utf-8"))
raw = text.encode("utf-8")
compression = {
    "text_chars": len(text),
    "text_bytes": len(raw),
    "classic": {
        "gzip": 8 * len(gzip.compress(raw, compresslevel=9)) / len(text),
        "bzip2": 8 * len(bz2.compress(raw, compresslevel=9)) / len(text),
        "xz": 8 * len(lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)) / len(text),
    },
    "models": [],
}
for name in COMPRESSORS:
    tokenizer, model = load(name)
    tokens = tokenizer(text, add_special_tokens=False).input_ids
    window = 1000  # every model here reads at least 1,024 tokens; the text is cut into windows that fit
    bits = 0.0
    start_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    for k in range(0, len(tokens), window):
        chunk = torch.tensor([[start_id] + tokens[k:k + window]])
        with torch.no_grad():
            logp = model(chunk).logits[0, :-1].log_softmax(-1)
        bits -= logp.gather(1, chunk[0, 1:, None]).sum().item() / math.log(2)
    params = sum(p.numel() for p in model.parameters())
    compression["models"].append({"model": name, "parameters": params, "tokens": len(tokens),
                                  "bits_per_char": bits / len(text)})
    print(name, f"{params / 1e6:.0f}M", f"{bits / len(text):.3f} bits/char")
print(compression["classic"])

# ---------------------------------------------------------------- a document completer and an assistant

chat = {"base": BASE, "chat": CHAT, "rows": []}
base_tokenizer, base_model = load(BASE)
chat_tokenizer, chat_model = load(CHAT)
for question in QUESTIONS:
    row = {"question": question}
    ids = base_tokenizer(question, return_tensors="pt").input_ids
    with torch.no_grad():
        out = base_model.generate(ids, max_new_tokens=60, do_sample=False, repetition_penalty=1.0, max_length=None)
    row["base"] = base_tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    ids = chat_tokenizer.apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True,
                                             return_tensors="pt", return_dict=True)["input_ids"]
    with torch.no_grad():
        out = chat_model.generate(ids, max_new_tokens=60, do_sample=False, repetition_penalty=1.0, max_length=None)
    row["chat"] = chat_tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    row["chat_prompt"] = chat_tokenizer.decode(ids[0])
    chat["rows"].append(row)
    print(json.dumps(row, indent=1, ensure_ascii=False))

OUT.parent.mkdir(exist_ok=True)
made_up["chat"] = []
for question in ["Who is Tom Cruise's mother?", "Who is Mary Lee Pfeiffer's son?"]:
    ids = chat_tokenizer.apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True,
                                             return_tensors="pt", return_dict=True)["input_ids"]
    with torch.no_grad():
        out = chat_model.generate(ids, max_new_tokens=60, do_sample=False, repetition_penalty=1.0, max_length=None)
    made_up["chat"].append({"question": question,
                            "answer": chat_tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)})
    print(made_up["chat"][-1])

OUT.write_text(json.dumps({"next_token": next_token, "made_up": made_up, "compression": compression, "chat": chat},
                          indent=1, ensure_ascii=False), encoding="utf-8")
print("saved", OUT)
