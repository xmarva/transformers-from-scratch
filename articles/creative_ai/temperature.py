"""Variation without selection, for the post "Can AI be creative?".

Qwen2.5-1.5B-Instruct continues one sentence about a brick eight times at each of three sampling temperatures (plain
softmax sampling: no top-k, no top-p, no repetition penalty, seed 0). For every continuation we also score how
likely the model itself finds it at temperature 1: the mean log-probability of its tokens.

Writes data/temperature.json.

    uv run python articles/creative_ai/temperature.py
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT = "Besides building walls, a brick can be used as a"
TEMPERATURES = [0.3, 1.0, 1.8]
SAMPLES, NEW_TOKENS = 8, 9
OUT = Path(__file__).parent / "data" / "temperature.json"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
prompt = tokenizer(PROMPT, return_tensors="pt")
start = prompt["input_ids"].shape[1]

rows = []
for temperature in TEMPERATURES:
    torch.manual_seed(0)
    with torch.no_grad():
        out = model.generate(
            **prompt, do_sample=True, temperature=temperature, top_k=0, top_p=1.0, repetition_penalty=1.0,
            max_new_tokens=NEW_TOKENS, num_return_sequences=SAMPLES, pad_token_id=tokenizer.eos_token_id,
        )
        logits = model(out).logits[:, start - 1:-1]  # the prediction for every generated token
    logprobs = torch.log_softmax(logits.float(), dim=-1).gather(-1, out[:, start:, None])[..., 0]
    samples = [tokenizer.decode(seq[start:], skip_special_tokens=True) for seq in out]
    rows.append({
        "temperature": temperature,
        "samples": samples,
        "mean_logprob": logprobs.mean(dim=1).tolist(),
    })
    print(temperature, [round(x, 2) for x in logprobs.mean(dim=1).tolist()])
    for s in samples:
        print("   ", repr(s))

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps({"model": MODEL, "prompt": PROMPT, "rows": rows}, indent=1, ensure_ascii=False),
               encoding="utf-8")
