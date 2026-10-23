"""Greedy speculative decoding on a CPU, for the post "LLM inference optimization": Qwen2.5-0.5B-Instruct drafts,
Qwen2.5-1.5B-Instruct checks.

Every round the draft model proposes GAMMA tokens, one at a time. The target model reads all of them in one forward
pass and keeps the longest prefix that agrees with its own greedy choice, then adds one token of its own: the
correction where they disagree, or the next token if all were accepted. The text is the one the target would have
written alone, which the script checks against plain greedy decoding.

Writes data/speculative.json: every round's counts, the acceptance rate estimated from them, and one round traced
token by token for the figure.

    uv run python articles/inference_optimization/speculative.py
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DRAFT, TARGET = "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"
GAMMA = 5
NEW_TOKENS = 200
QUESTION = "Why do we forget most of our dreams within minutes of waking up?"
OUT = Path(__file__).parent / "data" / "speculative.json"

torch.manual_seed(0)
tokenizer = AutoTokenizer.from_pretrained(TARGET)
draft = AutoModelForCausalLM.from_pretrained(DRAFT, dtype=torch.float32).eval()
target = AutoModelForCausalLM.from_pretrained(TARGET, dtype=torch.float32).eval()

messages = [{"role": "user", "content": QUESTION}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, return_tensors="pt")
prompt = prompt["input_ids"]


@torch.no_grad()
def propose(ids):
    proposal = ids
    for _ in range(GAMMA):
        logits = draft(proposal, logits_to_keep=1).logits
        proposal = torch.cat([proposal, logits[:, -1:].argmax(-1)], dim=1)
    return proposal[:, ids.shape[1]:]


@torch.no_grad()
def verify(ids, proposed):
    logits = target(torch.cat([ids, proposed], dim=1), logits_to_keep=GAMMA + 1).logits
    choices = logits[0].argmax(-1)  # the target's own token at each of the GAMMA + 1 positions
    accepted = 0
    while accepted < GAMMA and proposed[0, accepted] == choices[accepted]:
        accepted += 1
    return accepted, choices


ids, rounds, traced = prompt, [], None
while ids.shape[1] - prompt.shape[1] < NEW_TOKENS:
    proposed = propose(ids)
    accepted, choices = verify(ids, proposed)
    new = torch.cat([proposed[:, :accepted], choices[accepted].view(1, 1)], dim=1)
    rounds.append(accepted)
    if traced is None and 2 <= accepted < GAMMA:
        traced = {
            "context": [tokenizer.decode(t) for t in ids[0, -8:]],
            "proposed": [tokenizer.decode(t) for t in proposed[0]],
            "target": [tokenizer.decode(t) for t in choices],
            "accepted": accepted,
        }
    ids = torch.cat([ids, new], dim=1)
    print(f"round {len(rounds):3d}: {accepted} of {GAMMA} accepted", flush=True)

generated = ids[0, prompt.shape[1]:]
with torch.no_grad():
    # Qwen's generation_config.json samples and penalizes repetition; plain greedy decoding needs both turned off
    plain = target.generate(
        prompt, max_new_tokens=len(generated), do_sample=False, repetition_penalty=1.0,
        temperature=None, top_p=None, top_k=None,
    )[0, prompt.shape[1]:]
same = torch.equal(generated, plain)

# with acceptance i.i.d. at rate alpha, a round accepts k < GAMMA tokens with probability alpha^k (1 - alpha);
# the maximum-likelihood alpha is the accepted tokens over the accepted tokens plus the rejections
rejections = sum(1 for a in rounds if a < GAMMA)
alpha = sum(rounds) / (sum(rounds) + rejections)
result = {
    "draft": DRAFT,
    "target": TARGET,
    "gamma": GAMMA,
    "question": QUESTION,
    "rounds": rounds,
    "tokens": len(generated),
    "alpha": alpha,
    "same_as_greedy": same,
    "traced": traced,
    "text": tokenizer.decode(generated),
}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"{len(generated)} tokens in {len(rounds)} target passes, {len(generated) / len(rounds):.2f} per pass; "
      f"alpha = {alpha:.2f}; same text as plain greedy decoding: {same}")
