# Does Centaur read the instructions? Minitaur, the 8B sibling of Centaur (Binz et al., Nature 2025), and the
# Llama 3.1 8B it was tuned from, on two experiments held out of its training data: the magic-carpet two-step task
# (Feher da Silva & Hare 2020) and twenty logical reasoning problems (Jansen et al. 2021). We score the human choices
# with and without the parts of the prompt a reader would need to understand the task, and let the model play the
# two-step task itself.
# Data and the 4-bit setup follow the paper's repository: github.com/marcelbinz/Llama-3.1-Centaur-70B
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

OUT = Path(os.environ.get("CENTAUR_OUT", "/kaggle/working/centaur"))
DATA = Path("/kaggle/working/data")
REPO = "https://media.githubusercontent.com/media/marcelbinz/Llama-3.1-Centaur-70B/main/"
FILES = ["generalization/feher2020humans/prompts.jsonl", "generalization/feher2020humans/exp1.csv",
         "generalization/jansen2021logic/prompts.jsonl"]
BASE = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
ADAPTER = "marcelbinz/Llama-3.1-Minitaur-8B-adapter"
N_LOGIC = 300  # participants of the logic study scored, out of 3,543
PARTS = ["twostep", "logic"]  # version 1 ran "openloop" too; its results are kept from that run
T0 = time.time()


def log(*args):
    print(f"[{(time.time() - T0) / 60:6.1f} min]", *args, flush=True)


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1))


def load_model():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16, device_map={"": 0},
                                                 attn_implementation="sdpa")
    model = PeftModel.from_pretrained(model, ADAPTER).eval()
    n = 0
    for module in model.modules():  # the checkpoint computes in bf16, which a T4 or a P100 does not have
        if hasattr(module, "compute_dtype"):
            module.compute_dtype = torch.float16
            n += 1
    log("model loaded,", n, "4-bit layers, GPU", torch.cuda.get_device_name(0))
    return tok, model


class Scorer:
    def __init__(self, tok, model):
        import torch
        self.torch, self.tok, self.model = torch, tok, model
        self.inner = model.base_model.model  # LlamaForCausalLM with the LoRA layers in place

    def adapter(self, on):
        import contextlib
        return contextlib.nullcontext() if on else self.model.disable_adapter()

    def hidden(self, ids, chunk=1024):
        # in chunks through a KV cache: a whole session at once makes the T4's attention allocate an L x L matrix
        from transformers import DynamicCache
        cache, out = DynamicCache(), []
        for start in range(0, len(ids), chunk):
            o = self.inner.model(input_ids=self.torch.tensor([ids[start:start + chunk]], device="cuda"),
                                 past_key_values=cache, use_cache=True)
            out.append(o.last_hidden_state[0])
        return self.torch.cat(out)

    def nll(self, text):
        # negative log-likelihood of every token inside << >>, the human responses, as in the paper's collator
        enc = self.tok(text, return_offsets_mapping=True)
        spans = [(m.start(1), m.end(1)) for m in re.finditer(r"<<(.*?)>>", text)]
        resp, j = [], 0
        for i, (a, b) in enumerate(enc["offset_mapping"]):
            while j < len(spans) and spans[j][1] <= a:
                j += 1
            if b > a and j < len(spans) and spans[j][0] <= a < spans[j][1]:
                resp.append((i, j))
        with self.torch.no_grad():
            h = self.hidden(enc["input_ids"])
            pos = self.torch.tensor([i - 1 for i, _ in resp], device="cuda")
            logp = self.inner.lm_head(h[pos]).float().log_softmax(-1)
            target = self.torch.tensor([enc["input_ids"][i] for i, _ in resp], device="cuda")
            nll = (-logp.gather(1, target[:, None])[:, 0]).tolist()
        per_response = [0.0] * len(spans)
        for (_, j), x in zip(resp, nll, strict=True):
            per_response[j] += x
        return per_response, len(resp)

    def next_probs(self, context, options):
        # probabilities of the option letters right after a context ending in " <<"
        ids = self.tok(context)["input_ids"]
        with self.torch.no_grad():
            logp = self.inner.lm_head(self.hidden(ids)[-1:]).float().log_softmax(-1)[0]
        return {o: math.exp(logp[self.tok.encode(o, add_special_tokens=False)[0]].item()) for o in options}


# ---------------------------------------------------------------------------------------------------- the two-step task
TRIAL = re.compile(r"You are presented with magic carpets (\w) and (\w)\. You press <<(\w)>>\. You end up on (\w+) "
                   r"Mountain\. You see lamp (\w) and lamp (\w)\. You rub lamp <<(\w)>>\. You receive (\d) coins\.")


def split_header(text):
    i = text.index("\n\n")
    return text[:i], text[i + 2:]


def task_letters(header):
    m = re.search(r"Pink Mountain has genies (\w) and (\w), and Blue Mountain has genies (\w) and (\w)", header)
    lamps = {"Pink": [m.group(1), m.group(2)], "Blue": [m.group(3), m.group(4)]}
    m = re.search(r"Magic carpet (\w) generally flies to (\w+) Mountain, "
                  r"and magic carpet (\w) generally flies to (\w+)", header)
    return {m.group(1): m.group(2), m.group(3): m.group(4)}, lamps


def stay_table(days):
    # days: [(carpet, common, reward)] of completed days in order; returns the four stay counts used in the classic plot
    t = {f"{r}{'c' if c else 'r'}": [0, 0] for r in (0, 1) for c in (True, False)}
    for (c0, common, r), (c1, _, _) in zip(days, days[1:], strict=False):
        if c0 is None or c1 is None:
            continue
        key = f"{r}{'c' if common else 'r'}"
        t[key][0] += c0 == c1
        t[key][1] += 1
    return t


def human_days(text):
    header, body = split_header(text)
    common, _ = task_letters(header)
    days = []
    for line in body.split("\n"):
        m = TRIAL.fullmatch(line)
        if m:
            carpet, mountain, coins = m.group(3), m.group(4), int(m.group(8))
            days.append((carpet, common[carpet] == mountain, coins))
        elif line.startswith("You are presented"):
            days.append((None, None, None))  # a timeout breaks the chain of stays
    return days


def lookup_baselines(rows):
    # two models that know nothing about carpets or genies, fitted on the other participants (leave one out):
    # "repeat" is one probability of repeating the last choice at each stage; "table" conditions it on the last reward
    # and, for the carpets, on whether the last flight was common: the four numbers of the stay plot plus two for lamps
    events = []  # (participant, stage, condition or None when there is no history yet, repeated)
    for p, r in enumerate(rows):
        header, body = split_header(r["text"])
        common, _ = task_letters(header)
        last, last_lamp = None, {}
        for line in body.split("\n"):
            m = re.match(r"You are presented with magic carpets \w and \w\. You press <<(\w)>>\. You end up on (\w+) "
                         r"Mountain\. You see lamp \w and lamp \w\. "
                         r"(?:You rub lamp <<(\w)>>\. You receive (\d) coins)?", line)
            if not m:
                last = None  # a day without a choice breaks the chain
                continue
            carpet, mountain, lamp, coins = m.group(1), m.group(2), m.group(3), m.group(4)
            events.append((p, "s1", None if last is None else last, last is not None and carpet == last[0]))
            if lamp is None:
                last = None
                continue
            prev = last_lamp.get(mountain)
            events.append((p, "s2", None if prev is None else prev, prev is not None and lamp == prev[0]))
            last, last_lamp[mountain] = (carpet, int(coins), common[carpet] == mountain), (lamp, int(coins))
    keys = {"repeat": lambda stage, h: stage,
            "table": lambda stage, h: (stage, h[1]) if stage == "s2" else (stage, h[1], h[2])}
    out = {}
    for name, key in keys.items():
        per = []
        for p in range(len(rows)):
            fit = {}
            for q, stage, h, rep in events:
                if q != p and h is not None:
                    k = key(stage, h)
                    fit[k] = [fit.get(k, [0, 0])[0] + rep, fit.get(k, [0, 0])[1] + 1]
            nll = []
            for q, stage, h, rep in events:
                if q == p:
                    if h is None:
                        nll.append(math.log(2))  # no history yet at this stage: a coin flip
                    else:
                        s, t = fit[key(stage, h)]
                        nll.append(-math.log(s / t if rep else 1 - s / t))
            per.append(sum(nll) / len(nll))
        out[name] = per
    for name, key in keys.items():
        fit = {}
        for _, stage, h, rep in events:
            if h is not None:
                k = str(key(stage, h))
                fit[k] = [fit.get(k, [0, 0])[0] + rep, fit.get(k, [0, 0])[1] + 1]
        out[f"{name} counts"] = fit
    return out


def twostep():
    tok, model = load_model()
    sc = Scorer(tok, model)
    R = {}
    rows = [json.loads(line) for line in open(DATA / FILES[0])]
    variants = {
        "full": lambda t: t,
        "no instructions": lambda t: split_header(t)[1],
    }
    for adapter_on, name in [(True, "minitaur"), (False, "llama")]:
        with sc.adapter(adapter_on):
            for v, f in variants.items():
                per = []
                for r in rows:
                    nll, n_tok = sc.nll(f(r["text"]))
                    per.append({"participant": r["participant"], "nll": nll, "tokens": n_tok})
                R[f"twostep/{name}/{v}"] = per
                log(f"twostep {name} {v}: {sum(sum(p['nll']) / len(p['nll']) for p in per) / len(per):.4f}")
                save("likelihood.json", R)
            # no history: each choice predicted from the instructions and the current day alone
            per = []
            for r in rows:
                header, body = split_header(r["text"])
                cache, nll = {}, []
                for line in body.split("\n"):
                    for m in re.finditer(r"<<(\w)>>", line):
                        prefix = line[:m.start()]
                        context = header + "\n\n" + prefix + "<<"
                        if context not in cache:
                            options = re.findall(r"(?:carpets|lamp) (\w) and (?:lamp )?(\w)\.", prefix)[-1]
                            cache[context] = sc.next_probs(context, sorted(set(options)))
                        nll.append(-math.log(cache[context][m.group(1)]))
                per.append({"participant": r["participant"], "nll": nll, "contexts": len(cache)})
            R[f"twostep/{name}/no history"] = per
            log(f"twostep {name} no history: {sum(sum(p['nll']) / len(p['nll']) for p in per) / len(per):.4f}")
            save("likelihood.json", R)
    R["twostep/lookup"] = lookup_baselines(rows)
    R["twostep/human stays"] = [stay_table(human_days(r["text"])) for r in rows]
    save("likelihood.json", R)
    log("twostep done")


def logic():
    tok, model = load_model()
    sc = Scorer(tok, model)
    R = {}

    # the logic problems: the same twenty questions for every participant, answer letters shuffled per participant
    rows = [json.loads(line) for line in open(DATA / FILES[2])]
    rows = random.Random(0).sample(rows, N_LOGIC)
    block = re.compile(r"(Q\d+)\. (.*?)\nThe choices are:\n((?:\w: [^\n]*\n)+)You press <<(\w)>>\.", re.S)

    def no_questions(text):
        def strip(m):
            letters = "".join(f"{line[0]}\n" for line in m.group(3).splitlines())
            return f"{m.group(1)}.\nThe choices are:\n{letters}You press <<{m.group(4)}>>."
        return block.sub(strip, text)

    R["logic/meta"] = [{"participant": r["participant"],
                        "answers": [m.group(4) for m in block.finditer(r["text"])],
                        "options": [{line[0]: line[3:] for line in m.group(3).splitlines()}
                                    for m in block.finditer(r["text"])]}
                       for r in rows]
    for adapter_on, name in [(True, "minitaur"), (False, "llama")]:
        with sc.adapter(adapter_on):
            for v, f in {"full": lambda t: t, "no questions": no_questions}.items():
                per = []
                for r in rows:
                    text = f(r["text"])
                    nll, _ = sc.nll(text)
                    kinds = [m.group(1) for m in re.finditer(r"You (press|say) <<", text)]
                    per.append({"participant": r["participant"], "nll": nll, "kinds": kinds})
                R[f"logic/{name}/{v}"] = per
                press = [x for p in per for x, k in zip(p["nll"], p["kinds"], strict=True) if k == "press"]
                log(f"logic {name} {v}: {sum(press) / len(press):.4f} per answer, {len(per)} participants")
                save("logic.json", R)
    log("logic done")


# ------------------------------------------------------------------------------------------ the model plays the task
def openloop():
    import pandas as pd
    import torch
    from transformers import DynamicCache
    tok, model = load_model()
    inner = model.base_model.model
    rows = [json.loads(line) for line in open(DATA / FILES[0])]
    df = pd.read_csv(DATA / FILES[1])
    rng = random.Random(1)
    R = {}

    def run(header, schedule, with_header):
        cache = DynamicCache()
        common, lamps = task_letters(header)
        carpets = list(common)
        state = {"Pink": 0, "Blue": 1}
        mass, days = [], []

        def feed(text, options):
            ids = tok(text, add_special_tokens=not cache.get_seq_length())["input_ids"]
            with torch.no_grad():
                out = inner.model(input_ids=torch.tensor([ids], device="cuda"), past_key_values=cache, use_cache=True)
                logp = inner.lm_head(out.last_hidden_state[0, -1:]).float().log_softmax(-1)[0]
            p = [math.exp(logp[tok.encode(o, add_special_tokens=False)[0]].item()) for o in options]
            mass.append(sum(p))
            return rng.choices(options, weights=p)[0]

        text = (header + "\n\n") if with_header else ""
        for probs in schedule:
            a, b = rng.sample(carpets, 2)
            carpet = feed(text + f"You are presented with magic carpets {a} and {b}. You press <<", [a, b])
            mountain = common[carpet] if rng.random() < 0.7 else ({"Pink": "Blue", "Blue": "Pink"}[common[carpet]])
            l1, l2 = rng.sample(lamps[mountain], 2)
            lamp = feed(f"{carpet}>>. You end up on {mountain} Mountain. "
                        f"You see lamp {l1} and lamp {l2}. You rub lamp <<",
                        [l1, l2])
            reward = int(rng.random() < probs[state[mountain]][lamps[mountain].index(lamp)])
            days.append((carpet, common[carpet] == mountain, reward))
            text = f"{lamp}>>. You receive {reward} coins.\n"
        return days, mass

    conditions = [("minitaur", True, True), ("minitaur, no instructions", True, False), ("llama", False, True)]
    for r in rows:
        p = int(r["participant"])
        first = df[(df.participant == p) & (df.current_state == 999)]
        schedule = [[[x["reward.0.0"], x["reward.0.1"]], [x["reward.1.0"], x["reward.1.1"]]]
                    for _, x in first.iterrows()]
        header, _ = split_header(r["text"])
        header = re.sub(r"over the next \d+ days", f"over the next {len(schedule)} days", header)
        for name, adapter_on, with_header in conditions:
            with (contextlib_null() if adapter_on else model.disable_adapter()):
                days, mass = run(header, schedule, with_header)  # the environment is the same with or without the text
            R.setdefault(name, []).append({"participant": p, "days": days, "valid_mass": sum(mass) / len(mass),
                                           "stays": stay_table(days), "coins": sum(d[2] for d in days)})
            log(f"openloop {name} participant {p}: {sum(d[2] for d in days)} coins in {len(days)} days, "
                f"mass on valid options {sum(mass) / len(mass):.3f}")
            save("openloop.json", R)
    log("openloop done")


def contextlib_null():
    import contextlib
    return contextlib.nullcontext()


# ------------------------------------------------------------------------------------------------------------- driver
if __name__ == "__main__":
    if len(sys.argv) > 1:
        {"twostep": twostep, "logic": logic, "openloop": openloop}[sys.argv[1]]()
        sys.exit()
    subprocess.run(f"{sys.executable} -m pip install -q peft bitsandbytes", shell=True, check=False)
    for f in FILES:
        (DATA / f).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sfL", REPO + f, "-o", str(DATA / f)], check=True)
        log(f, (DATA / f).stat().st_size, "bytes")
    import torch
    n = torch.cuda.device_count()
    log(n, "GPUs:", [torch.cuda.get_device_name(i) for i in range(n)])
    versions = subprocess.run([sys.executable, "-c", "import torch, transformers, peft, bitsandbytes as b; "
                               "print(torch.__version__, transformers.__version__, peft.__version__, b.__version__)"],
                              capture_output=True, text=True).stdout.strip()
    save("env.json", {"gpus": [torch.cuda.get_device_name(i) for i in range(n)], "versions": versions})
    log("versions", versions)
    jobs = []
    for i, part in enumerate(PARTS):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i if n > 1 else 0))
        logf = open(OUT / f"{part}.log", "w")
        proc = subprocess.Popen([sys.executable, __file__, part], env=env, stdout=logf, stderr=subprocess.STDOUT)
        if n > 1:
            jobs.append((part, proc))
        else:
            proc.wait()
            log(part, "exit", proc.returncode)
    for part, proc in jobs:
        proc.wait()
        log(part, "exit", proc.returncode)
    for part in PARTS:
        print(f"===== {part}.log tail", flush=True)
        print("\n".join((OUT / f"{part}.log").read_text().splitlines()[-40:]), flush=True)
