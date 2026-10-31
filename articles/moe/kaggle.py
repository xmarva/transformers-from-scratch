"""Run all the experiments of the post on Kaggle's two T4 GPUs and download what the figures need.

This pushes a private Kaggle script with prepare.py, moe.py, train.py, routing.py and speed.py embedded in it. On
Kaggle it prepares the corpus, trains every model of train.py with each of SEEDS, the two GPUs taking runs from one
queue (seed 0 first), then runs routing.py and speed.py. When it is done, data/train.json, data/routing.json and
data/speed.json are replaced by its results. About two hours of GPU time.

    uv run python articles/moe/kaggle.py

Needs the Kaggle CLI credentials in ~/.kaggle.
"""

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).parent
SOURCES = ["prepare.py", "moe.py", "train.py", "routing.py", "speed.py"]
KERNEL = "qmarva/tfs-moe-experiments"
RESULTS = ["train.json", "routing.json", "speed.json"]

RUNNER = r'''
import os, subprocess, sys, threading, time
from pathlib import Path

FILES = __FILES__
SEEDS = [0, 1, 2]
BUDGET = 4.5 * 3600  # no training run starts after this many seconds
# rough cost of a run relative to the dense model; the queue starts the expensive ones first
COST = {"Dense": 1.0, "MoE, 8 experts": 1.7, "MoE, 8 experts, no balancing": 1.7,
        "MoE, 8 experts, bias balancing": 1.7, "Dense, 4x wider": 3.6, "MoE, 32 experts": 2.5}

WORK = Path("/kaggle/working")
CODE, LOGS = WORK / "moe", WORK / "logs"
for folder in (CODE, CODE / "data", LOGS):
    folder.mkdir(parents=True, exist_ok=True)
for name, source in FILES.items():
    (CODE / name).write_text(source)
ENV = dict(os.environ, DATA_DIR="/tmp/tfs-moe", OUT=str(WORK / "out"), PYTHONUNBUFFERED="1")
START = time.time()


def stamp():
    return f"[{(time.time() - START) / 60:6.1f} min]"


def run(script, log, **env):
    with open(LOGS / log, "a") as f:
        return subprocess.run([sys.executable, script], cwd=CODE, env={**ENV, **env}, stdout=f,
                              stderr=subprocess.STDOUT).returncode


def tail(log, lines=5):
    return "\n".join((LOGS / log).read_text().splitlines()[-lines:])


print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout, flush=True)
import torch
print("torch", torch.__version__, "GPUs:", torch.cuda.device_count(), flush=True)

if run("prepare.py", "prepare.log"):
    print(tail("prepare.log", 40)); raise SystemExit("prepare.py failed")
print(stamp(), "prepared\n" + tail("prepare.log", 6), flush=True)

jobs = [(seed, model) for seed in SEEDS for model in sorted(COST, key=lambda m: -COST[m])]
lock, failed = threading.Lock(), []


def worker(gpu):
    while True:
        with lock:
            if not jobs:
                return
            if time.time() - START > BUDGET:
                print(stamp(), "out of time, skipping:", jobs, flush=True)
                jobs.clear()
                return
            seed, model = jobs.pop(0)
        log = f"train-gpu{gpu}.log"
        code = run("train.py", log, MODELS=model, SEEDS=str(seed), CUDA_VISIBLE_DEVICES=str(gpu))
        print(stamp(), f"gpu {gpu}: {model} seed {seed}: exit {code}\n" + tail(log, 2), flush=True)
        if code:
            failed.append((model, seed))


threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in range(max(1, torch.cuda.device_count()))]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(stamp(), "training done; failed:", failed, flush=True)

for script in ("routing.py", "speed.py"):
    code = run(script, script.replace(".py", ".log"), CUDA_VISIBLE_DEVICES="0")
    print(stamp(), script, "exit", code, "\n" + tail(script.replace(".py", ".log"), 40), flush=True)
print(stamp(), "finished", flush=True)
'''


def kaggle(*args):
    return subprocess.run(["uvx", "kaggle", *args], capture_output=True, text=True, check=True).stdout


def main():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        files = {name: (HERE / name).read_text() for name in SOURCES}
        (folder / "run_moe.py").write_text(RUNNER.replace("__FILES__", repr(files)))
        metadata = {"id": KERNEL, "title": KERNEL.split("/")[-1].replace("-", " "), "code_file": "run_moe.py",
                    "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
                    "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
                    "model_sources": []}
        (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
        print(kaggle("kernels", "push", "-p", str(folder), "--accelerator", "NvidiaTeslaT4").strip())

    while True:
        time.sleep(120)
        status = kaggle("kernels", "status", KERNEL)
        if any(word in status for word in ("COMPLETE", "ERROR", "CANCEL")):
            print(status.strip())
            break

    with tempfile.TemporaryDirectory() as tmp:
        kaggle("kernels", "output", KERNEL, "-p", tmp, "--file-pattern", r"moe/data/.*\.json$")
        for name in RESULTS:
            shutil.copy(Path(tmp) / "moe" / "data" / name, HERE / "data" / name)
            print("->", HERE / "data" / name)


if __name__ == "__main__":
    main()
