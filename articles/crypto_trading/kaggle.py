"""Run the GPU experiments of the post on Kaggle's T4 GPUs and download their results.

    uv run python articles/crypto_trading/kaggle.py memory
    uv run python articles/crypto_trading/kaggle.py foundation
    uv run python articles/crypto_trading/kaggle.py memory-rerun Qwen/Qwen3-8B,Qwen/Qwen3.5-9B

`memory` pushes a private Kaggle script with memory.py and its questions (data/memory_items.json) embedded in it;
it updates transformers, asks every model of memory.py and brings back data/memory.json. About an hour.

`memory-rerun` runs only the models named, in a kernel of its own, and saves data/memory_rerun.json;
`memory.py merge memory_rerun.json` adds them to data/memory.json.

`foundation` pushes foundation.py with its inputs (made by `foundation.py inputs`); it installs Chronos-2 from PyPI
and the Kronos code from GitHub at a fixed commit, runs both models and brings back data/foundation_raw.json.
`foundation.py summarize` then scores the forecasts.

Needs the Kaggle CLI credentials in ~/.kaggle.
"""

import base64
import gzip
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).parent
KRONOS_COMMIT = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"

EXPERIMENTS = {
    "memory": {
        "kernel": "qmarva/tfs-crypto-llm-memory",
        "files": {"memory.py": HERE / "memory.py", "data/memory_items.json": HERE / "data" / "memory_items.json"},
        "setup": [["pip", "install", "-q", "-U", "transformers", "accelerate"]],
        "command": ["memory.py", "run"],
        "env": {},
        "result": "memory.json",
    },
    "memory-rerun": {
        "kernel": "qmarva/tfs-crypto-llm-memory-rerun",
        "files": {"memory.py": HERE / "memory.py", "data/memory_items.json": HERE / "data" / "memory_items.json"},
        "setup": [["pip", "install", "-q", "-U", "transformers", "accelerate"]],
        "command": ["memory.py", "run"],
        "env": {},
        "result": "memory.json",
        "save_as": "memory_rerun.json",
    },
    "foundation": {
        "kernel": "qmarva/tfs-crypto-foundation",
        "files": {"foundation.py": HERE / "foundation.py",
                  "inputs.json": Path.home() / ".cache" / "tfs-crypto" / "foundation_inputs.json"},
        "setup": [["pip", "install", "-q", "chronos-forecasting>=2.0", "einops"],
                  ["git", "clone", "-q", "https://github.com/shiyu-coder/Kronos", "kronos"],
                  ["git", "-C", "kronos", "checkout", "-q", KRONOS_COMMIT]],
        "command": ["foundation.py", "run"],
        "env": {"INPUTS": "inputs.json", "PYTHONPATH": "kronos"},
        "result": "foundation_raw.json",
    },
}

RUNNER = r'''
import base64, gzip, json, os, subprocess, sys
from pathlib import Path

FILES = json.loads(gzip.decompress(base64.b64decode(__FILES__)))
SETUP, COMMAND, ENV = __SETUP__, __COMMAND__, __ENV__
CODE = Path("/kaggle/working/crypto")
for name, source in FILES.items():
    (CODE / name).parent.mkdir(parents=True, exist_ok=True)
    (CODE / name).write_text(source)
(CODE / "data").mkdir(exist_ok=True)
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout, flush=True)
for step in SETUP:
    if step[0] == "pip":
        step = [sys.executable, "-m"] + step
    subprocess.run(step, cwd=CODE, check=True)
env = dict(os.environ, OUT=str(CODE / "data"), PYTHONUNBUFFERED="1", **ENV)
code = subprocess.run([sys.executable] + COMMAND, cwd=CODE, env=env).returncode
print(" ".join(COMMAND), "exit", code, flush=True)
'''


def kaggle(*args):
    return subprocess.run(["uvx", "kaggle", *args], capture_output=True, text=True, check=True).stdout


def main():
    experiment = EXPERIMENTS[sys.argv[1]]
    if len(sys.argv) > 2:
        experiment["env"] = {**experiment["env"], "ONLY": sys.argv[2]}
    kernel = experiment["kernel"]
    files = {name: path.read_text() for name, path in experiment["files"].items()}
    packed = base64.b64encode(gzip.compress(json.dumps(files).encode())).decode()
    runner = (RUNNER.replace("__FILES__", repr(packed)).replace("__SETUP__", repr(experiment["setup"]))
              .replace("__COMMAND__", repr(experiment["command"])).replace("__ENV__", repr(experiment["env"])))
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        (folder / "run.py").write_text(runner)
        metadata = {"id": kernel, "title": kernel.split("/")[-1].replace("-", " "), "code_file": "run.py",
                    "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
                    "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
                    "model_sources": []}
        (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
        print(kaggle("kernels", "push", "-p", str(folder), "--accelerator", "NvidiaTeslaT4").strip(), flush=True)

    while True:
        time.sleep(120)
        status = kaggle("kernels", "status", kernel)
        if any(word in status for word in ("COMPLETE", "ERROR", "CANCEL")):
            print(status.strip(), flush=True)
            break

    name = experiment["result"]
    with tempfile.TemporaryDirectory() as tmp:
        kaggle("kernels", "output", kernel, "-p", tmp, "--file-pattern", rf"crypto/data/{name.replace('.', '[.]')}$")
        target = HERE / "data" / experiment.get("save_as", name)
        shutil.copy(Path(tmp) / "crypto" / "data" / name, target)
        print("->", target)


if __name__ == "__main__":
    main()
