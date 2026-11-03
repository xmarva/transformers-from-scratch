"""Run the post's experiments on Kaggle's T4 GPUs and download their results.

    uv run python articles/ai_psychosis/kaggle.py generate             # every model: data/replies.json
    uv run python articles/ai_psychosis/kaggle.py generate "Qwen3 8B"  # only the models named (comma-separated)
    uv run python articles/ai_psychosis/kaggle.py extra                # custom instructions, reassurance
                                                                       #   -> data/replies_extra.json
    uv run python articles/ai_psychosis/kaggle.py judge                # the judge reads every reply: data/verdicts.json

Each pushes a private Kaggle script with the post's scripts and data embedded in it, waits for it and brings back
the result. Needs the Kaggle CLI credentials in ~/.kaggle.
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

EXPERIMENTS = {
    "generate": {
        "kernel": "qmarva/tfs-ai-psychosis-generate",
        "files": {"run.py": HERE / "run.py", "data/stimuli.json": HERE / "data" / "stimuli.json"},
        "setup": [["pip", "install", "-q", "-U", "transformers", "accelerate"]],
        "command": ["run.py", "run"],
        "env": {},
        "result": "replies.json",
    },
    "extra": {
        "kernel": "qmarva/tfs-ai-psychosis-extra",
        "files": {"run.py": HERE / "run.py", "data/stimuli.json": HERE / "data" / "stimuli.json"},
        "setup": [["pip", "install", "-q", "-U", "transformers", "accelerate"]],
        "command": ["run.py", "run"],
        # the thinking model is left out: its long thoughts would double the run for little
        "env": {"EXTRA": "1", "ONLY": "Qwen3.5 9B,Qwen3 8B,Llama 3.1 8B,Gemma 3 4B,Mistral 7B v0.3,OLMo 3 7B,"
                                     "Phi-4 mini,Qwen3 0.6B,Qwen3 1.7B,Qwen3 4B,Qwen3 14B"},
        "result": "replies_extra.json",
    },
    "judge": {
        "kernel": "qmarva/tfs-ai-psychosis-judge",
        "files": {"judge.py": HERE / "judge.py", "data/stimuli.json": HERE / "data" / "stimuli.json",
                  "data/replies.json": HERE / "data" / "replies.json",
                  "data/replies_extra.json": HERE / "data" / "replies_extra.json"},
        "setup": [["pip", "install", "-q", "-U", "transformers", "accelerate", "bitsandbytes"]],
        "command": ["judge.py", "run"],
        "env": {},
        "result": "verdicts.json",
    },
}

RUNNER = r'''
import base64, gzip, json, os, subprocess, sys
from pathlib import Path

FILES = json.loads(gzip.decompress(base64.b64decode(__FILES__)))
SETUP, COMMAND, ENV = __SETUP__, __COMMAND__, __ENV__
CODE = Path("/kaggle/working/psychosis")
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
    return subprocess.run(["uvx", "--with", "six", "kaggle", *args], capture_output=True, text=True, check=True).stdout


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
        kaggle("kernels", "output", kernel, "-p", tmp, "--file-pattern", rf"psychosis/data/{name.replace('.', '[.]')}$")
        target = HERE / "data" / experiment.get("save_as", name)
        shutil.copy(Path(tmp) / "psychosis" / "data" / name, target)
        print("->", target)


if __name__ == "__main__":
    main()
