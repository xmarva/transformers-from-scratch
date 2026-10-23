"""Where is the notebook running, where should it keep files, and how does it read secrets."""

from __future__ import annotations

import os
import random
import warnings
from pathlib import Path


def runtime() -> str:
    """Return "kaggle", "colab" or "local"."""
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    try:
        import google.colab  # noqa: F401

        return "colab"
    except ImportError:
        return "local"


def work_dir() -> Path:
    """A writable directory for data and checkpoints that survives until the session ends."""
    base = {"kaggle": Path("/kaggle/working"), "colab": Path("/content")}.get(runtime(), Path.cwd())
    path = base / "tfs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def secret(name: str) -> str | None:
    """Read a secret from the environment, Kaggle Secrets or Colab Secrets. Returns None when it isn't set."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        if runtime() == "kaggle":
            from kaggle_secrets import UserSecretsClient

            return UserSecretsClient().get_secret(name)
        if runtime() == "colab":
            from google.colab import userdata

            return userdata.get(name)
    except Exception:  # the secret isn't attached to this notebook
        return None
    return None


def quiet() -> None:
    """Keep notebook outputs clean: no download progress bars, no warnings about the environment itself.

    Call it before importing `datasets` or `huggingface_hub`: they read these settings at import time.
    """
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
    # the Hub answers anonymous downloads with a "please set HF_TOKEN" header, which the library logs as a warning
    os.environ["HF_HUB_VERBOSITY"] = "error"
    try:
        from huggingface_hub.utils import logging as hub_logging

        hub_logging.set_verbosity_error()  # in case the library was imported before quiet() ran
    except ImportError:
        pass
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # silences a warning when a notebook forks processes
    warnings.filterwarnings("ignore", message="IProgress not found")  # tqdm without ipywidgets
    warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")  # Colab: public downloads need no token
    warnings.filterwarnings("ignore", message=".*Colab secrets.*")


def smoke() -> bool:
    """True for a fast dev run (tiny data, few steps). CI runs every lesson this way."""
    return os.environ.get("TFS_SMOKE") == "1"


def seed_everything(seed: int) -> None:
    """Make Python, NumPy and (if installed) PyTorch random generators reproducible."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
