"""Where is the notebook running, where should it keep files, and how does it read secrets."""

from __future__ import annotations

import os
import random
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
