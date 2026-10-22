# %% [markdown]
# ## Setup
#
# The notebook installs one small helper package, `tfs`: the course plot style, drawing of token sequences and
# detection of where the notebook runs (Kaggle, Colab or your machine). Every piece of code this lesson is
# about is written out below, in the notebook itself.

# %%
import importlib.util
import subprocess
import sys

TFS = "git+https://github.com/xmarva/transformers-from-scratch@{{TFS_REF}}"
if importlib.util.find_spec("tfs") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", TFS])

from tfs import env
from tfs.viz import token_chips, use_course_style  # noqa: F401  (token_chips is used by the lesson cells)

SEED = 42
env.seed_everything(SEED)
COLORS = use_course_style()
WORK_DIR = env.work_dir()
SMOKE = env.smoke()  # CI runs every lesson on a tiny slice of data; leave it off
print(f"runtime: {env.runtime()} | work dir: {WORK_DIR} | smoke run: {SMOKE}")
