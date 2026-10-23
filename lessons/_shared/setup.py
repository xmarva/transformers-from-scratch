# %% [markdown]
# ## Setup
#
# One small helper package, `tfs`, draws the course's plots and token chips and knows where the notebook runs
# (Kaggle, Colab or your own machine). Everything the lesson teaches is written out in the notebook itself.

# %%
import importlib.util
import subprocess
import sys

TFS = "git+https://github.com/xmarva/transformers-from-scratch@{{TFS_REF}}"
if importlib.util.find_spec("tfs") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", TFS])

from tfs import env
from tfs.viz import table, token_chips, use_course_style

env.quiet()  # no progress bars or environment warnings in the outputs
SEED = 42
env.seed_everything(SEED)
COLORS = use_course_style()
WORK_DIR = env.work_dir()
