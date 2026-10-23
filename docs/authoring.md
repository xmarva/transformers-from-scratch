# Writing a lesson

Read the [style guide](style-guide.md) first; it is the definition of done.

## Layout

```
lessons/
  _shared/setup.py          cells included in every lesson (install helpers, quiet outputs, seed, plot style)
  01_tokenization/
    lesson.py               the lesson, jupytext "percent" format: the only file you edit
    lesson.toml             number, short title, cover alt text, gpu
    figures.py              static concept diagrams and covers (optional)
notebooks/                  generated, committed: Colab opens them from GitHub, Kaggle receives them
  colab/01_tokenization.ipynb
  kaggle/01_tokenization/notebook.ipynb + kernel-metadata.json
assets/figures/01/          rendered figures (PNG for notebooks, SVG for the site), light and dark
assets/covers/              rendered covers
src/tfs/                    helper package the notebooks install: environment, plot style, token chips
tools/                      build, render, run
```

## The source format

`lesson.py` is a notebook written as a Python file:

```python
# %% [markdown]
# ## Why subwords
# Text with math: $\lvert\mathcal{V}\rvert$ grows with the corpus.

# %% tags=["include:setup"]

# %%
words = text.split()

# %% tags=["only-kaggle"]
# runs only in the Kaggle build
```

Cell tags:

| Tag | Effect |
|---|---|
| `include:setup` | replaced by the cells of `lessons/_shared/setup.py` |
| `only-kaggle`, `only-colab` | kept only in that platform's notebook |

Other tags pass through untouched.

## Workflow

```bash
uv sync --group dev                      # once
uv run python tools/render.py 01         # figures and covers -> assets/
uv run python tools/build.py 01          # notebooks -> notebooks/
uv run python tools/run.py 01 --smoke    # top to bottom on a tiny slice (lessons with a SMOKE setting)
uv run python tools/run.py 01            # full run, outputs kept in build/executed/
uv run --with nbconvert jupyter nbconvert --to html build/executed/01_tokenization.ipynb  # look at it
uv run pytest -q && uv run ruff check .
```

Open a lesson in Jupyter while writing it: `uv run --with jupyterlab jupyter lab lessons/01_tokenization/lesson.py`
(jupytext pairs the file with a notebook view).

Every change goes through a pull request. CI checks lint, tests, that `notebooks/` matches the sources, and runs
every lesson (in smoke mode where a lesson has one).

## Figures and covers

`figures.py` exposes two dictionaries:

```python
FIGURES = {"bpe-merge": make_bpe_merge}  # each returns tools.figures.lib.Figure
COVERS = {"en": cover_en, "ru": cover_ru}  # each returns the cover SVG
```

`tools/render.py` refuses figures with problems (elements outside the canvas, overlapping labels) and English
covers containing Cyrillic. Look at both themes before committing; the checks don't see meaning.

Reference a rendered figure from a markdown cell with its raw GitHub URL and its own width:

```markdown
<img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/01/bpe-update-light.png" width="760" alt="Merging н + и in ▁понимать: only the pairs around the merge change">
```

The build rewrites `main` in these links to the ref it builds for, so a draft shows the figures of its branch.

## Publishing

1. Merge the lesson PR.
2. Push the Kaggle build (`kaggle kernels push -p notebooks/kaggle/01_tokenization`), run it, review it on Kaggle.
   Before the merge, push a draft built from the branch instead: `build.py 01 --ref <branch> --out-dir <tmp>`.
3. Set `published = true` in `lesson.toml`, rebuild and push again to make it public.
4. Export the executed notebook to a post for xmarva.github.io and open a PR there.
