# Transformers: from scratch to production

A practical course: build an English→Russian translator with a transformer written from scratch, measure it
honestly, improve it, and ship it as a tested package and a reproducible Docker image.

Every lesson is a self-contained notebook that runs on **Kaggle** and **Google Colab** with free hardware.
A lesson starts by downloading what the previous ones produced (tokenizer, checkpoints), so you can open any of
them on its own.

> Work in progress: lessons are added one pull request at a time.

## Lessons

| # | Lesson | You learn | Status |
|---|---|---|---|
| 1 | **How language models read: tokenization and BPE** | inside the tokenizers of GPT-4o and Llama 3, cleaning parallel data, leak-free splits, BPE by hand, Hugging Face `tokenizers` | [read](https://xmarva.github.io/blog/2026/transformers-from-scratch-01-tokenization/) · [Kaggle](https://www.kaggle.com/code/qmarva/transformers-from-scratch-01-tokenization) · [Colab](https://colab.research.google.com/github/xmarva/transformers-from-scratch/blob/main/notebooks/colab/01_tokenization.ipynb) |
| 2 | **Implementing the Transformer in PyTorch: why attention divides by √d_k** | real models' blueprints, attention and why it divides by √d_k, masks, heads, positions, the encoder–decoder, checks against PyTorch | [read](https://xmarva.github.io/blog/2026/transformers-from-scratch-02-transformer/) · [Kaggle](https://www.kaggle.com/code/qmarva/transformers-from-scratch-02-the-transformer) · [Colab](https://colab.research.google.com/github/xmarva/transformers-from-scratch/blob/main/notebooks/colab/02_transformer.ipynb) |
| 3 | **Training a Transformer: warm-up, label smoothing, beam search, sampling** | token-based batches, label smoothing, warm-up (and what happens without it), mixed precision, greedy, beam search and sampling, attention after training | [read](https://xmarva.github.io/blog/2026/transformers-from-scratch-03-training/) · [Kaggle](https://www.kaggle.com/code/qmarva/transformers-from-scratch-03-training-decoding) · [Colab](https://colab.research.google.com/github/xmarva/transformers-from-scratch/blob/main/notebooks/colab/03_training.ipynb) |
| 4 | **Evaluating machine translation with chrF and COMET** | BLEU and chrF from scratch, COMET by hand, sacreBLEU signatures, multiple references, metrics against MQM, paired bootstrap, test-set contamination, error analysis | [read](https://xmarva.github.io/blog/2026/transformers-from-scratch-04-evaluation/) · [Kaggle](https://www.kaggle.com/code/qmarva/transformers-from-scratch-04-bleu-chrf-comet) · [Colab](https://colab.research.google.com/github/xmarva/transformers-from-scratch/blob/main/notebooks/colab/04_evaluation.ipynb) |
| 5 | **What Llama changed in the Transformer: RoPE, RMSNorm, SwiGLU** | Llama 3.1 read from its config and weight headers, pre-norm, RMSNorm, SwiGLU, RoPE, fused attention, a KV cache, an ablation ladder against the noise of random seeds | [read](https://xmarva.github.io/blog/2026/transformers-from-scratch-05-llama/) · [Kaggle](https://www.kaggle.com/code/qmarva/transformers-from-scratch-05-what-llama-changed) · [Colab](https://colab.research.google.com/github/xmarva/transformers-from-scratch/blob/main/notebooks/colab/05_llama.ipynb) |
| 6 | **Hyperparameter tuning on a free GPU with W&B Sweeps** | experiment tracking with W&B, a learning-rate range test, random search and successive halving from scratch, W&B Sweeps with Bayesian optimization and early stopping, the gain checked against seeds | [read](https://xmarva.github.io/blog/2026/transformers-from-scratch-06-sweeps/) · [Kaggle](https://www.kaggle.com/code/qmarva/transformers-from-scratch-06-w-b-sweeps-tuning) · [Colab](https://colab.research.google.com/github/xmarva/transformers-from-scratch/blob/main/notebooks/colab/06_sweeps.ipynb) |
| 7 | **Your notebook is not a product: a Python package with tests and CI** | the src layout, `pyproject.toml`, configs that refuse bad values, a command line, four kinds of tests, mutation testing, GitHub Actions, a model card with measured limitations | [read](https://xmarva.github.io/blog/2026/transformers-from-scratch-07-package/) · [Kaggle](https://www.kaggle.com/code/qmarva/transformers-from-scratch-07-package-tests-ci) · [Colab](https://colab.research.google.com/github/xmarva/transformers-from-scratch/blob/main/notebooks/colab/07_package.ipynb) |
| 8 | **Serving a Transformer with FastAPI and Gradio: latency under load** | where inference time goes, the KV cache in beam search, int8 quantization, FastAPI and asyncio, load tests without coordinated omission, queueing theory checked by simulation, dynamic batching, Gradio | [read](https://xmarva.github.io/blog/2026/transformers-from-scratch-08-serving/) · [Kaggle](https://www.kaggle.com/code/qmarva/transformers-from-scratch-08-serving-under-load) · [Colab](https://colab.research.google.com/github/xmarva/transformers-from-scratch/blob/main/notebooks/colab/08_serving.ipynb) |
| 9 | **It works on my laptop: a reproducible Docker image of a model** | Docker images and layers, lock files with hashes, reproducible builds, image digests, GitHub Container Registry, CI that rebuilds and compares | planned |
| 10 | **Knowledge distillation, data and scale: what makes a translator better** | sequence-level knowledge distillation from opus-mt-en-ru, synthetic in-domain data, a bigger student, checkpoint averaging and ensembles, NLLB and LLMs as reference points, quality against CPU latency | planned |

The running example: translate English into Russian, trained on [Tatoeba](https://tatoeba.org) sentence pairs
and evaluated on the Tatoeba Challenge test set, with
[`Helsinki-NLP/opus-mt-en-ru`](https://huggingface.co/Helsinki-NLP/opus-mt-en-ru) as the baseline to beat.

## Repository

```
lessons/      lesson sources (one Python file per lesson, jupytext format) and their figures
notebooks/    generated notebooks for Colab and Kaggle
assets/       rendered figures and covers
src/tfs/      small helper package the notebooks install (environment, plot style, token drawing)
tools/        build notebooks, render figures, run lessons
articles/     posts for xmarva.github.io outside the course: text, measurements, figures
docs/         style guide, notation, how to write a lesson
```

Writing or reviewing a lesson: start with [docs/authoring.md](docs/authoring.md),
[docs/style-guide.md](docs/style-guide.md) and [docs/notation.md](docs/notation.md).

## License

Code under the [MIT License](LICENSE). Bundled IBM Plex fonts under the
[SIL Open Font License](src/tfs/viz/fonts/OFL.txt).
