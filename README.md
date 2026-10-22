# Transformers: from scratch to production

A practical course: build an English→Russian translator with a transformer written from scratch, measure it
honestly, improve it, and ship it as a monitored service on Hugging Face Spaces.

Every lesson is a self-contained notebook that runs on **Kaggle** and **Google Colab** with free hardware.
A lesson starts by downloading what the previous ones produced (tokenizer, checkpoints), so you can open any of
them on its own.

> Work in progress: lessons are added one pull request at a time.

## Lessons

| # | Lesson | You learn | Status |
|---|---|---|---|
| 1 | **How language models read: tokenization and BPE** | inside the tokenizers of GPT-4o and Llama 3, cleaning parallel data, leak-free splits, BPE by hand, Hugging Face `tokenizers` | in review |
| 2 | **The Transformer, built by hand** | real models' blueprints, attention and why it divides by √d_k, masks, heads, positions, the encoder–decoder, checks against PyTorch | in review |
| 3 | **From parrot to translator: training and decoding** | token-based batches, label smoothing, warm-up (and what happens without it), mixed precision, greedy, beam search and sampling, attention after training | in review |
| 4 | **Evaluating translations** | sacreBLEU, chrF++, COMET, error analysis, experiment tracking with W&B | planned |
| 5 | **Modern transformer improvements** | pre-norm, RMSNorm, RoPE, SwiGLU, fused attention, KV cache, ablations | planned |
| 6 | **Hyperparameter search** | W&B Sweeps, early stopping, spending a GPU budget well | planned |
| 7 | **From notebook to package** | project layout, configs, CLI, tests, CI, publishing a model card | planned |
| 8 | **Serving: API & demo** | FastAPI, batching, Gradio, faster inference, latency under load | planned |
| 9 | **Deploy, then watch it** | Docker, Hugging Face Spaces, CI/CD, metrics, drift, feedback | planned |

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
docs/         style guide, notation, how to write a lesson
```

Writing or reviewing a lesson: start with [docs/authoring.md](docs/authoring.md),
[docs/style-guide.md](docs/style-guide.md) and [docs/notation.md](docs/notation.md).

## License

Code under the [MIT License](LICENSE). Bundled IBM Plex fonts under the
[SIL Open Font License](src/tfs/viz/fonts/OFL.txt).
