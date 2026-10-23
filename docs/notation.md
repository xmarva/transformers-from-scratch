# Notation

One set of symbols for the whole course. Code uses the same names wherever Python allows.

## Data

| Symbol | Meaning | Code |
|---|---|---|
| $\mathbf{x} = (x_1, \dots, x_n)$ | source sentence (English) as token ids, length $n$ | `src` |
| $\mathbf{y} = (y_1, \dots, y_m)$ | target sentence (Russian) as token ids, length $m$ | `tgt` |
| $y_{<t}$ | target tokens before position $t$ | `tgt[:, :t]` |
| $\mathcal{V}$, $\lvert\mathcal{V}\rvert$ | vocabulary and its size | `vocab`, `vocab_size` |
| $\mathcal{D}$ | dataset of sentence pairs $(\mathbf{x}, \mathbf{y})$ | `dataset` |
| $B$, $T$ | batch size, padded sequence length | `B`, `T` |

## Model

| Symbol | Meaning | Code |
|---|---|---|
| $d_{\text{model}}$ | width of every residual stream vector | `d_model` |
| $h$ | number of attention heads | `n_heads` |
| $d_k = d_v = d_{\text{model}} / h$ | width of one head's queries/keys and values | `d_k` |
| $d_{\text{ff}}$ | hidden width of the feed-forward block | `d_ff` |
| $N$ | number of encoder (decoder) layers | `n_layers` |
| $E \in \mathbb{R}^{\lvert\mathcal{V}\rvert \times d_{\text{model}}}$ | token embedding matrix | `embedding.weight` |
| $Q, K, V$ | queries, keys, values, shape $(T, d_k)$ per head | `q`, `k`, `v` |
| $W^Q, W^K, W^V, W^O$ | attention projections | `w_q`, `w_k`, `w_v`, `w_o` |
| $M$ | additive attention mask ($0$ or $-\infty$) | `mask` |
| $\theta$ | all trainable parameters | `model.parameters()` |

Vectors are **rows** (as in the code, where a batch of vectors is a matrix with one vector per row), so a linear
layer is $\mathbf{x} W$. Matrices are capital italic letters, vectors bold lowercase.

The core formula, as it will appear in lesson 2:

$$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V$$

## Training

| Symbol | Meaning | Code |
|---|---|---|
| $p_\theta(y_t \mid y_{<t}, \mathbf{x})$ | model's probability of the next target token | `logits.softmax(-1)` |
| $\mathcal{L}$ | loss, mean token-level cross-entropy | `loss` |
| $\eta$ | learning rate | `lr` |
| $\varepsilon_{\text{ls}}$ | label smoothing | `label_smoothing` |
| $S$, $S_{\text{warm}}$ | total and warm-up optimisation steps | `total_steps`, `warmup_steps` |

$$\mathcal{L}(\theta) = -\frac{1}{m} \sum_{t=1}^{m} \log p_\theta(y_t \mid y_{<t}, \mathbf{x})$$

## Evaluation

| Symbol | Meaning |
|---|---|
| $\hat{\mathbf{y}}$ | model output (hypothesis) |
| $p_n$ | clipped $n$-gram precision of the hypothesis |
| $\mathrm{BP}$ | BLEU brevity penalty |
| $w_n$ | weight of $n$-gram order $n$, usually $1/4$ |
