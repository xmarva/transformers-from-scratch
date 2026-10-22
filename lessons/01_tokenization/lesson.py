# %% [markdown]
# # Lesson 1. Data and tokenization: BPE from scratch
#
# Here is one sentence in two languages:
#
# > The researcher said she was exhausted.
# >
# > Исследовательница сказала, что вымоталась.
#
# A neural network reads neither. It reads integers, and before any model exists someone has to decide how text
# turns into them. That decision fixes how long every input is, which words the model can represent at all, and how
# much each sentence costs to process. GPT-2, for example, spends several times more integers on the Russian sentence
# than on the English one; you will measure exactly how many in a minute.
#
# This course builds an English→Russian translator from scratch, so the decision is ours. In this lesson we make it.
#
# **What you'll build**
#
# 1. A clean English–Russian corpus with training, validation and test splits that do not leak into each other.
# 2. A byte-pair encoding (BPE) tokenizer written from scratch, fast enough for half a million sentences.
# 3. The tokenizer the rest of the course uses, trained with Hugging Face `tokenizers` and published with the data.
#
# **What you need**: Python and a little pandas. No previous lesson. A CPU runtime is enough; the whole notebook runs
# in about ten minutes on a free CPU runtime.

# %% tags=["include:setup"]

# %%
import collections
import csv
import json
import math
import os
import re
import time
import unicodedata

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

# %% [markdown]
# Every knob of the lesson lives in one cell. `SMOKE` is set only by the course's automated tests, which run the
# notebook on a small slice of data; for you it is `False`.

# %%
VAL_SHARE = 0.02  # share of the data (by sentence groups) held out for validation
NAIVE_SENTENCES, NAIVE_MERGES = (2_000, 50) if SMOKE else (40_000, 300)  # the slow BPE runs on a sample
N_MERGES = 300 if SMOKE else 8_000  # merges for our from-scratch tokenizer
VOCAB_SIZES = [500, 1_000] if SMOKE else [2_000, 4_000, 8_000, 16_000, 32_000]
VOCAB_SIZE = 1_000 if SMOKE else 16_000  # the tokenizer the course keeps
D_MODEL = 256  # width of the model in lesson 2, needed here to price the embedding table

# %% [markdown]
# ## 1. How many integers is a sentence?
#
# GPT-2's tokenizer is a good place to start, because it was built for English. Count how many tokens each version
# of our sentence becomes.

# %%
gpt2 = Tokenizer.from_pretrained("gpt2")
EN_EXAMPLE = "The researcher said she was exhausted."
RU_EXAMPLE = "Исследовательница сказала, что вымоталась."

for sentence in (EN_EXAMPLE, RU_EXAMPLE):
    print(f"{len(gpt2.encode(sentence).ids):>3} tokens: {sentence}")

# %% [markdown]
# The same meaning, a very different bill. Look at what the Russian tokens actually are:

# %%
print(gpt2.encode(RU_EXAMPLE).tokens[:12])
print(f"{len(RU_EXAMPLE)} characters, {len(RU_EXAMPLE.encode('utf-8'))} bytes in UTF-8")

# %% [markdown]
# Those are not letters. GPT-2 tokenizes **bytes**: it first writes the text in UTF-8 and shows each byte as a
# printable stand-in character (`Ð`, `Ñ`). Every Cyrillic letter takes two bytes in UTF-8, and the merges GPT-2 learned
# come from mostly English web text, so they rarely glue Russian bytes back into whole syllables. Russian pays a
# "token tax": longer sequences, more compute, a smaller share of the context window per idea
# ([Petrov et al., 2023](https://arxiv.org/abs/2305.15425) measure it across many languages).
#
# A translator that writes Russian should not pay it. The fix is a tokenizer trained on our own data. First we need
# that data, clean, and split so that our measurements later mean something.

# %% [markdown]
# ## 2. The data
#
# [Tatoeba](https://tatoeba.org) is a collection of sentences and their translations written by volunteers,
# licensed under CC BY 2.0 FR. We use two pieces of it:
#
# - **training data**: the English–Russian pairs prepared by the sentence-transformers project;
# - **test data**: the English–Russian test set of the [Tatoeba Translation Challenge](https://arxiv.org/abs/2010.06354)
#   (Tiedemann, 2020), a public benchmark. In lesson 4 we compare our model with published systems on it.
#
# Both downloads are pinned to an exact revision, so the numbers in this notebook do not drift when the sources change.

# %%
TRAIN_REVISION = "cec1343ab5a7a8befe99af4a2d0ca847b6c84743"
TEST_URL = (
    "https://raw.githubusercontent.com/Helsinki-NLP/Tatoeba-Challenge/"
    "5509af44a74016748c26f41ab164b64f48445f3f/data/test/eng-rus/test.txt"
)

train_raw = (
    load_dataset("sentence-transformers/parallel-sentences-tatoeba", "en-ru", split="train", revision=TRAIN_REVISION)
    .to_pandas()
    .rename(columns={"english": "en", "non_english": "ru"})
)
test_raw = pd.read_csv(
    TEST_URL,
    sep="\t",
    names=["src_lang", "tgt_lang", "en", "ru"],
    quoting=csv.QUOTE_NONE,
    dtype=str,
    keep_default_na=False,
)[["en", "ru"]]

if SMOKE:
    train_raw = train_raw.sample(20_000, random_state=SEED)
    test_raw = test_raw.sample(2_000, random_state=SEED)
print(f"training pairs: {len(train_raw):,}   test pairs: {len(test_raw):,}")
train_raw.sample(5, random_state=SEED)

# %% [markdown]
# ### One sentence, many translations
#
# The data has a property that matters for everything that follows: one English sentence often comes with several
# Russian translations, and one Russian sentence can translate several English ones.

# %%
per_english = train_raw.groupby("en").size()
print(f"{per_english.size:,} distinct English sentences, {per_english.mean():.2f} translations each on average")

most_translated = per_english.idxmax()
print(f"\n{most_translated!r} has {per_english.max()} translations:")
print(", ".join(train_raw.loc[train_raw.en == most_translated, "ru"]))

# %% [markdown]
# Volunteers have been creative. None of these translations is wrong; they differ in register, imagery and rudeness.
# Keep this in mind for lesson 4: a metric that compares the model's output with one reference translation punishes
# perfectly good answers.
#
# For now the consequence is about splitting the data, and we come back to it after cleaning.

# %% [markdown]
# ### Normalization
#
# The same text can be typed in different ways. Count the quote marks and dashes in the corpus:

# %%
chars = collections.Counter("".join(train_raw.en) + "".join(train_raw.ru))
print({c: chars[c] for c in "\"«»„“”'‘’-–—…" if chars[c]})

# %% [markdown]
# Some variants are a style choice (`«ёлочки»` versus `"straight quotes"`), others are invisible. The letter `й`, for
# instance, can be stored as one code point or as `и` followed by a combining breve. Python considers the two strings
# different, although every reader sees the same letter:

# %%
composed, decomposed = "й", "и\u0306"
print(composed == decomposed, len(composed), len(decomposed))
print(unicodedata.normalize("NFKC", decomposed) == composed)
print(unicodedata.normalize("NFKC", "H₂O"), repr(unicodedata.normalize("NFKC", "two\u00a0words")))

# %% [markdown]
# Unicode defines normal forms for exactly this. **NFKC** composes letters with their accents and also maps
# "compatibility" characters to plain ones: subscript digits, non-breaking spaces, the one-character ellipsis `…`.
#
# On top of NFKC we make four choices of our own, and write them down because the model will inherit them:
#
# 1. drop combining marks that survive composition: Tatoeba marks stress in some Russian sentences (`То́му`), which is a
#    pronunciation hint, not spelling;
# 2. use straight quotes and apostrophes, and one dash `—`;
# 3. read a hyphen surrounded by spaces as a dash;
# 4. collapse runs of whitespace into one space.

# %%
QUOTES = {"«": '"', "»": '"', "„": '"', "“": '"', "”": '"', "‘": "'", "’": "'", "–": "—"}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")  # combining marks left after NFKC
    text = text.translate(str.maketrans(QUOTES))
    text = re.sub(r"\s+-\s+", " — ", text)
    return re.sub(r"\s+", " ", text).strip()


examples = ["Он сказал: «Я не знаю».", "То́му было нечего пить.", "Wait\u00a0a second - what?", "It’s 5 o’clock…"]
for s in examples:
    print(f"{s!r:38} -> {normalize(s)!r}")

# %%
train = train_raw.assign(en=train_raw.en.map(normalize), ru=train_raw.ru.map(normalize))
test = test_raw.assign(en=test_raw.en.map(normalize), ru=test_raw.ru.map(normalize))
changed = int((train.en != train_raw.en).sum() + (train.ru != train_raw.ru).sum())
print(f"normalization changed {changed:,} of {2 * len(train):,} training sentences")

# %% [markdown]
# ### Filters, and one we don't apply
#
# A pair is dropped when it is clearly broken: Cyrillic on the English side, no Cyrillic on the Russian side, or more
# than 40 words, which in Tatoeba means a paragraph that slipped in among sentences. After normalization some pairs
# become identical, so we drop duplicates last.

# %%
CYRILLIC = re.compile("[а-яё]", re.IGNORECASE)


def n_words(texts: pd.Series) -> pd.Series:
    return texts.str.split().str.len()


rules = {
    "English side contains Cyrillic": train.en.str.contains(CYRILLIC),
    "Russian side has no Cyrillic": ~train.ru.str.contains(CYRILLIC),
    "longer than 40 words": (n_words(train.en) > 40) | (n_words(train.ru) > 40),
}
keep = pd.Series(True, index=train.index)
for name, broken in rules.items():
    print(f"{name:<32} {int((broken & keep).sum()):>6,} pairs")
    keep &= ~broken
before = int(keep.sum())
train = train[keep].drop_duplicates(["en", "ru"])
print(f"{'duplicates after normalization':<32} {before - len(train):>6,} pairs")

# %% [markdown]
# Machine translation pipelines often also drop pairs whose lengths differ too much, on the theory that a translation
# three times longer than its source must be misaligned. Before applying a recipe, look at what it would remove:

# %%
length_ratio = train.en.str.len() / train.ru.str.len()
flagged = train[(length_ratio > 3) | (length_ratio < 1 / 3)]
print(f"the length-ratio filter would remove {len(flagged):,} pairs, for example:")
flagged.sample(min(6, len(flagged)), random_state=SEED)

# %% [markdown]
# Most of them are good translations: idioms and compressed colloquial answers, exactly what makes a translator
# sound human. The filter was designed for noisy web-crawled corpora; Tatoeba is written by people, so we keep these
# pairs. The general rule: a cleaning step is a hypothesis about your data, and you test it by reading what it removes.

# %% [markdown]
# ### The test set must stay unseen
#
# The Tatoeba test set was cut from the same pool of sentences as our training data. If a test sentence also appears
# in training, the model can recite its answer and the test score stops measuring translation.
#
# Comparing exact strings is not enough: `Хватит!` and `хватит.` are the same sentence for our purposes. We compare
# **fingerprints** instead: case-folded, without punctuation and spaces, and with `ё` read as `е` (Russian texts use
# them interchangeably).


# %%
def fingerprint(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.casefold().replace("ё", "е"))


test_en, test_ru = set(test.en.map(fingerprint)), set(test.ru.map(fingerprint))
exact = train.en.isin(set(test.en)) | train.ru.isin(set(test.ru))
leaked = train.en.map(fingerprint).isin(test_en) | train.ru.map(fingerprint).isin(test_ru)
print(
    f"training pairs sharing a sentence with the test set: {int(exact.sum()):,} exactly, "
    f"{int(leaked.sum()):,} by fingerprint"
)
train = train[~leaked]
print(f"training pairs left: {len(train):,}")

# %% [markdown]
# ### A validation split that doesn't leak
#
# We also need a validation set: data the model never trains on, used to choose settings and to stop training at the
# right moment. The usual move is to hold out a random 2% of the pairs. Here that would be a mistake, and the picture
# shows why.
#
# ![Random split versus split by connected component](https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/01/leakage-light.png)
#
# These are real pairs from our data. Think of every sentence as a node and every translation pair as an edge. A random
# split cuts edges independently, so a validation pair like `Cut it out!` → `Прекрати!` still has both of its
# sentences in training pairs. The model has seen the source and has learned to produce the target; validation would
# reward memory.
#
# The fix is to split whole **connected components**: groups of sentences linked by translation, directly or through
# each other. We find them with a **union–find** structure: every sentence starts as its own group, every pair merges
# two groups, and `find` follows parent links to a group's representative. Path compression re-points every node on
# the way straight at the root, so later lookups are short. With compression alone, $m$ operations on $n$ nodes cost
# $O(m \log n)$ in the worst case; adding union by rank gives the famous near-linear bound $O(m\,\alpha(n))$
# ([Tarjan, 1975](https://doi.org/10.1145/321879.321884)). For our graph compression alone is plenty.


# %%
def find(parent: dict, node) -> object:
    root = node
    while parent.setdefault(root, root) != root:
        root = parent[root]
    while parent[node] != root:  # path compression
        parent[node], node = root, parent[node]
    return root


def connected_components(pairs: list[tuple[str, str]]) -> list:
    parent = {}
    for en, ru in pairs:
        a, b = find(parent, ("en", en)), find(parent, ("ru", ru))
        if a != b:
            parent[a] = b
    return [find(parent, ("en", en)) for en, _ in pairs]


start = time.perf_counter()
roots = connected_components(list(zip(train.en, train.ru)))
train = train.assign(group=pd.factorize(pd.Series(roots, index=train.index).map(str))[0])
sizes = train.groupby("group").size()
print(
    f"{sizes.size:,} components in {time.perf_counter() - start:.1f}s; "
    f"{int((sizes > 1).sum()):,} of them hold more than one pair, the largest holds {sizes.max()}"
)

# %% [markdown]
# The largest component is a whole family of ways to tell someone to stop, or to go away:

# %%
largest = train[train.group == sizes.idxmax()]
print(", ".join(sorted(set(largest.en))[:12]), "...")
print(", ".join(sorted(set(largest.ru))[:12]), "...")

# %% [markdown]
# Now compare the two ways of splitting. `leak_rate` is the share of validation pairs whose English or Russian
# sentence also occurs in training.


# %%
def leak_rate(train_part: pd.DataFrame, valid_part: pd.DataFrame) -> float:
    seen_en, seen_ru = set(train_part.en), set(train_part.ru)
    return float((valid_part.en.isin(seen_en) | valid_part.ru.isin(seen_ru)).mean())


rng = np.random.default_rng(SEED)
random_valid = rng.random(len(train)) < VAL_SHARE
print(f"random split:   {leak_rate(train[~random_valid], train[random_valid]):.1%} of validation pairs leak")

groups = train.group.unique()
valid_groups = set(rng.permutation(groups)[: int(VAL_SHARE * len(groups))])
is_valid = train.group.isin(valid_groups)
valid, train = train[is_valid].drop(columns="group"), train[~is_valid].drop(columns="group")
print(f"grouped split:  {leak_rate(train, valid):.1%} of validation pairs leak")
assert leak_rate(train, valid) == 0
print(f"\ntrain {len(train):,} · validation {len(valid):,} · test {len(test):,} pairs")

# %% [markdown]
# The splits are final. We save them; at the end of the lesson they are published so that lesson 2 can start from
# exactly these sentences.

# %%
DATA_DIR = WORK_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
splits = {"train": train, "validation": valid, "test": test}
for name, frame in splits.items():
    frame.reset_index(drop=True).to_parquet(DATA_DIR / f"{name}.parquet")
print(sorted(p.name for p in DATA_DIR.iterdir()))

# %% [markdown]
# ## 3. Why not words, and why not characters
#
# The simplest tokenizer splits text into words and gives each distinct word an id. Its problem is that the number of
# distinct words never stops growing. Watch it grow as we read through the training data:

# %%
WORD = re.compile(r"\w+")


def vocabulary_growth(texts: pd.Series, every: int) -> tuple[np.ndarray, np.ndarray]:
    """Number of distinct words V after reading N words, sampled every `every` words."""
    seen, n, ns, vs = set(), 0, [], []
    for text in texts:
        for word in WORD.findall(text.lower()):
            seen.add(word)
            n += 1
            if n % every == 0:
                ns.append(n)
                vs.append(len(seen))
    return np.array(ns), np.array(vs)


growth = {lang: vocabulary_growth(train[lang].sample(frac=1, random_state=SEED), every=1_000) for lang in ("en", "ru")}

# %% [markdown]
# Empirically, vocabulary growth follows **Heaps' law**:
#
# $$V(N) = K N^{\beta}, \qquad 0 < \beta < 1,$$
#
# where $N$ is the number of words read, $V(N)$ the number of distinct words among them, and $K$, $\beta$ constants
# of the language and the text. $\beta$ is the interesting one: it says how fast new word forms keep appearing. With
# $\beta = 0.5$, reading four times more text doubles the vocabulary.
#
# Taking logarithms turns the law into a straight line, $\log V = \log K + \beta \log N$, so we can estimate $\beta$ as
# the slope of a least-squares fit in log–log coordinates. In the plot, English is teal and Russian is rose, the
# colors of the encoder (which will read English) and the decoder (which will write Russian) for the rest of the
# course.

# %%
fig, ax = plt.subplots()
for lang, color in (("en", COLORS["enc"]), ("ru", COLORS["dec"])):
    n, v = growth[lang]
    beta, log_k = np.polyfit(np.log(n), np.log(v), 1)
    ax.loglog(n, v, color=color, label=f"{lang}: β = {beta:.2f}, {v[-1]:,} distinct words")
    ax.loglog(n, np.exp(log_k) * n**beta, color=color, linestyle=(0, (2, 3)), linewidth=1.2)
ax.set_xlabel("words read, N")
ax.set_ylabel("distinct words, V(N)")
ax.set_title("Russian keeps producing new word forms")
ax.legend(loc="upper left")
plt.show()

# %% [markdown]
# Russian grows faster and ends with several times more distinct words, although the Russian side has fewer words in
# total. The reason is morphology: `понимать, понимаю, понимаешь, понимание, недопонимание, понимания, …` are all
# different strings to a word tokenizer, while English mostly reuses `understand`.
#
# A real vocabulary must stop somewhere. Cap it at the 30 000 most frequent words and count how many validation words
# fall outside, the **out-of-vocabulary (OOV)** rate:


# %%
def oov_rate(train_texts: pd.Series, valid_texts: pd.Series, size: int = 30_000) -> float:
    counts = collections.Counter(w for text in train_texts for w in WORD.findall(text.lower()))
    vocabulary = {w for w, _ in counts.most_common(size)}
    words = [w for text in valid_texts for w in WORD.findall(text.lower())]
    return sum(w not in vocabulary for w in words) / len(words)


for lang in ("en", "ru"):
    print(f"{lang}: {oov_rate(train[lang], valid[lang]):.2%} of validation words are unknown")

# %% [markdown]
# Every unknown word becomes the same placeholder, and the model can neither read nor write it. For Russian that
# happens an order of magnitude more often than for English.
#
# The opposite extreme, one token per character, has no unknown words at all, but the sequences get long:

# %%
for lang in ("en", "ru"):
    chars_per_sentence = valid[lang].str.len().mean()
    words_per_sentence = n_words(valid[lang]).mean()
    print(
        f"{lang}: {chars_per_sentence:.1f} characters vs {words_per_sentence:.1f} words per sentence "
        f"({chars_per_sentence / words_per_sentence:.1f}x longer)"
    )

# %% [markdown]
# Length is expensive. In lesson 2 you will see that self-attention compares every position with every other, so its
# cost grows with the square of the sequence length $T$:
#
# $$\text{cost} \propto T^2 \quad\Rightarrow\quad \frac{\text{cost}_{\text{chars}}}{\text{cost}_{\text{words}}} =
# \left(\frac{T_{\text{chars}}}{T_{\text{words}}}\right)^2.$$
#
# Sequences five to six times longer make attention twenty-five to thirty-six times more expensive, and the model has
# to learn spelling before it can learn meaning.
#
# **Subword** tokenization is the compromise: frequent words stay whole, rare words are split into frequent pieces,
# and every string can still be written with the basic alphabet. Byte-pair encoding is the most widely used way to
# find those pieces.

# %% [markdown]
# ## 4. BPE from scratch
#
# BPE began as a compression algorithm (Gage, 1994) and was brought to machine
# translation by [Sennrich et al. (2016)](https://arxiv.org/abs/1508.07909). The idea fits in one picture:
#
# ![The BPE training loop](https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/01/bpe-loop-light.png)
#
# Start with every word spelled out as single characters. Then repeat: find the pair of adjacent symbols that occurs
# most often in the corpus, merge it into one new symbol everywhere, and add that symbol to the vocabulary.
#
# Formally, let $f(w)$ be the frequency of word $w$ in the corpus and $n_{ab}(w)$ the number of times symbol $a$ is
# immediately followed by symbol $b$ in the current spelling of $w$. The frequency of the pair is
#
# $$c(a, b) = \sum_{w} f(w)\, n_{ab}(w),$$
#
# and each step merges
#
# $$(a^*, b^*) = \operatorname*{arg\,max}_{(a, b)} c(a, b).$$
#
# If two pairs are equally frequent we take the one that compares greater as a pair of Python strings, that is, by
# Unicode code points. Any fixed rule works; what matters is that it is deterministic, so the same corpus always
# yields the same tokenizer. After $M$ merges the
# vocabulary holds the initial alphabet $\Sigma$ plus one symbol per merge: $\lvert\mathcal{V}\rvert =
# \lvert\Sigma\rvert + M$.

# %% [markdown]
# ### Pre-tokenization
#
# Merges never cross word boundaries, so we first split text into words and punctuation. A word keeps the space in
# front of it, written as `▁` (the convention of SentencePiece, [Kudo and Richardson, 2018](https://arxiv.org/abs/1808.06226)).
# With the spaces inside the tokens, decoding is just concatenation, and `▁` lets the model tell the beginning of a
# word from the middle of one.

# %%
PRE_TOKEN = re.compile(r" ?\w+| ?[^\w\s]+")


def pretokenize(text: str) -> list[str]:
    return [piece.replace(" ", "▁") for piece in PRE_TOKEN.findall(" " + text)]


print(pretokenize(RU_EXAMPLE))
print(pretokenize("Let's go — now!"))

# %% [markdown]
# Training only needs to know which words occur and how often. Each word becomes a tuple of symbols:


# %%
def word_counts(texts) -> dict[tuple[str, ...], int]:
    counts = collections.Counter(piece for text in texts for piece in pretokenize(text))
    return {tuple(word): n for word, n in counts.items()}


sample = pd.concat([train.en, train.ru]).sample(NAIVE_SENTENCES, random_state=SEED)
sample_words = word_counts(sample)
print(f"{len(sample_words):,} distinct words in {len(sample):,} sentences")
print(sorted(sample_words.items(), key=lambda kv: -kv[1])[:5])

# %% [markdown]
# ### The straightforward trainer
#
# Three small functions implement the loop. `pair_counts` computes $c(a, b)$ for every pair, `best_pair` takes the
# arg max with our tie-breaking rule, and `merge_word` rewrites one word with the new symbol.

# %%
Word = tuple[str, ...]
Pair = tuple[str, str]


def pair_counts(words: dict[Word, int]) -> collections.Counter:
    counts = collections.Counter()
    for word, freq in words.items():
        for pair in zip(word, word[1:]):
            counts[pair] += freq
    return counts


def best_pair(counts: dict[Pair, int]) -> Pair:
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]  # frequency first, then the pair itself


def merge_word(word: Word, pair: Pair) -> Word:
    out, i = [], 0
    while i < len(word):
        if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
            out.append(word[i] + word[i + 1])
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


# %% [markdown]
# Training is the loop itself:


# %%
def train_bpe_naive(words: dict[Word, int], n_merges: int) -> list[Pair]:
    merges = []
    for _ in range(n_merges):
        counts = pair_counts(words)
        if not counts:
            break
        pair = best_pair(counts)
        words = {merge_word(word, pair): freq for word, freq in words.items()}
        merges.append(pair)
    return merges


start = time.perf_counter()
naive_merges = train_bpe_naive(sample_words, NAIVE_MERGES)
naive_seconds = time.perf_counter() - start
print(f"{len(naive_merges)} merges in {naive_seconds:.1f}s")

# %% [markdown]
# The first merges are the most frequent pieces of both languages, interleaved:

# %%
token_chips(
    [["".join(pair) for pair in naive_merges[i : i + 10]] for i in (0, 10, 20)],
    labels=["merges 1–10", "11–20", "21–30"],
)
plt.show()

# %% [markdown]
# Word starts like `▁t`, `▁н`, `▁п` come first, then frequent syllables and whole short words. `▁T` seems to appear
# twice in the second row: one is the Latin `T`, the other the Cyrillic `Т` that starts `Том`. They look identical and
# are different characters, which is exactly why we never compare text by how it looks.
#
# The trainer is correct but slow. Every merge recounts all pairs in all words, so the cost is
# $O(M \cdot L)$, where $L$ is the total number of symbols in all distinct words. Both factors grow for the real
# run: the full training set has several times more distinct words than this sample, and we want thousands of merges,
# not hundreds.

# %% [markdown]
# ### The incremental trainer
#
# A merge only changes the words that contain the merged pair; every other count stays the same. So we keep the pair
# counts between steps, plus an index from each pair to the words that contain it. After a merge we visit only the
# affected words: subtract their old pairs, rewrite them, add their new pairs.


# %%
def train_bpe(words: dict[Word, int], n_merges: int) -> tuple[list[Pair], dict[Word, Word]]:
    """Return the merges and the final segmentation of every training word."""
    spellings = [list(word) for word in words]
    freqs = list(words.values())
    counts, where = collections.Counter(), collections.defaultdict(set)
    for i, word in enumerate(spellings):
        for pair in zip(word, word[1:]):
            counts[pair] += freqs[i]
            where[pair].add(i)

    merges = []
    for _ in range(n_merges):
        if not counts:
            break
        pair = best_pair(counts)
        merges.append(pair)
        for i in list(where[pair]):  # only the words that contain the pair
            old = spellings[i]
            for p in zip(old, old[1:]):
                counts[p] -= freqs[i]
                if counts[p] == 0:
                    del counts[p]
                where[p].discard(i)
            new = list(merge_word(tuple(old), pair))
            for p in zip(new, new[1:]):
                counts[p] += freqs[i]
                where[p].add(i)
            spellings[i] = new
    return merges, {word: tuple(s) for word, s in zip(words, spellings)}


# %% [markdown]
# A faster algorithm is only useful if it computes the same thing. Same input, same number of merges:

# %%
start = time.perf_counter()
fast_merges, _ = train_bpe(sample_words, NAIVE_MERGES)
fast_seconds = time.perf_counter() - start
assert fast_merges == naive_merges
print(f"identical merges; {naive_seconds:.1f}s -> {fast_seconds:.2f}s ({naive_seconds / fast_seconds:.0f}x faster)")

# %% [markdown]
# Now the full training set, both languages together. One shared vocabulary for English and Russian lets the model
# share one embedding table between encoder and decoder later on.

# %%
all_words = word_counts(pd.concat([train.en, train.ru]))
start = time.perf_counter()
merges, segmentation = train_bpe(all_words, N_MERGES)
print(f"{len(merges):,} merges over {len(all_words):,} distinct words in {time.perf_counter() - start:.0f}s")
alphabet = {symbol for word in all_words for symbol in word}
print(f"vocabulary: {len(alphabet):,} characters + {len(merges):,} merges = {len(alphabet) + len(merges):,} symbols")

# %% [markdown]
# ### Encoding: replay the merges in order
#
# To tokenize new text we do not count anything. We apply the learned merges in the order they were learned: in each
# word, find the adjacent pair with the lowest **rank** (the earliest merge), merge it, and repeat until no adjacent pair
# is in the merge table. Ranks matter because merges build on each other: `▁недо` exists only because `▁н`, `▁не` and
# `до` were merged first.

# %%
ranks = {pair: rank for rank, pair in enumerate(merges)}


def encode_word(word: str, ranks: dict[Pair, int], steps: list | None = None) -> list[str]:
    symbols = list(word)
    while len(symbols) > 1:
        rank, i = min((ranks.get(pair, math.inf), i) for i, pair in enumerate(zip(symbols, symbols[1:])))
        if rank == math.inf:
            break
        symbols[i : i + 2] = [symbols[i] + symbols[i + 1]]
        if steps is not None:
            steps.append((list(symbols), i))
    return symbols


steps = []
encode_word("▁недопонимание", ranks, steps)
token_chips([s for s, _ in steps], strong=lambda row, col, _: col == steps[row][1], size=11)
plt.show()

# %% [markdown]
# Each row is one merge; the highlighted symbol is the one just created. Replaying merges by rank reproduces exactly
# the segmentation that training arrived at, which we can check for every word in the training data:

# %%
mismatches = [w for w, seg in segmentation.items() if encode_word("".join(w), ranks) != list(seg)]
assert not mismatches, mismatches[:5]
print(f"encoding reproduces the training segmentation for all {len(segmentation):,} words")

# %% [markdown]
# Encoding a sentence is pre-tokenization plus `encode_word`, and decoding is concatenation with `▁` turned back
# into spaces. Nothing is lost on the way, which the round trip over the validation set confirms:


# %%
def encode(text: str) -> list[str]:
    return [token for word in pretokenize(text) for token in encode_word(word, ranks)]


def decode(tokens: list[str]) -> str:
    return "".join(tokens).replace("▁", " ")[1:]


assert all(decode(encode(s)) == s for s in pd.concat([valid.en, valid.ru]))
print(encode(RU_EXAMPLE))

# %% [markdown]
# ### What the merges learned
#
# Nobody told BPE about prefixes or suffixes, yet some appear on their own, because a frequent morpheme is a frequent
# string:

# %%
words = ["▁misunderstanding", "▁недопонимание", "▁исследовательница", "▁вымоталась", "▁researcher", "▁untranslatable"]
token_chips([encode_word(w, ranks) for w in words], labels=[w[1:] for w in words], size=11)
plt.show()

# %% [markdown]
# Some splits look like a linguist's (`недо`, the feminine suffix `ница`), others don't (`st` + `and`). BPE optimizes
# frequency, not meaning. The model will learn what the pieces mean from context; what we control is that frequent
# things are short and every rare thing can still be spelled.

# %% [markdown]
# ## 5. The production tokenizer
#
# Our implementation is readable and correct, and it would also be slow at inference and easy to break: the
# normalization lives in one function, the pre-tokenization in another, and nothing forces a server to call them the
# same way as training did. Hugging Face [`tokenizers`](https://huggingface.co/docs/tokenizers) packages the whole
# pipeline (normalizer, pre-tokenizer, model, decoder) into one serializable object, implemented in Rust.
#
# The rule we follow: **the tokenizer owns its normalization**. Then training, evaluation and the deployed service
# cannot normalize differently, because there is no separate step to forget.
#
# First the normalizer, rebuilt from the same choices as `normalize`:

# %%
hf_normalizer = normalizers.Sequence(
    [
        normalizers.NFKC(),
        normalizers.Replace(Regex(r"\p{Mn}"), ""),
        *[normalizers.Replace(a, b) for a, b in QUOTES.items()],
        normalizers.Replace(Regex(r"\s+-\s+"), " — "),
        normalizers.Replace(Regex(r"\s+"), " "),
        normalizers.Strip(),
    ]
)
raw_texts = pd.concat([train_raw.en, train_raw.ru, test_raw.en, test_raw.ru])
different = [s for s in raw_texts if hf_normalizer.normalize_str(s) != normalize(s)]
assert not different, different[:5]
print(f"identical normalization on all {len(raw_texts):,} raw sentences")

# %% [markdown]
# Then the pre-tokenizer. The same regular expression splits words and punctuation, and `Metaspace` turns the spaces
# into `▁`. Regular-expression engines don't always agree on what a "word character" is (Python's and the Rust
# library's disagree on combining marks and subscript digits), which is one more reason the normalizer removed those
# first. We check the result instead of assuming it:

# %%
hf_pre_tokenizer = pre_tokenizers.Sequence(
    [
        pre_tokenizers.Split(Regex(r" ?\w+| ?[^\w\s]+"), behavior="isolated"),
        pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="never", split=False),
    ]
)


def hf_pretokenize(text: str) -> list[str]:
    return [piece for piece, _ in hf_pre_tokenizer.pre_tokenize_str(" " + text)]


normalized = pd.concat([train.en, train.ru, valid.en, valid.ru, test.en, test.ru])
assert all(hf_pretokenize(s) == pretokenize(s) for s in normalized)
print(f"identical pre-tokenization on all {len(normalized):,} normalized sentences")

# %% [markdown]
# The tokenizer needs four special tokens, which the model will use from lesson 2 on: `<pad>` fills shorter
# sentences in a batch, `<unk>` stands for a character never seen in training, `<bos>` and `<eos>` mark the beginning
# and end of a sentence. They take ids 0 to 3.

# %%
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def build_tokenizer(vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence([hf_normalizer, normalizers.Prepend(" ")])
    tokenizer.pre_tokenizer = hf_pre_tokenizer
    tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always", split=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS, min_frequency=2, show_progress=False
    )
    tokenizer.train_from_iterator(list(train.en) + list(train.ru), trainer)
    return tokenizer


start = time.perf_counter()
tokenizer = build_tokenizer(VOCAB_SIZE)
print(f"vocabulary of {tokenizer.get_vocab_size():,} trained in {time.perf_counter() - start:.1f}s")
print(tokenizer.encode(RU_EXAMPLE).tokens)

# %% [markdown]
# Compare the two timings: the Rust trainer builds twice as many symbols in a fraction of the time our Python one
# needed for the same corpus. Both see exactly the same words, since we checked normalization and pre-tokenization, so their merges
# should agree except where the two implementations make different choices: Hugging Face breaks ties between equally
# frequent pairs differently and skips pairs seen only once (`min_frequency=2`). Compare the first thousand merges:
#
# %%
hf_merges = [
    tuple(m.split(" ")) if isinstance(m, str) else tuple(m) for m in json.loads(tokenizer.to_str())["model"]["merges"]
]
shared = set(hf_merges[:1_000]) & set(merges[:1_000])
print(f"{len(shared)} of the first 1,000 merges are the same in both")

# %% [markdown]
# The two implementations agree on 998 of the first thousand merges. The first disagreement is a tie: at merge 546,
# `▁d` + `e` and `▁l` + `ook` both occur 6 286 times, and Hugging Face takes the first while our rule takes the
# second. Same algorithm, different tie-breaking.
#
# The tokenizer is now self-contained. Feed it raw, unnormalized text, and decoding returns the normalized sentence:

# %%
messy = "  Он сказал:   «Я не  знаю» - и ушёл…"
ids = tokenizer.encode(messy).ids
print(ids)
print(repr(tokenizer.decode(ids)))
assert tokenizer.decode(ids) == normalize(messy)
assert [tokenizer.token_to_id(t) for t in SPECIAL_TOKENS] == [0, 1, 2, 3]

# %% [markdown]
# ### Choosing the vocabulary size
#
# A bigger vocabulary makes sequences shorter, but it is not free. Each token needs a row in the embedding table, and
# at every output position the model scores every token:
#
# $$\text{embedding parameters} = \lvert\mathcal{V}\rvert \cdot d_{\text{model}}.$$
#
# A rare token also gets few training examples, so its embedding is learned poorly. We measure both sides: the
# average number of tokens per validation sentence for several vocabulary sizes, and the parameter count of the
# embedding table for the model of lesson 2 ($d_{\text{model}} = 256$).

# %%
rows = []
for size in VOCAB_SIZES:
    candidate = build_tokenizer(size)
    row = {"vocab_size": size, "embedding_params": size * D_MODEL}
    for lang in ("en", "ru"):
        row[f"{lang}_tokens"] = np.mean([len(e.ids) for e in candidate.encode_batch(list(valid[lang]))])
    rows.append(row)
sweep = pd.DataFrame(rows).set_index("vocab_size")
sweep.round(2)

# %%
fig, ax = plt.subplots()
ax.plot(sweep.index, sweep.en_tokens, marker="o", color=COLORS["enc"], label="English")
ax.plot(sweep.index, sweep.ru_tokens, marker="o", color=COLORS["dec"], label="Russian")
ax.axvline(VOCAB_SIZE, color=COLORS["muted"], linestyle=(0, (3, 3)), linewidth=1)
ax.set_xscale("log", base=2)
ax.set_xticks(sweep.index, [f"{v // 1000}k" if v >= 1000 else str(v) for v in sweep.index])
ax.set_xlabel("vocabulary size")
ax.set_ylabel("tokens per sentence")
ax.set_title("Diminishing returns of a bigger vocabulary")
ax.legend()
plt.show()

# %% [markdown]
# Each doubling of the vocabulary saves fewer tokens than the previous one, while the embedding table doubles every
# time. We keep 16 000 symbols. Going to 32 000 would save about half a token per Russian sentence and a quarter of
# one per English sentence, and double an embedding table that already holds four million parameters, a sizeable
# share of a small translation model.
#
# ### The token tax, revisited
#
# Back to where we started. Here are both tokenizers on the whole validation set:

# %%
for lang in ("en", "ru"):
    ours = np.mean([len(e.ids) for e in tokenizer.encode_batch(list(valid[lang]))])
    theirs = np.mean([len(e.ids) for e in gpt2.encode_batch(list(valid[lang]))])
    print(f"{lang}: ours {ours:.1f} tokens per sentence, GPT-2 {theirs:.1f}, GPT-2 / ours = {theirs / ours:.2f}")
for sentence in (EN_EXAMPLE, RU_EXAMPLE):
    print(f"{len(tokenizer.encode(sentence).ids):>3} tokens: {sentence}")

# %% [markdown]
# On Russian our tokenizer needs a fraction of GPT-2's tokens. On English it is slightly worse, which is the honest
# price of sharing 16 000 symbols between two languages, while GPT-2 spends 50 000 mostly on English. For a
# translator that reads English and writes Russian, that is the right trade.
#
# ### Save and publish
#
# The tokenizer is one JSON file. The next lessons download it, together with the three splits, from the Hugging Face
# Hub, so that every lesson can start on its own. The course's copies are published from this notebook; to publish
# your own, set `TFS_HUB_USER` to your Hugging Face username and add a write token as the `HF_TOKEN` secret (Kaggle:
# *Add-ons → Secrets*; Colab: the key icon in the sidebar).

# %%
TOKENIZER_PATH = WORK_DIR / "tokenizer.json"
tokenizer.save(str(TOKENIZER_PATH))
assert Tokenizer.from_file(str(TOKENIZER_PATH)).encode(RU_EXAMPLE).ids == tokenizer.encode(RU_EXAMPLE).ids

DATASET_CARD = f"""---
license: cc-by-2.0
language: [en, ru]
task_categories: [translation]
---
# Tatoeba English–Russian, cleaned and split without leaks

Built in lesson 1 of [Transformers: from scratch to production](https://github.com/xmarva/transformers-from-scratch).

- train and validation: [sentence-transformers/parallel-sentences-tatoeba](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-tatoeba) (en-ru), revision `{TRAIN_REVISION[:7]}`
- test: Tatoeba Translation Challenge eng-rus test set (Tiedemann, 2020)
- NFKC normalization, combining marks removed, straight quotes, one dash
- training pairs sharing a sentence fingerprint with the test set removed
- validation split by connected components of the translation graph, so no sentence occurs in both

Sentences from [Tatoeba](https://tatoeba.org), CC BY 2.0 FR.

| split | pairs |
|---|---|
| train | {len(train):,} |
| validation | {len(valid):,} |
| test | {len(test):,} |
"""

HUB_USER = os.environ.get("TFS_HUB_USER")
hf_token = env.secret("HF_TOKEN")
if HUB_USER and hf_token and not SMOKE:
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    data_repo, tokenizer_repo = f"{HUB_USER}/tfs-tatoeba-en-ru", f"{HUB_USER}/tfs-bpe-en-ru-16k"
    api.create_repo(data_repo, repo_type="dataset", exist_ok=True)
    for name in splits:
        api.upload_file(
            path_or_fileobj=str(DATA_DIR / f"{name}.parquet"),
            path_in_repo=f"data/{name}.parquet",
            repo_id=data_repo,
            repo_type="dataset",
        )
    api.upload_file(
        path_or_fileobj=DATASET_CARD.encode(), path_in_repo="README.md", repo_id=data_repo, repo_type="dataset"
    )
    api.create_repo(tokenizer_repo, exist_ok=True)
    api.upload_file(path_or_fileobj=str(TOKENIZER_PATH), path_in_repo="tokenizer.json", repo_id=tokenizer_repo)
    print(f"published https://huggingface.co/datasets/{data_repo} and https://huggingface.co/{tokenizer_repo}")
else:
    print(
        "not publishing; lesson 2 downloads the course's copies from xmarva/tfs-tatoeba-en-ru and xmarva/tfs-bpe-en-ru-16k"
    )

# %% [markdown]
# ## Where this goes wrong
#
# - **Splitting pairs at random when sentences repeat.** Here a random split leaked more than half of the validation
#   set. Any dataset with paraphrases, multiple references or near-duplicates needs grouped splits.
# - **Checking test overlap with exact matches.** Case, punctuation and `ё`/`е` hid hundreds of leaked pairs from the
#   exact comparison.
# - **Cleaning by recipe.** The length-ratio filter would have removed some of the most natural translations in the
#   corpus. Read what a filter removes before you apply it.
# - **Normalizing in two places.** If the server normalizes differently from training, the model sees inputs it was
#   never trained on, and nothing crashes to tell you. Put normalization inside the tokenizer.
# - **Assuming two implementations agree.** Python's and Rust's regular expressions disagree on what a word character
#   is. The asserts in this notebook are cheap; silent mismatches are not.
# - **Judging a tokenizer by its vocabulary size.** Measure tokens per sentence on your own data, per language.

# %% [markdown]
# ## Exercises
#
# **1. BPE-dropout.** [Provilkov et al. (2020)](https://arxiv.org/abs/1910.13267) randomly skip merges at encoding
# time, so the same word is segmented differently on different epochs. Write `encode_word_dropout(word, ranks, p, rng)`
# that ignores each candidate merge with probability `p`, and print ten segmentations of `▁недопонимание` with
# `p = 0.1`. Why could seeing several segmentations of a word help a translation model?
#
# <details><summary>Solution</summary>
#
# ```python
# def encode_word_dropout(word, ranks, p, rng):
#     symbols = list(word)
#     while len(symbols) > 1:
#         candidates = [(ranks[pair], i) for i, pair in enumerate(zip(symbols, symbols[1:]))
#                       if pair in ranks and rng.random() >= p]
#         if not candidates:
#             break
#         _, i = min(candidates)
#         symbols[i:i + 2] = [symbols[i] + symbols[i + 1]]
#     return symbols
#
# rng = np.random.default_rng(0)
# for _ in range(10):
#     print(encode_word_dropout("▁недопонимание", ranks, 0.1, rng))
# ```
#
# The model sees the word through different pieces, so it learns what the pieces mean instead of memorizing one
# segmentation, and it copes better with rare words and typos, whose segmentation is unusual anyway. Use dropout only
# for training data; evaluation needs the deterministic `encode_word`.
# </details>
#
# **2. Group by one side only.** Split the data by the English sentence alone (all translations of one English
# sentence on the same side) and measure `leak_rate`. Which sentences still leak, and why?
#
# <details><summary>Solution</summary>
#
# ```python
# english = pd.concat([train, valid])
# held_out = set(rng.permutation(english.en.unique())[: int(VAL_SHARE * english.en.nunique())])
# part_valid = english.en.isin(held_out)
# print(f"{leak_rate(english[~part_valid], english[part_valid]):.1%}")
# ```
#
# No English sentence leaks, but Russian ones do: `Хватит!` translates both `Cut it out!` and `Enough!`, and the two
# English sentences can land on different sides. Only connected components close both doors.
# </details>
#
# **3. One vocabulary or two?** Train our from-scratch tokenizer on English only and on Russian only, with the same
# number of merges as the joint one, and compare tokens per validation sentence in each language with the joint
# tokenizer. Which language loses more from sharing, and what would you change in the joint training data?
#
# <details><summary>Hint</summary>
#
# Call `train_bpe(word_counts(train.en), N_MERGES)` and `train_bpe(word_counts(train.ru), N_MERGES)`, and build an
# `encode` for each set of ranks. A natural fix is to balance the languages in the training data, for instance by
# sampling so that both contribute the same number of words.
# </details>

# %% [markdown]
# ## Summary
#
# - Text reaches a model as integers, and the tokenizer decides how many. For Russian, an English-centric tokenizer
#   multiplies the cost; a tokenizer trained on our data removes most of that tax.
# - The data comes first. We normalized it, filtered only what was clearly broken, removed every training pair that
#   shares a sentence with the test set, and split validation by connected components, so that our measurements
#   measure translation and not memory.
# - Words make vocabularies explode (Heaps' law, especially for Russian), characters make sequences long. BPE finds
#   frequent subwords in between: count pairs, merge the most frequent, repeat; encode by replaying merges in order.
# - The tokenizer owns its normalization, so training and serving can't drift apart.
#
# **Published artifacts**: the dataset `xmarva/tfs-tatoeba-en-ru` (train, validation, test) and the tokenizer
# `xmarva/tfs-bpe-en-ru-16k`.
#
# **Next**: lesson 2 builds the Transformer that reads these token ids, piece by piece, from embeddings to attention.

# %% [markdown]
# ## References
#
# - Gage, P. (1994). A new algorithm for data compression. *The C Users Journal*, 12(2).
# - Sennrich, R., Haddow, B., Birch, A. (2016). [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909). ACL.
# - Kudo, T., Richardson, J. (2018). [SentencePiece: A simple and language independent subword tokenizer and detokenizer](https://arxiv.org/abs/1808.06226). EMNLP.
# - Provilkov, I., Emelianenko, D., Voita, E. (2020). [BPE-Dropout: Simple and Effective Subword Regularization](https://arxiv.org/abs/1910.13267). ACL.
# - Petrov, A., La Malfa, E., Torr, P., Bibi, A. (2023). [Language Model Tokenizers Introduce Unfairness Between Languages](https://arxiv.org/abs/2305.15425). NeurIPS.
# - Tiedemann, J. (2020). [The Tatoeba Translation Challenge](https://arxiv.org/abs/2010.06354). WMT.
# - Heaps, H. S. (1978). *Information Retrieval: Computational and Theoretical Aspects*. Academic Press.
# - Tarjan, R. E. (1975). Efficiency of a good but not linear set union algorithm. *Journal of the ACM*, 22(2).
# - [Unicode Standard Annex #15: Unicode Normalization Forms](https://unicode.org/reports/tr15/).
# - [Hugging Face Tokenizers documentation](https://huggingface.co/docs/tokenizers).
