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
# **What you need**: Python and a little pandas. No previous lesson. A free CPU runtime is enough; the whole notebook
# runs in about ten minutes.

# %% tags=["include:setup"]

# %%
import collections
import csv
import json
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
# All the settings of the lesson live in one cell. `SMOKE` is switched on only by the course's automated tests, which
# run the notebook on a small slice of data; for you it is `False`.

# %%
VAL_SHARE = 0.02  # share of the data held out for validation

if SMOKE:
    SAMPLE_SENTENCES, SAMPLE_MERGES = 2_000, 50
    N_MERGES = 300
    VOCAB_SIZES = [500, 1_000]
    VOCAB_SIZE = 1_000
else:
    SAMPLE_SENTENCES, SAMPLE_MERGES = 40_000, 300  # the slow BPE trainer runs on a sample
    N_MERGES = 8_000  # merges for our from-scratch tokenizer
    VOCAB_SIZES = [2_000, 4_000, 8_000, 16_000, 32_000]  # sizes we compare
    VOCAB_SIZE = 16_000  # the tokenizer the course keeps

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

print("English:", len(gpt2.encode(EN_EXAMPLE).ids), "tokens")
print("Russian:", len(gpt2.encode(RU_EXAMPLE).ids), "tokens")

# %% [markdown]
# The same meaning, a very different bill. Look at what the Russian tokens actually are:

# %%
print(gpt2.encode(RU_EXAMPLE).tokens[:12])
print("characters:", len(RU_EXAMPLE))
print("bytes in UTF-8:", len(RU_EXAMPLE.encode("utf-8")))

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
# Both downloads are pinned to an exact revision, so the numbers in this notebook do not change when the sources do.

# %%
TRAIN_REVISION = "cec1343ab5a7a8befe99af4a2d0ca847b6c84743"

dataset = load_dataset(
    "sentence-transformers/parallel-sentences-tatoeba", "en-ru", split="train", revision=TRAIN_REVISION
)
train_raw = dataset.to_pandas()
train_raw = train_raw.rename(columns={"english": "en", "non_english": "ru"})

# %% [markdown]
# The test set is a plain text file. Each line holds four fields separated by tabs: the two language codes, the
# English sentence and its Russian translation.

# %%
TEST_URL = (
    "https://raw.githubusercontent.com/Helsinki-NLP/Tatoeba-Challenge/"
    "5509af44a74016748c26f41ab164b64f48445f3f/data/test/eng-rus/test.txt"
)

test_raw = pd.read_csv(
    TEST_URL,
    sep="\t",
    names=["src_lang", "tgt_lang", "en", "ru"],
    quoting=csv.QUOTE_NONE,  # quote marks are part of the sentences, not of the file format
    dtype=str,
    keep_default_na=False,  # a sentence like "NA" must stay a sentence
)
test_raw = test_raw[["en", "ru"]]

if SMOKE:
    train_raw = train_raw.sample(20_000, random_state=SEED)
    test_raw = test_raw.sample(2_000, random_state=SEED)

print("training pairs:", len(train_raw))
print("test pairs:", len(test_raw))
train_raw.sample(5, random_state=SEED)

# %% [markdown]
# ### One sentence, many translations
#
# The data has a property that matters for everything that follows: one English sentence often comes with several
# Russian translations, and one Russian sentence can translate several English ones. Count the translations of every
# English sentence:

# %%
translations_per_sentence = train_raw.groupby("en").size()

print("distinct English sentences:", len(translations_per_sentence))
print("translations per sentence, on average:", round(translations_per_sentence.mean(), 2))

# %% [markdown]
# And look at the record holder:

# %%
champion = translations_per_sentence.idxmax()
its_translations = train_raw[train_raw.en == champion].ru

print(champion, "has", len(its_translations), "translations:")
print(", ".join(its_translations))

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
char_counts = collections.Counter("".join(train_raw.en) + "".join(train_raw.ru))

for char in "\"«»„“”'‘’-–—…":
    if char_counts[char] > 0:
        print(repr(char), char_counts[char])

# %% [markdown]
# Some variants are a style choice (`«ёлочки»` versus `"straight quotes"`), others are invisible. The letter `й`, for
# instance, can be stored as one code point, or as `и` followed by a combining breve (`̆`). Python considers the
# two strings different, although every reader sees the same letter:

# %%
one_code_point = "й"
two_code_points = "й"

print(one_code_point == two_code_points)
print(len(one_code_point), len(two_code_points))

# %% [markdown]
# Unicode defines **normal forms** for exactly this. Normal form **NFKC** composes letters with their accents, and
# also maps "compatibility" characters to plain ones: subscript digits, non-breaking spaces, the one-character
# ellipsis `…`.

# %%
print(unicodedata.normalize("NFKC", two_code_points) == one_code_point)
print(unicodedata.normalize("NFKC", "H₂O"))
print(repr(unicodedata.normalize("NFKC", "two words")))  #   is a non-breaking space

# %% [markdown]
# On top of NFKC we make four choices of our own, and write them down because the model will inherit them:
#
# 1. drop combining marks that survive NFKC: Tatoeba marks stress in some Russian sentences (`То́му`), which is a
#    pronunciation hint, not spelling;
# 2. use straight quotes and apostrophes, and one kind of dash, `—`;
# 3. read a hyphen surrounded by spaces as a dash;
# 4. collapse runs of whitespace into one space.

# %%
QUOTES = {"«": '"', "»": '"', "„": '"', "“": '"', "”": '"', "‘": "'", "’": "'", "–": "—"}


def normalize(text):
    text = unicodedata.normalize("NFKC", text)

    # Unicode category "Mn" is a combining mark, such as a stress accent
    kept = [char for char in text if unicodedata.category(char) != "Mn"]
    text = "".join(kept)

    for fancy, plain in QUOTES.items():
        text = text.replace(fancy, plain)

    text = re.sub(r"\s+-\s+", " — ", text)  # " - " is a dash
    text = re.sub(r"\s+", " ", text)  # one space between words
    return text.strip()


# %% [markdown]
# Try it on a few sentences before we trust it:

# %%
for before in ["Он сказал: «Я не знаю».", "То́му было нечего пить.", "Wait a second - what?", "It’s 5 o’clock…"]:
    print(repr(before), "->", repr(normalize(before)))

# %% [markdown]
# Now normalize both sides of both datasets, and count how many training sentences actually changed:

# %%
train = train_raw.copy()
train["en"] = train_raw.en.map(normalize)
train["ru"] = train_raw.ru.map(normalize)

test = test_raw.copy()
test["en"] = test_raw.en.map(normalize)
test["ru"] = test_raw.ru.map(normalize)

changed = (train.en != train_raw.en).sum() + (train.ru != train_raw.ru).sum()
print("changed:", changed, "of", 2 * len(train), "training sentences")

# %% [markdown]
# ### Filters, and one we don't apply
#
# A pair is dropped when it is clearly broken: Cyrillic on the English side, no Cyrillic on the Russian side, or more
# than 40 words, which in Tatoeba means a paragraph that slipped in among sentences. Each check gives a column of
# `True`/`False`, one value per pair:

# %%
CYRILLIC = r"[а-яё]"


def word_count(sentences):
    return sentences.str.split().str.len()


cyrillic_in_english = train.en.str.contains(CYRILLIC, case=False)
no_cyrillic_in_russian = ~train.ru.str.contains(CYRILLIC, case=False)
too_long = (word_count(train.en) > 40) | (word_count(train.ru) > 40)

print("Cyrillic on the English side:", cyrillic_in_english.sum())
print("no Cyrillic on the Russian side:", no_cyrillic_in_russian.sum())
print("longer than 40 words:", too_long.sum())

# %% [markdown]
# What does Cyrillic on the English side look like?

# %%
train[cyrillic_in_english].head(3)

# %% [markdown]
# Mislabeled pairs. We drop all the broken ones, and then the duplicates: after normalization some pairs became
# identical.

# %%
broken = cyrillic_in_english | no_cyrillic_in_russian | too_long
train = train[~broken]

size_before = len(train)
train = train.drop_duplicates()
print("duplicates removed:", size_before - len(train))

# %% [markdown]
# Machine translation pipelines often also drop pairs whose lengths differ too much, on the theory that a translation
# three times longer than its source must be misaligned. Before applying a recipe, look at what it would remove:

# %%
length_ratio = train.en.str.len() / train.ru.str.len()
flagged = train[(length_ratio > 3) | (length_ratio < 1 / 3)]

print("the length-ratio filter would remove", len(flagged), "pairs, for example:")
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
# **fingerprints** instead: lowercase, without punctuation and spaces, and with `ё` read as `е` (Russian texts use them
# interchangeably).


# %%
def fingerprint(text):
    text = text.casefold().replace("ё", "е")  # casefold is lowercase that also handles special cases like ß
    return re.sub(r"[\W_]+", "", text)  # drop everything that is not a letter or a digit


print(fingerprint("Хватит!"), fingerprint("хватит."), fingerprint("Ты безнадёжна!"))

# %% [markdown]
# A training pair leaks when its English or its Russian sentence has the fingerprint of a test sentence. Compare the
# two ways of checking:

# %%
test_en_exact, test_ru_exact = set(test.en), set(test.ru)
test_en_prints, test_ru_prints = set(test.en.map(fingerprint)), set(test.ru.map(fingerprint))

leaks_exactly = train.en.isin(test_en_exact) | train.ru.isin(test_ru_exact)
leaks_by_fingerprint = train.en.map(fingerprint).isin(test_en_prints) | train.ru.map(fingerprint).isin(test_ru_prints)

print("pairs sharing a sentence with the test set, exact strings:", leaks_exactly.sum())
print("pairs sharing a sentence with the test set, fingerprints: ", leaks_by_fingerprint.sum())

train = train[~leaks_by_fingerprint]
print("training pairs left:", len(train))

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
# each other. A classic structure finds them, **union–find**. Every sentence starts as its own group. Every pair says
# "these two sentences belong together", and we merge their groups. To merge, each group needs a representative, its
# *root*: every sentence keeps a link to a "parent" sentence, and following parents leads to the root.


# %%
class UnionFind:
    def __init__(self):
        self.parent = {}  # sentence -> its parent; a root is its own parent

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x  # a sentence seen for the first time is a group of one

        root = x
        while self.parent[root] != root:
            root = self.parent[root]

        # path compression: point everything on the way straight at the root, so the next lookup is short
        while self.parent[x] != root:
            next_x = self.parent[x]
            self.parent[x] = root
            x = next_x
        return root

    def union(self, a, b):
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[root_a] = root_b  # the two groups become one


# %% [markdown]
# A sentence is stored together with its language, `("en", "Stop it.")`, because the same string can occur on both
# sides (names, numbers). Try it on the eight pairs from the picture, plus one unrelated pair:

# %%
example_pairs = [
    ("Cut it out!", "Перестань!"),
    ("Stop it.", "Перестань!"),
    ("Cut it out!", "Прекрати!"),
    ("Stop that!", "Прекрати!"),
    ("Stop it.", "Прекрати это!"),
    ("Stop that!", "Прекрати это!"),
    ("Cut it out!", "Хватит!"),
    ("Enough!", "Хватит!"),
    ("I'm hungry.", "Я голоден."),
]

groups = UnionFind()
for en, ru in example_pairs:
    groups.union(("en", en), ("ru", ru))

for en, ru in example_pairs:
    print(en, "/", ru, "-> group", groups.find(("en", en)))

# %% [markdown]
# All eight pairs from the picture end up in one group, named after its root, and the unrelated pair forms its own.
#
# With path compression alone, $m$ operations on $n$ sentences cost $O(m \log n)$ in the worst case; also attaching
# the smaller group under the larger one (union by rank) gives the famous near-linear bound $O(m\,\alpha(n))$
# ([Tarjan, 1975](https://doi.org/10.1145/321879.321884)). For our data compression alone is plenty. Now the real
# thing: every training pair, and a group label for each.

# %%
start = time.perf_counter()

groups = UnionFind()
for en, ru in zip(train.en, train.ru):
    groups.union(("en", en), ("ru", ru))

labels = [str(groups.find(("en", en))) for en in train.en]  # every pair is labeled with the root of its group
train = train.assign(group=labels)
print(f"done in {time.perf_counter() - start:.1f}s")

group_sizes = train.groupby("group").size()
print("groups:", len(group_sizes))
print("groups with more than one pair:", (group_sizes > 1).sum())
print("pairs in the largest group:", group_sizes.max())

# %% [markdown]
# The largest group is a whole family of ways to tell someone to stop, or to go away:

# %%
largest = train[train.group == group_sizes.idxmax()]
print(sorted(set(largest.en))[:12])
print(sorted(set(largest.ru))[:12])

# %% [markdown]
# Now compare the two ways of splitting. `leak_rate` measures the share of validation pairs whose English or Russian
# sentence also occurs somewhere in training:


# %%
def leak_rate(train_part, valid_part):
    seen_english = set(train_part.en)
    seen_russian = set(train_part.ru)
    leaked = valid_part.en.isin(seen_english) | valid_part.ru.isin(seen_russian)
    return leaked.mean()


# %% [markdown]
# First the random split: every pair goes to validation with probability 2%.

# %%
rng = np.random.default_rng(SEED)

random_valid = rng.random(len(train)) < VAL_SHARE
print(f"random split: {leak_rate(train[~random_valid], train[random_valid]):.1%} of validation pairs leak")

# %% [markdown]
# Then the split by groups: we pick 2% of the groups, and their pairs, all of them, go to validation.

# %%
all_groups = train.group.unique()
valid_groups = rng.choice(all_groups, size=int(VAL_SHARE * len(all_groups)), replace=False)
in_valid = train.group.isin(set(valid_groups))

valid = train[in_valid].drop(columns="group")
train = train[~in_valid].drop(columns="group")

print(f"split by groups: {leak_rate(train, valid):.1%} of validation pairs leak")
assert leak_rate(train, valid) == 0  # stops the notebook if a single pair leaks

print("train:", len(train), "validation:", len(valid), "test:", len(test))

# %% [markdown]
# The splits are final. We save them; at the end of the lesson they are published so that lesson 2 can start from
# exactly these sentences.

# %%
DATA_DIR = WORK_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

splits = {"train": train, "validation": valid, "test": test}
for name, frame in splits.items():
    frame.reset_index(drop=True).to_parquet(DATA_DIR / f"{name}.parquet")

print(sorted(path.name for path in DATA_DIR.iterdir()))

# %% [markdown]
# ## 3. Why not words, and why not characters
#
# The simplest tokenizer splits text into words and gives each distinct word an id. Its problem is that the number of
# distinct words never stops growing. Watch it grow as we read through the training data, recording the number of
# distinct words after every thousand words read:

# %%
WORD = re.compile(r"\w+")  # a run of letters or digits


def vocabulary_growth(sentences, every=1_000):
    seen = set()
    words_read = 0
    n_points, v_points = [], []
    for sentence in sentences:
        for word in WORD.findall(sentence.lower()):
            seen.add(word)
            words_read += 1
            if words_read % every == 0:
                n_points.append(words_read)
                v_points.append(len(seen))
    return np.array(n_points), np.array(v_points)


shuffled = train.sample(frac=1, random_state=SEED)
growth = {"en": vocabulary_growth(shuffled.en), "ru": vocabulary_growth(shuffled.ru)}

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
# the slope of a least-squares line through the points in log–log coordinates; `np.polyfit(x, y, 1)` returns the slope
# and the intercept. In the plot, English is teal and Russian is rose, the colors of the encoder (which will read
# English) and the decoder (which will write Russian) for the rest of the course.

# %%
fig, ax = plt.subplots()

for lang, color in [("en", COLORS["enc"]), ("ru", COLORS["dec"])]:
    n, v = growth[lang]
    beta, log_k = np.polyfit(np.log(n), np.log(v), 1)
    ax.loglog(n, v, color=color, label=f"{lang}: β = {beta:.2f}, {v[-1]:,} distinct words")
    ax.loglog(n, np.exp(log_k) * n**beta, color=color, linestyle=(0, (2, 3)), linewidth=1.2)  # the fitted law

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
# A real vocabulary must stop somewhere. Keep the 30 000 most frequent training words and count how many validation
# words fall outside, the **out-of-vocabulary (OOV)** rate:


# %%
def oov_rate(train_sentences, valid_sentences, size=30_000):
    counts = collections.Counter()
    for sentence in train_sentences:
        counts.update(WORD.findall(sentence.lower()))
    vocabulary = {word for word, _ in counts.most_common(size)}

    valid_words = [word for sentence in valid_sentences for word in WORD.findall(sentence.lower())]
    unknown = [word for word in valid_words if word not in vocabulary]
    return len(unknown) / len(valid_words)


print(f"English: {oov_rate(train.en, valid.en):.2%} of validation words are unknown")
print(f"Russian: {oov_rate(train.ru, valid.ru):.2%} of validation words are unknown")

# %% [markdown]
# Every unknown word becomes the same placeholder, and the model can neither read nor write it. For Russian that
# happens an order of magnitude more often than for English.
#
# The opposite extreme, one token per character, has no unknown words at all, but the sequences get long:

# %%
for lang in ["en", "ru"]:
    chars = valid[lang].str.len().mean()
    words = word_count(valid[lang]).mean()
    print(f"{lang}: {chars:.1f} characters vs {words:.1f} words per sentence, {chars / words:.1f} times longer")

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
# BPE began as a compression algorithm (Gage, 1994) and was brought to machine translation by
# [Sennrich et al. (2016)](https://arxiv.org/abs/1508.07909). The idea fits in one picture:
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
# yields the same tokenizer. After $M$ merges the vocabulary holds the initial alphabet $\Sigma$ plus one symbol per
# merge: $\lvert\mathcal{V}\rvert = \lvert\Sigma\rvert + M$.

# %% [markdown]
# ### Pre-tokenization
#
# Merges never cross word boundaries, so we first split text into words and punctuation. A word keeps the space in
# front of it, written as `▁` (the convention of SentencePiece, [Kudo and Richardson, 2018](https://arxiv.org/abs/1808.06226)).
# With the spaces inside the tokens, decoding is just concatenation, and `▁` lets the model tell the beginning of a
# word from the middle of one.
#
# The regular expression has two alternatives: ` ?\w+` is an optional space followed by letters or digits, and
# ` ?[^\w\s]+` is an optional space followed by punctuation (anything that is neither a word character nor
# whitespace).

# %%
PRE_TOKEN = re.compile(r" ?\w+| ?[^\w\s]+")


def pretokenize(text):
    pieces = PRE_TOKEN.findall(" " + text)  # the leading space marks the first word too
    return [piece.replace(" ", "▁") for piece in pieces]


print(pretokenize(RU_EXAMPLE))
print(pretokenize("Let's go — now!"))

# %% [markdown]
# ### Counting pairs on a toy corpus
#
# Before touching half a million sentences, run every step on three Russian words whose counts you can check by hand.
# A word is stored as a tuple of its symbols, with its frequency:

# %%
toy = {
    tuple("▁понимание"): 5,
    tuple("▁недопонимание"): 3,
    tuple("▁понимать"): 4,
}


def pair_counts(words):
    counts = collections.Counter()
    for word, freq in words.items():
        for left, right in zip(word, word[1:]):  # every adjacent pair in the word
            counts[left, right] += freq
    return counts


pair_counts(toy).most_common(5)

# %% [markdown]
# `н` + `и` wins: it occurs twice in `понимание` (five times), twice in `недопонимание` (three times) and once in
# `понимать` (four times), which gives $2 \cdot 5 + 2 \cdot 3 + 1 \cdot 4 = 20$, the formula for $c(a, b)$ above.
#
# Picking the winner, with our rule for ties:


# %%
def most_frequent_pair(counts):
    best_pair, best_count = None, 0
    for pair, count in counts.items():
        if count > best_count or (count == best_count and pair > best_pair):
            best_pair, best_count = pair, count
    return best_pair


most_frequent_pair(pair_counts(toy))

# %% [markdown]
# Merging a pair rewrites a word, scanning it from left to right:


# %%
def merge_word(word, pair):
    merged = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
            merged.append(word[i] + word[i + 1])  # two symbols become one
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return tuple(merged)


token_chips([tuple("▁недопонимание"), merge_word(tuple("▁недопонимание"), ("н", "и"))], labels=["before", "after"])
plt.show()

# %% [markdown]
# ### The straightforward trainer
#
# Training is these three steps in a loop:


# %%
def train_bpe_simple(words, n_merges):
    merges = []
    for _ in range(n_merges):
        counts = pair_counts(words)
        if not counts:  # every word is a single symbol, nothing left to merge
            break
        pair = most_frequent_pair(counts)
        merges.append(pair)
        words = {merge_word(word, pair): freq for word, freq in words.items()}
    return merges


train_bpe_simple(toy, 6)

# %% [markdown]
# Six merges on the toy corpus, each one built on the previous ones: `ни`, then `по`, then `по` + `ни` = `пони`, and so
# on until the whole stem `▁понима`. Now a real sample: we count the words of 40 000 sentences, English and Russian
# together.


# %%
def word_counts(sentences):
    counts = collections.Counter()
    for sentence in sentences:
        counts.update(pretokenize(sentence))
    return {tuple(word): freq for word, freq in counts.items()}


sample = pd.concat([train.en, train.ru]).sample(SAMPLE_SENTENCES, random_state=SEED)
sample_words = word_counts(sample)
print("distinct words in the sample:", len(sample_words))

start = time.perf_counter()
simple_merges = train_bpe_simple(sample_words, SAMPLE_MERGES)
simple_seconds = time.perf_counter() - start
print(f"{len(simple_merges)} merges in {simple_seconds:.1f}s")

# %% [markdown]
# The first merges are the most frequent pieces of both languages, interleaved:

# %%
first_30 = ["".join(pair) for pair in simple_merges[:30]]
token_chips([first_30[0:10], first_30[10:20], first_30[20:30]], labels=["merges 1–10", "11–20", "21–30"])
plt.show()

# %% [markdown]
# Word starts like `▁t`, `▁н`, `▁п` come first, then frequent syllables and whole short words. `▁T` seems to appear
# twice in the second row: one is the Latin `T`, the other the Cyrillic `Т` that starts `Том`. They look identical and
# are different characters, which is exactly why we never compare text by how it looks.
#
# The trainer is correct but slow. Every merge recounts all pairs in all words, so the cost is $O(M \cdot L)$, where
# $L$ is the total number of symbols in all distinct words. Both factors grow for the real run: the full training set
# has several times more distinct words than this sample, and we want thousands of merges, not hundreds.

# %% [markdown]
# ### The incremental trainer
#
# A merge only changes the words that contain the merged pair; every other count stays the same. So instead of
# recounting, we keep two tables between steps:
#
# - `pair_freq`: how often each pair occurs in the corpus, $c(a, b)$;
# - `words_with_pair`: for each pair, which words contain it.
#
# After a merge we visit only the affected words: take their old pairs out of the tables, rewrite the word, and put its
# new pairs in.


# %%
def train_bpe(word_freqs, n_merges):
    words = [list(word) for word in word_freqs]  # current spelling of every word
    freqs = list(word_freqs.values())
    pair_freq = collections.Counter()
    words_with_pair = collections.defaultdict(set)  # pair -> indices of the words that contain it

    def add_pairs(i):
        for pair in zip(words[i], words[i][1:]):
            pair_freq[pair] += freqs[i]
            words_with_pair[pair].add(i)

    def remove_pairs(i):
        for pair in zip(words[i], words[i][1:]):
            pair_freq[pair] -= freqs[i]
            if pair_freq[pair] == 0:
                del pair_freq[pair]
            words_with_pair[pair].discard(i)

    for i in range(len(words)):
        add_pairs(i)

    merges = []
    for _ in range(n_merges):
        if not pair_freq:
            break
        pair = most_frequent_pair(pair_freq)
        merges.append(pair)
        for i in list(words_with_pair[pair]):  # only the words that contain the pair change
            remove_pairs(i)
            words[i] = list(merge_word(words[i], pair))
            add_pairs(i)

    final_spelling = {word: tuple(spelling) for word, spelling in zip(word_freqs, words)}
    return merges, final_spelling


# %% [markdown]
# A faster algorithm is only useful if it computes the same thing. Same input, same number of merges:

# %%
start = time.perf_counter()
fast_merges, _ = train_bpe(sample_words, SAMPLE_MERGES)
fast_seconds = time.perf_counter() - start

assert fast_merges == simple_merges
print(
    f"identical merges, {simple_seconds:.1f}s -> {fast_seconds:.2f}s, {simple_seconds / fast_seconds:.0f} times faster"
)

# %% [markdown]
# Now the full training set, both languages together. One shared vocabulary for English and Russian lets the model
# share one embedding table between encoder and decoder later on.

# %%
all_words = word_counts(pd.concat([train.en, train.ru]))

start = time.perf_counter()
merges, segmentation = train_bpe(all_words, N_MERGES)
print(f"{len(merges)} merges over {len(all_words)} distinct words in {time.perf_counter() - start:.0f}s")

alphabet = {symbol for word in all_words for symbol in word}
print("vocabulary:", len(alphabet), "characters +", len(merges), "merges =", len(alphabet) + len(merges), "symbols")

# %% [markdown]
# ### Encoding: replay the merges in order
#
# To tokenize new text we do not count anything. We apply the learned merges in the order they were learned: in each
# word, find the adjacent pair that was merged earliest during training, its **rank**, merge it, and repeat until no
# adjacent pair is in the merge table. Ranks matter because merges build on each other: `▁недо` exists only because
# `▁н`, `▁не` and `до` were merged first.

# %%
ranks = {pair: rank for rank, pair in enumerate(merges)}  # pair -> when it was merged: 0, 1, 2, ...


def encode_word(word, ranks, steps=None):
    symbols = list(word)
    while True:
        # find the adjacent pair with the lowest rank
        best_i, best_rank = None, None
        for i in range(len(symbols) - 1):
            rank = ranks.get((symbols[i], symbols[i + 1]))
            if rank is not None and (best_rank is None or rank < best_rank):
                best_i, best_rank = i, rank

        if best_i is None:  # no pair left that training ever merged
            return symbols

        symbols[best_i : best_i + 2] = [symbols[best_i] + symbols[best_i + 1]]
        if steps is not None:
            steps.append((list(symbols), best_i))  # remember every step, to draw it below


steps = []
encode_word("▁недопонимание", ranks, steps)
token_chips([symbols for symbols, _ in steps], strong=lambda row, col, _: col == steps[row][1], size=11)
plt.show()

# %% [markdown]
# Each row is one merge; the highlighted symbol is the one just created. Replaying merges by rank reproduces exactly
# the segmentation that training arrived at. Rather than believe it, check it for every word in the training data:

# %%
mismatches = []
for word, trained in segmentation.items():
    if encode_word("".join(word), ranks) != list(trained):
        mismatches.append(word)

assert not mismatches, mismatches[:5]
print("encoding reproduces the training segmentation for all", len(segmentation), "words")

# %% [markdown]
# Encoding a sentence is pre-tokenization plus `encode_word`, and decoding is concatenation with `▁` turned back
# into spaces. Nothing is lost on the way; the round trip over the whole validation set confirms it:


# %%
def encode(text):
    tokens = []
    for word in pretokenize(text):
        tokens.extend(encode_word(word, ranks))
    return tokens


def decode(tokens):
    text = "".join(tokens).replace("▁", " ")
    return text[1:]  # drop the space we added in front of the first word


for sentence in pd.concat([valid.en, valid.ru]):
    assert decode(encode(sentence)) == sentence

print(encode(RU_EXAMPLE))

# %% [markdown]
# ### What the merges learned
#
# Nobody told BPE about prefixes or suffixes, yet some appear on their own, because a frequent morpheme is a frequent
# string:

# %%
words = ["▁misunderstanding", "▁недопонимание", "▁исследовательница", "▁вымоталась", "▁researcher", "▁untranslatable"]
token_chips([encode_word(word, ranks) for word in words], labels=[word[1:] for word in words], size=11)
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
# pipeline (normalizer, pre-tokenizer, model, decoder) into one object that is saved as a single file, implemented in
# Rust.
#
# The rule we follow: **the tokenizer owns its normalization**. Then training, evaluation and the deployed service
# cannot normalize differently, because there is no separate step to forget.
#
# First the normalizer, rebuilt from the same choices as `normalize`, step for step:

# %%
hf_normalizer = normalizers.Sequence(
    [
        normalizers.NFKC(),
        normalizers.Replace(Regex(r"\p{Mn}"), ""),  # combining marks
        *[normalizers.Replace(fancy, plain) for fancy, plain in QUOTES.items()],
        normalizers.Replace(Regex(r"\s+-\s+"), " — "),
        normalizers.Replace(Regex(r"\s+"), " "),
        normalizers.Strip(),
    ]
)

print(hf_normalizer.normalize_str("Он сказал:   «Я не знаю» - и ушёл…"))

# %% [markdown]
# "Rebuilt from the same choices" is a claim; check it on every raw sentence we have:

# %%
raw_sentences = pd.concat([train_raw.en, train_raw.ru, test_raw.en, test_raw.ru])

different = [s for s in raw_sentences if hf_normalizer.normalize_str(s) != normalize(s)]
assert not different, different[:5]
print("identical normalization on all", len(raw_sentences), "raw sentences")

# %% [markdown]
# Then the pre-tokenizer: the same regular expression splits words and punctuation, and `Metaspace` turns the spaces
# into `▁`. Regular-expression engines don't always agree on what a "word character" is (Python's and the Rust
# library's disagree on combining marks and subscript digits), which is one more reason the normalizer removed those
# first. Check this one too:

# %%
hf_pre_tokenizer = pre_tokenizers.Sequence(
    [
        pre_tokenizers.Split(Regex(r" ?\w+| ?[^\w\s]+"), behavior="isolated"),
        pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="never", split=False),
    ]
)


def hf_pretokenize(text):
    pieces = hf_pre_tokenizer.pre_tokenize_str(" " + text)  # a list of (piece, position) pairs
    return [piece for piece, _ in pieces]


normalized_sentences = pd.concat([train.en, train.ru, valid.en, valid.ru, test.en, test.ru])
for sentence in normalized_sentences:
    assert hf_pretokenize(sentence) == pretokenize(sentence), sentence
print("identical pre-tokenization on all", len(normalized_sentences), "normalized sentences")

# %% [markdown]
# The tokenizer needs four special tokens, which the model will use from lesson 2 on: `<pad>` fills shorter
# sentences in a batch, `<unk>` stands for a character never seen in training, `<bos>` and `<eos>` mark the beginning
# and end of a sentence. They take ids 0 to 3.

# %%
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def build_tokenizer(vocab_size):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence([hf_normalizer, normalizers.Prepend(" ")])
    tokenizer.pre_tokenizer = hf_pre_tokenizer
    tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always", split=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,  # ignore pairs seen only once
        show_progress=False,
    )
    tokenizer.train_from_iterator(list(train.en) + list(train.ru), trainer)
    return tokenizer


start = time.perf_counter()
tokenizer = build_tokenizer(VOCAB_SIZE)
print(f"vocabulary of {tokenizer.get_vocab_size()} trained in {time.perf_counter() - start:.1f}s")
print(tokenizer.encode(RU_EXAMPLE).tokens)

# %% [markdown]
# Compare the two timings: the Rust trainer builds twice as many symbols in a fraction of the time our Python one
# needed for the same corpus. Both see exactly the same words, since we checked normalization and pre-tokenization, so
# their merges should agree except where the two implementations make different choices: Hugging Face breaks ties
# between equally frequent pairs differently and skips pairs seen only once (`min_frequency=2`). The saved tokenizer
# lists its merges in order; compare the first thousand with ours:

# %%
saved = json.loads(tokenizer.to_str())

hf_merges = []
for merge in saved["model"]["merges"]:
    if isinstance(merge, str):  # older versions of tokenizers store a merge as "a b"
        merge = merge.split(" ")
    hf_merges.append(tuple(merge))

shared = set(hf_merges[:1000]) & set(merges[:1000])
print(len(shared), "of the first 1000 merges are the same in both")

first_difference = next(k for k in range(len(merges)) if hf_merges[k] != merges[k])
print("the lists first differ at merge", first_difference + 1)
print("Hugging Face merged", hf_merges[first_difference], "and we merged", merges[first_difference])

# %% [markdown]
# Almost all of the first thousand merges are the same. Why does the first difference happen? Rebuild the corpus as
# both trainers saw it at that moment: replaying the merges that came before it, with our `encode_word`, gives every
# word's spelling at that step, and `pair_counts` gives the frequencies.

# %%
earlier_ranks = {pair: rank for rank, pair in enumerate(merges[:first_difference])}

corpus_at_that_step = {}
for word, freq in all_words.items():
    spelling = tuple(encode_word("".join(word), earlier_ranks))
    corpus_at_that_step[spelling] = freq

counts_then = pair_counts(corpus_at_that_step)
print("Hugging Face's pair:", hf_merges[first_difference], counts_then[hf_merges[first_difference]])
print("our pair:", merges[first_difference], counts_then[merges[first_difference]])

# %% [markdown]
# A tie: both pairs were equally frequent, and the two implementations break ties differently. Same algorithm, same
# data, a different rule for a coin toss.
#
# The tokenizer is now self-contained. Feed it raw, unnormalized text, and decoding returns the normalized sentence:

# %%
messy = "  Он сказал:   «Я не  знаю» - и ушёл…"
ids = tokenizer.encode(messy).ids

print(ids)
print(repr(tokenizer.decode(ids)))
assert tokenizer.decode(ids) == normalize(messy)
assert [tokenizer.token_to_id(token) for token in SPECIAL_TOKENS] == [0, 1, 2, 3]

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
def tokens_per_sentence(tokenizer, sentences):
    encodings = tokenizer.encode_batch(list(sentences))
    return np.mean([len(encoding.ids) for encoding in encodings])


rows = []
for size in VOCAB_SIZES:
    candidate = build_tokenizer(size)
    rows.append(
        {
            "vocab_size": size,
            "embedding_params": size * D_MODEL,
            "en_tokens": tokens_per_sentence(candidate, valid.en),
            "ru_tokens": tokens_per_sentence(candidate, valid.ru),
        }
    )

sweep = pd.DataFrame(rows).set_index("vocab_size")
sweep.round(2)

# %%
fig, ax = plt.subplots()
ax.plot(sweep.index, sweep.en_tokens, marker="o", color=COLORS["enc"], label="English")
ax.plot(sweep.index, sweep.ru_tokens, marker="o", color=COLORS["dec"], label="Russian")
ax.axvline(VOCAB_SIZE, color=COLORS["muted"], linestyle=(0, (3, 3)), linewidth=1)  # the size we keep

ax.set_xscale("log", base=2)
ax.set_xticks(sweep.index, [f"{size // 1000}k" if size >= 1000 else str(size) for size in sweep.index])
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
for lang in ["en", "ru"]:
    ours = tokens_per_sentence(tokenizer, valid[lang])
    theirs = tokens_per_sentence(gpt2, valid[lang])
    print(f"{lang}: ours {ours:.1f} tokens per sentence, GPT-2 {theirs:.1f}, GPT-2 / ours = {theirs / ours:.2f}")

print()
print("English example:", len(tokenizer.encode(EN_EXAMPLE).ids), "tokens")
print("Russian example:", len(tokenizer.encode(RU_EXAMPLE).ids), "tokens")

# %% [markdown]
# On Russian our tokenizer needs a fraction of GPT-2's tokens. On English it is slightly worse, which is the honest
# price of sharing 16 000 symbols between two languages, while GPT-2 spends 50 000 mostly on English. For a
# translator that reads English and writes Russian, that is the right trade.
#
# ### Save and publish
#
# The tokenizer is one JSON file. Saving and loading it back must give the same tokens:

# %%
TOKENIZER_PATH = WORK_DIR / "tokenizer.json"
tokenizer.save(str(TOKENIZER_PATH))

reloaded = Tokenizer.from_file(str(TOKENIZER_PATH))
assert reloaded.encode(RU_EXAMPLE).ids == tokenizer.encode(RU_EXAMPLE).ids

# %% [markdown]
# The next lessons download the tokenizer and the three splits from the Hugging Face Hub, so that every lesson can
# start on its own. The course's copies are published from this notebook. To publish your own, set the environment
# variable `TFS_HUB_USER` to your Hugging Face username and add a write token as the `HF_TOKEN` secret (Kaggle:
# *Add-ons → Secrets*; Colab: the key icon in the sidebar).
#
# A dataset on the Hub needs a card that says where the data comes from and how it was processed:

# %%
DATASET_CARD = f"""---
license: cc-by-2.0
language: [en, ru]
task_categories: [translation]
---
# Tatoeba English–Russian, cleaned and split without leaks

Built in lesson 1 of [Transformers: from scratch to production](https://github.com/xmarva/transformers-from-scratch).

- train and validation: sentence-transformers/parallel-sentences-tatoeba (en-ru), revision `{TRAIN_REVISION[:7]}`
- test: Tatoeba Translation Challenge eng-rus test set (Tiedemann, 2020)
- NFKC normalization, combining marks removed, straight quotes, one dash
- training pairs sharing a sentence fingerprint with the test set removed
- validation split by connected components of the translation graph, so no sentence occurs in both

Sentences from [Tatoeba](https://tatoeba.org), CC BY 2.0 FR.

| split | pairs |
|---|---|
| train | {len(train)} |
| validation | {len(valid)} |
| test | {len(test)} |
"""

# %%
HUB_USER = os.environ.get("TFS_HUB_USER")
hf_token = env.secret("HF_TOKEN")

if HUB_USER and hf_token and not SMOKE:
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    data_repo = f"{HUB_USER}/tfs-tatoeba-en-ru"
    tokenizer_repo = f"{HUB_USER}/tfs-bpe-en-ru-16k"

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
    print("published", data_repo, "and", tokenizer_repo)
else:
    print(
        "not publishing; lesson 2 downloads the course's copies from xmarva/tfs-tatoeba-en-ru and xmarva/tfs-bpe-en-ru-16k"
    )

# %% [markdown]
# ## References
#
# - Gage, P. (1994). A new algorithm for data compression. *The C Users Journal*, 12(2).
# - Sennrich, R., Haddow, B., Birch, A. (2016). [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909). ACL.
# - Kudo, T., Richardson, J. (2018). [SentencePiece: A simple and language independent subword tokenizer and detokenizer](https://arxiv.org/abs/1808.06226). EMNLP.
# - Petrov, A., La Malfa, E., Torr, P., Bibi, A. (2023). [Language Model Tokenizers Introduce Unfairness Between Languages](https://arxiv.org/abs/2305.15425). NeurIPS.
# - Tiedemann, J. (2020). [The Tatoeba Translation Challenge](https://arxiv.org/abs/2010.06354). WMT.
# - Heaps, H. S. (1978). *Information Retrieval: Computational and Theoretical Aspects*. Academic Press.
# - Tarjan, R. E. (1975). [Efficiency of a good but not linear set union algorithm](https://doi.org/10.1145/321879.321884). *Journal of the ACM*, 22(2).
# - [Unicode Standard Annex #15: Unicode Normalization Forms](https://unicode.org/reports/tr15/).
# - [Hugging Face Tokenizers documentation](https://huggingface.co/docs/tokenizers).
