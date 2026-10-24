"""Static figures and covers of lesson 4.

Facts are computed at render time: the references come from the course's test set on the Hugging Face Hub, the
translation from opus-mt-en-ru itself, the scores from sacreBLEU and from the same n-gram counting the lesson
writes. The bootstrap figure is a toy whose numbers are drawn here with a fixed seed.
"""

import random
from collections import Counter

from tools.figures import cover as cv
from tools.figures.lib import Figure

HERO_EN = "Are you coming alone?"


def test_references(sentence):
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("xmarva/tfs-tatoeba-en-ru", "data/test.parquet", repo_type="dataset")
    test = pd.read_parquet(path)
    return list(test[test.en == sentence].ru)


def opus_translation(sentence):
    """What opus-mt-en-ru writes for a sentence, with its own default decoding (4 beams)."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    name = "Helsinki-NLP/opus-mt-en-ru"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name).eval()
    with torch.no_grad():
        output = model.generate(**tokenizer([sentence], return_tensors="pt"))
    return tokenizer.decode(output[0], skip_special_tokens=True)


# ---------------------------------------------------------------- hero: one translation, fifteen references

def hero() -> Figure:
    import sacrebleu

    references = test_references(HERO_EN)
    translation = opus_translation(HERO_EN)
    scored = [(sacrebleu.sentence_bleu(translation, [r]).score, r) for r in references]
    best = max(score for score, _ in scored)
    columns = {"ты": [], "вы": []}
    for score, reference in sorted(scored, reverse=True):
        columns["ты" if reference.startswith("Ты") else "вы"].append((score, reference))
    rows = max(len(column) for column in columns.values())
    fig = Figure(
        760,
        172 + rows * 34 + 46,
        alt=f"opus-mt-en-ru translates '{HERO_EN}' as {translation} The test set has {len(references)} Russian "
        f"translations of the sentence, each a test pair of its own. Scored against each of them, the same correct "
        f"translation gets sentence BLEU between {min(s for s, _ in scored):.0f} and {best:.0f}",
    )
    fig.text(14, 28, "One correct translation, scored against fifteen correct references", kind="title")
    fig.text(14, 50, f"«{HERO_EN}» in the course's test set: {len(references)} pairs, one per translation",
             kind="note")
    fig.text(14, 90, "opus-mt-en-ru writes", kind="label")
    fig.chip(170, 70, translation, role="data", strong=True)
    titles = {"ты": "references with ты (informal you)", "вы": "references with вы (formal or plural you)"}
    for k, (key, column) in enumerate(columns.items()):
        x0 = 14 + k * 380
        fig.text(x0, 138, titles[key], kind="subtitle")
        for row, (score, reference) in enumerate(column):
            y = 152 + row * 34
            fig.chip(x0, y, reference, role="dec", h=26, size=12)
            bar_x = x0 + 168
            fig.rect(bar_x, y + 9, 120, 8, role="data", tint=True, r=2)
            fig.rect(bar_x, y + 9, max(2, 1.2 * score), 8, role="data", r=2)
            fig.text(bar_x + 128, y + 18, f"BLEU {score:.0f}", kind="small")
    note_y = 152 + rows * 34 + 24
    fig.text(14, note_y, "Every reference fits some speaker and listener: ты or вы, a man or a woman, on foot or by "
             "car.", kind="note")
    fig.text(14, note_y + 20, f"The best match gives BLEU {best:.0f} of 100.", kind="note")
    return fig


# ---------------------------------------------------------------- BLEU on one sentence: a synonym

BLEU_EN = "Tell Tom that I may be late."
BLEU_EXPECTED = "Скажи Тому, что я, возможно, опоздаю."  # the lesson's code uses this translation too
BLEU_REF = "Скажи Тому, что я, может быть, опоздаю."


def our_translation(sentence):
    """Our lesson 3 model with its beam search, on the CPU."""
    import json

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from tfs import decoding
    from tfs.model import Transformer
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(hf_hub_download("xmarva/tfs-bpe-en-ru-16k", "tokenizer.json"))
    with open(hf_hub_download("xmarva/tfs-transformer-en-ru", "config.json")) as f:
        model = Transformer(**json.load(f))
    model.load_state_dict(load_file(hf_hub_download("xmarva/tfs-transformer-en-ru", "model.safetensors")))
    return decoding.beam_search(model.eval(), tokenizer, [sentence])[0]


def bleu_ngrams() -> Figure:
    import math

    from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a

    hypothesis = our_translation(BLEU_EN)
    assert hypothesis == BLEU_EXPECTED, hypothesis
    tokenize = Tokenizer13a()
    hyp, ref = tokenize(hypothesis).split(), tokenize(BLEU_REF).split()
    rows = []
    for n in range(1, 5):
        hyp_grams = Counter(tuple(hyp[i : i + n]) for i in range(len(hyp) - n + 1))
        ref_grams = Counter(tuple(ref[i : i + n]) for i in range(len(ref) - n + 1))
        matched = sum(min(count, ref_grams[gram]) for gram, count in hyp_grams.items())
        rows.append((n, matched, sum(hyp_grams.values())))
    brevity = 1.0 if len(hyp) > len(ref) else math.exp(1 - len(ref) / len(hyp))
    score = 100 * brevity * math.exp(sum(math.log(m / t) for _, m, t in rows) / 4)
    only_hyp = [w for w in hyp if w not in ref]
    only_ref = [w for w in ref if w not in hyp]
    fig = Figure(
        760,
        400,
        alt=f"BLEU on one sentence. Our model translates '{BLEU_EN}' as {hypothesis} The reference says "
        f"{BLEU_REF} The single different word breaks n-grams of every length: "
        + ", ".join(f"{m} of {t} {n}-grams match" for n, m, t in rows)
        + f". With the brevity penalty {brevity:.2f}, BLEU is {score:.0f}",
    )
    fig.text(14, 28, "BLEU on one sentence: a synonym breaks n-grams of every length", kind="title")
    fig.text(14, 50, f"English: {BLEU_EN}", kind="note")
    for label, words, differ, y in [("our model", hyp, only_hyp, 70), ("reference", ref, only_ref, 112)]:
        fig.text(14, y + 20, label, kind="label")
        x = 110
        for word in words:
            w = fig.chip(x, y, word, role="dec" if word in differ else "data", strong=word in differ, h=28,
                         size=12)
            x += w + 5
        assert x <= 750, f"{label} reaches {x:.0f}"
    fig.text(14, 182, "n", kind="label")
    fig.text(60, 182, "n-grams of the translation found in the reference", kind="label")
    fig.text(560, 182, "precision", kind="label")
    for k, (n, matched, total) in enumerate(rows):
        y = 196 + k * 34
        fig.text(14, y + 17, str(n), kind="mono")
        fig.rect(60, y + 6, 480, 14, role="data", tint=True, r=3)
        fig.rect(60, y + 6, 480 * matched / total, 14, role="data", r=3)
        fig.text(560, y + 17, f"{matched}/{total} = {matched / total:.2f}", kind="count")
    fig.text(14, 352, f"brevity penalty: {len(hyp)} words against {len(ref)}, BP = {brevity:.2f}", kind="body")
    fig.text(14, 376, f"BLEU = BP × geometric mean of the four precisions = {score:.0f}, for a correct translation",
             kind="body")
    return fig


# ---------------------------------------------------------------- the plan

def plan() -> Figure:
    fig = Figure(
        760,
        330,
        alt="The plan of the lesson: the 36 385 test pairs are translated by three systems (our model with greedy "
        "decoding, our model with beam search, opus-mt-en-ru); the translations are scored by BLEU, chrF and "
        "COMET against the references; the metrics are themselves checked against professional translators' "
        "judgments and against translations broken on purpose; differences between systems get confidence "
        "intervals from the bootstrap",
    )
    fig.text(14, 28, "Translate, score, then check the scores", kind="title")
    fig.box(14, 118, 128, 56, role="data", label="test set")
    fig.text(78, 196, "36 385 pairs", kind="label", anchor="middle")
    translators = ["ours, greedy", "ours, beam of 4", "opus-mt-en-ru"]
    metrics = [("BLEU", "word n-grams"), ("chrF", "character n-grams"), ("COMET", "a neural network")]
    for k, name in enumerate(translators):
        y = 76 + k * 52
        fig.box(186, y, 150, 40, role="plain", label=name)
        fig.arrow((144, 146), (182, y + 20))
    fig.box(374, 108, 94, 76, role="data", label="")
    fig.text(421, 142, "translations", kind="body", anchor="middle")
    fig.text(421, 160, "+ references", kind="label", anchor="middle")
    for k in range(3):
        fig.arrow((338, 96 + k * 52), (370, 126 + k * 20))
    for k, (name, how) in enumerate(metrics):
        y = 76 + k * 52
        fig.box(508, y, 92, 40, role="enc" if name == "COMET" else "neutral", label=name)
        fig.text(608, y + 25, how, kind="label")
        fig.arrow((470, 146), (504, y + 20))
    fig.line((14, 240), (746, 240), role="muted", width=1)
    fig.text(14, 266, "Then check the measuring stick:", kind="subtitle")
    fig.text(14, 290, "professional translators' scores of 15 WMT systems · translations broken on purpose", kind="note")
    fig.text(14, 312, "the bootstrap: which differences would survive another test set · what opus-mt saw in training",
             kind="note")
    return fig


# ---------------------------------------------------------------- chrF on one word

CHRF_HYP, CHRF_REF = "сказал", "сказала"


def char_ngrams(text, n):
    counts = Counter()
    for i in range(len(text) - n + 1):
        counts[text[i : i + n]] += 1
    return counts


def chrf_word() -> Figure:
    beta = 2
    rows = []
    for n in range(1, 7):
        hyp, ref = char_ngrams(CHRF_HYP, n), char_ngrams(CHRF_REF, n)
        matched = sum(min(count, ref[gram]) for gram, count in hyp.items())
        missed = []
        for gram, count in ref.items():
            missed += [gram] * max(0, count - hyp[gram])
        rows.append((n, list(hyp.elements()), missed, matched, sum(hyp.values()), sum(ref.values())))
    precision = sum(m / h for _, _, _, m, h, _ in rows) / len(rows)
    recall = sum(m / r for _, _, _, m, _, r in rows) / len(rows)
    chrf = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
    fig = Figure(
        760,
        360,
        alt=f"chrF on the word {CHRF_HYP} against the reference {CHRF_REF}: for n from 1 to 6, every character "
        f"n-gram of the translation is found in the reference, so precision is 1; the reference has one n-gram "
        f"more at every n, the ones with its last letter, so recall falls as n grows. Averaged, precision "
        f"{precision:.2f} and recall {recall:.2f} give chrF {100 * chrf:.0f}; a word-level metric sees one wrong "
        "word",
    )
    fig.text(14, 28, f"chrF gives partial credit: «{CHRF_HYP}» against «{CHRF_REF}»", kind="title")
    fig.text(14, 50, "the translation's character n-grams (filled: found in the reference), and the ones it missed",
             kind="note")
    fig.text(14, 82, "n", kind="label")
    fig.text(52, 82, "the translation's n-grams", kind="label")
    fig.text(452, 82, "missed", kind="label")
    fig.text(560, 82, "precision", kind="label")
    fig.text(660, 82, "recall", kind="label")
    for k, (n, grams, missed, matched, n_hyp, n_ref) in enumerate(rows):
        y = 94 + k * 34
        fig.text(14, y + 18, str(n), kind="mono")
        x = 52
        for gram in grams:
            w = fig.chip(x, y, gram, role="data", strong=True, h=26, size=12)
            x += w + 4
        assert x <= 444, f"row {n} reaches {x:.0f}"
        x = 452
        for gram in missed:
            w = fig.chip(x, y, gram, role="dec", h=26, size=12)
            x += w + 4
        fig.text(560, y + 18, f"{matched}/{n_hyp}", kind="count")
        fig.text(660, y + 18, f"{matched}/{n_ref}", kind="count")
    fig.text(14, 324, f"average over n: precision {precision:.2f}, recall {recall:.2f}; with recall weighted "
             f"β = {beta} times, chrF = {100 * chrf:.0f}", kind="body")
    fig.text(14, 346, "BLEU compares whole words: «сказал» is simply not «сказала».", kind="note")
    return fig


# ---------------------------------------------------------------- COMET

def comet() -> Figure:
    fig = Figure(
        760,
        380,
        alt="COMET: the source sentence, the translation and the reference are each read by the same XLM-R large "
        "encoder; a weighted mix of its 25 layers is averaged over the tokens into one vector of 1024 numbers per "
        "sentence, s, h and r; six combinations of them, h, r, the products h times r and h times s and the "
        "distances between them, make 6144 features, and a small feed-forward network turns them into one score. "
        "It was trained to predict human scores of translations",
    )
    fig.text(14, 28, "COMET: three sentences, one encoder, a small regressor", kind="title")
    inputs = [("source (English)", "s"), ("translation", "h"), ("reference", "r")]
    for k, (name, symbol) in enumerate(inputs):
        y = 76 + k * 64
        fig.box(14, y, 170, 40, role="data", label=name)
        fig.arrow((186, y + 20), (232, y + 20))
        fig.box(422, y, 70, 40, role="neutral", label=symbol)
        fig.arrow((388, y + 20), (418, y + 20))
    fig.box(236, 70, 150, 180, role="enc", label="")
    fig.text(311, 150, "XLM-R large", kind="block", role="enc", anchor="middle")
    fig.text(311, 172, "24 layers", kind="label", anchor="middle")
    fig.text(236, 270, "one encoder, the same", kind="label")
    fig.text(236, 286, "weights for all three", kind="label")
    fig.text(408, 270, "1024 numbers", kind="label")
    fig.text(408, 286, "per sentence", kind="label")
    features = ["h", "r", "h ⊙ r", "|h − r|", "h ⊙ s", "|h − s|"]
    fig.box(540, 70, 206, 196, role="plain", label="")
    fig.text(556, 92, "features, 6 × 1024", kind="label")
    for k, feature in enumerate(features):
        fig.chip(556 + (k % 2) * 92, 104 + (k // 2) * 50, feature, role="att", h=30, size=13)
    for k in range(3):
        fig.arrow((494, 96 + k * 64), (536, 140 + k * 28))
    fig.box(236, 318, 340, 40, role="neutral", label="feed-forward: 6144 → 3072 → 1024 → 1")
    fig.arrow((643, 268), (643, 338), (580, 338))
    fig.box(14, 318, 170, 40, role="enc", label="score", strong=True)
    fig.arrow((234, 338), (188, 338))
    fig.text(14, 304, "trained on human scores", kind="label")
    return fig


# ---------------------------------------------------------------- the paired bootstrap, on a toy test set

def bootstrap() -> Figure:
    rng = random.Random(4)
    n = 8
    a = [rng.choice([1, 1, 1, 0]) for _ in range(n)]  # 1: system A's translation of the sentence is good
    b = [rng.choice([1, 1, 0, 0]) for _ in range(n)]
    resamples = []
    for _ in range(3):
        rows = sorted(rng.randrange(n) for _ in range(n))
        diff = (sum(a[i] for i in rows) - sum(b[i] for i in rows)) / n
        resamples.append((rows, diff))
    fig = Figure(
        760,
        340,
        alt="The paired bootstrap on a toy test set of 8 sentences: draw 8 sentences with replacement, so that some "
        "appear twice and others not at all, score both systems on the same draw, and record the difference; "
        "repeated a thousand times, the differences form a distribution, and the share of draws where A beats B "
        "says how sure we can be",
    )
    fig.text(14, 28, "The paired bootstrap: other test sets, drawn from the one we have", kind="title")
    fig.text(14, 50, "a toy test set of 8 sentences; each translation scores 1 (good) or 0 (bad)", kind="note")
    fig.text(14, 92, "test set", kind="label")
    for i in range(n):
        x = 110 + i * 50
        fig.chip(x, 72, f"s{i + 1}", role="data", h=28, size=12)
        fig.text(x + 15, 118, f"A {a[i]}", kind="small", anchor="middle")
        fig.text(x + 15, 134, f"B {b[i]}", kind="small", anchor="middle")
    whole = (sum(a) - sum(b)) / n
    fig.text(520, 91, f"A − B = {whole:+.2f}".replace("-", "−"), kind="count_hot")
    for k, (rows, diff) in enumerate(resamples):
        y = 164 + k * 44
        fig.text(14, y + 19, f"draw {k + 1}", kind="label")
        for j, i in enumerate(rows):
            fig.chip(110 + j * 50, y, f"s{i + 1}", role="data", strong=rows.count(i) > 1, h=28, size=12)
        fig.text(520, y + 19, f"A − B = {diff:+.2f}".replace("-", "−"), kind="count")
    fig.text(14, 304, "Filled: drawn more than once. Both systems are scored on the same draw, which makes it paired.",
             kind="note")
    fig.text(14, 324, "Repeat a thousand times and count how often A − B > 0.", kind="note")
    return fig


# ---------------------------------------------------------------- covers: the hero's real numbers

LATIN = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
                 ["a", "b", "v", "g", "d", "e", "yo", "zh", "z", "i", "y", "k", "l", "m", "n", "o", "p", "r", "s", "t",
                  "u", "f", "kh", "ts", "ch", "sh", "shch", "", "y", "'", "e", "yu", "ya"]))


def transliterate(text):
    out = []
    for char in text:
        latin = LATIN.get(char.lower(), char)
        out.append(latin.capitalize() if char.isupper() and latin else latin)
    return "".join(out)


def _cover(lang):
    import sacrebleu

    english = lang == "en"
    shown = transliterate if english else (lambda text: text)
    references = test_references(HERO_EN)
    translation = opus_translation(HERO_EN)
    scored = sorted(((sacrebleu.sentence_bleu(translation, [r]).score, r) for r in references), reverse=True)
    rows = scored[:2] + scored[len(scored) // 2 - 1 : len(scored) // 2 + 1] + scored[-2:]
    out = [cv.label(cv.MOTIF_X0, 128, f"«{HERO_EN.upper()}»" if english else "«ARE YOU COMING ALONE?»")]
    chip, w = cv.chip(cv.MOTIF_X0, 146, shown(translation), size=18, kind="hot")
    cv.fits(cv.MOTIF_X0 + w)
    out.append(chip)
    caption = f"AGAINST {len(references)} CORRECT REFERENCES" if english else f"ПРОТИВ {len(references)} ВЕРНЫХ ЭТАЛОНОВ"
    out.append(cv.label(cv.MOTIF_X0, 230, caption))
    widest = max(cv.chip_w(shown(reference), 16) for _, reference in rows)
    for k, (score, reference) in enumerate(rows):
        y = 250 + k * 46
        chip, w = cv.chip(cv.MOTIF_X0, y, shown(reference), size=16, h=36)
        out.append(chip)
        bar_x = cv.MOTIF_X0 + widest + 24
        out.append(f'<rect x="{bar_x}" y="{y + 14}" width="150" height="8" rx="3" fill="{cv.RED_TINT}"/>')
        out.append(f'<rect x="{bar_x}" y="{y + 14}" width="{1.5 * score:.1f}" height="8" rx="3" fill="{cv.RED}"/>')
        out.append(cv.text(bar_x + 162, y + 24, f"BLEU {score:.0f}", 16, cv.MUTED, cv.MONO))
        cv.fits(bar_x + 162 + cv.chip_w("BLEU 35", 16, 0))
        assert cv.MOTIF_X0 + w < bar_x - 8, f"{reference} runs into its bar"
    title = ["Evaluating", "translation"] if english else ["Оценка", "перевода"]
    hook = ["One right answer,", "fifteen references"] if english else ["Один верный ответ,", "пятнадцать эталонов"]
    return cv.frame("04_evaluation.ipynb", title, hook, "".join(out))


FIGURES = {
    "hero": hero,
    "bleu-ngrams": bleu_ngrams,
    "plan": plan,
    "chrf-word": chrf_word,
    "comet": comet,
    "bootstrap": bootstrap,
}
COVERS = {"en": lambda: _cover("en"), "ru": lambda: _cover("ru")}
