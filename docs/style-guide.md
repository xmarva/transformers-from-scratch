# Style guide

The acceptance criteria for every lesson, figure and cover. A lesson is ready when it passes all of them.

## Voice

- Talk to one reader, as "you". Short, direct sentences; one idea per paragraph.
- Open each lesson and each major section with something concrete: a sentence the model gets wrong, a number
  that surprises, a question the reader can't answer yet. Then explain.
- Be exact rather than impressive. No hype words ("revolutionary", "staggering", "game-changer", "delve",
  "it's worth noting"), no rhetorical drama. A dry joke now and then is welcome.
- Define a term the first time it appears. State assumptions out loud ("we lowercase everything, which loses
  ...").
- Every number in the text comes from an output the reader can see. Never type a number that the code computes.

## Examples

- No textbook clichés ("the cat sat on the mat", "king − man + woman", "low/lower/newest/widest"). An example
  has to earn its place by showing the phenomenon better than any other: Russian morphology for subwords,
  gender agreement for attention, a synonym for the weakness of BLEU.
- Prefer sentences about language, the mind and science; they keep the course coherent and memorable.
- Check every linguistic claim (case, gender, suffix vs ending) before it goes into a figure or the text.

## Structure of a lesson

1. Header (generated): cover, platform badges.
2. Title and a hook paragraph.
3. **What you'll build** and **what you need** (previous lesson's artifacts, CPU/GPU, expected run time).
4. Setup (shared include).
5. Sections, each: why it matters → idea (with a figure where a picture explains better than words) → math →
   code → check → what the output tells us.
6. **Where this goes wrong**: the mistakes people actually make, and how to notice them.
7. **Exercises**: two or three, one of them open-ended. Solutions in a collapsed `<details>` block.
8. **Summary**: what you can do now, which artifacts the lesson published, what comes next.
9. **References**: papers and docs, with links.

## Math

Explain every formula in the same order:

1. **Intuition**: one or two sentences, before any symbol.
2. **Formula**, displayed with `$$...$$`, using the course [notation](notation.md).
3. **Each term**: what it is, its shape or range, why it is there.
4. **A worked example** small enough to check by hand, computed in a code cell.
5. **Code** whose variable names match the symbols (`d_k`, `Q`, `logits`).

Derive a result when the derivation is the explanation (why attention divides by $\sqrt{d_k}$); cite it when it
isn't. Inline math uses `$...$`. Write shapes as $(B, T, d_{\text{model}})$.

## Code

- One idea per cell, at most ~25 lines. Explain before the cell, interpret after it.
- Functions first; a class only when state really belongs together.
- Type hints on functions the lesson defines. Names a reader can say out loud.
- Check understanding with `assert`s and tiny examples, not with walls of prints. Print only what the text
  discusses.
- No defensive boilerplate (`try/except` that re-raises, retries, logging setup) unless it is the topic.
- Reproducible: seeds from the setup cell, pinned dataset revisions, deterministic splits.
- Everything runs on Kaggle and Colab with the free tier: CPU for lesson 1, one T4 for the rest, about an
  hour per lesson. `SMOKE` (set by CI) shrinks data and steps; it must not change what the code teaches.

## Figures

- Draw a figure where a picture explains better than text: data flow, architecture, an algorithm step,
  a metric's mechanics, a system diagram. Not as decoration.
- **Concept diagrams** are code in `lessons/NN/figures.py`, rendered by `tools/render.py` in light and dark.
  `Figure.problems()` must be empty, and both themes are looked at before the PR.
- **Anything that shows computed numbers** (merges, attention maps, curves) is drawn by the notebook itself at
  run time, with `use_course_style()` / `token_chips()`. Then the picture can't disagree with the code.
- Color means a concept everywhere: amber data and tokens, teal encoder, rose decoder, violet attention.
- Text in a figure is at least 12 px at the width of a Kaggle cell (about 900 px).
- Every image has alt text. External images only with a compatible license and a link; prefer redrawing
  them in our style with "adapted from".
- Screenshots (W&B, Gradio, Grafana) are taken at one size, in one theme, with the same frame.

## Covers

- 1200 × 630, dark background, the site's red. Notebook tab, title of two to four words, a hook question,
  a motif showing the lesson's idea. No author name.
- English covers use Latin script only (`check_latin` enforces it): English examples, or transliterated
  Russian when the point is Russian grammar. Russian covers may use Cyrillic.
- Illustrative motifs are replaced by real outputs (our tokenizer, our model) once they exist.
