# Style guide

The acceptance criteria for every lesson, figure and cover. A lesson is ready when it passes all of them.

## Voice

- Talk to one reader, as "you". Short, direct sentences; one idea per paragraph.
- Start from what readers already care about, which is how today's large models behave (GPT-4o, Llama,
  DeepSeek), and lead to the technical part step by step. Lesson 1 opens with what GPT-4o actually receives
  when you ask it about strawberries; only then come byte pairs and merge tables.
- Open each major section with something concrete too: a sentence the model gets wrong, a number that
  surprises, a question the reader can't answer yet. Then explain.
- Explain what each step does *and why we do it*, including the choices we didn't make ("the length-ratio filter
  would remove these pairs, and they are good translations").
- Add the facts that make a topic memorable: where an algorithm came from, a famous failure, a quirk of the data
  (Tom in 23% of Tatoeba's sentences, `SolidGoldMagikarp`). Every such fact is checked against a source and
  linked where it first appears.
- Be exact rather than impressive. No hype ("revolutionary", "game-changer"), no rhetorical drama, and none of
  the stock phrases of generated text: "let's dive in", "delve", "crucial", "it's worth noting", "in this section
  we will", "the magic of", an aphorism closing every paragraph, em dashes in every sentence. A dry joke now and
  then is welcome.
- Define a term the first time it appears, in bold. State assumptions out loud ("we lowercase everything, which
  loses ...").
- Every number in the text comes from an output the reader can see. Never type a number that the code computes;
  describe it ("several times more", "less than a token per sentence") and let the output give the value.

## Examples

- No textbook clichés ("the cat sat on the mat", "king − man + woman", "low/lower/newest/widest"). An example
  has to earn its place by showing the phenomenon better than any other: Russian morphology for subwords,
  gender agreement for attention, a synonym for the weakness of BLEU.
- Prefer sentences about language, the mind and science; they keep the course coherent and memorable.
- Check every linguistic claim (case, gender, suffix vs ending) before it goes into a figure or the text.

## Structure of a lesson

The arc that lesson 1 settled on:

1. **Header** (generated): cover, one line with the course, the lesson number and the other platform's badge.
2. **Title and hook**: a concrete scene with a real model, a hero figure of it, and why it matters; a few short
   paragraphs, no more.
3. **What you'll do** and **what you need** (previous lessons' artifacts, CPU/GPU, run time).
4. **Setup** (shared include, silent).
5. **Real models first.** Take apart the real thing in code (a tokenizer, a config file, a few megabytes, never
   gigabytes of weights unless the lesson needs them) and let the reader see the phenomenon the lesson explains.
6. **The plan**: one figure of what we are about to build, then build it.
7. **Build sections**, each: why it matters → idea, with a figure → math → a tiny example → code → a check
   (`assert`) → what the output tells us.
8. **Payoff**: put what we built next to the real models from step 5, on the same measurement. Say honestly
   where ours wins, where it loses and why.
9. **Save and publish** what the next lesson downloads.
10. **References**: papers, docs and the sources of every fact, with links.

No "common mistakes", exercises or summary sections: the lesson ends with its last real step and the references.

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

The reader is a beginner who reads every line. Code that looks generated loses them.

- **Show before you scale.** Run every new function on a tiny example the reader can check by hand (three words, the
  eight pairs from a figure), print the result, and only then run it on the real data.
- **Plain loops and `if`s over clever one-liners.** No `max(..., key=lambda kv: (kv[1], kv[0]))`, tuple-swap tricks,
  `setdefault` tricks or long pandas chains. Name the intermediate results.
- **No type hints, type aliases or docstrings** in lesson code: the text around the cell explains. Short comments
  say *why*, in plain words.
- **Plain prints** (`print("train:", len(train))`), no alignment tricks. An f-string is fine for rounding and
  percentages (`f"{share:.1%}"`).
- **Code lines up to 92 characters**, so a Kaggle or Colab cell shows them without scrolling
  (`tests/test_lessons.py` checks it).
- Invisible characters (combining marks, non-breaking spaces) are written as `\u` escapes, never pasted.
- One idea per cell, at most ~25 lines. Explain before the cell, interpret after it.
- Checks are `assert`s introduced in the text ("rather than believe it, check it").
- When the text needs a number the code computes, make the notebook print it right there and refer to the output.
  Numbers typed into the text go stale as soon as something upstream changes.
- Settings live next to the code that uses them (`VOCAB_SIZE = 16_000` right before the tokenizer is trained), not
  in one cell of constants at the top.
- Reproducible: seeds from the setup cell, pinned dataset revisions, deterministic splits.
- Everything runs on Kaggle and Colab with the free tier: CPU for lesson 1, one T4 for the rest, about an
  hour per lesson. A lesson that trains for long defines `SMOKE = env.smoke()` in its own settings cell; CI sets
  it to shrink data and steps, and it must not change what the code teaches. A lesson that runs in minutes, like
  lesson 1, has no smoke path at all.

## Outputs

A notebook is read as much as it is run, so every output is designed:

- DataFrames are shown with `table(df)` (no index, wrapped text, numbers aligned), never as raw reprs.
- Tokens, and short lists of sentences, are shown as chips with `token_chips` (with `ids=` when the ids
  matter).
- Setup is silent: `env.quiet()` hides progress bars and environment warnings. No stderr in a finished notebook.
- Key rules are blockquotes (`> A cleaning step is a hypothesis about your data.`), at most a few per lesson.
- Plots use `use_course_style()`; two series that mean English and Russian are teal and rose, with a legend and
  direct labels on bars.

## Figures

- Draw a figure where a picture explains better than text: data flow, architecture, an algorithm step,
  a metric's mechanics, a system diagram. Not as decoration. In practice every section has at least one
  picture; lesson 1 has nine static figures and a dozen run-time ones.
- **Concept diagrams** are code in `lessons/NN/figures.py`, rendered by `tools/render.py` in light and dark.
  `Figure.problems()` must be empty (nothing outside the canvas, no overlapping labels, no text crossing a box),
  and both themes are looked at before the PR.
- A static figure that shows facts about real systems or algorithms **computes them at render time**: the hero of
  lesson 1 calls GPT-4o's tokenizer, the union–find figure runs the same algorithm as the notebook on the same
  pairs, the BPE update figure counts the pairs.
- **Anything that shows computed numbers** (merges, attention maps, curves) is drawn by the notebook itself at
  run time, with `use_course_style()` / `token_chips()`. Then the picture can't disagree with the code.
- Color means a concept everywhere: amber data and tokens, teal encoder, rose decoder, violet attention.
- Text in a figure is at least 12 px at the width of a Kaggle cell (about 900 px).
- Static figures go into markdown as `<img src="…/NAME-light.png" width="W" alt="…">` with the figure's own
  width, so a narrow diagram isn't blown up to the full cell.
- Every image has alt text. External images only with a compatible license and a link; prefer redrawing
  them in our style with "adapted from".
- Screenshots (W&B, Gradio, Grafana) are taken at one size, in one theme, with the same frame.

## Covers

- 1200 × 630, dark background, the site's red. Notebook tab, title of two to four words, a hook question,
  a motif showing the lesson's idea. No author name.
- English covers use Latin script only (`check_latin` enforces it): English examples, or transliterated
  Russian when the point is Russian grammar. Russian covers may use Cyrillic.
- Illustrative motifs are replaced by real outputs (our tokenizer, our model) once they exist.

## Before the PR

A lesson is reviewed the way a reader will meet it:

1. Run it in full (`tools/run.py`): no errors, nothing on stderr.
2. Dump the outputs and check every claim in the text against them, one by one.
3. Render the executed notebook to HTML (`jupyter nbconvert --to html`) and look at every screen: nothing raw,
   clipped, cramped or tiny.
4. `ruff`, `pytest` (code line length included) and `build.py --check` pass.
5. Push a private Kaggle draft built from the branch (`build.py --ref <branch>`) and let it run there.
