# Style guide

The acceptance criteria for every lesson, figure and cover. A lesson is ready when it passes all of them.

## Voice

The course has an author, and it should sound like one: clear, a little sharp, sometimes ironic or sarcastic,
witty when the material allows it. A reader should be able to tell a paragraph of this course from a paragraph of
a generated tutorial.

- Talk to one reader, as "you". Short, direct sentences; one idea per paragraph, one to three sentences long.
- Have opinions and say them. If a common practice is wasteful, say so and show why; if a famous result is
  oversold, say that too, with the evidence. Irony goes to methods, hype and our own model's failures, never to the
  reader.
- Don't over-polish. A text written by a person has rough edges: an aside, a sentence that starts with "And", a
  joke that only half lands. Smoothing every sentence into the same even tone is what makes text read as
  generated.
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
- Be exact rather than impressive. Sharp is not loud: no hype ("revolutionary", "game-changer"), no rhetorical
  drama.
- Define a term the first time it appears, in bold. State assumptions out loud ("we lowercase everything, which
  loses ...").
- Every number in the text comes from an output the reader can see. Never type a number that the code computes;
  describe it ("several times more", "less than a token per sentence") and let the output give the value.

**Phrasing that reads as generated**, in titles and in text alike. None of it goes into a lesson:

- A noun with a tacked-on participle or adverb as a pseudo-title: "improvements, measured", "the Transformer,
  revisited", "attention, explained".
- Two clipped clauses joined by "then": "deploy, then watch it", "train, then trust".
- Vague "from X to Y" and "X: a Y and a Z" headings that name no technique: "from notebook to package", "serving:
  an API and a demo".
- An aphorism closing every paragraph; a pair of short contrasting sentences ("It isn't magic. It's math.");
  rhetorical questions answered in the next word; a clickbait question as a title ("Is BLEU lying?").
- Stock phrases: "let's dive in", "delve", "crucial", "it's worth noting", "in this section we will", "the magic
  of", "a testament to", "game-changer", "in today's fast-paced world"; em dashes in every sentence.

## Titles

A lesson's title is what a search engine shows and what decides whether anyone clicks. It must be:

- **clear**: a reader knows from the title alone what the lesson teaches;
- **findable**: it contains the words people search for (the technique, the library, the task: "BLEU", "RoPE",
  "FastAPI", "machine translation");
- **intriguing** through content, not tricks: a concrete detail the reader didn't expect ("What Llama changed in
  the Transformer"). No clickbait questions ("Is BLEU lying?") or teasers: they are how generated titles sound;
- **specific**: it names techniques, not categories. "Serving a Transformer with FastAPI and Gradio" says more than
  "Serving: an API and a demo".

About 50–70 characters, with the main keyword early. The same rules hold for section headings: plain and
descriptive, no slogans.

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
- One idea per cell, at most ~25 lines; a longer function gets a cell of its own with a sentence before it. Explain
  before the cell, interpret after it.
- When a lesson writes an algorithm that a library also provides (attention, LayerNorm, Adam), check the two
  against each other with an `assert`.
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

- Draw a lot. Every important concept gets its own picture: data flow, architecture, an algorithm step, a
  metric's mechanics, a system diagram, a number format, a failure case. A section without a picture needs a
  reason. Pictures explain; they are never decoration. Lesson 1 has nine static figures and a dozen run-time ones,
  and later lessons should have at least as many.
- **Concept diagrams** are code in `lessons/NN/figures.py`, rendered by `tools/render.py` in light and dark.
  `Figure.problems()` must be empty (nothing outside the canvas, no overlapping labels, no text crossing a box),
  and both themes are looked at before the PR.
- Every key concept of a lesson gets a picture, not only the architecture: the loss, the optimizer, the learning
  rate, a number format, a decoding step. A formula and its picture go together.
- A picture must show what the text claims, checked on its numbers. A toy example is tuned honestly: if the simple
  method wins somewhere (gradient descent on a mildly uneven valley), the lesson says so and shows where it loses.
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

## Post components

Long posts (`articles/*/post.md`, exported to the site with `tools/export_article.py`) are read twice: a quick
pass through the summary, headings, charts and marked lines, then a slow one through everything else. The site
has blocks for the quick pass, styled in its `_sass/_components.scss`, with live examples and markup on its
`/components/` page. They are plain HTML in the markdown; `markdown="1"` renders the markdown inside.

- **Summary** (`aside.tldr`, label "In short"): first in a post that runs past a few screens. Three or four
  sentences a reader could stop after, and up to three `.stat` numbers, each with its unit and what it measures.
- **Headings** state the finding or answer the question: "Calibration: a week out, prices are off by 1.8
  points", not "Calibration". `###` only splits a section longer than about five screens.
- **Method** (`details.method`): sampling, full derivations, assumptions. Closed by default; the summary line says
  what is inside. The main text must read whole without it.
- **Callouts**: `callout-key` for the point of a section, `callout-warn` for an easy mistake, a plain `callout` for
  an aside with its own label ("For traders", "For engineers"). One per section at most.
- **Takeaway** (`p.takeaway`): one sentence after a chart or section, a finding with its number, not a slogan.
- **Flow** (`p.flow`): a chain of three to five short steps; the last, the consequence, is drawn in red.
- **Pull quote** (`blockquote.pull` with `cite`): one per post, for a sentence worth reading twice.
- **Bold and italics** are the only marks inside a sentence; highlighters, coloured or underlined words, inline
  bars and labels were tried and look garish. In posts, bold is a whole sentence, the one in a section a skimming
  reader must not miss, rarely more than one per section, never a fragment or a lone number. Italics introduce a new term
  and carry a stressed word; lessons keep bold definitions. Names of layers, functions, files and APIs go in
  code: `RMSNorm`, `prices-history`.
- **Pointer** (`a.arrow-link`): a link the reader is invited to follow to another section or post.

The rule for all of them is scarcity: a page where everything is highlighted has nothing highlighted. Look at the
post in light and dark at 1280 and 390 px before the PR.

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
