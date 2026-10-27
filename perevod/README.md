# perevod

English-to-Russian translation with the small Transformer built in the course
[Transformers: from scratch to production][course]. The model and its card are on the
Hugging Face Hub: [xmarva/perevod-en-ru](https://huggingface.co/xmarva/perevod-en-ru).

```bash
REPO=https://github.com/xmarva/transformers-from-scratch
pip install "perevod @ git+$REPO#subdirectory=perevod"
perevod "I think, therefore I am."
perevod --beam-size 1 < sentences.txt > translations.txt
```

```python
from perevod import Translator

translator = Translator.from_pretrained()
translator(["The doctor will see you now.", "Tom is reading."])
```

The model was trained on short everyday sentences from Tatoeba. Read its model card before
trusting it with anything else.

Tests: `pip install -e ".[test]"`, then `pytest -m "not slow"` (seconds, offline) or
`pytest` (downloads the model).

[course]: https://github.com/xmarva/transformers-from-scratch
