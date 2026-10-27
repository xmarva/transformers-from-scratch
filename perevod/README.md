# perevod

English-to-Russian translation with the small Transformer built in the course
[Transformers: from scratch to production][course]. The model and its card are on the
Hugging Face Hub: [xmarva/perevod-en-ru](https://huggingface.co/xmarva/perevod-en-ru).

```bash
REPO=https://github.com/xmarva/transformers-from-scratch
pip install "perevod @ git+$REPO#subdirectory=perevod"
perevod "Tom is afraid of the dark."
perevod --beam-size 1 < sentences.txt > translations.txt
```

```python
from perevod import Translator

translator = Translator.from_pretrained()
translator(["The doctor will see you now.", "Tom is reading."])
```

As a service, with an HTTP API at `/translate`, its documentation at `/docs`, and with
`--demo` a page at `/`:

```bash
pip install "perevod[demo] @ git+$REPO#subdirectory=perevod"
perevod-server --port 8000 --demo
```

Or from the image of a release, with the model inside, which needs no network to run:

```bash
docker run --rm -p 8000:8000 ghcr.io/xmarva/perevod:0.3.0
curl -X POST localhost:8000/translate -H "Content-Type: application/json" -d '{"text": "Hello!"}'
```

The model was trained on short everyday sentences from Tatoeba. Read its model card before
trusting it with anything else.

Tests: `pip install -e ".[test]"`, then `pytest -m "not slow"` (seconds, offline) or
`pytest` (downloads the model).

[course]: https://github.com/xmarva/transformers-from-scratch
