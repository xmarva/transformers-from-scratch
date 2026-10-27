import json
import warnings
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from tokenizers import Tokenizer

from . import decoding
from .config import GenerationConfig, ModelConfig
from .model import Seq2Seq

MODEL = "xmarva/perevod-en-ru"
REVISION = "4abc0d63e862ca99d2bd99cd1007e4a9fe47c358"  # tested with this commit
FILES = ["config.json", "generation_config.json", "model.safetensors", "tokenizer.json"]


class Translator:
    """Translates English into Russian with the course's Transformer.

    >>> translator = Translator.from_pretrained()
    >>> translator("I think, therefore I am.")
    """

    def __init__(self, model, tokenizer, generation):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.generation = generation

    @classmethod
    def from_pretrained(cls, model=MODEL, revision=None, device=None):
        """Load a model from a Hugging Face repository or from a local directory.

        The default model comes at the revision this version of perevod was tested with.
        """
        if model == MODEL and revision is None:
            revision = REVISION
        path = Path(model)
        if not path.is_dir():
            path = Path(snapshot_download(model, revision=revision, allow_patterns=FILES))
        config = ModelConfig.from_dict(json.loads((path / "config.json").read_text()))
        settings = json.loads((path / "generation_config.json").read_text())
        network = Seq2Seq(config)
        network.load_state_dict(load_file(path / "model.safetensors"))
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        return cls(network.to(device), tokenizer, GenerationConfig.from_dict(settings))

    def translate(self, texts, beam_size=None):
        """Translate a string or a list of strings; beam_size=1 is greedy decoding."""
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        cleaned = []
        for text in texts:
            cleaned.append(self.clean(text))
        # an empty input has an empty translation: the model never sees it
        to_translate = [text for text in cleaned if text]
        done = iter(self.decode(to_translate, beam_size or self.generation.beam_size))
        translations = []
        for text in cleaned:
            translations.append(next(done) if text else "")
        return translations[0] if single else translations

    __call__ = translate

    def clean(self, text):
        text = " ".join(text.split())  # line breaks and runs of spaces become one space
        limit = self.model.config.max_len
        if len(self.tokenizer.encode(text).ids) > limit:
            warnings.warn(f"input longer than {limit} tokens is cut to fit", stacklevel=3)
        return text

    def decode(self, sentences, beam_size):
        settings = self.generation
        if beam_size == 1:
            return decoding.greedy(self.model, self.tokenizer, sentences,
                                   settings.max_tokens)
        return decoding.beam_search(self.model, self.tokenizer, sentences, beam_size,
                                    settings.alpha, settings.max_tokens)
