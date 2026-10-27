import json

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

from perevod import Translator
from perevod.config import GenerationConfig, ModelConfig
from perevod.model import Seq2Seq

# a model small enough to build in a millisecond: twenty words, random weights. It
# translates nonsense, which is fine: these tests check what must hold for any weights
WORDS = ["<pad>", "<unk>", "<bos>", "<eos>", "i", "think", "therefore", "am", ".", ",",
         "tom", "reads", "a", "book", "я", "думаю", "том", "читает", "книгу", "?"]


@pytest.fixture(scope="session")
def tiny_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny-model")
    vocab = {}
    for token_id, word in enumerate(WORDS):
        vocab[word] = token_id
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(path / "tokenizer.json"))

    config = ModelConfig(setup={"norm": "rms", "ffn": "swiglu", "pos": "rope"},
                         vocab_size=len(WORDS), d_model=16, n_heads=2, d_ff=32, n_layers=2,
                         dropout=0.1, max_len=32)
    torch.manual_seed(0)
    save_file(Seq2Seq(config).state_dict(), path / "model.safetensors")
    (path / "config.json").write_text(json.dumps(config.to_dict()))
    generation = GenerationConfig(beam_size=3, max_tokens=10)
    (path / "generation_config.json").write_text(json.dumps(generation.to_dict()))
    return path


@pytest.fixture
def translator(tiny_dir):
    return Translator.from_pretrained(str(tiny_dir), device="cpu")
