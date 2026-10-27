import json

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

from perevod import Translator
from perevod.config import GenerationConfig, ModelConfig
from perevod.decoding import pad
from perevod.model import Seq2Seq

# a model small enough to train in a second: twenty words, four sentence pairs learned by
# heart, so that different sentences get different translations
SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>"]
WORDS = SPECIAL + ["i", "think", "therefore", "am", ".", ",", "tom", "reads", "a", "book",
                   "?", "я", "думаю", "том", "читает", "книгу"]
PAIRS = {"i think therefore i am .": "я думаю .", "tom reads a book": "том читает книгу",
         "i am ?": "я ?", "tom": "том"}


def tiny_tokenizer():
    vocab = {}
    for token_id, word in enumerate(WORDS):
        vocab[word] = token_id
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.add_special_tokens(SPECIAL)  # so that decoding leaves them out
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return tokenizer


def learn_pairs(model, tokenizer, steps=200):
    bos, eos = tokenizer.token_to_id("<bos>"), tokenizer.token_to_id("<eos>")
    sources, targets = [], []
    for english, russian in PAIRS.items():
        sources.append(tokenizer.encode(english).ids)
        targets.append([bos] + tokenizer.encode(russian).ids + [eos])
    src, tgt = pad(sources, 0, "cpu"), pad(targets, 0, "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(steps):
        logits = model(src, tgt[:, :-1])
        loss = F.cross_entropy(logits.transpose(1, 2), tgt[:, 1:], ignore_index=0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


@pytest.fixture(scope="session")
def tiny_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny-model")
    tokenizer = tiny_tokenizer()
    tokenizer.save(str(path / "tokenizer.json"))
    config = ModelConfig(setup={"norm": "rms", "ffn": "swiglu", "pos": "rope"},
                         vocab_size=len(WORDS), d_model=16, n_heads=2, d_ff=32, n_layers=2,
                         dropout=0.1, max_len=32)
    torch.manual_seed(0)
    model = Seq2Seq(config)
    learn_pairs(model, tokenizer)
    save_file(model.state_dict(), path / "model.safetensors")
    (path / "config.json").write_text(json.dumps(config.to_dict()))
    generation = GenerationConfig(beam_size=3, max_tokens=10)
    (path / "generation_config.json").write_text(json.dumps(generation.to_dict()))
    return path


@pytest.fixture
def pairs():
    return PAIRS


@pytest.fixture
def translator(tiny_dir):
    return Translator.from_pretrained(str(tiny_dir), device="cpu")
