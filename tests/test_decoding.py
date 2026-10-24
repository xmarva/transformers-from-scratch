import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from tfs import decoding
from tfs.model import Transformer

WORDS = ["<pad>", "<bos>", "<eos>", "<unk>"] + [f"w{k}" for k in range(36)]
SENTENCES = ["w1 w2 w3", "w4", "w5 w6 w7 w8 w9 w10", "w11 w12", "w13 w14 w15 w16", "w3 w3 w3"]


def tiny_setup():
    tokenizer = Tokenizer(WordLevel({word: i for i, word in enumerate(WORDS)}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    torch.manual_seed(0)
    model = Transformer(vocab_size=len(WORDS), d_model=16, n_heads=4, d_ff=32, n_layers=2, dropout=0.0, pad_id=0)
    return model.eval(), tokenizer


def test_greedy_batches_like_one_sentence_at_a_time():
    model, tokenizer = tiny_setup()
    batched = decoding.greedy(model, tokenizer, SENTENCES, max_tokens=8)
    alone = [decoding.greedy(model, tokenizer, [s], max_tokens=8)[0] for s in SENTENCES]
    assert batched == alone


def test_beam_search_batches_like_one_sentence_at_a_time():
    model, tokenizer = tiny_setup()
    batched = decoding.beam_search(model, tokenizer, SENTENCES, max_tokens=8)
    alone = [decoding.beam_search(model, tokenizer, [s], max_tokens=8)[0] for s in SENTENCES]
    assert batched == alone


def test_a_beam_of_one_is_greedy():
    model, tokenizer = tiny_setup()
    assert decoding.beam_search(model, tokenizer, SENTENCES, beam_size=1, max_tokens=8) == decoding.greedy(
        model, tokenizer, SENTENCES, max_tokens=8
    )
