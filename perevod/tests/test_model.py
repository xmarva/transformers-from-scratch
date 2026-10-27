import torch

from perevod.config import ModelConfig
from perevod.model import Seq2Seq


def tiny_model():
    torch.manual_seed(0)
    config = ModelConfig(setup={"norm": "rms", "ffn": "swiglu", "pos": "rope"},
                         vocab_size=20, d_model=16, n_heads=2, d_ff=32, n_layers=2,
                         max_len=32)
    return Seq2Seq(config).eval()


def test_a_position_does_not_see_the_tokens_after_it():
    model = tiny_model()
    src = torch.tensor([[5, 6, 7, 8]])
    tgt = torch.tensor([[2, 9, 10, 11, 12]])
    changed = tgt.clone()
    changed[0, 3:] = torch.tensor([13, 14])
    before, after = model(src, tgt), model(src, changed)
    assert torch.allclose(before[0, :3], after[0, :3], atol=1e-6)
    assert not torch.allclose(before[0, 3:], after[0, 3:])


def test_padding_the_source_changes_nothing():
    model = tiny_model()
    src = torch.tensor([[5, 6, 7]])
    padded = torch.tensor([[5, 6, 7, 0, 0]])
    tgt = torch.tensor([[2, 9, 10]])
    assert torch.allclose(model(src, tgt), model(padded, tgt), atol=1e-5)
