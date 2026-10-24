import torch

from tfs.model import Transformer
from tfs.modern import Seq2Seq, apply_rope, rope_angles

PAD = 0
SETUPS = [
    {"norm": "post", "ffn": "relu", "pos": "sinusoidal"},
    {"norm": "pre", "ffn": "relu", "pos": "sinusoidal"},
    {"norm": "rms", "ffn": "swiglu", "pos": "rope"},
]


def tiny(setup):
    torch.manual_seed(0)
    model = Seq2Seq(setup, vocab_size=50, d_model=16, n_heads=4, d_ff=32, n_layers=2, dropout=0.0, pad_id=PAD)
    return model.eval()


def test_the_2017_setting_is_lesson_2s_model():
    torch.manual_seed(0)
    lesson2 = Transformer(vocab_size=50, d_model=16, n_heads=4, d_ff=32, n_layers=2, dropout=0.0, pad_id=PAD).eval()
    rebuilt = tiny(SETUPS[0])
    rebuilt.load_state_dict(lesson2.state_dict())  # the same parameter names, nothing missing
    src = torch.tensor([[5, 6, 7, PAD], [8, 9, 10, 11]])
    tgt = torch.tensor([[1, 12, 13], [1, 14, PAD]])
    assert torch.allclose(lesson2(src, tgt), rebuilt(src, tgt), atol=1e-5)


def test_the_cache_gives_the_scores_of_a_full_pass():
    src = torch.tensor([[5, 6, 7, 8]])
    tgt = torch.tensor([[1, 12, 13, 14, 15]])
    for setup in SETUPS:
        model = tiny(setup)
        with torch.no_grad():
            memory = model.encode(src)
            full = model.decode(tgt, memory, src)
            caches = model.new_caches(memory)
            for position in range(tgt.shape[1]):
                step = model.decode_step(tgt[:, position : position + 1], memory, src, caches, position)
                assert torch.allclose(step, full[:, position], atol=1e-5), setup


def test_rope_scores_depend_on_distance_only():
    q, k = torch.randn(1, 1, 10, 8), torch.randn(1, 1, 10, 8)
    angles = rope_angles(40, 8)
    here = apply_rope(q, angles[:10]) @ apply_rope(k, angles[:10]).transpose(-1, -2)
    later = apply_rope(q, angles[20:30]) @ apply_rope(k, angles[20:30]).transpose(-1, -2)
    assert torch.allclose(here, later, atol=1e-5)
