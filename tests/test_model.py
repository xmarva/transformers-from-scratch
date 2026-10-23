import torch

from tfs.model import Transformer

PAD = 0


def tiny_model():
    torch.manual_seed(0)
    model = Transformer(vocab_size=50, d_model=16, n_heads=4, d_ff=32, n_layers=2, dropout=0.0, pad_id=PAD)
    return model.eval()


def test_scores_have_one_row_per_target_position():
    model = tiny_model()
    src = torch.tensor([[5, 6, 7, PAD], [8, 9, 10, 11]])
    tgt = torch.tensor([[1, 12, 13], [1, 14, PAD]])
    assert model(src, tgt).shape == (2, 3, 50)


def test_decoder_cannot_see_the_future():
    model = tiny_model()
    src = torch.tensor([[5, 6, 7]])
    a = model(src, torch.tensor([[1, 12, 13, 14]]))
    b = model(src, torch.tensor([[1, 12, 13, 40]]))
    assert torch.allclose(a[:, :3], b[:, :3], atol=1e-6)
    assert not torch.allclose(a[:, 3], b[:, 3])


def test_padding_changes_nothing_for_real_tokens():
    model = tiny_model()
    tgt = torch.tensor([[1, 12, 13]])
    alone = model(torch.tensor([[5, 6, 7]]), tgt)
    padded = model(torch.tensor([[5, 6, 7, PAD, PAD]]), tgt)
    assert torch.allclose(alone, padded, atol=1e-5)


def test_output_layer_is_the_embedding():
    model = tiny_model()
    names = [name for name, _ in model.named_parameters()]
    assert names.count("embedding.weight") == 1 and not any("output" in name for name in names)
