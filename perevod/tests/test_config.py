import pytest

from perevod.config import GenerationConfig, ModelConfig

LLAMA_STYLE = {"norm": "rms", "ffn": "swiglu", "pos": "rope"}


def test_a_typo_in_a_key_is_an_error():
    with pytest.raises(ValueError, match="n_head"):
        ModelConfig.from_dict({"setup": LLAMA_STYLE, "vocab_size": 100, "n_head": 4})


def test_another_design_is_an_error():
    with pytest.raises(ValueError, match="setup"):
        ModelConfig(setup={**LLAMA_STYLE, "norm": "layer"}, vocab_size=100)


def test_the_heads_must_split_the_width():
    with pytest.raises(ValueError, match="heads"):
        ModelConfig(setup=LLAMA_STYLE, vocab_size=100, d_model=250, n_heads=4)


def test_a_beam_needs_at_least_one_hypothesis():
    with pytest.raises(ValueError, match="beam_size"):
        GenerationConfig(beam_size=0)


def test_a_config_survives_a_round_trip():
    config = ModelConfig(setup=LLAMA_STYLE, vocab_size=100)
    assert ModelConfig.from_dict(config.to_dict()) == config
