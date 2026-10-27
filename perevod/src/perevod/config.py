from dataclasses import asdict, dataclass, fields

DESIGN = {"norm": "rms", "ffn": "swiglu", "pos": "rope"}  # the one design perevod runs


def from_dict(cls, data):
    known = []
    for field in fields(cls):
        known.append(field.name)
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {unknown}, known are {known}")
    return cls(**data)


@dataclass(frozen=True)
class ModelConfig:
    setup: dict
    vocab_size: int
    d_model: int = 256
    n_heads: int = 4
    d_ff: int = 1024
    n_layers: int = 3
    dropout: float = 0.1
    pad_id: int = 0
    max_len: int = 256

    def __post_init__(self):
        problems = []
        if self.setup != DESIGN:
            problems.append(f"setup is {self.setup}, but perevod runs {DESIGN} only")
        if self.d_model % self.n_heads != 0:
            problems.append(f"{self.n_heads} heads don't split d_model={self.d_model}")
        elif self.d_model // self.n_heads % 2 != 0:
            problems.append("RoPE rotates pairs of numbers: d_model / n_heads must be even")
        if not 0 <= self.dropout < 1:
            problems.append(f"dropout={self.dropout} is not a probability below 1")
        if problems:
            raise ValueError("ModelConfig: " + "; ".join(problems))

    @classmethod
    def from_dict(cls, data):
        return from_dict(cls, data)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class GenerationConfig:
    beam_size: int = 4
    alpha: float = 0.6
    max_tokens: int = 128

    def __post_init__(self):
        if self.beam_size < 1:
            raise ValueError(f"GenerationConfig: beam_size={self.beam_size} is below 1")

    @classmethod
    def from_dict(cls, data):
        return from_dict(cls, data)

    def to_dict(self):
        return asdict(self)
