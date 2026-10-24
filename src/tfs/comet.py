"""COMET (Unbabel/wmt22-comet-da) as lesson 4 writes it, so that later lessons can score translations.

Lesson 4 builds the forward pass from the checkpoint: XLM-R large, a sparsemax mix of its 25 layers, the mean over
tokens, six combinations of the three sentence vectors and a small feed-forward regressor. It checks the result
against scores of the official `unbabel-comet` package, which this module doesn't need.
"""

from __future__ import annotations

import torch
from torch import nn

REPO = "Unbabel/wmt22-comet-da"
XLMR = "FacebookAI/xlm-roberta-large"


def sparsemax(z):
    """Martins and Astudillo (2016): like softmax, but weights below a threshold become exactly zero."""
    z_sorted, _ = z.sort(descending=True)
    k = torch.arange(1, len(z) + 1, dtype=z.dtype)
    cumulative = z_sorted.cumsum(dim=0)
    survivors = (1 + k * z_sorted > cumulative).sum()
    tau = (cumulative[survivors - 1] - 1) / survivors
    return (z - tau).clamp(min=0)


class Comet:
    """`Comet(device).score(sources, translations, references)` -> one score per translation.

    Every distinct text is embedded once and kept, so scoring several systems on one test set costs little more
    than scoring one.
    """

    def __init__(self, device="cuda"):
        from huggingface_hub import hf_hub_download
        from transformers import AutoConfig, AutoTokenizer, XLMRobertaModel

        self.device = device
        path = hf_hub_download(REPO, "checkpoints/model.ckpt")
        weights = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
        self.tokenizer = AutoTokenizer.from_pretrained(XLMR)
        encoder = XLMRobertaModel(AutoConfig.from_pretrained(XLMR), add_pooling_layer=False)
        encoder_weights = {}
        for name, tensor in weights.items():
            if name.startswith("encoder.model."):
                encoder_weights[name.removeprefix("encoder.model.")] = tensor
        encoder.load_state_dict(encoder_weights, strict=False)  # newer transformers compute position_ids
        self.encoder = encoder.to(device).eval()

        scores = [weights[f"layerwise_attention.scalar_parameters.{k}"] for k in range(25)]
        self.layer_weights = sparsemax(torch.cat(scores)).tolist()
        self.gamma = weights["layerwise_attention.gamma"].item()

        self.estimator = nn.Sequential(
            nn.Linear(6144, 3072), nn.Tanh(), nn.Linear(3072, 1024), nn.Tanh(), nn.Linear(1024, 1)
        )
        for ours, theirs in [(0, 0), (2, 3), (4, 6)]:  # the checkpoint counts the dropout layers too
            self.estimator[ours].weight.data = weights[f"estimator.ff.{theirs}.weight"]
            self.estimator[ours].bias.data = weights[f"estimator.ff.{theirs}.bias"]
        self.estimator = self.estimator.to(device).eval()
        self.embeddings = {}

    @torch.no_grad()
    def embed(self, texts):
        batch = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=510)
        batch = batch.to(self.device)
        layers = self.encoder(**batch, output_hidden_states=True).hidden_states  # 25 × (B, T, 1024)
        mixed = 0
        for weight, layer in zip(self.layer_weights, layers, strict=True):
            mixed = mixed + weight * layer
        mixed = self.gamma * mixed
        mask = batch["attention_mask"].unsqueeze(-1).float()
        return (mixed * mask).sum(dim=1) / mask.sum(dim=1)

    def embed_all(self, texts, batch_size=64):
        todo = [text for text in dict.fromkeys(texts) if text not in self.embeddings]
        todo.sort(key=len)  # similar lengths in one batch: little padding
        for start in range(0, len(todo), batch_size):
            chunk = todo[start : start + batch_size]
            for text, vector in zip(chunk, self.embed(chunk).cpu(), strict=True):
                self.embeddings[text] = vector

    @torch.no_grad()
    def score(self, sources, translations, references, batch_size=512):
        sources, translations, references = list(sources), list(translations), list(references)
        self.embed_all(sources + translations + references)
        scores = []
        for start in range(0, len(translations), batch_size):
            end = start + batch_size
            s = torch.stack([self.embeddings[t] for t in sources[start:end]]).to(self.device)
            h = torch.stack([self.embeddings[t] for t in translations[start:end]]).to(self.device)
            r = torch.stack([self.embeddings[t] for t in references[start:end]]).to(self.device)
            features = torch.cat([h, r, h * r, (h - r).abs(), h * s, (h - s).abs()], dim=1)
            scores += self.estimator(features).squeeze(1).tolist()
        return scores
