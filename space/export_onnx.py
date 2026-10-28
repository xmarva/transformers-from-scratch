# Export perevod's model to the two ONNX graphs of the in-browser demo:
#   encoder.onnx: source ids -> memory
#   decoder.onnx: hypotheses, memory, source ids -> log-probabilities of the next token
# and the tokenizer, which the page loads as it is.
# Usage: python export_onnx.py --out model
import argparse
import logging
from pathlib import Path

import torch
from torch.export import Dim

from perevod import Translator, decoding


class Encoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, src):
        return self.model.encode(src)


class DecoderStep(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tgt, memory, src):
        return self.model.decode(tgt, memory, src)[:, -1].log_softmax(-1)


def main():
    parser = argparse.ArgumentParser(description="Export perevod's model to ONNX")
    parser.add_argument("--out", default="model")
    out = Path(parser.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    logging.getLogger("torch.onnx").setLevel(logging.ERROR)  # the exporter is chatty
    translator = Translator.from_pretrained(device="cpu")
    model = translator.model
    _, bos, _ = decoding.special_ids(translator.tokenizer)
    ids = decoding.source_ids(translator.tokenizer, "Tom is afraid of the dark.", 256)
    src = torch.tensor([ids, ids])  # two rows, three target tokens: any sizes above one do
    tgt = torch.tensor([[bos, bos, bos], [bos, bos, bos]])
    batch = Dim("batch")
    source = Dim("source", max=256)  # the model's positions
    target = Dim("target", max=256)
    # export leaves a module in the mode it found it in, and a new module starts in training
    # mode: without eval() the wrappers would switch the model's dropout back on afterwards
    encoder = Encoder(model).eval()
    step = DecoderStep(model).eval()
    with torch.no_grad():
        memory = model.encode(src)
        torch.onnx.export(encoder, (src,), out / "encoder.onnx", input_names=["src"],
                          output_names=["memory"], external_data=False, verbose=False,
                          dynamic_shapes={"src": {0: batch, 1: source}})
        torch.onnx.export(step, (tgt, memory, src), out / "decoder.onnx",
                          input_names=["tgt", "memory", "src"], output_names=["logp"],
                          external_data=False, verbose=False,
                          dynamic_shapes={"tgt": {0: batch, 1: target},
                                          "memory": {0: batch, 1: source},
                                          "src": {0: batch, 1: source}})
    translator.tokenizer.save(str(out / "tokenizer.json"))  # the page needs it too
    for path in sorted(out.glob("*.onnx")):
        print(f"{path.name}: {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
