---
title: perevod
emoji: 🔤
colorFrom: red
colorTo: gray
sdk: static
pinned: false
license: mit
short_description: English to Russian with a Transformer, in your browser
models:
  - xmarva/perevod-en-ru
---

# perevod, in your browser

English to Russian with the small Transformer built from scratch in the course
[Transformers: from scratch to production](https://github.com/xmarva/transformers-from-scratch), running on the
visitor's device with [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/): no server, nothing sent anywhere.

The model is [xmarva/perevod-en-ru](https://huggingface.co/xmarva/perevod-en-ru) at commit `4abc0d6`, exported by
`export_onnx.py` to two ONNX graphs: the encoder, and one step of the decoder that returns the log-probabilities of
the next token. The tokenizer (`tokenizer.js`) and the beam search (`translate.js`) are rewritten in JavaScript from
the [perevod](https://github.com/xmarva/transformers-from-scratch/tree/main/perevod) package and checked against it:
`check_tokenizer.py` compares the tokenizers id for id on every sentence of the test set, and the translations match
the package's word for word. The source is the
[space folder](https://github.com/xmarva/transformers-from-scratch/tree/main/space) of the course repository, and
lesson 9 of the course explains it.

It was trained on short everyday sentences; the model card says where it fails.
