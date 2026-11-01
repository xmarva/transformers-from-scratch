# Transformer architectures: from BERT to diffusion

The second course of this repository. The first one builds a Transformer and ships it; this one meets the family
that grew out of it. Each lesson is one architecture: what it changed and why, its key part written from scratch,
its history, where it is the right tool today and where it isn't, and a practical task it solves well. The lessons go
from the simplest to the most complex, and each one fixes a limit of an earlier one.

Every lesson follows the same plan:

- **What it changed.** The story: the paper, the people, what it beat, the limit of its predecessor it removed.
- **The key part from scratch.** Intuition, a picture, the exact maths, and the component in 30–150 lines of PyTorch.
- **Weights transplant.** The real pretrained weights loaded into our code give the same outputs as the reference
  implementation, to about 1e-4. That is how we know we wrote the real thing and not a toy that looks like it.
- **A small experiment** where it shows the idea's effect best: a short run on a free T4, not a reproduction.
- **A practical task** with a pretrained model and an interesting dataset, a different one in every lesson.
- **Where to use it today**, and where something else is better, with numbers.
- **A passport**: the same card in every lesson (objective, attention, positions, normalisation, activation, context,
  sizes), and a family tree that grows by one branch per lesson.

Lessons run on Kaggle or Colab with a free T4, each on its own; static figures are drawn by `figures.py`, and every
number in the text comes from an output of the notebook. The course shares the tooling of the first one
(`tools/courses.py`): lesson folders are named `aNN_name`, so notebooks, figures and covers never clash.

## Part 1. Encoders: models that understand text

| # | Lesson | From scratch, and the check | Practical task |
|---|---|---|---|
| a01 | **BERT**: masked language modelling and fine-tuning. Google Search, 2019; cloze tests; why "bidirectional" needed a mask | embeddings (token, segment, position), the encoder with post-norm and GELU, the MLM head; bert-base weights; the 80/10/10 mask; what BERT knows and its biases, probed through its own predictions; what its heads look at | a detector of personal data in financial documents (gretelai/synthetic_pii_finance_multilingual, Apache-2.0), with 512-token windows over long documents, against a regex baseline |
| a02 | **From RoBERTa to ModernBERT**: what seven years did to BERT. RoBERTa's recipe, DeBERTa, ELECTRA, ModernBERT | DeBERTa's disentangled attention, ELECTRA's replaced-token detection, ModernBERT's alternating local and global attention; their weights; ELECTRA against MLM on the same small budget | one hard classification task, three generations of encoders, accuracy against speed |
| a03 | **Embeddings and search**: Sentence-BERT, cross-encoders, ColBERT | BM25, mean pooling, InfoNCE with in-batch negatives, MaxSim; MiniLM, bge, a cross-encoder and ColBERTv2 in our code; a bi-encoder trained from scratch and one adapted to the corpus | a literature search over 58,000 ML and NLP papers (LitSearch): recall@k, time and index size of every method, fusion and reranking |

## Part 2. Decoders and the encoder–decoder: models that generate

| # | Lesson | From scratch, and the check | Practical task |
|---|---|---|---|
| a04 | **GPT**: from GPT-1 to GPT-3, and a model that learns chess | the GPT-2 block (pre-norm, learned positions, tied embeddings), generation with a KV cache; gpt2 weights; the logit lens | a small GPT trained on Lichess games: legal-move rate, and a linear probe that finds the board inside it |
| a05 | **T5 and BART**: the encoder–decoder | relative position buckets, span corruption, cross-attention; t5-small weights; why decoder-only won | text-to-SQL, checked by running the queries |
| a06 | **Inside modern LLMs**: what Llama, Mistral, Gemma and Qwen do differently | grouped-query attention, sliding windows, local and global layers, QK-norm, logit soft-capping; Qwen3-0.6B weights; a table of a dozen models from their configs | structured extraction with a small model, fine-tuned with LoRA |

## Part 3. Scale and efficiency

| # | Lesson | From scratch, and the check | Practical task |
|---|---|---|---|
| a07 | **Mixture of experts**: Switch Transformer, Mixtral, DeepSeek-V3 | top-k routing, load balancing, shared and fine-grained experts; a small real MoE's weights | which experts fire for code, maths, Russian and English, measured on a real MoE |
| a08 | **The KV cache budget**: MQA, GQA and DeepSeek's latent attention | MLA with decoupled RoPE; MiniCPM3 weights; memory per token by architecture | how many concurrent chats fit on a T4, by architecture |
| a09 | **Long context**: from Longformer to YaRN | position interpolation, NTK scaling, YaRN, attention sinks, FlashAttention as tiled online softmax checked against plain attention | a needle in a long haystack, and questions over a whole book |
| a10 | **Beyond attention**: Mamba and hybrids | the selective scan; mamba-130m weights; associative recall; Jamba and Gated DeltaNet | Mamba on long DNA sequences (Genomic Benchmarks) |

## Part 4. Beyond text

| # | Lesson | From scratch, and the check | Practical task |
|---|---|---|---|
| a11 | **Vision Transformer**: ViT, DeiT, DINOv2 | patch embeddings, the class token, positions; vit-base weights; ViT against a CNN when data is scarce | land use from satellite images (EuroSAT), and what the attention looks at |
| a12 | **CLIP and SigLIP** | the contrastive and the sigmoid losses, both towers; clip-vit-b/32 weights | zero-shot classification and image search; typographic attacks |
| a13 | **Vision-language models**: from LLaVA to Qwen-VL | the projector, visual tokens and their cost, dynamic resolution; a small LLaVA-OneVision's weights | a small VLM reading charts and receipts |
| a14 | **Whisper**: speech recognition as an encoder–decoder | the log-mel spectrogram, task tokens, timestamps; whisper-tiny weights | word error rate in English and Russian (FLEURS), and hallucinations on silence |
| a15 | **Diffusion Transformers**: DiT and Stable Diffusion 3 | denoising diffusion and flow matching, adaLN-Zero; MM-DiT | a small DiT trained on butterflies |

Datasets and models are checked for licence and availability when each lesson is written; the table is the plan,
not a promise.
