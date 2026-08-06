# quant-head

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.40%2B-yellow.svg)](https://huggingface.co/docs/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Model-agnostic INT8 quantization for LLM Embedding Tables & LM Heads.**

</div>

---

### Tech Stack
`Python` • `PyTorch` • `Hugging Face Transformers` • `Setuptools`

---

Model-agnostic INT8 quantization for the **embedding table and LM head** of
any Hugging Face causal LM — two layers that can dominate memory footprint
for models with large vocabularies relative to hidden size (e.g. Gemma's
~256K-token vocab on a 2B model, or Qwen2.5's ~152K-token vocab on a 0.5B
model), but which general-purpose quantization tools typically skip.

```python
from transformers import AutoModelForCausalLM
from quant_head import quantize_embeddings_and_head

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
model, report = quantize_embeddings_and_head(model)
```

## Why this exists

Quantization libraries like bitsandbytes and GPTQ compress the transformer
block, but generally leave the embedding table and LM head in full
precision — these can be 50%+ of a model's parameter memory once vocab
size is large (Llama 3: 128K tokens, Qwen2.5: ~152K tokens). No reusable,
model-agnostic library existed to fill that specific gap — see
[huggingface/transformers#31474](https://github.com/huggingface/transformers/issues/31474)
for the original motivation and the perplexity measurements that inspired
this project (8-bit compression measured as nearly lossless; NF4 introduces
a small but real error).

### Motivation (from [#31474](https://github.com/huggingface/transformers/issues/31474), opened by [@galqiwi](https://github.com/galqiwi))

The maintainer who raised this (closed without resolution as of writing —
this project doesn't close it, just builds a standalone workaround for the
underlying idea) pointed out that as the vocabulary sizes of
open models keep growing, the embedding and head layers are becoming a
disproportionately large and increasingly overlooked source of memory
overhead — a bottleneck that general quantization tooling hadn't caught up
to yet. They'd already measured that 8-bit compression of these layers is
close to lossless, and flagged that a working implementation could unlock
real use cases: running large models (e.g. 70B-class) on free-tier
single-GPU setups without offloading, and fitting small models like Gemma
2B into around 1GB of RAM for on-device or embedded use — but that actually
shipping it required solving two integration problems (no support for
mixing multiple quantization backends in `transformers`, and no
`nn.Embedding`-compatible quantization path in `bitsandbytes`) that hadn't
been resolved. This project doesn't solve those two integration problems —
it deliberately sidesteps them by implementing the compression as a
standalone, model-agnostic module rather than a `transformers`/bitsandbytes
core-library patch, to validate and ship the underlying idea without
waiting on that upstream work.

## What makes this not a one-off script

Instead of hardcoding attribute paths (`model.model.embed_tokens`, which
differs across architectures — `transformer.wte` in GPT-2,
`gpt_neox.embed_in` in Pythia, etc.), this uses the standard
`PreTrainedModel` interface (`get_input_embeddings` / `set_input_embeddings`
/ `get_output_embeddings` / `set_output_embeddings`), so it works on any
model that implements it.

It also correctly handles **tied embeddings** — many models (Qwen2.5, GPT-2
by default, Gemma) share the same weight tensor between the input embedding
and the output head. Naively quantizing both independently re-rounds the
same weights twice with different error, silently breaking the tie and
wasting memory. This library detects tying (both via
`config.tie_word_embeddings` and by checking weight identity directly, in
case the config flag is wrong) and reuses one shared quantized buffer.

Validated across 4 architecturally distinct model families — see
results below.

## Install

```bash
pip install git+https://github.com/ShrutiShravani/quant_head

```

Run [`examples/quickstart.py`](examples/quickstart.py) to understand how to use it.

## Quickstart

To test with a GPU, open [`QUANT_HEAD.ipynb`](examples/QUANT_HEAD.ipynb) in Colab (Runtime → Change runtime type → GPU) and run all cells.

or  Reproduce locally : `python benchmarks/benchmark.py --all`

Benchmark protocol: WikiText-2-raw perplexity (test split, sliding-window
evaluation, stride=512, 50 documents) — the same dataset/protocol used to
report numbers in GPTQ, AWQ, and LLM.int8(). Latency is mean forward-pass
time over 20 runs after 3 warmup runs. Run on a single T4 GPU (Colab).

Results 

| Model | Variant | Storage (GB) | Peak Infer (GB) | PPL | ΔPPL | Latency (ms) | Tied |
|---|---|---|---|---|---|---|---|
| gpt2 | fp16_baseline | 0.247 | 0.572 | 37.647 | 0.0 | 16.31 | True |
| gpt2 | int8_embeddings | 0.211 | 0.535 | 37.773 | +0.126 | 12.94 | True |
| gpt2 | int8_embeddings_and_head | 0.211 | 0.535 | 37.773 | +0.126 | 13.44 | True |
| pythia-160m | fp16_baseline | 0.318 | 0.642 | 36.187 | 0.0 | 18.78 | False |
| pythia-160m | int8_embeddings | 0.281 | 0.605 | 36.278 | +0.091 | 11.41 | False |
| pythia-160m | int8_embeddings_and_head | 0.245 | 0.569 | 37.057 | +0.870 | 14.12 | False |
| Qwen2.5-0.5B | fp16_baseline | 0.937 | 1.819 | 17.728 | 0.0 | 43.68 | True |
| Qwen2.5-0.5B | int8_embeddings | 0.810 | 1.692 | 17.721 | −0.007 | 47.18 | True |
| Qwen2.5-0.5B | int8_embeddings_and_head | 0.810 | 1.692 | 17.721 | −0.007 | 47.10 | True |
| TinyLlama-1.1B | fp16_baseline | 2.060 | 2.268 | 10.378 | 0.0 | 38.12 | False |
| TinyLlama-1.1B | int8_embeddings | 1.998 | 2.205 | 10.375 | −0.003 | 36.77 | False |
| TinyLlama-1.1B | int8_embeddings_and_head | 1.936 | 2.236 | 10.377 | −0.001 | 39.74 | False |

**What to notice:**
- **Storage and peak inference memory both decrease, consistently, across every model and every variant.** That's the core claim, and it now holds without exception.
- **Tied models (gpt2, Qwen2.5) show identical numbers for `int8_embeddings` and `int8_embeddings_and_head`.** This is expected, not a bug — see [Tied-weight handling](#tied-weight-handling-a-real-bug-found-and-fixed) below for why.
- **PPL deltas are all small** (≤0.13 except one outlier), consistent with the "8-bit is nearly lossless" finding reported in the motivating GitHub issue. This validates that the quantization math itself is correct — it isn't just compiling, it's numerically sound.
- **pythia-160m's `int8_embeddings_and_head` shows a larger ΔPPL (+0.87)** than every other cell. Still small in absolute terms, but worth a follow-up run to see if it's a stable effect (e.g. smaller hidden size being less forgiving of int8 rounding) or noise from the 50-document eval slice — flagged here rather than smoothed over.
- **Latency numbers are noisy at this model scale** (kernel-launch overhead and warmup variance dominate actual compute time for sub-1B models on a single forward pass) and shouldn't be read as a precise before/after comparison — included for completeness, not as the main result.

## Tied-weight handling: a real bug found and fixed

Many models (GPT-2, Qwen2.5, Gemma) tie the input embedding and output head
to the *same* underlying weight tensor (`config.tie_word_embeddings=True`).
This creates a genuine correctness hazard for partial quantization that the
motivating GitHub issue flagged as a caveat but didn't solve: **if you
quantize only the embedding and leave the head alone, the head keeps a live
reference to the original full-precision tensor.** Nothing gets freed — you
end up holding both the new int8 copy *and* the original fp16 tensor
simultaneously, so memory goes **up**, not down.

This was caught empirically, not by inspection: an early benchmark run
showed `int8_embeddings` storage *increasing* relative to baseline on gpt2
and Qwen2.5 (both tied) while correctly *decreasing* on pythia and TinyLlama
(both untied) — an exact split by tying status, which pointed straight at
the cause.

The fix: `quantize_embeddings_and_head` detects tying (via
`config.tie_word_embeddings` and independently via direct weight-identity
comparison, in case the config flag is wrong) and treats quantization as
all-or-nothing for tied models — requesting only one side forces both,
avoiding the stale-reference/doubled-memory failure mode entirely. The
returned `QuantizationReport.tied_forced_both_sides` flag tells you when
this coercion happened. Covered by a regression test in `tests/test_modules.py`.

## Honest limitation: the LM head implementation is not compute-optimal

`QuantizedEmbedding` gives a real memory reduction during inference because
it only dequantizes the rows actually indexed per batch. `QuantizedLinear`
(the LM head), in this reference implementation, dequantizes the *entire*
weight matrix on every forward call — so it reduces resident/storage memory
but does not reduce peak compute memory as much as it could, and adds a
small latency overhead relative to fp16. A real speed/peak-memory win for
the head requires a fused kernel (e.g. Triton) that dequantizes row-by-row
inside the matmul itself, never materializing the full fp16 matrix — a
natural follow-up, not implemented here.

## Stage 2 (roadmap, not yet implemented): a fused kernel for the head

The current `QuantizedLinear` is a correctness-first reference
implementation, not a performance one. The planned follow-up is a Triton
kernel that fuses dequantization directly into the matmul:

- Load int8 weight tiles and their per-row scales
- Dequantize each tile in registers/shared memory, immediately before it's
  consumed by the matmul accumulation
- Never materialize the full fp16 weight matrix in global memory at any point

This removes the two costs the current implementation pays on every forward
call: the extra elementwise multiply to dequantize, and the peak-memory
spike from briefly holding both the int8 buffer and a full fp16 copy at
once. Expected result: `QuantizedLinear` moves from "saves storage only" to
"saves storage, peak memory, and latency," matching what `QuantizedEmbedding`
already achieves for the same reason (only ever materializing the small
slice actually needed for the current computation, not the whole table).

Not started yet — tracked here so the gap between "what's built" and "what's
designed" stays explicit rather than implied.

## Limitations

- Row-wise symmetric INT8 quantization only (no NF4/AQLM-style codecs)
- `QuantizedLinear`'s forward pass is not compute-optimal (see above)
- No training support — inference only
- Requires models to implement the standard `get_input_embeddings()` /
  `get_output_embeddings()` interface (true for effectively all
  `transformers` causal LMs)

## Project layout

```
quant_head/
  modules.py   # QuantizedEmbedding, QuantizedLinear
  utils.py     # quantize_embeddings_and_head() -- the main entry point
benchmarks/
  benchmark.py # standardized WikiText-2 perplexity benchmark, 4 models
examples/
  quickstart.py
  QUANT_HEAD.ipynb
tests/
  test_modules.py  # CPU-only, no downloads required, runs in CI
```
