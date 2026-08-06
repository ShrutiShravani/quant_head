"""
benchmarks/benchmark.py

Standardized benchmark for the quant_head library.

Dataset & protocol: WikiText-2-raw (test split), evaluated with the
standard sliding-window perplexity method (stride = max_length = 512).
This is the same dataset and protocol used to report numbers in GPTQ,
AWQ, LLM.int8(), and the HF "Perplexity of fixed-length models" guide,
so results here are directly comparable to numbers reported elsewhere.

Models: chosen to cover different architectures, attribute-naming
conventions, and weight-tying behavior, so a passing run is real evidence
of "works on any HF causal LM" rather than "works on one model I tuned it for":

    - gpt2                                  untied embeddings, classic architecture
    - EleutherAI/pythia-160m                 GPT-NeoX style, different attribute names
    - Qwen/Qwen2.5-0.5B                      modern architecture, large (~152k) vocab
    - TinyLlama/TinyLlama-1.1B-Chat-v1.0      Llama family, TIED embeddings

Usage:
    python benchmarks/benchmark.py --model Qwen/Qwen2.5-0.5B
    python benchmarks/benchmark.py --all
    python benchmarks/benchmark.py --all --device cpu   # slower, no VRAM numbers
"""
import argparse
import gc
import json
import time
from dataclasses import dataclass, asdict

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from quant_head.utils import quantize_embeddings_and_head

DEFAULT_MODELS = [
    "gpt2",
    "EleutherAI/pythia-160m",
    "Qwen/Qwen2.5-0.5B",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]

STRIDE = 512
MAX_LENGTH = 512
NUM_DOCS = 50  # WikiText-2 test documents concatenated for evaluation

VARIANTS = ["fp16_baseline", "int8_embeddings", "int8_embeddings_and_head"]


@dataclass
class BenchResult:
    model_id: str
    variant: str
    storage_vram_gb: float
    peak_inference_vram_gb: float
    perplexity: float
    perplexity_delta: float
    forward_latency_ms: float
    tied_weights: bool


def load_wikitext2_text():
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return ds["text"]


def compute_perplexity(model, tokenizer, text_lines, device):
    model.eval()
    encodings = tokenizer("\n\n".join(text_lines[:NUM_DOCS]), return_tensors="pt")
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, STRIDE):
        end_loc = min(begin_loc + MAX_LENGTH, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
        nlls.append(outputs.loss)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    return torch.exp(torch.stack(nlls).mean()).item()


def measure_forward_latency(model, tokenizer, device, n_runs=20):
    inputs = tokenizer("The quick brown fox jumps over the lazy dog. " * 8, return_tensors="pt").to(device)
    with torch.no_grad():
        for _ in range(3):
            model(**inputs)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_runs):
            model(**inputs)
        if device == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - start) / n_runs * 1000


def cleanup(device):
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def run_one_variant(model_id, variant, text_lines, device, baseline_ppl=None):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cleanup(device)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
    tied = bool(getattr(model.config, "tie_word_embeddings", False))

    if variant == "int8_embeddings":
        model, _ = quantize_embeddings_and_head(model, quantize_embeddings=True, quantize_head=False)
    elif variant == "int8_embeddings_and_head":
        model, _ = quantize_embeddings_and_head(model, quantize_embeddings=True, quantize_head=True)

    # Steady-state storage size: how much VRAM the model occupies at rest,
    # right after load+quantize, before any forward pass. This is the
    # number that reflects "does quantization shrink the checkpoint/resident
    # weights" -- separate from peak memory used *during* inference.
    if device == "cuda":
        torch.cuda.synchronize()
    storage_vram = torch.cuda.memory_allocated() / (1024 ** 3) if device == "cuda" else 0.0

    # Reset the peak counter *after* load+quantize so peak_inference_vram
    # isolates memory used during actual inference (forward pass
    # activations, logits tensors) -- not the transient overhead of the
    # quantization conversion itself. This separation matters most for
    # small models, where activation/logit memory can be comparable to or
    # larger than the embedding table being quantized.
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    ppl = compute_perplexity(model, tokenizer, text_lines, device)
    latency = measure_forward_latency(model, tokenizer, device)
    peak_inference_vram = torch.cuda.max_memory_allocated() / (1024 ** 3) if device == "cuda" else 0.0
    delta = 0.0 if baseline_ppl is None else ppl - baseline_ppl

    result = BenchResult(model_id, variant, round(storage_vram, 3), round(peak_inference_vram, 3),
                          round(ppl, 3), round(delta, 3), round(latency, 2), tied)
    del model
    cleanup(device)
    return result


def run_model_suite(model_id, text_lines, device):
    results = []
    baseline_ppl = None
    for variant in VARIANTS:
        print(f"  -> {model_id} [{variant}]")
        r = run_one_variant(model_id, variant, text_lines, device, baseline_ppl)
        if variant == "fp16_baseline":
            baseline_ppl = r.perplexity
        results.append(r)
    return results


def print_table(results):
    header = (f"{'Model':<38}{'Variant':<26}{'Storage(GB)':<13}{'PeakInfer(GB)':<15}"
              f"{'PPL':<8}{'ΔPPL':<8}{'Latency(ms)':<12}{'Tied'}")
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(f"{r.model_id:<38}{r.variant:<26}{r.storage_vram_gb:<13}{r.peak_inference_vram_gb:<15}"
              f"{r.perplexity:<8}{r.perplexity_delta:<8}{r.forward_latency_ms:<12}{r.tied_weights}")


def main():
    parser = argparse.ArgumentParser(description="quant_head standardized benchmark (WikiText-2 perplexity)")
    parser.add_argument("--model", type=str, default=None, help="Single HF model id to benchmark")
    parser.add_argument("--all", action="store_true", help="Run the full 4-model validation suite")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU (perplexity numbers still valid, VRAM will read 0).")
        args.device = "cpu"

    text_lines = load_wikitext2_text()
    models = DEFAULT_MODELS if args.all else [args.model or "Qwen/Qwen2.5-0.5B"]

    all_results = []
    for model_id in models:
        print(f"\n=== {model_id} ===")
        all_results.extend(run_model_suite(model_id, text_lines, args.device))

    if  args.device == "cpu":
        print("\n[NOTE] Running on CPU.")
        print("  • Memory tracking (VRAM) is disabled (will display 0.0 GB).")
        print("  • Latency will be higher due to lack of CUDA hardware acceleration.")
        print("  • For peak memory savings and speedup benchmarks, run on a CUDA GPU.\n")

    print_table(all_results)

    with open(args.output, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\nSaved results to {args.output}")



if __name__ == "__main__":
    main()