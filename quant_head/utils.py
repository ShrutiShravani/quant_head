"""
quant_head.utils

Model-agnostic helper for quantizing the input embedding and output (LM
head) layers of any Hugging Face causal LM.

Design goal: work on ANY model implementing the standard PreTrainedModel
interface -- get_input_embeddings / set_input_embeddings /
get_output_embeddings / set_output_embeddings -- rather than assuming a
hardcoded attribute path like `model.model.embed_tokens`, which differs
across architectures (e.g. `transformer.wte` in GPT-2, `model.embed_tokens`
in Llama/Qwen/Gemma, `gpt_neox.embed_in` in GPT-NeoX/Pythia).
"""
from dataclasses import dataclass

import torch.nn as nn

from quant_head.quant_modules import QuantizedEmbedding, QuantizedLinear


@dataclass
class QuantizationReport:
    quantized_embeddings: bool
    quantized_head: bool
    tied_weights: bool
    head_shares_embedding_table: bool
    tied_forced_both_sides: bool = False
    embedding_memory_bytes: int = 0
    head_memory_bytes: int = 0
    baseline_memory_bytes: int=0
    
    @property
    def total_quantized_bytes(self) -> int:
        """Total memory consumed by quantized layers."""
        if self.head_shares_embedding_table:
            return self.embedding_memory_bytes
        return self.embedding_memory_bytes + self.head_memory_bytes

    @property
    def memory_saved_bytes(self) -> int:
        """Memory saved compared to original baseline."""
        return max(0, self.baseline_memory_bytes - self.total_quantized_bytes)

    @property
    def reduction_percentage(self) -> float:
        """Percentage of memory saved relative to baseline."""
        if self.baseline_memory_bytes == 0:
            return 0.0
        return (self.memory_saved_bytes / self.baseline_memory_bytes) * 100.0

    def print_summary(self):
        """Prints a clean executive summary for quickstart and user scripts."""
        print("\n" + "=" * 55)
        print("             QUANT-HEAD MEMORY REPORT             ")
        print("=" * 55)
        print(f"  Tied Weights                 : {self.tied_weights}")
        print(f"  Quantized Embeddings         : {self.quantized_embeddings}")
        print(f"  Quantized Head               : {self.quantized_head}")
        print(f"  Head Reuses Embedding Table  : {self.head_shares_embedding_table}")
        print("-" * 55)
        print(f"  Baseline Weight Memory       : {self.baseline_memory_bytes / 1e6:.2f} MB")
        print(f"  Quantized Weight Memory      : {self.total_quantized_bytes / 1e6:.2f} MB")
        print(f"  Memory Saved                 : {self.memory_saved_bytes / 1e6:.2f} MB ({self.reduction_percentage:.1f}% reduction)")
        print("=" * 55 + "\n")

def _make_shared_head(quantized_embedding: QuantizedEmbedding) -> QuantizedLinear:
    head = QuantizedLinear.__new__(QuantizedLinear)
    nn.Module.__init__(head)
    head.in_features = quantized_embedding.embedding_dim
    head.out_features = quantized_embedding.num_embeddings
    head.register_buffer("qweight", quantized_embedding.qweight)
    head.register_buffer("scales", quantized_embedding.scales)
    head.bias = None
    return head


def quantize_embeddings_and_head(
    model: nn.Module,
    quantize_embeddings: bool = True,
    quantize_head: bool = True,
) -> tuple[nn.Module, QuantizationReport]:
    if not (hasattr(model, "get_input_embeddings") and hasattr(model, "get_output_embeddings")):
        raise TypeError(
            "Model does not expose get_input_embeddings/get_output_embeddings; "
            "quant_head only supports transformers PreTrainedModel-style causal LMs."
        )

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()

    config_tied = bool(getattr(getattr(model, "config", None), "tie_word_embeddings", False))
    weight_identity_tied = (
        output_embeddings is not None
        and input_embeddings is not None
        and isinstance(output_embeddings, nn.Linear)
        and output_embeddings.weight is input_embeddings.weight
    )
    tied = config_tied or weight_identity_tied


    # --- 1. CALCULATE BASELINE MEMORY HERE BEFORE REPLACING LAYERS ---
    baseline_bytes = 0

    # Read input embedding size (use hasattr instead of strict isinstance check)
    if input_embeddings is not None and hasattr(input_embeddings, "weight") and input_embeddings.weight is not None:
        baseline_bytes += input_embeddings.weight.numel() * input_embeddings.weight.element_size()

    # Read output head size (only add if untied, so we don't double count shared weights)
    if output_embeddings is not None and hasattr(output_embeddings, "weight") and output_embeddings.weight is not None:
        if not tied:
            baseline_bytes += output_embeddings.weight.numel() * output_embeddings.weight.element_size()
            if getattr(output_embeddings, "bias", None) is not None:
                baseline_bytes += output_embeddings.bias.numel() * output_embeddings.bias.element_size()

    # For a tied model, embedding and head are the SAME tensor. Quantizing
    # only one side leaves the untouched side holding a live reference to
    # the original fp16 tensor, so memory goes UP instead of down. Force
    # both sides together whenever either was requested on a tied model.
    tied_forced_both = False
    if tied and quantize_embeddings != quantize_head:
        tied_forced_both = True
        quantize_embeddings = quantize_head = (quantize_embeddings or quantize_head)

    quantized_input = None
    if quantize_embeddings and isinstance(input_embeddings, nn.Embedding):
        quantized_input = QuantizedEmbedding.from_float(input_embeddings)
        model.set_input_embeddings(quantized_input)

    quantized_head_done = False
    if quantize_head and output_embeddings is not None:
        if tied and quantized_input is not None:
            model.set_output_embeddings(_make_shared_head(quantized_input))
            quantized_head_done = True
        elif tied and quantized_input is None:
            if isinstance(output_embeddings, nn.Linear):
                model.set_output_embeddings(QuantizedLinear.from_float(output_embeddings))
                quantized_head_done = True
        elif isinstance(output_embeddings, nn.Linear):
            model.set_output_embeddings(QuantizedLinear.from_float(output_embeddings))
            quantized_head_done = True

    embed_mem = model.get_input_embeddings().weight_memory_bytes if quantized_input is not None else 0
    head_module = model.get_output_embeddings()
    head_mem = getattr(head_module, "weight_memory_bytes", 0) if quantized_head_done else 0

    report = QuantizationReport(
        quantized_embeddings=quantized_input is not None,
        quantized_head=quantized_head_done,
        tied_weights=tied,
        head_shares_embedding_table=tied and quantized_input is not None,
        tied_forced_both_sides=tied_forced_both,
        embedding_memory_bytes=embed_mem,
        head_memory_bytes=head_mem,
        baseline_memory_bytes=baseline_bytes,
    )
    return model, report