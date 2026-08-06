import torch
import torch.nn as nn



"""
quant_head.modules

Row-wise INT8 quantized drop-in replacements for nn.Embedding and nn.Linear,
intended specifically for compressing the input embedding table and the
LM head (output projection) of a causal language model -- typically the
largest non-transformer-block parameters for models with big vocabularies.
"""
import torch
import torch.nn as nn

"""
quant_head.modules

Row-wise INT8 quantized drop-in replacements for nn.Embedding and nn.Linear,
intended specifically for compressing the input embedding table and the
LM head (output projection) of a causal language model -- typically the
largest non-transformer-block parameters for models with big vocabularies.
"""
import torch
import torch.nn as nn


class QuantizedEmbedding(nn.Module):
    """INT8 row-wise quantized drop-in replacement for nn.Embedding."""

    def __init__(self, original_embedding: nn.Embedding):
        super().__init__()
        self.num_embeddings = original_embedding.num_embeddings
        self.embedding_dim = original_embedding.embedding_dim
        self.padding_idx = original_embedding.padding_idx

        # Detect model precision (float16 on GPU, float32 on CPU)
        orig_dtype = original_embedding.weight.dtype
        weight = original_embedding.weight.data.float()

        max_vals = weight.abs().amax(dim=1, keepdim=True)
        scales = (max_vals / 127.0).clamp(min=1e-8)
        qweight = torch.round(weight / scales).clamp(-127, 127).to(torch.int8)

        self.register_buffer("qweight", qweight)
        # FIX 1: Store scales in the layer's original dtype (float32 on CPU, float16 on GPU)
        self.register_buffer("scales", scales.to(orig_dtype))

    @classmethod
    def from_float(cls, original_embedding: nn.Embedding) -> "QuantizedEmbedding":
        return cls(original_embedding)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        q_rows = self.qweight[input_ids]        # [*, embed_dim], int8
        scales = self.scales[input_ids]         # [*, 1]
        
        # FIX 2 & 3: Dequantize using self.scales.dtype dynamically
        target_dtype = self.scales.dtype
        return q_rows.to(target_dtype) * scales.to(target_dtype)

    def dequantize_full(self) -> torch.Tensor:
        """Materialize the full fp16/fp32 table. Used internally when a tied
        LM head needs to reuse this table's weights."""
        # FIX 4: Dynamically match scale dtype instead of hardcoded .half()
        return self.qweight.to(self.scales.dtype) * self.scales

    @property
    def weight_memory_bytes(self) -> int:
        return self.qweight.numel() * 1 + self.scales.numel() * 2

    def extra_repr(self) -> str:
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, dtype=int8"


class QuantizedLinear(nn.Module):
    """INT8 row-wise quantized drop-in replacement for nn.Linear."""

    def __init__(self, original_linear: nn.Linear):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features

        orig_dtype = original_linear.weight.dtype
        weight = original_linear.weight.data.float()

        max_vals = weight.abs().amax(dim=1, keepdim=True)
        scales = (max_vals / 127.0).clamp(min=1e-8)
        qweight = torch.round(weight / scales).clamp(-127, 127).to(torch.int8)

        self.register_buffer("qweight", qweight)
        self.register_buffer("scales", scales.to(orig_dtype))

        self.bias = None
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone())

    @classmethod
    def from_float(cls, original_linear: nn.Linear) -> "QuantizedLinear":
        return cls(original_linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dynamically match input tensor 'x' dtype
        weight = self.qweight.to(x.dtype) * self.scales.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return nn.functional.linear(x, weight, bias)

    @property
    def weight_memory_bytes(self) -> int:
        return self.qweight.numel() * 1 + self.scales.numel() * 2

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, dtype=int8"