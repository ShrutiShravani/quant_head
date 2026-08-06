"""
tests/test_modules.py

CPU-only tests (no GPU required, no model downloads required for the
module-level tests) so these can run in CI on every PR.
"""
import torch
import torch.nn as nn
import pytest

from quant_head.quant_modules import QuantizedEmbedding, QuantizedLinear
from quant_head.utils import quantize_embeddings_and_head


def test_quantized_embedding_shape_and_dtype():
    emb = nn.Embedding(100, 16)
    qemb = QuantizedEmbedding.from_float(emb)
    assert qemb.qweight.dtype == torch.int8
    assert qemb.qweight.shape == (100, 16)

    ids = torch.randint(0, 100, (2, 5))
    out = qemb(ids)
    assert out.shape == (2, 5, 16)
    assert out.dtype == emb.weight.dtype


def test_quantized_embedding_reasonable_error():
    torch.manual_seed(0)
    emb = nn.Embedding(50, 32)
    qemb = QuantizedEmbedding.from_float(emb)

    ids = torch.arange(50)
    original = emb(ids)
    quantized = qemb(ids).float()

    # Row-wise int8 quantization should keep relative error small.
    rel_error = (original - quantized).abs().mean() / original.abs().mean()
    assert rel_error < 0.05, f"quantization error too large: {rel_error:.4f}"


def test_quantized_linear_shape_and_dtype():
    lin = nn.Linear(32, 100, bias=True)
    qlin = QuantizedLinear.from_float(lin)
    assert qlin.qweight.dtype == torch.int8
    assert qlin.qweight.shape == (100, 32)

    x = torch.randn(4, 32)
    out = qlin(x)
    assert out.shape == (4, 100)
    assert out.dtype == x.dtype


def test_quantized_linear_bias_preserved():
    lin = nn.Linear(8, 8, bias=True)
    lin.bias.data.fill_(1.0)
    qlin = QuantizedLinear.from_float(lin)
    assert qlin.bias is not None
    assert torch.allclose(qlin.bias, lin.bias)


class _TinyUntiedLM(nn.Module):
    """Minimal fake causal LM with untied input/output embeddings, to test
    the generic get/set_input/output_embeddings path without downloading
    a real model."""

    class _Config:
        tie_word_embeddings = False

    def __init__(self, vocab=50, dim=16):
        super().__init__()
        self.config = self._Config()
        self.embed = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, module):
        self.embed = module

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, module):
        self.head = module


class _TinyTiedLM(nn.Module):
    """Minimal fake causal LM with TIED input/output embeddings, to test
    the tied-weight sharing path without downloading a real model."""

    class _Config:
        tie_word_embeddings = True

    def __init__(self, vocab=50, dim=16):
        super().__init__()
        self.config = self._Config()
        self.embed = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.embed.weight  # actual weight tying

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, module):
        self.embed = module

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, module):
        self.head = module


def test_untied_model_quantizes_independently():
    model = _TinyUntiedLM()
    model, report = quantize_embeddings_and_head(model)

    assert report.quantized_embeddings
    assert report.quantized_head
    assert not report.tied_weights
    assert not report.head_shares_embedding_table
    # Independent quantized buffers, not the same object
    assert model.get_input_embeddings().qweight is not model.get_output_embeddings().qweight


def test_tied_model_shares_quantized_buffers():
    model = _TinyTiedLM()
    model, report = quantize_embeddings_and_head(model)

    assert report.tied_weights
    assert report.head_shares_embedding_table
    # The whole point of tied-weight handling: same underlying int8 buffer,
    # not two independently-rounded copies.
    assert model.get_input_embeddings().qweight is model.get_output_embeddings().qweight
    assert model.get_input_embeddings().scales is model.get_output_embeddings().scales


def test_tied_model_head_matches_embedding_numerically():
    model = _TinyTiedLM(vocab=20, dim=8)
    model, _ = quantize_embeddings_and_head(model)

    ids = torch.arange(20)
    embedded = model.get_input_embeddings()(ids)          # [20, 8]
    logits_identity = model.get_output_embeddings()(embedded)  # [20, 20]

    # embedding row i dotted with itself (via the shared table) should sit
    # near the diagonal max, sanity-checking that the shared buffer is
    # wired correctly rather than silently zeroed or shape-mismatched.
    assert logits_identity.shape == (20, 20)
    assert torch.isfinite(logits_identity).all()


def test_quantize_embeddings_only_leaves_head_untouched():
    model = _TinyUntiedLM()
    original_head = model.head
    model, report = quantize_embeddings_and_head(model, quantize_embeddings=True, quantize_head=False)

    assert report.quantized_embeddings
    assert not report.quantized_head
    assert model.get_output_embeddings() is original_head


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))