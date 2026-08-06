"""
examples/quickstart.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from quant_head.utils import quantize_embeddings_and_head

MODEL_ID = "gpt2"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Loading {MODEL_ID} on {device.upper()}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)

model, report = quantize_embeddings_and_head(model, quantize_embeddings=True, quantize_head=True)

print("\nQuantization report:")
print(f"  quantized_embeddings        : {report.quantized_embeddings}")
print(f"  quantized_head              : {report.quantized_head}")
print(f"  tied_weights                : {report.tied_weights}")
print(f"  head_shares_embedding_table : {report.head_shares_embedding_table}")
print(f"  embedding table now uses    : {report.embedding_memory_bytes / 1e6:.1f} MB (int8)")
if not report.head_shares_embedding_table:
    print(f"  head now uses               : {report.head_memory_bytes / 1e6:.1f} MB (int8)")

inputs = tokenizer("The capital of France is", return_tensors="pt").to(device)
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)

print("\nGenerated text with quantized embeddings/head:")
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))


print(report.print_summary())