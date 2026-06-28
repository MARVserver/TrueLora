# LoRA Merge

True-LoRA supports merging LoRA adapters directly into base model weights.

## Overview

LoRA merge permanently fuses the adapter's low-rank updates into the base model's weight matrices, producing a standalone model that no longer needs the adapter at inference time.

```
Original Weights:  W
LoRA Delta:        Δ = (B @ A) × (α / r)
Merged Weights:    W' = W + Δ
```

## CLI Usage

### Basic Merge

```bash
true-lora merge-adapter \
  --adapter generated.pt \
  --model gpt2 \
  --out merged_model/
```

### With Options

```bash
true-lora merge-adapter \
  --adapter generated.pt \
  --model gpt2 \
  --out merged_model/ \
  --dtype float16 \
  --device cuda \
  --max-length 512 \
  --allow-download
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--adapter` | (required) | Path to LoRA adapter (.pt) |
| `--model` | (required) | HuggingFace model name/path |
| `--out` | (required) | Output directory |
| `--dtype` | float32 | Output dtype: float32, float16, bfloat16 |
| `--device` | cpu | Device: cpu, cuda, cuda:0 |
| `--allow-download` | False | Allow downloading model from HF Hub |

## Python API

### Merge into Model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from true_lora import (
    load_torch_state_dict, infer_lora_tensor_specs,
    merge_lora_into_linear, save_peft_adapter
)

# Load adapter
state_dict = load_torch_state_dict("generated.pt")
specs = infer_lora_tensor_specs(state_dict)

# Load base model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Merge permanently
merged = merge_lora_into_linear(model, state_dict, specs, strict=False)

# Save merged model
model.save_pretrained("merged_model/")
tokenizer.save_pretrained("merged_model/")
```

### Temporary Merge (Context Manager)

```python
from true_lora import temporary_lora

with temporary_lora(model, state_dict, specs, strict=False) as applied:
    # Model has adapter applied
    output = model.generate(...)
# Adapter automatically removed
```

### Multiple Adapter Merge

```python
from true_lora import load_torch_state_dict, infer_lora_tensor_specs, merge_lora_into_linear

# Load multiple adapters
adapter_a = load_torch_state_dict("adapter_a.pt")
adapter_b = load_torch_state_dict("adapter_b.pt")

# Merge sequentially
specs_a = infer_lora_tensor_specs(adapter_a)
merge_lora_into_linear(model, adapter_a, specs_a, strict=False)

specs_b = infer_lora_tensor_specs(adapter_b)
merge_lora_into_linear(model, adapter_b, specs_b, strict=False)

# Or merge with custom weights
merged = {}
for name in set(list(adapter_a.keys()) + list(adapter_b.keys())):
    if name in adapter_a and name in adapter_b:
        merged[name] = adapter_a[name] * 0.5 + adapter_b[name] * 0.5
    elif name in adapter_a:
        merged[name] = adapter_a[name]
    else:
        merged[name] = adapter_b[name]

specs = infer_lora_tensor_specs(merged)
merge_lora_into_linear(model, merged, specs, strict=False)
```

## Supported Models

Any model with `nn.Linear` layers is supported:

| Model | Tested | Notes |
|-------|--------|-------|
| GPT-2 | ✅ | Full support |
| GPT-Neo | ✅ | Full support |
| LLaMA | ✅ | Full support |
| Mistral | ✅ | Full support |
| Phi | ✅ | Full support |
| BERT | ✅ | For classification |
| RoBERTa | ✅ | For classification |
| T5 | ✅ | Encoder-decoder |

## Output Formats

### True-LoRA Format (.pt)

```python
{
    "state_dict": {name: tensor, ...},
    "true_lora_report": {...}
}
```

### PEFT Directory Format

```
merged_model/
├── adapter_config.json
├── adapter_model.bin
└── (model weights if saved)
```

## Quantization Support

### Save as FP16

```python
import torch
model = model.half()  # Convert to FP16
model.save_pretrained("merged_fp16/")
```

### Save as BF16

```python
model = model.to(torch.bfloat16)
model.save_pretrained("merged_bf16/")
```

## Performance

| Operation | GPT-2 (124M) | LLaMA-7B |
|-----------|--------------|----------|
| Merge time | ~0.1s | ~2.5s |
| Memory (FP32) | ~500MB | ~28GB |
| Memory (FP16) | ~250MB | ~14GB |
| Output size | ~500MB | ~14GB |

## Troubleshooting

### Shape Mismatch

```
ValueError: delta shape (768, 768) does not match module weight shape (768, 2304)
```

**Solution**: The adapter was created for a different model architecture. Check tensor specs match the target model.

### Missing Tensors

```
KeyError: 'transformer.h.0.attn.c_attn.lora_A.weight'
```

**Solution**: Use `strict=False` to skip missing layers, or verify the adapter covers the target modules.

### Device Mismatch

```
RuntimeError: tensors are on different devices
```

**Solution**: Ensure both model and adapter are on the same device:
```python
model.to("cpu")
state_dict = {k: v.to("cpu") for k, v in state_dict.items()}
```
