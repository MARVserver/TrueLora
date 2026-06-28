# Quick Start

Get up and running with True-LoRA in 5 minutes.

## Step 1: Install

```bash
pip install git+https://github.com/MARVserver/TrueLora.git
```

## Step 2: Demo (No Setup Required)

Generate a LoRA adapter from a text prompt:

```bash
true-lora demo --prompt "python code generation debugging" --out adapter.pt
```

This creates a LoRA adapter file `adapter.pt` that can be applied to any compatible model.

## Step 3: Train on Adapter Bank

Create a manifest file (`adapters.jsonl`) listing your pre-trained adapters:

```json
{"description": "code generation python", "path": "adapter_00.pt", "metrics": {"score": 0.72}}
{"description": "japanese translation", "path": "adapter_01.pt", "metrics": {"score": 0.48}}
{"description": "creative writing", "path": "adapter_02.pt", "metrics": {"score": 0.61}}
```

Train the hypernetwork:

```bash
true-lora train \
  --manifest adapters.jsonl \
  --out checkpoint.pt \
  --steps 500
```

## Step 4: Generate Adapters

Use the trained checkpoint to generate new adapters:

```bash
true-lora generate \
  --manifest adapters.jsonl \
  --checkpoint checkpoint.pt \
  --prompt "scientific research paper" \
  --out generated.pt
```

## Step 5: Evaluate on GPT-2

Test the generated adapter on a real model:

```bash
true-lora gate \
  --adapter generated.pt \
  --hf-generation-benchmark benchmark.jsonl \
  --model gpt2
```

## Step 6: Merge into Model

Merge the LoRA adapter directly into model weights:

```bash
true-lora merge-adapter \
  --adapter generated.pt \
  --model gpt2 \
  --out merged_model/
```

## Python API Quick Start

```python
import torch
from true_lora import (
    TrueLoraGenerator, AdapterBank, AdapterSpec,
    LoraTensorSpec, HashingTextEncoder, temporary_lora
)

# 1. Create encoder and adapters
encoder = HashingTextEncoder(dim=256)
adapters = [
    AdapterSpec("code generation", encoder.encode("code generation"), {
        "layer.lora_A.weight": torch.randn(4, 16),
        "layer.lora_B.weight": torch.randn(16, 4),
    }),
]

# 2. Create generator
specs = [LoraTensorSpec("layer", 16, 16, 4)]
bank = AdapterBank(adapters)
model = TrueLoraGenerator(specs, bank)

# 3. Generate adapter
adapter, report = model.generate("python debugging tests")
print(f"Uncertainty: {report['uncertainty']:.3f}")

# 4. Apply to model
from transformers import AutoModelForCausalLM
gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")
specs = infer_lora_tensor_specs(adapter)
with temporary_lora(gpt2, adapter, specs, strict=False):
    # Model now has the adapter applied
    output = gpt2.generate(...)
```
