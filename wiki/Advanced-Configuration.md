# Advanced Configuration

## Hypernetwork Architecture

### Default Configuration

```python
TrueLoraGenerator(
    tensor_specs=specs,
    adapter_bank=bank,
    text_dim=256,       # Text embedding dimension
    hidden_dim=512,     # Hidden layer dimension
    max_tensor_norm=1.0, # Maximum tensor norm
    ood_shrink_factor=0.25, # Out-of-distribution shrink factor
)
```

### Custom Architecture

```python
# Larger model for more complex tasks
model = TrueLoraGenerator(
    specs, bank,
    text_dim=512,       # Larger embedding
    hidden_dim=1024,    # Larger hidden layer
    max_tensor_norm=4.0, # Allow larger tensors
    ood_shrink_factor=0.5, # Less aggressive shrinking
)
```

### Architecture Options

| Parameter | Small | Default | Large |
|-----------|-------|---------|-------|
| text_dim | 128 | 256 | 512 |
| hidden_dim | 256 | 512 | 1024 |
| Parameters | ~100K | ~400K | ~1.5M |
| Training Speed | Fast | Medium | Slow |

## Retrieval Configuration

### Top-K Selection

```python
# More adapters = more diverse blending
adapter, report = model.generate(prompt, retrieval_k=8)

# Fewer adapters = more focused
adapter, report = model.generate(prompt, retrieval_k=2)
```

### Metric-Weighted Retrieval

```python
# Use quality metrics to guide retrieval
adapter, report = model.generate(
    prompt,
    retrieval_metric="score",
    metric_weight=0.5,  # 50% semantic, 50% quality
)
```

### Minimum Retrieval Score

```python
# Abstain if no good match found
adapter, report = model.generate(
    prompt,
    min_retrieval_score=0.5,
)

if report["abstained"] == 1.0:
    print("No good adapter match found")
```

## Training Configuration

### Learning Rate Schedule

```python
# Default
losses = train_on_adapter_bank(model, adapters, steps=500, lr=1e-3)

# Conservative (larger models)
losses = train_on_adapter_bank(model, adapters, steps=1000, lr=5e-4)

# Aggressive (small models)
losses = train_on_adapter_bank(model, adapters, steps=200, lr=3e-3)
```

### Training Steps

| Dataset Size | Recommended Steps |
|--------------|-------------------|
| 5 adapters | 100-200 |
| 20 adapters | 300-500 |
| 100 adapters | 500-1000 |
| 1000+ adapters | 1000-2000 |

## Quality Gate Configuration

### Strict Gate

```python
gate = QualityGate(
    min_accuracy_delta=0.1,    # Require 10% accuracy improvement
    max_uncertainty=0.5,       # Low uncertainty threshold
    max_tensor_norm=4.0,       # Moderate tensor size
    max_consistency_mse=0.01,  # High consistency required
    min_prompt_sensitivity_mse=0.1,  # Must be sensitive to prompts
    min_retrieval_score_delta=0.0,   # Must improve retrieval
)
```

### Lenient Gate

```python
gate = QualityGate(
    min_accuracy_delta=0.0,    # No accuracy requirement
    max_uncertainty=0.9,       # Allow high uncertainty
    max_tensor_norm=8.0,       # Allow large tensors
)
```

### Custom Gate

```python
gate = QualityGate()
# Modify individual thresholds
gate.min_accuracy_delta = 0.05
gate.max_uncertainty = 0.7
```

## Text Encoder Configuration

### Custom Dimension

```python
# Smaller encoder (faster, less accurate)
encoder = HashingTextEncoder(dim=128)

# Larger encoder (slower, more accurate)
encoder = HashingTextEncoder(dim=512)
```

### Token Hash Cache

The encoder automatically caches token hashes for repeated tokens:

```python
encoder = HashingTextEncoder(dim=256)

# First call: computes hash
encoder.encode("python code generation")

# Second call: uses cache
encoder.encode("python code generation")  # ~5x faster
```

## Reproducibility

### Fixed Seed

```python
from true_lora import set_seed

set_seed(42)  # Set global seed

# All operations now deterministic
adapter1, _ = model.generate("test prompt")
adapter2, _ = model.generate("test prompt")
assert torch.equal(adapter1["layer.lora_A.weight"], 
                   adapter2["layer.lora_A.weight"])
```

### Seed per Adapter

```python
for i, prompt in enumerate(prompts):
    set_seed(42 + i)
    adapter, report = model.generate(prompt)
```

## Memory Optimization

### Gradient Checkpointing

```python
# For large models
model.hyper.gradient_checkpointing_enable()
```

### Mixed Precision

```python
# FP16 training
with torch.cuda.amp.autocast():
    losses = train_on_adapter_bank(model, adapters, steps=500)
```

### CPU Offloading

```python
# For very large models
model.to("cpu")
state_dict = {k: v.to("cpu") for k, v in state_dict.items()}
```

## Performance Tuning

### Batch Processing

```python
# Generate multiple adapters
adapters = []
for prompt in prompts:
    adapter, report = model.generate(prompt)
    adapters.append(adapter)
```

### Parallel Retrieval

```python
# Score all adapters at once
scores = bank.score(query_embedding)
# Then retrieve top-k
```

### Cached Embeddings

```python
# Pre-encode all prompts
embeddings = {prompt: encoder.encode(prompt) for prompt in prompts}

# Reuse embeddings
for prompt in prompts:
    embedding = embeddings[prompt]
    adapter, report = model.generate_from_embedding(embedding)
```
