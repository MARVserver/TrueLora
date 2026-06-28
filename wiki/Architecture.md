# Architecture

## Overview

True-LoRA combines retrieval-based adapter blending with neural generation to produce LoRA adapters from text prompts.

```
Text Prompt
    │
    ▼
┌─────────────────┐
│ Text Encoder    │  Feature Hashing (blake2b)
│ (HashingText)   │  dim: 256
└────────┬────────┘
         │
         ├──────────────────────────────────────┐
         ▼                                      ▼
┌─────────────────┐                    ┌─────────────────┐
│ Adapter Bank    │─── Top-K Search ──▶│ HyperAdapter    │
│ (Pre-trained)   │                    │ (Neural Net)    │
└────────┬────────┘                    └────────┬────────┘
         │                                      │
         ▼                                      ▼
┌─────────────────┐                    ┌─────────────────┐
│ Interpolation   │                    │ Generation      │
│ (Weighted Avg)  │                    │ (Mean + LogVar) │
└────────┬────────┘                    └────────┬────────┘
         │                                      │
         └──────────────┬───────────────────────┘
                        ▼
              ┌───────────────────┐
              │ Blending Layer    │
              │ (Uncertainty ×    │
              │  Retrieved +      │
              │  Generated)       │
              └─────────┬─────────┘
                        ▼
              ┌───────────────────┐
              │ Norm Clipping     │
              │ (Max Tensor Norm) │
              └─────────┬─────────┘
                        ▼
                LoRA Adapter Output
```

## Components

### 1. HashingTextEncoder (`text.py`)

Converts text prompts to fixed-dimensional vectors using feature hashing.

```python
class HashingTextEncoder:
    dim: int = 256  # Output dimension
    
    def encode(text: str) -> torch.Tensor:
        # Tokenize with regex
        # Hash each token with blake2b
        # Map to bucket + sign
        # L2 normalize
```

**Performance**: Token hash cache for repeated tokens.

### 2. AdapterBank (`adapter.py`)

Stores pre-trained LoRA adapters and retrieves the most relevant ones.

```python
class AdapterBank:
    adapters: list[AdapterSpec]      # All adapters
    embeddings: torch.Tensor         # Pre-normalized embeddings (n, dim)
    _metric_prior_cache: dict        # Cached metric priors
    
    def retrieve_with_max_score(query, k, metric, metric_weight):
        # Single-pass: score + top-k
        # Returns: adapters, weights, all_scores
    
    def interpolate_retrieved(adapters, weights):
        # Batched: torch.stack + matrix multiply
        # Returns: merged_tensors, uncertainty
```

**Performance**: 
- Cached metric priors
- Single-pass retrieval (no double scoring)
- Batched interpolation

### 3. HyperAdapter (`generator.py`)

Neural network that generates LoRA tensors from text embeddings.

```python
class HyperAdapter(nn.Module):
    # Input: text_dim (256)
    # Output: 2 × total_params (mean + log_var for VAE-style generation)
    
    net = Sequential(
        Linear(text_dim, hidden_dim),
        SiLU(),
        LayerNorm(hidden_dim),
        Linear(hidden_dim, total * 2),
    )
```

**Architecture**:
- Pre-computed offset tables for fast tensor splitting
- Uncertainty estimated from log_var mean

### 4. TrueLoraGenerator (`generator.py`)

Orchestrates the full generation pipeline.

```python
class TrueLoraGenerator:
    def generate_components(prompt, retrieval_k, ...) -> (components, report):
        # 1. Encode prompt
        # 2. Retrieve adapters (single-pass scoring)
        # 3. Interpolate retrieved adapters (batched)
        # 4. Generate via hypernetwork
        # 5. Blend with uncertainty weighting
        # 6. Clip norms
```

### 5. apply.py - LoRA Application

Applies LoRA adapters to PyTorch models.

```python
def temporary_lora(model, state_dict, specs):
    # Only clone targeted module weights (memory optimization)
    # Apply LoRA delta: W += (B @ A) * (alpha / rank)
    # Restore original weights on exit
```

### 6. Quality Gating (`quality.py`)

Automatically accept/reject adapters based on quality criteria.

```python
class QualityGate:
    min_accuracy_delta: float = 0.0
    max_uncertainty: float = 0.8
    max_tensor_norm: float = 8.0
    max_consistency_mse: float | None = None
    min_prompt_sensitivity_mse: float | None = None
```

## Data Flow

### Generation Pipeline

1. **Encode**: `prompt → embedding (256,)`
2. **Score**: `embedding × adapter_embeddings.T → scores (n,)`
3. **Retrieve**: `topk(scores, k) → adapters, weights`
4. **Interpolate**: `Σ(adapter_tensors × weights) → retrieved_tensors`
5. **Generate**: `embedding → hypernet → generated_tensors + uncertainty`
6. **Blend**: `retrieved × uncertainty + generated × (1 - uncertainty)`
7. **Clip**: `clip_norm(blended, max_norm)`

### Training Pipeline

1. **Forward**: `adapter.embedding → hypernet → predicted_tensors`
2. **Loss**: `MSE(predicted, target_tensors)`
3. **Backward**: Standard PyTorch backpropagation
4. **Update**: AdamW optimizer on hypernetwork parameters

## Performance Characteristics

| Operation | Complexity | Optimization |
|-----------|-----------|--------------|
| Text Encoding | O(tokens) | Token hash cache |
| Adapter Scoring | O(n × dim) | Pre-normalized embeddings |
| Top-K Retrieval | O(n log k) | Single-pass with max_score |
| Interpolation | O(k × tensors) | Batched torch.stack |
| Hypernetwork Forward | O(params) | Pre-computed offsets |
| Norm Clipping | O(tensors) | Scalar comparison |

## Memory Usage

- **Adapter Bank**: `n × (dim + tensor_params)` float32
- **Hypernetwork**: `text_dim × hidden + hidden × total × 2` parameters
- **Temporary**: Only targeted module weights cloned (90%+ savings)
