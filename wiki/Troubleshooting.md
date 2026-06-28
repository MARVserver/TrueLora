# Troubleshooting

## Installation Issues

### `pip install` fails with "externally-managed-environment"

```bash
# Option 1: Use --break-system-packages
pip install --break-system-packages -e .

# Option 2: Use a virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install -e .
```

### `torch` not found

```bash
pip install torch>=2.0
# For CUDA:
pip install torch>=2.0 --index-url https://download.pytorch.org/whl/cu121
```

### `numpy` not found

```bash
pip install numpy>=1.24
```

### `transformers` not found

```bash
pip install transformers>=4.0
```

## Runtime Errors

### `ModuleNotFoundError: No module named 'true_lora'`

```bash
# Option 1: Install in development mode
pip install -e .

# Option 2: Set PYTHONPATH
export PYTHONPATH=/path/to/True-lora/src
```

### `RuntimeError: Numpy is not available`

```bash
pip install numpy
```

### `ValueError: AdapterBank requires at least one adapter`

The adapter bank is empty. Ensure your manifest file contains valid adapters.

### `ValueError: No matching LoRA A/B tensors found`

The adapter file doesn't contain valid LoRA tensors. Check:
1. The state dict has keys ending in `.lora_A.weight` and `.lora_B.weight`
2. The tensors are 2D matrices
3. The rank dimension matches between A and B

### `RuntimeError: tensors are on different devices`

```python
# Ensure both model and adapter are on the same device
model.to("cpu")
state_dict = {k: v.to("cpu") for k, v in state_dict.items()}

# Or move to GPU
model.to("cuda")
state_dict = {k: v.to("cuda") for k, v in state_dict.items()}
```

### `KeyError: 'transformer.h.0.attn.c_attn.lora_A.weight'`

The adapter doesn't cover the requested module. Options:
1. Use `strict=False` to skip missing layers
2. Check tensor specs match the target model
3. Ensure the adapter was created for the same architecture

### `RuntimeError: a leaf Variable that requires grad is being used in an in-place operation`

```python
# Use torch.no_grad() when restoring weights
with torch.no_grad():
    module.weight.copy_(original_weight)
```

## Training Issues

### Loss not decreasing

1. Check learning rate (try 1e-3 or 3e-3)
2. Increase training steps
3. Verify adapter tensors are valid
4. Check for NaN values in tensors

### Training is too slow

1. Reduce text_dim and hidden_dim
2. Use fewer training steps
3. Use smaller rank (rank=2 instead of rank=4)
4. Train on GPU if available

### Overfitting

1. Reduce training steps
2. Use more adapters in the bank
3. Add dropout to the hypernetwork
4. Use data augmentation on prompts

## Generation Issues

### Generated adapter has high uncertainty

This is normal for prompts far from the training data. Options:
1. Add more diverse adapters to the bank
2. Increase retrieval_k
3. Use metric-weighted retrieval
4. Accept the high uncertainty and use quality gating

### Generated adapter doesn't work well

1. Check uncertainty score (should be < 0.8)
2. Verify adapter covers the target model's modules
3. Try different prompts
4. Check tensor norms (should be < max_tensor_norm)

### `abstained` flag is 1.0

No adapter in the bank matches the prompt well enough. Options:
1. Lower `min_retrieval_score`
2. Add more adapters to the bank
3. Use a different prompt

## Merge Issues

### Shape mismatch during merge

```
ValueError: delta shape (768, 768) does not match module weight shape (768, 2304)
```

The adapter was created for a different architecture. Check:
1. Tensor specs match the target model
2. The adapter covers the correct layers

### Missing tensors during merge

```
KeyError: 'some.module.lora_A.weight'
```

Use `strict=False` or verify the adapter covers all target modules.

### Device mismatch during merge

```python
# Move everything to CPU first
model.cpu()
state_dict = {k: v.cpu() for k, v in state_dict.items()}
```

## Performance Issues

### Memory usage is high

1. Use `temporary_lora()` instead of permanent merge
2. Only clone targeted module weights
3. Use FP16 for large models
4. Offload to CPU if needed

### Generation is slow

1. Use the token hash cache (automatic)
2. Pre-encode prompts
3. Batch multiple generations
4. Use GPU for the hypernetwork

### Retrieval is slow

1. Reduce the adapter bank size
2. Use smaller text_dim
3. Cache embeddings
4. Use approximate nearest neighbor search for very large banks

## Getting Help

If you encounter an issue not covered here:

1. Check the [FAQ](FAQ.md)
2. Search GitHub Issues
3. Open a new issue with:
   - Error message
   - Full traceback
   - Python version
   - PyTorch version
   - Steps to reproduce
