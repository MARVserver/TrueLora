# FAQ

## General

### What is True-LoRA?

True-LoRA is a framework that generates LoRA (Low-Rank Adaptation) adapters directly from text prompts without fine-tuning. It combines retrieval-based adapter blending with neural generation.

### How is this different from regular LoRA?

Regular LoRA requires fine-tuning for each task. True-LoRA generates adapters instantly from text prompts by blending pre-trained adapters and generating new ones via a hypernetwork.

### Do I need to fine-tune anything?

Only the hypernetwork needs training (one-time). After that, any new adapter can be generated instantly from a text prompt.

### What models are supported?

Any model with `nn.Linear` layers: GPT-2, LLaMA, Mistral, Phi, BERT, RoBERTa, T5, and more.

## Installation

### What Python version is required?

Python >= 3.10 is required.

### Do I need CUDA?

No. True-LoRA works on CPU. CUDA is optional for faster training/inference.

### Do I need transformers?

Only if you want to evaluate on HuggingFace models. The core library works without it.

## Usage

### How do I create my own adapters?

1. Train LoRA adapters for your tasks using standard methods
2. Create a manifest file listing them
3. Train the TrueLoraGenerator on the manifest
4. Generate new adapters from text prompts

### How do I know if a generated adapter is good?

Check the uncertainty score:
- `uncertainty < 0.5`: High confidence
- `uncertainty 0.5-0.8`: Medium confidence
- `uncertainty > 0.8`: Low confidence (may not work well)

Use the quality gate for automatic acceptance/rejection.

### Can I merge adapters into models?

Yes! Use `true-lora merge-adapter` or the `merge_lora_into_linear()` function.

### How do I export to PEFT format?

Use `true-lora export-peft` to create a standard HuggingFace PEFT directory.

## Performance

### How fast is adapter generation?

~2ms per adapter on CPU, ~0.1ms on GPU.

### How much memory does it use?

- Adapter bank: ~1KB per adapter
- Hypernetwork: ~400K parameters (default)
- Generation: Minimal (single forward pass)

### Can I scale to 1000+ adapters?

Yes. The retrieval uses O(n) scoring with pre-normalized embeddings.

## Troubleshooting

### "ModuleNotFoundError: No module named 'true_lora'"

```bash
pip install -e .
# or
export PYTHONPATH=src
```

### "RuntimeError: Numpy is not available"

```bash
pip install numpy
```

### "ValueError: No matching LoRA A/B tensors found"

The adapter file doesn't contain valid LoRA tensors. Check the state dict keys.

### "RuntimeError: tensors are on different devices"

Ensure model and adapter are on the same device:
```python
model.to("cpu")
state_dict = {k: v.to("cpu") for k, v in state_dict.items()}
```

### "KeyError: 'some.module.lora_A.weight'"

The adapter doesn't cover the requested module. Use `strict=False` or check tensor specs.

## Advanced

### How do I add support for a new model architecture?

1. Identify the `nn.Linear` module names in your model
2. Create `LoraTensorSpec` entries for each target layer
3. Train adapters for those layers
4. Use TrueLoraGenerator as usual

### Can I use this with DeepSpeed/FSDP?

The hypernetwork is small enough for single-GPU training. For distributed training, you'd need to modify the training loop.

### How do I contribute?

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

## License

MIT License. See [LICENSE](LICENSE) for details.
