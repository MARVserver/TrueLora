# True-LoRA Wiki

**Retrieval-Grounded, Uncertainty-Aware Text-to-LoRA Generation**

Welcome to the True-LoRA wiki. This wiki provides comprehensive documentation for using, configuring, and extending True-LoRA.

---

## Table of Contents

- [[Quick-Start]] - Get up and running in 5 minutes
- [[Installation]] - Installation options and requirements
- [[Architecture]] - How True-LoRA works internally
- [[CLI-Reference]] - Complete command-line interface documentation
- [[API-Reference]] - Python API documentation
- [[LoRA-Merge]] - Merging LoRA adapters into base models
- [[GPT-2-Experiment]] - Running experiments with GPT-2
- [[Advanced-Configuration]] - Fine-tuning parameters and optimization
- [[FAQ]] - Frequently asked questions
- [[Troubleshooting]] - Common issues and solutions

---

## What is True-LoRA?

True-LoRA generates LoRA (Low-Rank Adaptation) adapters **directly from text prompts** without any fine-tuning. Instead of training a separate adapter for each task, True-LoRA:

1. Stores pre-trained adapters in an **Adapter Bank**
2. **Retrieves** the most relevant adapters for a given prompt
3. **Blends** them using learned interpolation weights
4. **Generates** new adapter tensors via a hypernetwork
5. Estimates **uncertainty** for quality control

## Key Features

| Feature | Description |
|---------|-------------|
| Zero-Shot Adaptation | Generate LoRA adapters for new tasks without fine-tuning |
| Uncertainty Estimation | Know when the generated adapter is reliable |
| Quality Gating | Automatically accept/reject adapters based on confidence |
| PEFT Compatible | Export to standard HuggingFace PEFT format |
| LoRA Merge | Merge adapters directly into base model weights |
| Batch Processing | Optimized tensor operations for high throughput |
| Reproducibility | Deterministic generation with seed control |

## Quick Example

```python
from true_lora import TrueLoraGenerator, AdapterBank, AdapterSpec, LoraTensorSpec

# Create adapter bank with pre-trained adapters
specs = [LoraTensorSpec("layer", out_features=768, in_features=768, rank=4)]
bank = AdapterBank(adapters)

# Create generator
model = TrueLoraGenerator(specs, bank, text_dim=256, hidden_dim=512)

# Generate adapter from text prompt
adapter, report = model.generate("python code generation debugging")
print(f"Uncertainty: {report['uncertainty']:.3f}")
```

## Community

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share experiments

## License

MIT License
