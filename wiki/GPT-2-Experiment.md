# GPT-2 Experiment

Running experiments with GPT-2 to demonstrate True-LoRA's capabilities.

## Setup

```bash
# Install with HuggingFace support
pip install "true-lora[hf]" git+https://github.com/MARVserver/TrueLora.git

# Navigate to experiments
cd experiments/gpt2/
```

## Quick Run

```bash
python run_experiment.py
```

## What the Experiment Does

### 1. Creates GPT-2 Compatible Adapters

Defines LoRA tensor specs matching GPT-2's architecture:

```python
SPECS = [
    LoraTensorSpec("transformer.h.0.attn.c_attn", out_features=2304, in_features=768, rank=4),
    LoraTensorSpec("transformer.h.0.attn.c_proj", out_features=768, in_features=768, rank=4),
]
```

### 2. Creates Training Adapters

Generates 5 adapters for different tasks:

| Description | Score |
|-------------|-------|
| code generation python javascript typescript | 0.72 |
| japanese translation polite casual business | 0.48 |
| creative writing storytelling narrative prose | 0.61 |
| scientific explanation physics chemistry biology | 0.55 |
| chat conversation question answering dialogue | 0.65 |

### 3. Trains TrueLoraGenerator

Trains for 200 steps on the adapter bank:

```
First loss: 1.383255
Last loss:  0.000814
```

### 4. Generates New Adapters

Generates adapters from prompts:

```python
prompts = [
    "python code generation debugging unit tests",
    "japanese polite conversation business email",
    "creative fantasy storytelling world building",
    "scientific research paper methodology",
]
```

### 5. Evaluates on GPT-2

Applies generated adapters to GPT-2 and compares outputs:

**Prompt**: `def hello_world():`

| Adapter | Output |
|---------|--------|
| baseline | `Hello World\n\nThe package is available if you run it with the` |
| creative_fantasy | `from pycurl.urlopen import PY_POST` |
| japanese_polite | `print "Hello world"\n\nprint` |
| python_code | `"Hello, world!" hello_world.init()` |
| scientific | `from hello_world import hello_world` |

### 6. Ablation Report

| Metric | Value |
|--------|-------|
| mean_blended_mse | 0.008242 |
| mean_retrieval_mse | 0.010435 |
| mean_generated_mse | 0.006293 |
| blended_wins | 3 |
| generated_wins | 2 |

## Custom Experiments

### Create Your Own Adapters

```python
import torch
from true_lora import AdapterSpec, HashingTextEncoder, save_peft_adapter

encoder = HashingTextEncoder(dim=256)

# Create adapter tensors
tensors = {
    "transformer.h.0.attn.c_attn.lora_A.weight": torch.randn(4, 768) * 0.1,
    "transformer.h.0.attn.c_attn.lora_B.weight": torch.randn(2304, 4) * 0.1,
}

# Save adapter
save_peft_adapter(Path("my_adapter.pt"), tensors, {"score": 0.8})

# Create adapter spec
spec = AdapterSpec(
    description="my custom task",
    embedding=encoder.encode("my custom task"),
    tensors=tensors,
    metrics={"score": 0.8},
)
```

### Custom Manifest

```jsonl
{"description": "code generation", "path": "adapters/code.pt", "metrics": {"score": 0.72}}
{"description": "translation", "path": "adapters/translate.pt", "metrics": {"score": 0.48}}
{"description": "creative writing", "path": "adapters/creative.pt", "metrics": {"score": 0.61}}
```

### Custom Evaluation

```python
from true_lora import TrueLoraGenerator, AdapterBank, LoraTensorSpec

# Create bank
bank = AdapterBank(adapters)
specs = [LoraTensorSpec("transformer.h.0.attn.c_attn", 2304, 768, 4)]

# Create generator
model = TrueLoraGenerator(specs, bank)

# Generate
adapter, report = model.generate("your prompt here")
```

## Output Files

```
experiments/gpt2/outputs/
├── training_adapters/
│   ├── adapter_00.pt
│   ├── adapter_01.pt
│   ├── adapter_02.pt
│   ├── adapter_03.pt
│   └── adapter_04.pt
├── generated_python_code_generation_debugging_unit_te.pt
├── generated_japanese_polite_conversation_business_em.pt
├── generated_creative_fantasy_storytelling_world_buil.pt
└── generated_scientific_research_paper_methodology.pt
```
