# Installation

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- (Optional) transformers >= 4.0 for HuggingFace model evaluation
- (Optional) numpy >= 1.24 for fingerprinting

## Install from GitHub (Recommended)

```bash
pip install git+https://github.com/MARVserver/TrueLora.git
```

## Install in Development Mode

```bash
git clone https://github.com/MARVserver/TrueLora.git
cd TrueLora
pip install -e .
```

## Install with Optional Dependencies

```bash
# Full install with HuggingFace support
pip install "true-lora[hf]"

# Full install with all optional dependencies
pip install "true-lora[all]"
```

## Verify Installation

```bash
# Check CLI is available
true-lora --help

# Run smoke tests
python -m pytest tests/ -v
```

## Docker Installation

```dockerfile
FROM python:3.12-slim
RUN pip install torch>=2.0
RUN pip install git+https://github.com/MARVserver/TrueLora.git
```

## System Dependencies

No system-level dependencies are required beyond Python and pip. The package is pure Python + PyTorch.

## Troubleshooting Installation

### torch not found
```bash
pip install torch>=2.0
```

### numpy not found
```bash
pip install numpy>=1.24
```

### Permission denied
```bash
pip install --user git+https://github.com/MARVserver/TrueLora.git
```
