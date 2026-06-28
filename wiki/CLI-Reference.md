# CLI Reference

True-LoRA provides a comprehensive CLI with 15 commands.

## General Usage

```bash
true-lora <command> [options]
```

## Commands

### `demo`

Generate a LoRA adapter from a text prompt (quick demo).

```bash
true-lora demo \
  --prompt "python code generation debugging" \
  --out adapter.pt \
  --steps 80 \
  --k 2 \
  --max-norm 4.0 \
  --seed 42
```

| Option | Default | Description |
|--------|---------|-------------|
| `--prompt` | "write reliable python tests for debugging" | Text prompt |
| `--out` | true-lora-demo.pt | Output adapter path |
| `--steps` | 80 | Training steps |
| `--k` | 2 | Retrieval top-k |
| `--max-norm` | 4.0 | Maximum tensor norm |
| `--seed` | None | Random seed |

### `train`

Train the hypernetwork on an adapter bank.

```bash
true-lora train \
  --manifest adapters.jsonl \
  --out checkpoint.pt \
  --steps 500 \
  --lr 1e-3 \
  --text-dim 256 \
  --hidden-dim 512
```

| Option | Default | Description |
|--------|---------|-------------|
| `--manifest` | (required) | Adapter manifest JSONL |
| `--out` | (required) | Output checkpoint path |
| `--steps` | 500 | Training steps |
| `--lr` | 1e-3 | Learning rate |
| `--text-dim` | 256 | Text encoder dimension |
| `--hidden-dim` | 512 | Hidden layer dimension |

### `generate`

Generate a LoRA adapter using a trained checkpoint.

```bash
true-lora generate \
  --manifest adapters.jsonl \
  --checkpoint checkpoint.pt \
  --prompt "creative writing storytelling" \
  --out generated.pt \
  --k 4
```

### `eval`

Evaluate reconstruction quality on the adapter bank.

```bash
true-lora eval \
  --manifest adapters.jsonl \
  --checkpoint checkpoint.pt \
  --k 4 \
  --ablation
```

### `gate`

Quality gate: evaluate adapter on benchmark and accept/reject.

```bash
true-lora gate \
  --adapter generated.pt \
  --hf-generation-benchmark benchmark.jsonl \
  --model gpt2 \
  --min-accuracy-delta 0.1 \
  --max-uncertainty 0.8
```

| Option | Default | Description |
|--------|---------|-------------|
| `--adapter` | (required) | Adapter path |
| `--benchmark` | None | Local benchmark JSONL |
| `--hf-benchmark` | None | HF classification benchmark |
| `--hf-generation-benchmark` | None | HF generation benchmark |
| `--model` | "" | HF model name |
| `--min-accuracy-delta` | 0.0 | Min accuracy improvement |
| `--max-uncertainty` | 0.8 | Max uncertainty threshold |
| `--max-tensor-norm` | 8.0 | Max tensor norm |

### `merge-adapter`

Merge LoRA adapter into base model weights and save.

```bash
true-lora merge-adapter \
  --adapter generated.pt \
  --model gpt2 \
  --out merged_model/ \
  --dtype float16
```

### `export-peft`

Export adapter to HuggingFace PEFT format.

```bash
true-lora export-peft \
  --adapter generated.pt \
  --out peft_output/ \
  --base-model gpt2
```

### `inspect-peft`

Inspect a PEFT directory.

```bash
true-lora inspect-peft --adapter-dir peft_output/
```

### `validate-manifest`

Validate an adapter manifest file.

```bash
true-lora validate-manifest \
  --manifest adapters.jsonl \
  --required-metric score
```

### `bank-summary`

Show summary statistics for an adapter bank.

```bash
true-lora bank-summary --manifest adapters.jsonl
```

### `prompt-consistency`

Analyze consistency across similar prompts.

```bash
true-lora prompt-consistency \
  --manifest adapters.jsonl \
  --prompts prompts.jsonl \
  --checkpoint checkpoint.pt
```

### `prompt-sensitivity`

Analyze sensitivity to prompt variations.

```bash
true-lora prompt-sensitivity \
  --manifest adapters.jsonl \
  --contrasts contrasts.jsonl \
  --checkpoint checkpoint.pt
```

### `pipeline`

Full pipeline: train → generate → evaluate → gate → export.

```bash
true-lora pipeline \
  --manifest adapters.jsonl \
  --benchmark benchmark.jsonl \
  --prompt "code generation" \
  --adapter-out generated.pt \
  --export-dir peft_output/ \
  --train-steps 200 \
  --base-model gpt2
```

### `loo-eval`

Leave-one-out cross-validation.

```bash
true-lora loo-eval \
  --manifest adapters.jsonl \
  --steps 100 \
  --k 4
```

### `compare-reports`

Compare multiple JSON reports by metric.

```bash
true-lora compare-reports report1.json report2.json --metric accuracy_delta
```

### `audit`

Audit multiple reports against quality thresholds.

```bash
true-lora audit report1.json report2.json --min-accuracy-delta 0.1
```

## Global Options

All commands support:

| Option | Description |
|--------|-------------|
| `--report-out PATH` | Save JSON report to file |
| `--seed INT` | Random seed for reproducibility |
