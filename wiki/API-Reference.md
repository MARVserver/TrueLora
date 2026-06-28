# API Reference

## Core Classes

### `LoraTensorSpec`

```python
@dataclass(frozen=True)
class LoraTensorSpec:
    name: str              # Module name (e.g., "transformer.h.0.attn.c_attn")
    out_features: int      # Output dimension
    in_features: int       # Input dimension
    rank: int              # LoRA rank
    alpha: float = 1.0     # LoRA alpha scaling
```

**Properties**:
- `a_shape -> tuple[int, int]`: Shape of LoRA A matrix `(rank, in_features)`
- `b_shape -> tuple[int, int]`: Shape of LoRA B matrix `(out_features, rank)`

---

### `AdapterSpec`

```python
@dataclass
class AdapterSpec:
    description: str                    # Text description
    embedding: torch.Tensor             # Text embedding vector
    tensors: dict[str, torch.Tensor]    # LoRA tensors
    metrics: dict[str, float] | None    # Quality metrics
    source: str | None                  # Source file path
    fingerprint: str | None             # SHA256 fingerprint
```

---

### `AdapterBank`

```python
class AdapterBank:
    def __init__(self, adapters: Iterable[AdapterSpec]) -> None:
        """Create adapter bank from list of adapters."""
    
    def score(
        self,
        query_embedding: torch.Tensor,
        metric: str | None = None,
        metric_weight: float = 0.0,
    ) -> torch.Tensor:
        """Score all adapters against query."""
    
    def retrieve(
        self,
        query_embedding: torch.Tensor,
        k: int = 4,
        metric: str | None = None,
        metric_weight: float = 0.0,
    ) -> tuple[list[AdapterSpec], torch.Tensor]:
        """Retrieve top-k adapters with softmax weights."""
    
    def retrieve_with_max_score(
        self,
        query_embedding: torch.Tensor,
        k: int = 4,
        metric: str | None = None,
        metric_weight: float = 0.0,
    ) -> tuple[list[AdapterSpec], torch.Tensor, torch.Tensor]:
        """Retrieve top-k and return all scores (avoids double computation)."""
    
    def interpolate(
        self,
        query_embedding: torch.Tensor,
        k: int = 4,
        metric: str | None = None,
        metric_weight: float = 0.0,
    ) -> tuple[dict[str, torch.Tensor], float]:
        """Retrieve and interpolate adapters."""
    
    def interpolate_retrieved(
        self,
        adapters: list[AdapterSpec],
        weights: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], float]:
        """Interpolate pre-retrieved adapters (batched)."""
    
    @staticmethod
    def retrieval_provenance(
        adapters: list[AdapterSpec],
        weights: torch.Tensor,
    ) -> list[dict[object]]:
        """Get provenance info for retrieved adapters."""
```

---

### `TrueLoraGenerator`

```python
class TrueLoraGenerator:
    def __init__(
        self,
        tensor_specs: list[LoraTensorSpec],
        adapter_bank: AdapterBank,
        text_dim: int = 256,
        hidden_dim: int = 512,
        max_tensor_norm: float = 1.0,
        ood_shrink_factor: float = 0.25,
    ) -> None:
        """Create TrueLoraGenerator."""
    
    def generate(
        self,
        prompt: str,
        retrieval_k: int = 4,
        retrieval_metric: str | None = None,
        metric_weight: float = 0.0,
        min_retrieval_score: float | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
        """Generate blended LoRA adapter from prompt."""
    
    def generate_components(
        self,
        prompt: str,
        retrieval_k: int = 4,
        retrieval_metric: str | None = None,
        metric_weight: float = 0.0,
        min_retrieval_score: float | None = None,
    ) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, object]]:
        """Generate all components (blended, retrieval, generated)."""
```

---

## Text Encoding

### `HashingTextEncoder`

```python
class HashingTextEncoder:
    def __init__(self, dim: int = 256) -> None:
        """Create encoder with output dimension."""
    
    def encode(self, text: str) -> torch.Tensor:
        """Encode text to normalized vector of shape (dim,)."""
```

---

## LoRA Application

### `lora_delta`

```python
def lora_delta(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float | None = None,
) -> torch.Tensor:
    """Compute LoRA delta: (B @ A) * (alpha / rank)"""
```

### `merge_lora_into_linear`

```python
def merge_lora_into_linear(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    specs: list[LoraTensorSpec],
    strict: bool = True,
) -> list[str]:
    """Merge LoRA into linear layers permanently."""
```

### `temporary_lora`

```python
@contextmanager
def temporary_lora(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    specs: list[LoraTensorSpec],
    strict: bool = True,
) -> Iterator[list[str]]:
    """Context manager that applies LoRA temporarily."""
```

---

## Adapter I/O

### `save_peft_adapter`

```python
def save_peft_adapter(
    path: Path,
    state_dict: dict[str, torch.Tensor],
    report: dict[str, object] | None = None,
) -> None:
    """Save adapter in True-LoRA format."""
```

### `save_peft_directory`

```python
def save_peft_directory(
    path: Path,
    state_dict: dict[str, torch.Tensor],
    specs: list[LoraTensorSpec],
    report: dict[str, object] | None = None,
    base_model_name_or_path: str = "",
) -> None:
    """Save adapter in HuggingFace PEFT directory format."""
```

### `load_adapter_manifest`

```python
def load_adapter_manifest(
    path: Path,
    encoder: HashingTextEncoder,
) -> tuple[list[LoraTensorSpec], AdapterBank, list[AdapterSpec]]:
    """Load adapter bank from JSONL manifest."""
```

### `infer_lora_tensor_specs`

```python
def infer_lora_tensor_specs(
    state_dict: dict[str, torch.Tensor],
) -> list[LoraTensorSpec]:
    """Infer LoRA tensor specs from state dict."""
```

---

## Quality

### `QualityGate`

```python
@dataclass(frozen=True)
class QualityGate:
    min_accuracy_delta: float = 0.0
    max_uncertainty: float = 0.8
    max_tensor_norm: float = 8.0
    max_consistency_mse: float | None = None
    min_prompt_sensitivity_mse: float | None = None
    min_retrieval_score_delta: float | None = None
```

### `gate_adapter`

```python
def gate_adapter(
    state_dict: dict[str, torch.Tensor],
    eval_report: dict[str, float],
    generation_report: dict[str, float] | None = None,
    consistency_report: dict[str, float] | None = None,
    sensitivity_report: dict[str, float] | None = None,
    gate: QualityGate | None = None,
) -> dict[str, float | bool | str]:
    """Apply quality gate to adapter."""
```

---

## Evaluation

### `evaluate_hf_causal_lm_generation`

```python
def evaluate_hf_causal_lm_generation(
    model_name_or_path: str,
    state_dict: dict[str, torch.Tensor],
    dataset_path: Path,
    max_new_tokens: int = 32,
    max_length: int = 512,
    device: str = "cpu",
    local_files_only: bool = True,
) -> dict[str, float]:
    """Evaluate adapter on HuggingFace causal LM."""
```

### `evaluate_hf_sequence_classification`

```python
def evaluate_hf_sequence_classification(
    model_name_or_path: str,
    state_dict: dict[str, torch.Tensor],
    dataset_path: Path,
    batch_size: int = 8,
    max_length: int = 256,
    device: str = "cpu",
    local_files_only: bool = True,
) -> dict[str, float]:
    """Evaluate adapter on HuggingFace sequence classifier."""
```

---

## Reproducibility

### `set_seed`

```python
def set_seed(seed: int | None) -> int | None:
    """Set random seed for reproducibility."""
```
