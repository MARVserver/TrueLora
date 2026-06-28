"""LoRA merge utilities for HuggingFace models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from true_lora.adapter import (
    infer_lora_tensor_specs,
    load_torch_state_dict,
    save_peft_adapter,
)
from true_lora.apply import merge_lora_into_linear


def merge_adapter_into_hf_model(
    adapter_path: Path,
    model_name_or_path: str,
    output_dir: Path,
    dtype: str = "float32",
    device: str = "cpu",
    allow_download: bool = False,
) -> dict[str, Any]:
    """Merge LoRA adapter into HuggingFace model and save.

    Args:
        adapter_path: Path to LoRA adapter (.pt file)
        model_name_or_path: HuggingFace model name or local path
        output_dir: Directory to save merged model
        dtype: Output dtype (float32, float16, bfloat16)
        device: Device to use for merging
        allow_download: Allow downloading model from HF Hub

    Returns:
        Report dictionary with merge statistics
    """
    import importlib.util

    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError("transformers is required for HuggingFace model merge")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load adapter
    state_dict = load_torch_state_dict(adapter_path)
    specs = infer_lora_tensor_specs(state_dict)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        local_files_only=not allow_download,
        torch_dtype=_parse_dtype(dtype),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        local_files_only=not allow_download,
    )

    # Move to device
    model = model.to(device)
    state_dict = {k: v.to(device) for k, v in state_dict.items()}

    # Merge
    applied = merge_lora_into_linear(model, state_dict, specs, strict=False)

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return {
        "adapter_path": str(adapter_path),
        "model_name_or_path": model_name_or_path,
        "output_dir": str(output_dir),
        "dtype": dtype,
        "device": device,
        "applied_layers": applied,
        "spec_count": len(specs),
        "tensor_count": len(state_dict),
    }


def merge_adapters(
    adapter_paths: list[Path],
    model_name_or_path: str,
    output_dir: Path,
    weights: list[float] | None = None,
    dtype: str = "float32",
    device: str = "cpu",
    allow_download: bool = False,
) -> dict[str, Any]:
    """Merge multiple LoRA adapters into HuggingFace model.

    Args:
        adapter_paths: List of adapter paths
        model_name_or_path: HuggingFace model name or local path
        output_dir: Directory to save merged model
        weights: Optional weights for each adapter (default: equal weights)
        dtype: Output dtype
        device: Device to use
        allow_download: Allow downloading model from HF Hub

    Returns:
        Report dictionary
    """
    import importlib.util

    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError("transformers is required for HuggingFace model merge")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if weights is None:
        weights = [1.0 / len(adapter_paths)] * len(adapter_paths)
    if len(weights) != len(adapter_paths):
        raise ValueError("weights must have same length as adapter_paths")

    # Load and merge adapters
    merged_state: dict[str, torch.Tensor] = {}
    for path, weight in zip(adapter_paths, weights):
        state_dict = load_torch_state_dict(path)
        for name, tensor in state_dict.items():
            if name in merged_state:
                merged_state[name] = merged_state[name] + tensor.float() * weight
            else:
                merged_state[name] = tensor.float() * weight

    specs = infer_lora_tensor_specs(merged_state)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        local_files_only=not allow_download,
        torch_dtype=_parse_dtype(dtype),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        local_files_only=not allow_download,
    )

    # Move to device
    model = model.to(device)
    merged_state = {k: v.to(device) for k, v in merged_state.items()}

    # Merge
    applied = merge_lora_into_linear(model, merged_state, specs, strict=False)

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return {
        "adapter_paths": [str(p) for p in adapter_paths],
        "weights": weights,
        "model_name_or_path": model_name_or_path,
        "output_dir": str(output_dir),
        "dtype": dtype,
        "device": device,
        "applied_layers": applied,
        "spec_count": len(specs),
    }


def _parse_dtype(dtype: str) -> torch.dtype:
    """Parse dtype string to torch.dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype}. Use float32, float16, or bfloat16")
    return dtype_map[dtype]
