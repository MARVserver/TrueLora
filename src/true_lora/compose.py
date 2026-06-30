"""Compositional LoRA generation (task arithmetic).

Text-to-LoRA generates an adapter for a *single* task description at a time. A
real workload is often a mix -- "Python code, explained in Japanese" -- or a style
delta -- "formal, minus casual". True-LoRA composes these without ever training on
the exact combination, in two complementary ways:

* **embedding-space** (``mode="embedding"``): blend the task embeddings, then run
  the hypernetwork once. This leans on the conditioned hypernetwork's smoothness so
  the composed point lands in a region the network actually reasons about; negative
  weights move *away* from a concept (subtraction).
* **delta-space** (``mode="delta"``): generate each task's LoRA independently, then
  take a weighted sum of the resulting deltas (norm-clipped). This is exact and
  predictable -- the composed adapter is literally the linear combination -- and
  works the same whether the generator is bankless or retrieval-grounded.

Both return a ``(state_dict, report)`` pair shaped like :meth:`TrueLoraGenerator.generate`,
with an extra ``"composition"`` block describing the prompts/weights/mode.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

if TYPE_CHECKING:
    from true_lora.generator import TrueLoraGenerator


def _normalize_weights(prompts: Sequence[str], weights: Sequence[float] | None) -> list[float]:
    if not prompts:
        raise ValueError("compose requires at least one prompt")
    if weights is None:
        return [1.0 / len(prompts)] * len(prompts)
    if len(weights) != len(prompts):
        raise ValueError("weights and prompts must have the same length")
    return [float(w) for w in weights]


def blend_embeddings(
    generator: "TrueLoraGenerator",
    prompts: Sequence[str],
    weights: Sequence[float],
) -> torch.Tensor:
    """Weighted sum of the prompts' embeddings, L2-normalized (encoder contract)."""
    blended: torch.Tensor | None = None
    for prompt, weight in zip(prompts, weights):
        vec = generator.encoder.encode(prompt).float() * weight
        blended = vec if blended is None else blended + vec
    assert blended is not None  # prompts is non-empty (checked by _normalize_weights)
    return blended / max(float(blended.norm()), 1e-8)


def compose_lora(
    generator: "TrueLoraGenerator",
    prompts: Sequence[str],
    weights: Sequence[float] | None = None,
    *,
    mode: str = "embedding",
    retrieval_k: int = 4,
    retrieval_metric: str | None = None,
    metric_weight: float = 0.0,
    min_retrieval_score: float | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Compose several task descriptions into one LoRA.

    Args:
        generator: a :class:`~true_lora.generator.TrueLoraGenerator`.
        prompts: the task descriptions to combine.
        weights: per-prompt weights (default: equal). May be negative to subtract.
        mode: ``"embedding"`` (blend embeddings, generate once) or ``"delta"``
            (generate each, weight-sum the deltas).

    Returns:
        ``(state_dict, report)`` like :meth:`TrueLoraGenerator.generate`, with an
        added ``report["composition"]``.
    """
    weights = _normalize_weights(prompts, weights)
    gen_kwargs = dict(
        retrieval_k=retrieval_k,
        retrieval_metric=retrieval_metric,
        metric_weight=metric_weight,
        min_retrieval_score=min_retrieval_score,
    )

    if mode == "embedding":
        embedding = blend_embeddings(generator, prompts, weights)
        components, report = generator.components_from_embedding(embedding, **gen_kwargs)
        state_dict = components["blended"]

    elif mode == "delta":
        state_dict = _compose_deltas(generator, prompts, weights, gen_kwargs)
        # Report mirrors a single generation but with averaged uncertainty so the
        # downstream gating / reliability code keeps working unchanged.
        uncertainties = []
        for prompt in prompts:
            _, r = generator.generate(prompt, **gen_kwargs)
            uncertainties.append(float(r["uncertainty"]))
        report = {
            "uncertainty": sum(uncertainties) / len(uncertainties),
            "generated_weight": 1.0,
            "abstained": 0.0,
            "retrieved_adapters": [],
        }
    else:
        raise ValueError(f"unknown compose mode {mode!r}; use 'embedding' or 'delta'")

    report["composition"] = {
        "prompts": list(prompts),
        "weights": list(weights),
        "mode": mode,
    }
    return state_dict, report


def _compose_deltas(
    generator: "TrueLoraGenerator",
    prompts: Sequence[str],
    weights: Sequence[float],
    gen_kwargs: dict,
) -> dict[str, torch.Tensor]:
    accumulated: dict[str, torch.Tensor] = {}
    for prompt, weight in zip(prompts, weights):
        state_dict, _ = generator.generate(prompt, **gen_kwargs)
        for name, tensor in state_dict.items():
            contribution = tensor.float() * weight
            if name in accumulated:
                accumulated[name] = accumulated[name] + contribution
            else:
                accumulated[name] = contribution
    # Clip the summed deltas to the same per-tensor norm budget as a single generation.
    return {name: generator._clip_norm(tensor) for name, tensor in accumulated.items()}


def task_arithmetic(
    generator: "TrueLoraGenerator",
    add: Sequence[str],
    subtract: Sequence[str] = (),
    *,
    add_weight: float = 1.0,
    subtract_weight: float = 1.0,
    mode: str = "embedding",
    **kwargs,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Add some task descriptions and subtract others (e.g. ``formal - casual``).

    Convenience over :func:`compose_lora` with signed weights: every prompt in
    ``add`` gets ``+add_weight``, every prompt in ``subtract`` gets ``-subtract_weight``.
    """
    if not add:
        raise ValueError("task_arithmetic requires at least one prompt to add")
    prompts = list(add) + list(subtract)
    weights = [add_weight] * len(add) + [-subtract_weight] * len(subtract)
    return compose_lora(generator, prompts, weights, mode=mode, **kwargs)
