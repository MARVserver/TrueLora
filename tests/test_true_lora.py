import torch

from true_lora.adapter import AdapterBank, AdapterSpec, LoraTensorSpec
from true_lora.generator import (
    ConditionedHyperAdapter,
    HyperAdapter,
    TrueLoraGenerator,
    layer_index,
    module_key,
)
from true_lora.text import HashingTextEncoder, SemanticTextEncoder
from true_lora.train import train_on_adapter_bank


def make_bank():
    encoder = HashingTextEncoder(dim=32)
    tensors_a = {
        "layer.lora_A.weight": torch.ones(2, 4),
        "layer.lora_B.weight": torch.ones(4, 2),
    }
    tensors_b = {
        "layer.lora_A.weight": -torch.ones(2, 4),
        "layer.lora_B.weight": -torch.ones(4, 2),
    }
    adapters = [
        AdapterSpec("math reasoning", encoder.encode("math reasoning"), tensors_a),
        AdapterSpec("translation writing", encoder.encode("translation writing"), tensors_b),
    ]
    return encoder, AdapterBank(adapters)


def test_retrieval_prefers_matching_description():
    encoder, bank = make_bank()
    adapters, weights = bank.retrieve(encoder.encode("math reasoning algebra"), k=2)
    assert adapters[0].description == "math reasoning"
    assert weights[0] > weights[1]


def test_generator_returns_clipped_lora_tensors():
    _, bank = make_bank()
    specs = [LoraTensorSpec("layer", out_features=4, in_features=4, rank=2)]
    model = TrueLoraGenerator(specs, bank, text_dim=32, hidden_dim=16, max_tensor_norm=0.5)
    state_dict, report = model.generate("math reasoning", retrieval_k=2)

    assert set(state_dict) == {"layer.lora_A.weight", "layer.lora_B.weight"}
    assert all(tensor.norm() <= 0.5001 for tensor in state_dict.values())
    assert 0.0 <= report["uncertainty"] <= 1.0


def make_gqa_specs(layers, ranks=4):
    # q_proj keeps full width, v_proj is narrower (grouped-query attention).
    specs = []
    for li in layers:
        specs.append(LoraTensorSpec(f"model.layers.{li}.self_attn.q_proj", 16, 16, ranks))
        specs.append(LoraTensorSpec(f"model.layers.{li}.self_attn.v_proj", 4, 16, ranks))
    return specs


def make_conditioned_bank(specs, encoder):
    descriptions = [("python code generation", 0.2), ("japanese translation", -0.2)]
    adapters = []
    for desc, scale in descriptions:
        tensors = {}
        for spec in specs:
            tensors[f"{spec.name}.lora_A.weight"] = torch.full(spec.a_shape, scale)
            tensors[f"{spec.name}.lora_B.weight"] = torch.full(spec.b_shape, scale / 2)
        adapters.append(AdapterSpec(desc, encoder.encode(desc), tensors, metrics={"score": 0.5}))
    return AdapterBank(adapters), adapters


def test_module_key_and_layer_index_parsing():
    assert module_key("model.layers.18.self_attn.q_proj") == "model.layers.{}.self_attn.q_proj"
    assert module_key("model.layers.0.self_attn.q_proj") == module_key("model.layers.18.self_attn.q_proj")
    assert layer_index("model.layers.18.self_attn.q_proj") == 18
    assert layer_index("layer") == 0


def test_conditioned_generator_handles_mixed_shapes():
    encoder = HashingTextEncoder(dim=32)
    specs = make_gqa_specs([0, 6, 12])
    bank, _ = make_conditioned_bank(specs, encoder)
    model = TrueLoraGenerator(
        specs, bank, text_dim=32, hidden_dim=16, max_tensor_norm=0.5,
        encoder=encoder, hyper_kind="conditioned",
    )
    state_dict, report = model.generate("python code generation", retrieval_k=2)

    # v_proj (narrow) and q_proj (full) deltas keep their distinct shapes.
    assert state_dict["model.layers.0.self_attn.v_proj.lora_B.weight"].shape == (4, 4)
    assert state_dict["model.layers.0.self_attn.q_proj.lora_B.weight"].shape == (16, 4)
    assert all(tensor.norm() <= 0.5001 for tensor in state_dict.values())
    assert 0.0 <= report["uncertainty"] <= 1.0


def test_conditioned_hypernetwork_scales_with_module_types_not_layers():
    # Many layers, two module types. The conditioned net shares heads across layers,
    # so it stays far smaller than the flat net whose output grows with total size.
    specs = make_gqa_specs(list(range(24)))
    conditioned = ConditionedHyperAdapter(32, 64, specs)
    flat = HyperAdapter(32, 64, specs)
    cond_params = sum(p.numel() for p in conditioned.parameters())
    flat_params = sum(p.numel() for p in flat.parameters())
    assert cond_params < flat_params


def test_conditioned_hypernetwork_trains():
    torch.manual_seed(0)
    encoder = HashingTextEncoder(dim=32)
    specs = make_gqa_specs([0, 6])
    bank, adapters = make_conditioned_bank(specs, encoder)
    model = TrueLoraGenerator(
        specs, bank, text_dim=32, hidden_dim=32, max_tensor_norm=4.0,
        encoder=encoder, hyper_kind="conditioned",
    )
    losses = train_on_adapter_bank(model, adapters, steps=60, lr=1e-2)
    assert losses[-1] < losses[0]


def test_semantic_encoder_interface():
    encoder = SemanticTextEncoder(fallback_dim=48)
    vector = encoder.encode("python code generation")
    assert vector.shape == (encoder.dim,)
    # Vectors are L2-normalized for the cosine-similarity retrieval contract.
    assert abs(float(vector.norm()) - 1.0) < 1e-4
    assert encoder.backend in {"sentence-transformers", "hashing-fallback"}
