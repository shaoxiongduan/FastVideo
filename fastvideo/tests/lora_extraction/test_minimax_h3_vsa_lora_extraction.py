"""CPU coverage for MiniMax-H3 mixed LoRA/dense-gate extraction."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

_REPO_ROOT = Path(__file__).parents[3]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "lora_extraction" / "extract_minimax_h3_lora.py"
_SPEC = importlib.util.spec_from_file_location("extract_minimax_h3_lora", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
extract_minimax_h3_loras = _MODULE.extract_minimax_h3_loras


def test_dcp_fastvideo_keys_map_to_diffusers_namespace() -> None:
    mapping = {
        "time_embedder.fc_in.weight": "time_embedder.linear_1.weight",
        "time_embedder.fc_out.bias": "time_embedder.linear_2.bias",
        "transformer_blocks.3.attn.to_out.weight": "transformer_blocks.3.attn.to_out.0.weight",
        "transformer_blocks.4.ff.fc_in.weight": "transformer_blocks.4.ff.net.0.proj.weight",
        "transformer_blocks.5.ff.fc_out.weight": "transformer_blocks.5.ff.net.2.weight",
        "transformer_blocks.6.attn.to_gate_compress.weight":
        "transformer_blocks.6.attn.to_gate_compress.weight",
    }
    for source, target in mapping.items():
        assert _MODULE._fastvideo_to_diffusers_key(source) == target


def _write_transformer(root: Path, state: dict[str, torch.Tensor]) -> None:
    transformer = root / "transformer"
    transformer.mkdir(parents=True)
    shard = "diffusion_pytorch_model-00001-of-00001.safetensors"
    save_file(state, str(transformer / shard))
    index = {
        "metadata": {
            "total_size": sum(tensor.numel() * tensor.element_size() for tensor in state.values())
        },
        "weight_map": {key: shard for key in state},
    }
    (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8")


def _toy_states() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(7)
    base = {
        "context_embedder.weight": torch.randn(6, 5, generator=generator),
        "norm_out.linear.weight": torch.randn(5, 4, generator=generator),
        "transformer_blocks.0.attn.to_q.weight": torch.randn(7, 6, generator=generator),
        "transformer_blocks.0.norm1.weight": torch.randn(6, generator=generator),
    }
    finetuned = {key: tensor.clone() for key, tensor in base.items()}
    finetuned["context_embedder.weight"] += torch.randn(6, 2, generator=generator) @ torch.randn(
        2, 5, generator=generator)
    finetuned["norm_out.linear.weight"] += torch.randn(5, 2, generator=generator) @ torch.randn(
        2, 4, generator=generator)
    finetuned["transformer_blocks.0.attn.to_q.weight"] += torch.randn(
        7, 2, generator=generator) @ torch.randn(2, 6, generator=generator)
    finetuned["transformer_blocks.0.norm1.weight"] += 0.25
    finetuned["transformer_blocks.0.attn.to_gate_compress.weight"] = torch.randn(
        7, 6, generator=generator)
    return base, finetuned


def _reconstructed_delta(adapter: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    module_name = key.removesuffix(".weight")
    return adapter[f"{module_name}.lora_B.weight"] @ adapter[f"{module_name}.lora_A.weight"]


def test_extracts_nested_ranks_and_exact_dense_auxiliary_tensors(tmp_path: Path) -> None:
    base, finetuned = _toy_states()
    base_dir = tmp_path / "base"
    finetuned_dir = tmp_path / "finetuned"
    output_dir = tmp_path / "output"
    _write_transformer(base_dir, base)
    _write_transformer(finetuned_dir, finetuned)

    checkpoints = extract_minimax_h3_loras(
        base=str(base_dir),
        finetuned=str(finetuned_dir),
        output_dir=str(output_dir),
        ranks=(1, 2),
        device="cpu",
        oversample=1,
        niter=2,
        factor_dtype="float32",
        output_dtype="float32",
        expected_gate_count=1,
    )

    assert checkpoints == [
        output_dir / "rank-1" / "adapter_model.safetensors",
        output_dir / "rank-2" / "adapter_model.safetensors",
    ]
    rank_one = load_file(str(checkpoints[0]))
    rank_two = load_file(str(checkpoints[1]))
    gate_key = "transformer_blocks.0.attn.to_gate_compress.weight"
    assert torch.equal(rank_one[gate_key], finetuned[gate_key])
    assert torch.equal(rank_two[gate_key], finetuned[gate_key])

    # Boundary projections and one-dimensional state are cheap enough to keep
    # exact. Only the large block matrix is represented as LoRA factors.
    assert torch.equal(rank_two["norm_out.linear.weight"], finetuned["norm_out.linear.weight"])
    assert torch.equal(rank_two["context_embedder.weight"], finetuned["context_embedder.weight"])
    assert torch.equal(rank_two["transformer_blocks.0.norm1.weight"],
                       finetuned["transformer_blocks.0.norm1.weight"])
    assert "transformer_blocks.0.attn.to_q.lora_A.weight" in rank_two

    key = "transformer_blocks.0.attn.to_q.weight"
    expected_delta = finetuned[key] - base[key]
    error_one = torch.linalg.vector_norm(expected_delta - _reconstructed_delta(rank_one, key))
    error_two = torch.linalg.vector_norm(expected_delta - _reconstructed_delta(rank_two, key))
    assert error_two <= error_one + 1e-5
    torch.testing.assert_close(_reconstructed_delta(rank_two, key), expected_delta, atol=2e-5, rtol=2e-5)

    with safe_open(checkpoints[1], framework="pt") as checkpoint:
        assert checkpoint.metadata()["dense_tensor_policy"].startswith("full VSA gates")
        assert checkpoint.metadata()["requested_rank"] == "2"


def test_rejects_unrecognized_finetuned_only_tensor(tmp_path: Path) -> None:
    base, finetuned = _toy_states()
    finetuned["unexpected.weight"] = torch.ones(2, 2)
    base_dir = tmp_path / "base"
    finetuned_dir = tmp_path / "finetuned"
    _write_transformer(base_dir, base)
    _write_transformer(finetuned_dir, finetuned)

    with pytest.raises(ValueError, match="unexpected tensors besides VSA gates"):
        extract_minimax_h3_loras(
            base=str(base_dir),
            finetuned=str(finetuned_dir),
            output_dir=str(tmp_path / "output"),
            ranks=(2, ),
            device="cpu",
            expected_gate_count=1,
        )
