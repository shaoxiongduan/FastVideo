# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the ComfyUI -> FastVideo MiniMax-H3 LoRA conversion."""

import pytest
import torch

from fastvideo.models.dits.minimax_h3_lora import (
    convert_comfy_h3_lora,
    fold_adaln_lora,
    is_comfy_h3_lora,
)

HIDDEN = 8
INNER = 3 * HIDDEN  # fused qkv rows, q then k then v
TIME_EMBED_DIM = 16
LORA_RANK = 4


def _comfy_state_dict() -> dict[str, torch.Tensor]:
    """One main block, one refiner block, and the final layer."""

    def pair(out_dim: int, in_dim: int, tag: str) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(abs(hash(tag)) % (2**31))
        return {
            "lora_A": torch.randn(LORA_RANK, in_dim, generator=generator),
            "lora_B": torch.randn(out_dim, LORA_RANK, generator=generator),
        }

    modules = {
        "blocks.0.attn.qkv_proj": pair(INNER, HIDDEN, "qkv"),
        "blocks.0.attn.out_proj": pair(HIDDEN, HIDDEN, "out"),
        "blocks.0.mlp.fc1": pair(4 * HIDDEN, HIDDEN, "fc1"),
        "blocks.0.mlp.fc2": pair(HIDDEN, 2 * HIDDEN, "fc2"),
        "blocks.0.adaln_proj.linear": pair(6 * 3 * HIDDEN, TIME_EMBED_DIM, "adaln"),
        "final_layer.adaln_proj.linear": pair(2 * HIDDEN, TIME_EMBED_DIM, "final"),
        "token_refiner.blocks.1.attn.qkv_proj": pair(INNER, HIDDEN, "rqkv"),
        "token_refiner.blocks.1.mlp.fc1": pair(4 * HIDDEN, HIDDEN, "rfc1"),
    }
    return {f"diffusion_model.{module}.{kind}.weight": value
            for module, parts in modules.items() for kind, value in parts.items()}


def test_detects_comfy_and_ignores_native_names():
    assert is_comfy_h3_lora(_comfy_state_dict())
    native = {"transformer_blocks.0.attn.to_q.lora_A.weight": torch.zeros(2, 2)}
    assert not is_comfy_h3_lora(native)


def test_renames_cover_every_published_module():
    converted = convert_comfy_h3_lora(_comfy_state_dict())
    modules = {key.rsplit(".lora_", 1)[0] for key in converted}
    assert modules == {
        "transformer_blocks.0.attn.to_q",
        "transformer_blocks.0.attn.to_k",
        "transformer_blocks.0.attn.to_v",
        "transformer_blocks.0.attn.to_out",
        "transformer_blocks.0.ff.fc_in",
        "transformer_blocks.0.ff.fc_out",
        "transformer_blocks.0.adaln_proj.linear",
        "norm_out.linear",
        "token_refiner.refiner_blocks.1.attn.to_q",
        "token_refiner.refiner_blocks.1.attn.to_k",
        "token_refiner.refiner_blocks.1.attn.to_v",
        "token_refiner.refiner_blocks.1.ff.fc_in",
    }


def test_fused_qkv_split_reproduces_the_fused_update():
    state_dict = _comfy_state_dict()
    converted = convert_comfy_h3_lora(state_dict)

    fused_A = state_dict["diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight"]
    fused_B = state_dict["diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight"]
    parts = []
    for name in ("to_q", "to_k", "to_v"):
        part_A = converted[f"transformer_blocks.0.attn.{name}.lora_A.weight"]
        part_B = converted[f"transformer_blocks.0.attn.{name}.lora_B.weight"]
        # every part reuses the fused right factor unchanged
        assert torch.equal(part_A, fused_A)
        assert part_B.shape == (HIDDEN, LORA_RANK)
        parts.append(part_B @ part_A)

    torch.testing.assert_close(torch.cat(parts, dim=0), fused_B @ fused_A)


def test_adaln_fold_is_exact_on_the_basis_subspace():
    """``(W + BA)u`` and the folded rank-reduced form agree when ``u`` is in span(V)."""
    torch.manual_seed(0)
    adaln_rank, out_dim = 5, 12
    basis, _ = torch.linalg.qr(torch.randn(TIME_EMBED_DIM, adaln_rank, dtype=torch.float64))
    u = basis @ torch.randn(adaln_rank, dtype=torch.float64)

    W = torch.randn(out_dim, TIME_EMBED_DIM, dtype=torch.float64)
    A = torch.randn(LORA_RANK, TIME_EMBED_DIM, dtype=torch.float64)
    B = torch.randn(out_dim, LORA_RANK, dtype=torch.float64)

    full_rank = (W + B @ A) @ u
    # the checkpoint stores W V, and adaln_basis.weight holds V transposed
    folded_A = fold_adaln_lora(A, basis.T)
    reduced = (W @ basis + B @ folded_A) @ (basis.T @ u)

    torch.testing.assert_close(reduced, full_rank)


def test_adaln_fold_applies_only_with_a_basis():
    state_dict = _comfy_state_dict()
    basis = torch.randn(5, TIME_EMBED_DIM)

    without = convert_comfy_h3_lora(state_dict)
    with_basis = convert_comfy_h3_lora(state_dict, adaln_basis_weight=basis)

    for module in ("transformer_blocks.0.adaln_proj.linear", "norm_out.linear"):
        assert without[f"{module}.lora_A.weight"].shape == (LORA_RANK, TIME_EMBED_DIM)
        assert with_basis[f"{module}.lora_A.weight"].shape == (LORA_RANK, 5)
        # lora_B is untouched either way: only the input side is reprojected
        torch.testing.assert_close(without[f"{module}.lora_B.weight"], with_basis[f"{module}.lora_B.weight"])

    # a non-AdaLN module keeps its width regardless
    assert with_basis["transformer_blocks.0.ff.fc_in.lora_A.weight"].shape == (LORA_RANK, HIDDEN)


def test_unknown_module_is_rejected():
    with pytest.raises(ValueError, match="No FastVideo parameter matches"):
        convert_comfy_h3_lora({"blocks.0.mystery_proj.lora_A.weight": torch.zeros(2, 2)})


def test_basis_width_mismatch_is_rejected():
    with pytest.raises(ValueError, match="stored basis expects"):
        fold_adaln_lora(torch.zeros(LORA_RANK, TIME_EMBED_DIM), torch.zeros(5, TIME_EMBED_DIM + 1))
