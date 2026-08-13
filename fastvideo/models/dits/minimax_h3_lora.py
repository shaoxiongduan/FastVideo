# SPDX-License-Identifier: Apache-2.0
"""Adapt ComfyUI-format MiniMax-H3 LoRA adapters to FastVideo parameter names.

The published H3 adapters target the ComfyUI repack, which fuses Q/K/V into one
projection and names the blocks differently from the diffusers-style checkpoint
FastVideo loads. Two of the required edits cannot be written as the regex rename
that ``lora_param_names_mapping`` supports, which is why this is a function:

* the fused ``attn.qkv_proj`` update becomes three updates that share ``lora_A``
  and slice ``lora_B`` by output rows;
* against a rank-reduced checkpoint the AdaLN ``lora_A`` has to be projected
  into the stored basis, see :func:`fold_adaln_lora`.

Everything else is a rename. The two AdaLN layouts already agree: both stacks
build the modulation table as ``[timestep, modality, 6, hidden]`` and both split
the fused QKV rows in q, k, v order, so no tensor is permuted here.
"""

from __future__ import annotations

import re
from collections import defaultdict

import torch

# Comfy module path -> FastVideo module path. The text-refiner patterns are
# tried first; the main-block patterns would otherwise also match them.
_RENAMES: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"^token_refiner\.blocks\.(\d+)\.attn\.out_proj$"), r"token_refiner.refiner_blocks.\1.attn.to_out"),
    (re.compile(r"^token_refiner\.blocks\.(\d+)\.mlp\.fc1$"), r"token_refiner.refiner_blocks.\1.ff.fc_in"),
    (re.compile(r"^token_refiner\.blocks\.(\d+)\.mlp\.fc2$"), r"token_refiner.refiner_blocks.\1.ff.fc_out"),
    (re.compile(r"^blocks\.(\d+)\.attn\.out_proj$"), r"transformer_blocks.\1.attn.to_out"),
    (re.compile(r"^blocks\.(\d+)\.mlp\.fc1$"), r"transformer_blocks.\1.ff.fc_in"),
    (re.compile(r"^blocks\.(\d+)\.mlp\.fc2$"), r"transformer_blocks.\1.ff.fc_out"),
    (re.compile(r"^blocks\.(\d+)\.adaln_proj\.linear$"), r"transformer_blocks.\1.adaln_proj.linear"),
    (re.compile(r"^final_layer\.adaln_proj\.linear$"), r"norm_out.linear"),
)

# Fused projections, mapped to the attention module holding their three parts.
_FUSED_QKV: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"^token_refiner\.blocks\.(\d+)\.attn\.qkv_proj$"), r"token_refiner.refiner_blocks.\1.attn"),
    (re.compile(r"^blocks\.(\d+)\.attn\.qkv_proj$"), r"transformer_blocks.\1.attn"),
)

# Comfy splits the fused projection as ``qkv.split(inner, dim=-1)``, so the
# output rows run q, then k, then v.
_QKV_PARTS = ("to_q", "to_k", "to_v")

# FastVideo modules whose input is the AdaLN time-embedding curve, and so are
# reprojected when the checkpoint carries a basis.
_ADALN_TARGETS = (".adaln_proj.linear", "norm_out.linear")

_ENTRY = re.compile(r"^(?P<module>.+)\.(?P<kind>lora_A|lora_B|lora_alpha|alpha)(?:\.weight)?$")
_STRIP_PREFIXES = ("diffusion_model.", "transformer.")


def _strip_prefix(name: str) -> str:
    for prefix in _STRIP_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _match(module: str, table: tuple[tuple[re.Pattern, str], ...]) -> str | None:
    for pattern, replacement in table:
        if pattern.match(module):
            return pattern.sub(replacement, module)
    return None


def is_comfy_h3_lora(state_dict: dict[str, torch.Tensor]) -> bool:
    """Does this adapter use the ComfyUI H3 parameter names?

    Adapters already written against FastVideo's names are left alone, so a
    future native adapter keeps working without a flag.
    """
    for key in state_dict:
        entry = _ENTRY.match(_strip_prefix(key))
        if entry is None:
            continue
        module = entry.group("module")
        if _match(module, _FUSED_QKV) is not None or _match(module, _RENAMES) is not None:
            return True
    return False


def fold_adaln_lora(lora_A: torch.Tensor, basis_weight: torch.Tensor) -> torch.Tensor:
    """Project an AdaLN ``lora_A`` into a rank-reduced checkpoint's stored basis.

    The released projection consumes ``u = silu(temb)`` directly, so the adapted
    layer computes ``(W + BA) u``. A rank-reduced checkpoint stores ``W V`` and
    feeds the block ``Vᵀu`` instead, so the same update is reproduced by
    ``A' = A V``::

        (W + B A) u  ==  (W V + B (A V)) (Vᵀ u)

    The two sides agree to the basis's own reconstruction error, since the fit
    makes ``V Vᵀ u == u`` over the timestep curve and no published adapter
    touches the time embedder that produces ``u``.

    Args:
        lora_A: ``[lora_rank, time_embed_dim]`` adapter factor.
        basis_weight: the checkpoint's ``adaln_basis.weight``, which stores
            ``Vᵀ`` with shape ``[adaln_rank, time_embed_dim]``.

    Returns:
        ``[lora_rank, adaln_rank]``, in ``lora_A``'s dtype.
    """
    if lora_A.shape[1] != basis_weight.shape[1]:
        raise ValueError(f"AdaLN lora_A has width {lora_A.shape[1]}, but the stored basis expects "
                         f"{basis_weight.shape[1]}.")
    # Contract in at least fp32, since both factors are fp16/bf16 on disk;
    # promote rather than force, so a wider input keeps its precision.
    compute_dtype = torch.promote_types(torch.promote_types(lora_A.dtype, basis_weight.dtype), torch.float32)
    # The adapter is freshly loaded on CPU while the basis already sits on the
    # worker's GPU, so contract on the basis's device and hand the result back
    # where the adapter came from.
    folded = (lora_A.to(device=basis_weight.device, dtype=compute_dtype) @ basis_weight.to(compute_dtype).T)
    return folded.to(device=lora_A.device, dtype=lora_A.dtype)


def _emit(out: dict[str, torch.Tensor], target: str, parts: dict[str, torch.Tensor],
          basis_weight: torch.Tensor | None) -> None:
    for kind, weight in parts.items():
        if kind in ("lora_alpha", "alpha"):
            out[f"{target}.lora_alpha"] = weight
            continue
        if (kind == "lora_A" and basis_weight is not None and target.endswith(_ADALN_TARGETS)):
            weight = fold_adaln_lora(weight, basis_weight)
        out[f"{target}.{kind}.weight"] = weight


def _emit_fused_qkv(out: dict[str, torch.Tensor], attn_module: str, parts: dict[str, torch.Tensor],
                    source: str) -> None:
    lora_A = parts.get("lora_A")
    lora_B = parts.get("lora_B")
    if lora_A is None or lora_B is None:
        raise ValueError(f"Fused QKV adapter {source!r} needs both lora_A and lora_B, got {sorted(parts)}.")
    rows = lora_B.shape[0]
    if rows % len(_QKV_PARTS):
        raise ValueError(f"Fused QKV adapter {source!r} has {rows} output rows, which is not divisible by "
                         f"{len(_QKV_PARTS)}.")
    inner = rows // len(_QKV_PARTS)
    alpha = parts.get("lora_alpha", parts.get("alpha"))
    for index, part in enumerate(_QKV_PARTS):
        target = f"{attn_module}.{part}"
        # lora_A is shared by construction: the fused update is B @ A, so every
        # row block sees the same right factor.
        out[f"{target}.lora_A.weight"] = lora_A
        out[f"{target}.lora_B.weight"] = lora_B[index * inner:(index + 1) * inner].clone()
        if alpha is not None:
            out[f"{target}.lora_alpha"] = alpha


def convert_comfy_h3_lora(
    state_dict: dict[str, torch.Tensor],
    *,
    adaln_basis_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Rewrite a ComfyUI H3 adapter against FastVideo's parameter names.

    Args:
        state_dict: the adapter as published.
        adaln_basis_weight: ``adaln_basis.weight`` when the target checkpoint is
            rank-reduced, else ``None``. Supplying it folds the AdaLN adapters
            into the basis so one published adapter serves both checkpoints.

    Returns:
        A state dict keyed the way :meth:`LoRAPipeline.set_lora_adapter` expects.
    """
    grouped: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for key, weight in state_dict.items():
        entry = _ENTRY.match(_strip_prefix(key))
        if entry is None:
            raise ValueError(f"Unrecognized MiniMax-H3 LoRA key {key!r}: expected a lora_A/lora_B/alpha suffix.")
        grouped[entry.group("module")][entry.group("kind")] = weight

    converted: dict[str, torch.Tensor] = {}
    for module, parts in grouped.items():
        attn_module = _match(module, _FUSED_QKV)
        if attn_module is not None:
            _emit_fused_qkv(converted, attn_module, parts, module)
            continue
        target = _match(module, _RENAMES)
        if target is None:
            raise ValueError(f"No FastVideo parameter matches the MiniMax-H3 LoRA module {module!r}. The adapter "
                             f"may target a different model or a newer ComfyUI layout.")
        _emit(converted, target, parts, adaln_basis_weight)
    return converted
