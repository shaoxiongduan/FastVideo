# SPDX-License-Identifier: Apache-2.0
"""Convert a lightx2v Minimax-h3-Turbo LoRA into the key layout FastVideo loads.

lightx2v publishes MiniMax-H3 turbo LoRAs in the ``minimax-h3-diffusers`` key format,
which names each tensor::

    transformer_blocks.0.attn.to_q.lora_A.default.weight

FastVideo's loader (``LoRAPipeline.set_lora_adapter``) strips ``diffusion_model.`` and
``.weight``, runs the remainder through the model's ``param_names_mapping``, and then looks
the result up as ``<layer>.lora_A`` / ``<layer>.lora_B``. The PEFT-style ``.default`` adapter
infix survives every one of those steps, so every lookup misses and **the LoRA silently
loads zero layers** — no error, just an unchanged model. Dropping ``.default`` is the whole
fix; the rest of the key layout already matches, because H3's ``param_names_mapping`` rewrites
``to_out.0 -> to_out``, ``ff.net.0.proj -> ff.fc_in`` and ``ff.net.2 -> ff.fc_out``.

The alpha lives in the file's ``__metadata__`` rather than in per-layer ``.lora_alpha``
tensors, so it is materialized here as one scalar tensor per adapted layer. Without it
FastVideo falls back to its own default scaling, which is not necessarily ``alpha/rank``.

    python scripts/checkpoint_conversion/convert_minimax_h3_lightx2v_lora.py \\
        --input  minimax_h3_fl2v_turbo_4step_v1.1_768p_bf16.safetensors \\
        --output minimax_h3_fl2v_turbo_4step_v1.1_768p_fastvideo.safetensors
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ADAPTER_INFIX = ".default."


def convert(state: dict[str, torch.Tensor], alpha: float | None) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    layers: set[str] = set()
    for key, tensor in state.items():
        new_key = key.replace(ADAPTER_INFIX, ".") if ADAPTER_INFIX in key else key
        if new_key in out:
            raise ValueError(f"key collision after rename: {key} -> {new_key}")
        out[new_key] = tensor
        for tag in (".lora_A", ".lora_B"):
            if tag in new_key:
                layers.add(new_key.split(tag)[0])

    if alpha is not None:
        for layer in sorted(layers):
            out[f"{layer}.lora_alpha"] = torch.tensor(alpha, dtype=torch.float32)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--alpha", type=float, default=None,
                   help="override the alpha in the file's __metadata__ (default: use it)")
    p.add_argument("--no-alpha", action="store_true", help="do not write .lora_alpha tensors at all")
    args = p.parse_args()

    src = Path(args.input).expanduser().resolve()
    with safe_open(src, framework="pt") as f:  # type: ignore[no-untyped-call]
        meta = f.metadata() or {}
        state = {k: f.get_tensor(k) for k in f.keys()}

    alpha = args.alpha
    if alpha is None and not args.no_alpha and "alpha" in meta:
        alpha = float(meta["alpha"])
    if args.no_alpha:
        alpha = None

    converted = convert(state, alpha)

    renamed = sum(1 for k in state if ADAPTER_INFIX in k)
    n_alpha = sum(1 for k in converted if k.endswith(".lora_alpha"))
    ranks = {tuple(v.shape)[0] for k, v in converted.items() if ".lora_A" in k}
    print(f"in  : {len(state)} tensors   metadata: {meta}")
    print(f"out : {len(converted)} tensors   renamed {renamed}   wrote {n_alpha} alpha scalars")
    print(f"lora_A rank(s): {sorted(ranks)}   alpha: {alpha}")
    if ADAPTER_INFIX in "".join(converted):
        raise SystemExit("internal error: '.default.' survived the rename")

    dst = Path(args.output).expanduser().resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    save_file(converted, str(dst),
              metadata={k: str(v) for k, v in {**meta, "converted_for": "fastvideo",
                                               "source": src.name}.items()})
    print(f"wrote {dst}  ({dst.stat().st_size / 2**30:.2f} GiB)")
    print(json.dumps(sorted(converted)[:3], indent=1))


if __name__ == "__main__":
    main()
