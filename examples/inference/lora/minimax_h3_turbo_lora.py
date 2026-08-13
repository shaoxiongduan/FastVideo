# SPDX-License-Identifier: Apache-2.0
"""Few-step MiniMax H3 with a community turbo LoRA.

The published H3 adapters are ComfyUI-format; the pipeline converts them on
load, so pass the .safetensors path directly. Use 4-8 steps instead of 50 --
the adapter's author reports 6-8 as the useful range, with visible motion smear
at 4.

The same adapter also loads against a rank-reduced checkpoint (--model-path
noctuashap/MiniMax-H3-pruned-r16): its AdaLN factors are folded into the stored
basis automatically.

Note that a turbo adapter does not reproduce the base model's output for a given
seed. It relocates the sample, so treat a seed as a new draw rather than as the
50-step result made faster.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fastvideo import VideoGenerator
from fastvideo.api import (
    ComponentConfig,
    EngineConfig,
    GenerationRequest,
    GeneratorConfig,
    OffloadConfig,
    OutputConfig,
    ParallelismConfig,
    PipelineSelection,
    SamplingConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="MiniMaxAI/MiniMax-H3")
    parser.add_argument(
        "--lora-path",
        required=True,
        help="ComfyUI-format H3 adapter, e.g. a local "
        "minimax_h3_turbo_v4_step600_ema.safetensors from "
        "larryvrh/MiniMax-H3-Turbo-Lora",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="outputs/minimax_h3_turbo")
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = VideoGenerator.from_config(
        GeneratorConfig(
            model_path=args.model_path,
            pipeline=PipelineSelection(components=ComponentConfig(lora_path=args.lora_path)),
            engine=EngineConfig(
                num_gpus=args.num_gpus,
                # Resident placement: the adapter merges into plain tensors
                # rather than going through the sharded DTensor merge path.
                use_fsdp_inference=False,
                parallelism=ParallelismConfig(tp_size=1, sp_size=args.num_gpus),
                offload=OffloadConfig(
                    dit=False,
                    dit_layerwise=False,
                    text_encoder=True,
                    vae=True,
                    pin_cpu_memory=False,
                ),
            ),
        ))
    try:
        result = generator.generate(
            GenerationRequest(
                prompt=args.prompt,
                negative_prompt="",
                sampling=SamplingConfig(
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    fps=24,
                    num_inference_steps=args.steps,
                    guidance_scale=1.0,
                    batch_cfg=False,
                    seed=args.seed,
                ),
                output=OutputConfig(
                    output_path=str(output_dir / "minimax_h3_turbo.mp4"),
                    save_video=True,
                    return_frames=False,
                ),
            ))
        print(f"Output written to: {result.video_path}")
        if result.generation_time is not None:
            print(f"Generation time: {result.generation_time:.2f}s")
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
