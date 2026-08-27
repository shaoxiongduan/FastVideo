# SPDX-License-Identifier: Apache-2.0
"""Generate MiniMax H3 text-to-audio-video with an acceleration LoRA applied.

Same path as ``basic_minimax_h3_t2v.py`` plus ``--lora-path``, for running distilled
few-step LoRAs (e.g. lightx2v Minimax-h3-Turbo). H3 fits on a single GB200 (62 GiB of DiT
weights against 185 GiB of HBM), so ``--num-gpus 1`` gives one independent worker per GPU
rather than one sequence-parallel job across all of them.

lightx2v LoRAs must be converted first — see
``scripts/checkpoint_conversion/convert_minimax_h3_lightx2v_lora.py``; their PEFT-style
``.default`` adapter infix otherwise makes every layer lookup miss *silently*.

Schedule: ModelTC documents inference at video shift 12 / audio 3, which is what the H3
pipeline already asserts, with the N evaluation points at ``q_i = (N - i) / N``. So a
4-step turbo LoRA just needs ``--steps 4``; no schedule override.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from fastvideo import VideoGenerator
from fastvideo.pipelines.basic.minimax_h3.packing import (
    MINIMAX_H3_FPS,
    MINIMAX_H3_FRAMES_PER_CHUNK,
    MINIMAX_H3_LATENTS_PER_CHUNK,
    MINIMAX_H3_MAX_DURATION,
    align_num_frames,
)
from fastvideo.api import (
    EngineConfig,
    GenerationRequest,
    GeneratorConfig,
    OffloadConfig,
    OutputConfig,
    ParallelismConfig,
    PipelineSelection,
    SamplingConfig,
)


def clamp_to_supported_frames(num_frames: int) -> int:
    """Largest H3-valid frame count no longer than the model's 15s ceiling.

    H3 accepts 5-15s at 24fps and rounds frame counts UP to the next 17n+5. 18 of the 60
    validation prompts ask for 362 frames = 15.083s, which is over the ceiling by two
    frames, so the pipeline raises instead of generating and the whole worker dies. 345
    (14.375s) is the largest valid count that fits -- and is exactly what the published
    bundle arms used for those prompts, so clamping here also keeps arms comparable.
    """
    aligned = align_num_frames(num_frames)
    if aligned / MINIMAX_H3_FPS <= MINIMAX_H3_MAX_DURATION:
        return aligned
    ceiling = int(MINIMAX_H3_MAX_DURATION * MINIMAX_H3_FPS)
    best = MINIMAX_H3_LATENTS_PER_CHUNK
    while best + MINIMAX_H3_FRAMES_PER_CHUNK <= ceiling:
        best += MINIMAX_H3_FRAMES_PER_CHUNK
    return best


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", default="/mnt/lustre/vlm-k1kong/models/MiniMax-H3")
    p.add_argument("--lora-path", default=None, help="converted LoRA safetensors; omit for the base baseline")
    p.add_argument("--lora-strength", type=float, default=1.0)
    p.add_argument("--prompts", default="/mnt/lustre/vlm-s4duan/FastVideo/prompts.jsonl")
    p.add_argument("--indices", type=int, nargs="+", default=[0], help="row indices of prompts.jsonl to render")
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=None, help="cap num_frames (for quick smoke tests)")
    p.add_argument("--num-frames", type=int, default=None,
                   help="override num_frames for every prompt. H3 rounds UP to the next 17n+5, so 124 "
                        "(5.17s at 24fps) is the valid value nearest 5 seconds; 125 silently becomes 141.")
    p.add_argument("--shard", type=int, default=None, help="this worker's index, for splitting the prompt list")
    p.add_argument("--num-shards", type=int, default=None, help="total number of workers")
    p.add_argument("--skip-existing", action="store_true", help="skip prompts whose output file already exists")
    p.add_argument("--no-clamp-duration", dest="clamp_duration", action="store_false",
                   help="let over-long prompts raise instead of clamping to H3's 15s ceiling")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(x) for x in Path(args.prompts).read_text().splitlines() if x.strip()]

    indices = list(args.indices)
    if args.num_shards is not None:
        indices = [i for i in range(len(rows)) if i % args.num_shards == args.shard]
    if args.skip_existing:
        pending = []
        for i in indices:
            if (out_dir / f"{i:03d}_{rows[i]['case_id']}.mp4").exists():
                print(f"[skip] {i:03d} already rendered", flush=True)
            else:
                pending.append(i)
        indices = pending
    print(f"[plan] {len(indices)} prompts on this worker: {indices}", flush=True)
    if not indices:
        return

    generator = VideoGenerator.from_config(
        GeneratorConfig(
            model_path=args.model_path,
            pipeline=PipelineSelection(experimental={}),
            engine=EngineConfig(
                num_gpus=args.num_gpus,
                use_fsdp_inference=args.num_gpus > 1,
                parallelism=ParallelismConfig(tp_size=1, sp_size=args.num_gpus),
                offload=OffloadConfig(dit=False, dit_layerwise=False, text_encoder=True, vae=True,
                                      pin_cpu_memory=False),
            ),
        ))

    if args.lora_path:
        generator.set_lora_adapter("turbo", args.lora_path, strength=args.lora_strength)
        print(f"[lora] applied {args.lora_path} strength={args.lora_strength}", flush=True)

    try:
      for idx in indices:
        row = rows[idx]
        gen = row.get("generation", {})
        frames = int(args.num_frames or gen.get("num_frames", 125))
        if args.max_frames:
            frames = min(frames, args.max_frames)
        if args.clamp_duration:
            frames = clamp_to_supported_frames(frames)
        case = row["case_id"]
        dst = out_dir / f"{idx:03d}_{case}.mp4"

        t0 = time.time()
        result = generator.generate(
            GenerationRequest(
                prompt=row["prompt"],
                negative_prompt="",
                sampling=SamplingConfig(
                    height=int(gen.get("height", 1344)),
                    width=int(gen.get("width", 768)),
                    num_frames=frames,
                    fps=int(gen.get("fps", 24)),
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    batch_cfg=False,
                    seed=args.seed,
                ),
                output=OutputConfig(output_path=str(dst), save_video=True, return_frames=False),
            ))
        print(f"[done] {dst.name}  {gen.get('width')}x{gen.get('height')}  {frames}f  "
              f"wall {time.time() - t0:.1f}s  gen {result.generation_time or float('nan'):.1f}s  "
              f"-> {result.video_path}", flush=True)
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
