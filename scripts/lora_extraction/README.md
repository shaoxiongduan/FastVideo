# LoRA Extraction and Merging

Tools for extracting and merging LoRA adapters for FastVideo models.

## Extract LoRA Adapter

```bash
python extract_lora.py \
  --base Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --ft FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers \
  --out adapter_r32.safetensors \
  --rank 32
```

**Options:**
- `--base`: Base model (HuggingFace ID or local path)
- `--ft`: Fine-tuned model (HuggingFace ID or local path)
- `--out`: Output adapter file
- `--rank`: LoRA rank (16, 32, 64, 128)
- `--full-rank`: Extract full-rank adapter (optional)


> **Note:** The script automatically handles architectural differences (e.g., FastWan has extra `gate_compress` layers) by falling back to direct safetensors loading for both models if pipeline loading fails.

## Extract MiniMax-H3 VSA Adapters

MiniMax-H3 needs the streaming, multi-rank extractor because its transformer is
33B parameters and VSA students add trained dense gates that are absent from
the base checkpoint:

```bash
python scripts/lora_extraction/extract_minimax_h3_lora.py \
  --base /path/to/MiniMax-H3 \
  --base-model-id MiniMaxAI/MiniMax-H3 \
  --base-revision 9bfb6693f2cf6de171db46d1aa586f67d773a1da \
  --finetuned /path/to/training-output/checkpoint-1300 \
  --finetuned-role student \
  --finetuned-model-id FastVideo/FastVideo-FastH3-4-step-v1 \
  --finetuned-revision b790390377918066c5f5902ec6cc96e21a55926e \
  --output-dir /path/to/fasth3-loras \
  --ranks 64 128 256 \
  --device cuda:0
```

Use the fp32 DCP training checkpoint when it is available. Although the script
also accepts the published bf16 Diffusers export, subtracting two bf16 exports
produces a quantized shadow of the training update: most changed entries sit at
one bf16 ULP and the apparent delta has a long, nearly full-rank tail.

The largest requested rank is factorized once. Smaller ranks are prefix slices
of the same factors, and per-layer residual estimates are recorded in the
factorization manifest. Each `rank-<N>/adapter_model.safetensors` is a mixed
checkpoint containing:

- Diffusers-named `lora_A` and `lora_B` factors for the large transformer and
  token-refiner block matrices;
- exact small boundary, timestep, normalization, and bias tensors;
- the 50 full bf16 `to_gate_compress.weight` tensors copied exactly from the
  VSA-trained checkpoint.

Do not drop or low-rank the dense gates: the base has no counterpart for them,
and a zero/missing gate disables the trained VSA compression branch. Use
`--resume` to reuse completed per-layer factors after an interrupted run.

## Merge Adapter

```bash
python merge_lora.py \
  --base Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --adapter adapter_r32.safetensors \
  --ft FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers \
  --output merged_model
```

**Options:**
- `--base`: Base model (HuggingFace ID or local path)
- `--adapter`: LoRA adapter file (.safetensors)
- `--ft`: Fine-tuned model (for configuration)
- `--output`: Output directory

## Validate Quality (Optional)

```bash
python lora_inference_comparison.py \
  --base merged_model \
  --ft FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers \
  --adapter NONE \
  --output-dir results \
  --prompt "A cat sitting on a windowsill" \
  --seed 42 \
  --height 480 \
  --width 480 \
  --num-frames 49 \
  --num-inference-steps 32 \
  --compute-ssim \
  --compute-lpips
```

**Options:**
- `--base`: Merged model or base model path
- `--ft`: Fine-tuned model (reference)
- `--adapter`: Path to adapter or NONE
- `--output-dir`: Output directory
- `--prompt`: Text prompt (default: "A cat sitting on a windowsill")
- `--seed`: Random seed (default: 42)
- `--height`: Video height (default: 480)
- `--width`: Video width (default: 832)
- `--num-frames`: Number of frames (default: 49)
- `--num-inference-steps`: Inference steps (default: 32)
- `--compute-ssim`: Compute SSIM metric
- `--compute-lpips`: Compute LPIPS metric
