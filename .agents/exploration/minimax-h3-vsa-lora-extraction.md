# Exploration Log: MiniMax-H3 VSA LoRA Extraction

## Status: under_review

## Context

Extract rank-64, rank-128, and rank-256 adapters from
`FastVideo/FastVideo-FastH3-4-step-v1` against its exact MiniMax-H3 T2AV base.
The distilled student was trained with VSA-H3, so each portable checkpoint must
also carry the 50 trained, full-rank `to_gate_compress.weight` tensors that have
no counterpart in the base checkpoint.

Pinned inputs:

- FastVideo source: upstream `main` at `e9bbaca07d511b2ee7e16474dae6f923426223dc`
- Student: `FastVideo/FastVideo-FastH3-4-step-v1@b790390377918066c5f5902ec6cc96e21a55926e`
- Base: `MiniMaxAI/MiniMax-H3@9bfb6693f2cf6de171db46d1aa586f67d773a1da`
- Student transformer content digest: `b36987515e4c75fa4c7aaa632a7842c829ea141b235358a54d782b51230497b3`

## Progress

- [x] Create a clean branch and worktree from the latest upstream `main`.
- [x] Authenticate to Hugging Face with the user-provided token and inspect the private student repository.
- [x] Pin the exact base revision from `provenance.json`.
- [x] Download the student transformer shards into the user's Lustre HF cache.
- [x] Verify every downloaded student shard and the local base shards against their published SHA-256 digests.
- [x] Measure delta spectra and choose a deterministic truncated-SVD implementation.
- [x] Add a streaming, multi-rank extractor that includes full VSA gates.
- [x] Add CPU unit tests for rank truncation, mixed LoRA/full tensors, metadata, and reconstruction.
- [x] Produce and structurally validate rank-64, rank-128, and rank-256 artifacts on the existing four-GB200 `debug` allocation.
- [x] Run native-video divergence checks for all current mixed checkpoints with the checkpoint's FA4 + VSA-H3 profile.

## Findings

- The student has 688 transformer tensors and the base has 638. The 50 extra
  tensors are `transformer_blocks.<0..49>.attn.to_gate_compress.weight`, each
  bf16 `[7168, 5376]`. Together they are 3,853,516,800 bytes.
- These gates must be shipped as dense tensors. Treating them as ordinary LoRA
  deltas would compare against an implicit zero matrix and produce a lossy
  low-rank compression of a newly trained branch, not a low-rank fine-tuning
  delta. Omitting them disables the VSA gate branch entirely.
- The generic `extract_lora.py` silently skips fine-tuned-only tensors, performs
  full CPU SVD, reloads both models for every rank, holds both full state dicts
  in RAM, and filters out `norm_out.linear.weight` because its name contains
  `norm`. It therefore cannot produce a correct H3 VSA adapter unchanged.
- The available Slurm job `4156` (`debug`) provides one node, 4x NVIDIA GB200,
  956 GiB host RAM, and shared Lustre access.
- All 14 student shards and all 14 pinned-base shards passed SHA-256
  verification. The extractor factorized 370 common 2-D matrices once at
  rank 256 (randomized SVD, `q=320`, four power iterations) and derived all
  three requested ranks from the nested factors.
- The FP32 DCP update's aggregate theoretical residuals over the 370
  factorized matrices are 0.48627 (rank 64), 0.43328 (rank 128), and 0.37597
  (rank 256). Directly comparing `B @ A` with the exported BF16 delta is
  misleading: the update must be added to the BF16 base and rounded. On that
  effective merged-weight metric over the 258 core factorization groups, the
  residuals are 0.66065, 0.62349, and 0.58114.
- Produced checkpoints live under
  `/mnt/lustre/vlm-s4duan/models/FastVideo-FastH3-4-step-v1-loras/`. The current
  artifacts have 362 `lora_A`/`lora_B` pairs and 326 exact dense tensors: all
  50 VSA gates, 268 one-dimensional weights/biases, and 8 boundary matrices.
  File sizes are 4.97 GiB (rank 64), 6.21 GiB (rank 128), and 8.68 GiB
  (rank 256).
- The first native-video comparison used the earlier artifact that kept only
  the VSA gates dense. Without FA4, rank 256 reached video SSIM 0.6169 and
  audio cosine correlation 0.9346. Keeping auxiliary state exact improved the
  corresponding no-FA4 rank-256 result to SSIM 0.6617 and audio correlation
  0.9467.
- Final checkpoint-profile comparisons enabled FA4, VSA-H3 at 90% sparsity,
  64-token tiles, the sm100a kernel, and the exact `[999, 749, 500, 250]`
  denoising ladder. Against the native model, ranks 64/128/256 respectively
  measured video SSIM 0.5983/0.5619/0.5860 and audio cosine correlation
  0.8067/0.8076/0.7854. A repeated native FA4 run was bit-exact in decoded
  video and audio, so the divergence is not run-to-run nondeterminism. Sparse
  top-k selection makes output distance non-monotonic even though matrix
  reconstruction error decreases monotonically with rank. Replicated-DiT and
  FSDP-inference runs produced byte-identical MP4s for the native checkpoint
  and all three adapters; the final report uses the checkpoint's FSDP route.

## Community comparison

`drozbay/MiniMax-H3-FastH3-Preview-LoRA@4f95050e` extracts the older
FastH3 Preview v0.2 against a ComfyUI BF16 base with KJNodes `LoraExtractKJ`.
It uses exact `torch.linalg.svd`, fused-QKV factors, FP16 output, additive
`.diff`/`.diff_b` tensors, and optional exact AdaLN refits for an 8-dimensional
pruned base. Its published files contain no `to_gate_compress` tensors, so they
cannot reproduce the preview checkpoint's trained VSA-H3 route and were
validated under a different dense/ComfyUI setup at 960x544 and 73 frames.

Useful ideas to adopt are fused-QKV storage, additive auxiliary deltas, an
optional pruned-AdaLN target, and a dense-attention low-resolution diagnostic.
Exact SVD alone is only a marginal advantage: representative rank-256 layers
put the oversampled randomized solver within 0.00-0.07% of exact residual, and
ranks 64/128 were effectively identical. A controlled recreation of the full
community method on this same v1 target (exact SVD of the BF16 export delta,
fused QKV, FP16 factors, and an FP32 merge before the final BF16 cast) gave
aggregate effective residuals 0.91859, 0.89683, and 0.85667. The current
DCP/separate-QKV method reduced that error by 28.1%, 30.5%, and 32.2%. This
weight-fit control is distinct from the community's published quality numbers,
which remain incomparable because the checkpoint, base layout, attention
route, quantization ceiling, resolution, and duration all differ.

## Mistakes / Dead Ends

- The first `git fetch origin main` only updated `FETCH_HEAD` because this clone's
  `remote.origin.fetch` refspec tracks only `trackwan_bidir`. Fetching with the
  explicit refspec `+refs/heads/main:refs/remotes/origin/main` fixed it.
- Passing both explicit filenames and `--include` to `hf download` caused the
  include glob to be ignored. A second call using only `--include
  'transformer/*'` downloaded all transformer shards.

## Proposed Standardization

If the extraction and validation succeed, promote the streaming multi-rank
workflow into the existing `scripts/lora_extraction/` tools: support a declared
set of full-tensor passthrough keys, derive several smaller ranks from one
max-rank factorization, stream indexed safetensors instead of loading the whole
model, and emit reconstruction/error metadata for auditability.
