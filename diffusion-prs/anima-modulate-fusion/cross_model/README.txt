Shared modulation caller regression — correctness only

Purpose
  Check whether the two host-dispatch changes used by Anima cause numerical
  errors or routing/fallback changes in other callers. No benchmark, ABBA
  timing, model downloads, or image/video generation is performed.

Sources and environment
  Baseline: 6fa3fe69e2e5e19b75cadd9fc285b72634551992.
  Candidate: only modulate_scale_shift_jit.py overlaid on the same baseline.
  Candidate SHA256: 5705525464fedef85e83631e44aa53ce87bd7a7b1b83d7294a25c22621bcc8b0.
  The existing Anima call-site patch is not needed for these other callers.
  Modal base image: im-rRXDzAdc13HCyQHStCXqwE. Single H200.
  Baseline and candidate run in separate Python processes in the same container.

Coverage
  48 cases per source version: FP16/BF16/FP32 x batch 1/2, sequence length 17,
  using model hidden widths 3072 (FLUX), 4608 (Ideogram), 5120 (LingBot),
  4096/2048 (LTX video/audio).

  FLUX: real _flux_norm_modulate, both normal priority and a controlled
    disabled LayerNorm-fusion gate to exercise the shared modulation fallback.
    Default FLUX may use the shared kernel only for its first-sight reference.
  Ideogram: real _norm_scale and Ideogram4RMSNorm, enable_fused=False
    (the path that uses this shared kernel). Batch 2 takes its native fallback.
  LingBot: real CamConditioner.forward with the supported precomputed
    scale_shift argument. No camera-projection weights are allocated because
    this branch does not access them. Contiguous and non-contiguous inputs.
  LTX-2: real _ltx2_rms_norm_modulate with RMSNormNoWeight, unmounted high-quality
    fusion (the lossless path). Chunked row views and per-token fallback.

  Each result is compared byte-for-byte with its eager expression; outputs
  are also hashed to compare the baseline and candidate. Spies verify actual
  shared-kernel dispatch for eligible inputs, including first/repeated calls.
  Shared wrapper error injection checks exact fallback and a single failed
  invocation followed by disabled retries. Patched state is restored.

  Five warmed CUDA graph capture/replay cases per version change the input
  before replay to check that caching the callable never caches tensor data.
  Ideogram intentionally takes its native expression during graph capture.

Existing repository tests
  pytest -q test/registered/kernels/ops/diffusion/test_modulate.py -k modulate_scale_shift
  pytest -q test/registered/kernels/ops/diffusion/test_model_fast_paths.py -k 'flux_fused_ln_modulate or flux_norm_modulate or ltx2_lossless'
  pytest -q python/sglang/multimodal_gen/test/unit/test_ltx2_modulate_mount.py

Run (allocates a GPU)
  modal run --detach cross_model/run_modal.py

Download (from the parent anima directory)
  python download_results.py RUN_ID

Limits
  These are real model component entrypoints with synthetic inputs, not
  full checkpoint-based model inference. Results do not establish image/video
  quality, distributed inference compatibility, torch.compile behavior or
  performance for the four complete models. No repository tests are modified.
