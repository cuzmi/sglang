# Ming-Image RMSNorm experiment

## Scope and controls

- Branch: `perf/ming-image-rmsnorm`.
- Baseline: `6fa3fe69e2e5e19b75cadd9fc285b72634551992`; `git archive HEAD` excludes the unrelated uncommitted Anima change.
- Candidate: only `MingRMSNorm.forward_native` and its imports/gate differ in timed model source. The shared kernel is unchanged. See `candidate.patch` and `source_manifest.json`.
- Model: `inclusionAI/Ming-Image-0.1-Design`, revision `208087ada1486931692c1896f38d4cd16ff3df82`.
- One Modal H200, TP1, eager, breakable CUDA graph off, default model cache/offload behavior identical across arms. Eight CPU cores; 64 GiB host memory.
- Workload: 1024 x 1024, 12 denoise steps, CFG 1, seed 42, lossless PNG; prompt: `A minimalist exhibition poster with a red geometric sculpture`.
- Performance order: A B B A A B B A on one allocated GPU, one fresh process per arm and one request warmup with one denoise step per process. Report four samples per variant and within-arm ranges. Model loading is excluded from SGLang request timing.
- Primary metric: sum of unprofiled denoise-step timings. Secondary: SGLang total request duration. No profiler timings are used for speed claims.
- Validity gates: all processes succeed, all requests complete 12 steps and save a 1024 x 1024 image, all corresponding pixels agree, diagnostics show actual fused dispatch with no disabled gate. Any failure is preserved; do not average a favorable subset.
- Diagnostic A/B runs occur after all timed arms. Only diagnostics append `record_function` markers and counters; timed model files are exact baseline/candidate bytes.

## Numerical contract

FP32 conversion and square are fused; `aten::mean` receives a same-shape contiguous FP32 square buffer. The finishing kernel performs FP32 epsilon/rsqrt/multiply, casts the normalized activation to the input dtype, then multiplies weight and rounds the output. The original reduction remains unchanged.

The wrapper checks each `(device, dtype, shape, eps)` on first dispatch with `torch.equal`; mismatch disables fusion and returns the native result. FP32, unsupported layouts/platforms, residual calls, compile tracing, and unseen shapes during CUDA graph capture retain the native path. Verified graph replay can use the fusion.

## PR precedents

- [LTX-2 modulate mount #34315](https://github.com/sgl-project/sglang/pull/34315): a small model wiring change, live gate verification, output equality, kernel tests, exact hardware/configuration and before/after denoise/E2E numbers. Its reported speedup is author evidence, not a prediction for Ming.
- [Qwen-Image 2.1 #39983](https://github.com/sgl-project/sglang/pull/39983): provenance of `rmsnorm_preserve_reduction`; documents first-call checks, fresh-start comparisons, output parity, and limitations of pre-final-revision measurements.
- Local `.github/PULL_REQUEST_TEMPLATE.md`: Motivation, Modifications, Accuracy Tests, Speed Tests and Profiling, Checklist. No unrun upstream CI or broader model quality checks should be marked passed.

## Reproduction

Run from this directory using the Modal Python package and authenticated user workspace:

```sh
python prepare_inputs.py --sglang /path/to/sglang
modal run --detach experiment.py --mode micro
modal run --detach experiment.py --mode full
```

`inputs/baseline.tar.gz` is the exact baseline archive; `inputs/candidate_ming_image.py` and the two test files are explicit overlays. `experiment.py` uploads only these allowlisted files plus `verify_rmsnorm.py`. Results are saved to Modal Volume `ming-rmsnorm-experiments`; create a local directory before `modal volume get` to preserve folder layout. Analyze a downloaded full-run directory with `python analyze_results.py <directory>`.

The image starts from the moving `lmsysorg/sglang:dev` tag; the actual immutable Modal image ID is in launch logs and installed versions are in each manifest. Future rebuilds of the tag may differ. Current conclusions only apply to the recorded environment.

The baseline archive is generated locally and excluded from Git. The tracked input overlays are the exact GPU-tested files; preparing inputs verifies their hashes before rebuilding the archive. Full runs require the pinned checkpoint already present in the `ming-image-hf-cache` Modal Volume (the driver deliberately uses offline mode). See README.md for archived run IDs and retrieval commands.
