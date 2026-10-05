Anima FP32 split-half RoPE experiment

Evidence-only archive. Reproduction scripts and source patches are retained at:
https://github.com/cuzmi/sglang/tree/893495236a5ba25667313768b5e791fc51e355d8/diffusion-prs/anima-fp32-rope
Commands and source-file paths below refer to that fixed snapshot.

===================================

Source worktree: /Users/cuzimi/.codex/worktrees/anima-fp32-rope/sglang
Branch: perf/anima-fp32-rope
Baseline: 6fa3fe69e2e5e19b75cadd9fc285b72634551992
Only the Anima RoPE call site, diffusion registry/export facade, and new kernel differ.
No Ming-Image, AdaLN modulation, or InternVL optimization is included.
No new repository unit tests were added. verify_rope.py is an external experiment.

Mechanism and numerical contract
-------------------------------
AnimaAttention.forward: Q/K projection -> unchanged RMSNorm -> paired RoPE -> attention.
Input q/k: contiguous CUDA FP16/BF16 [B,S,H,128].
Cos/sin: contiguous FP32 [S,128], broadcast over batch and heads.
Read the opposite half directly instead of allocating cat(-x2,x1).
Cast loaded values to FP32, perform separate FP32 products and add, then cast on store.
Launch with enable_fp_fusion=False to preserve eager rounding boundaries.
A two-dimensional grid handles Q and K in one launch; outputs are fresh tensors.
Unsupported dtype/layout/platform/dimension or gradient use retains eager behavior.
torch.compile retains the original arithmetic for Inductor to compile.
Each new (device,dtype,shape) is checked with the shared BitExactFusionGate outside capture.
A mismatch or kernel exception permanently disables the fast path for the process.
Unverified CUDA-graph capture uses eager; a warmed verified signature uses fusion.
The finite experiment checks compare raw 16-bit output representations, including signed zero.
The runtime gate uses torch.equal, matching existing repository convention.

Reproduction
------------
The self-contained experiment.py uses cached Modal image im-rRXDzAdc13HCyQHStCXqwE,
restores its SHA256-checked public baseline archive, then installs only the three
SHA256-checked source files in inputs/manifest.json. Both timing arms use the same
facade/kernel files; the baseline's original anima.py never invokes the new kernel.
The original source archive SHA256 is
349f9021b4d7bcd7d3c42425d1691fccc42f8d75f2ac5054882407d621ebb233.

Model: circlestone-labs/Anima-Base-v1.0-Diffusers
Revision: 073c3a9db359c31ad0e8aa268d15775473c2176c
Hardware: one Modal H200, CPU=8, RAM=64 GiB; one-hour function timeout.
Cached model volume: ming-image-hf-cache (shared existing cache, preserved).
Result volume: anima-rope-experiments.

From this directory with a configured Modal SDK:
  python -m modal run --detach experiment.py
  python download.py <run-id>
  python analyze.py results/<run-id>

All timings use the canonical sglang generate command, 1024x1024, 30 steps,
guidance=4, seed=42, CPU generator, FlashAttention, compile=false, BCG=false.
Each fresh subprocess performs one warmup request with one denoising step.
Order: A/A control followed by A/B/B/A, all on the same container and GPU.
The workload is one synthetic fixed prompt, not a quality dataset.
Request IDs are not applicable to this fresh-process local generate workflow.
Every command, per-step time, source hash, decoded-pixel hash and PNG hash is retained.
Any subprocess failure or pixel mismatch invalidates the run.
Separate eager/candidate profiles are excluded from unprofiled timing results.
No other GPU jobs are launched by this campaign concurrently.

Metric interpretation and limits
-------------------------------
Operator microbenchmark uses triton.testing.do_bench with 100ms warmup, 300ms measurement
per cell; calls include guards/output allocation and are separate from profiler timing.
The operator trace contains 10 Q/K calls at [1,4096,16,128].
Full-model total_duration_ms is the framework-reported request duration from perf.json;
it is not process launch/model-load wall time or HTTP TTFT.
denoise_ms is the sum of framework denoise_steps_ms entries.
Only two independent subprocess samples per variant enter ABBA; 30 steps are not
30 independent experiments. A/A and within-variant spread are noise context, not CIs.
Compile/BCG model-level acceleration, multi-GPU behavior, other GPUs, prompts, seeds,
and large-resolution full-model performance are not established by this experiment.
Existing Anima CUDA tests use head_dim=32, exercising fallback; actual fused head_dim=128
coverage comes from the standalone screen and checkpoint inference/profile.

Versions: see manifest.json and generation logs. The image's Diffusers package metadata
reports 0.37.0 while its live runtime reports 0.39.0.dev0; preserve both rather than
claiming the metadata identifies the imported source version.

Artifacts
---------
inputs/manifest.json: exact uploaded runtime source hashes.
results/<run-id>/: raw manifests, outputs, logs, operator traces, full-model traces.
pre-commit-python313.log: required hooks run with Python 3.13.
pre-commit.log: initial macOS Python 3.9 hook failure retained for provenance.

Final observed result
---------------------
Run: rope-20261004T161949Z-d5819700, status complete.
Local runtime commit: 6143cce2d2f5701b07ae155ad652705cabe7b646 (published to origin/perf/anima-fp32-rope).
ABBA mean denoise: 2843.804 -> 2250.416 ms (-20.866%).
ABBA framework request duration: 3110.287 -> 2520.631 ms (-18.958%).
All 8 PNG and decoded-pixel hashes match, including separate profiling requests.
Candidate full-model trace: 336 CPU RoPE ranges and 336 fused launches; no disable.
Actual signature: BF16 [1,4096,16,128], FP32 cos/sin [4096,128].
Both exported model traces contain six ProfilerStep ranges despite the CLI argument
requesting five; report observed counts, not the argument-derived estimate.
The model trace confirms dispatch; operator traces isolate the 16 -> 1 launch change.
Summary: results/rope-20261004T161949Z-d5819700/summary.json.

Published archive
-----------------
This codex/pr-results branch contains evidence only. The runtime code is on
perf/anima-fp32-rope at 6143cce2d2f5701b07ae155ad652705cabe7b646.
The results are measurements of that exact runtime commit, not of future PR updates.
