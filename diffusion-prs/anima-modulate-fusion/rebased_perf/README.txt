Anima PR #2: post-rebase full-model performance

Base: affa261e3d289fe4f907c9b2e8d773fef0d36dba
Head: 05bee50b2f4fef4484bab3db3bff7ef4495e98e5
Native-metric Modal app: ap-GKazfgOuBGh2qUkatyF2HD
Native-metric run: anima-rebased-perf-20261004T163234Z-b2ba18da
Synchronized Modal app: ap-xjwWO4eKSR3qw2syUkWn3R
Synchronized run: anima-rebased-sync-perf-20261004T164048Z-a9b075f6

The candidate contains FFI callable caching and Anima eager modulation fusion.
Duplicate-check removal belongs to the baseline. CUDA arithmetic and launch
geometry are unchanged. See ../rebase_validation/candidate.patch and the named
Git commits for the exact production delta.

This run reuses the immutable image im-USSXj25TgwLcB1dg8jFGSS built during
rebase_validation: current baseline source in /workspace/sglang-rebased,
candidate file overlays, and the same dependency runtime. It does not reuse
the earlier live GPU/container. All eight arms here run in one new container.
The pinned model snapshot must already exist in the ming-image-hf-cache volume.
The image and volumes require access to the original Modal account.

Run from the parent anima directory:
  modal run --detach rebased_perf/run_modal.py
  modal run --detach rebased_perf/run_modal_synchronized.py
  python download_results.py RUN_ID
  python rebased_perf/analyze.py RUN_ID

Each arm starts a fresh CLI process, makes a one-step same-resolution warmup
request, then generates one 30-step image. The order is ABBA followed by BAAB.
The benchmark records worker total_duration_ms, excluding model load, warmup,
and PNG saving. It is not client/network end-to-end latency. Profiling,
torch.compile, and CUDA graph are disabled. Four process samples per version,
per experiment; the two experiments use separate containers and are not pooled.

Timing boundary audit: gpu_worker.py records total_duration_ms before output
transport, and Anima decoding leaves frames on the GPU. The native timer does
not explicitly synchronize at that boundary, so it may omit trailing CUDA work.
The synchronized experiment therefore adds the SAME timing-only overlay in
both arms: torch.cuda.synchronize() before timer start and after forward.
No per-stage or per-step synchronization is added. The instrumentation patch
and hashes are in the manifest and evidence/synchronized/timing.patch.
It prints a marker AFTER timer end; both warmup and measured markers must be
present exactly once. Production source commits are unchanged. Prefer the
synchronized experiment for device-completed worker request latency.

The driver checks exact source hashes and successful matched work in every
arm. Any failure or differing output pixel invalidates the run; no arms are
excluded. analyze.py verifies all eight arms, checks downloaded PNG hashes,
and preserves both block results plus the aggregate. It uses no external
Python dependencies. No performance threshold is used as a validity filter.

Compact evidence is tracked in evidence/native and evidence/synchronized:
manifest, all eight perf.json files, and summary with boundary GPU snapshots.
Full command output and generated
PNGs are in the Modal volume anima-modulate-experiments/RUN_ID and the local
ignored results/RUN_ID directory. EVIDENCE.json hashes the tracked archive.

Correctness component/unit-test results for these exact sources were already
archived in ../rebase_validation. This run adds full-model pixel comparison
and new performance evidence, without adding repository unit tests.

Historical measurements used an older base and another instance. They are
not pooled with these results. One GPU instance and one prompt/seed/resolution
cannot establish a universal speedup or rule out performance regressions.
