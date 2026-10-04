Anima modulation experiments — archive and reproduction

Scope
  SGLang baseline: 6fa3fe69e2e5e19b75cadd9fc285b72634551992
  Source branch: perf/anima-modulate-fusion
  Model: circlestone-labs/Anima-Base-v1.0-Diffusers
  Model revision: 073c3a9db359c31ad0e8aa268d15775473c2176c
  Final patch: patches/dispatch_candidate.patch (two production files).
  No new repository unit tests have been added. See UNIT_TEST_REVIEW.txt.

Evidence to read
  evidence/initial/REPORT.txt: initial integration-only ABBA, +4.93% latency.
  evidence/geometry/REPORT.txt: nine geometry variants and host-path screen.
  evidence/dispatch/REPORT.txt: final ABC-CBA, -4.15% vs eager, -2.35% vs initial.
  PR_BODY.txt: local PR draft, including limitations and negative evidence.
  EVIDENCE.json: run IDs, storage locations, scope and tracked-file checksums.

The final model experiment has only two requests per variant on one H200.
The integration-only result changed sign across Modal instances. Do not pool
latencies across instances or infer a universal speedup. CUDA geometry was not
changed; the final backend patch caches the FFI callable and avoids a duplicate
eligibility check. All six final outputs match exactly; 36 standalone numerical
cases, CUDA graph replay, 12 modulate tests and 17 Anima tests passed.

What is archived
  The executed benchmark/validation/analysis scripts, source patches, compact
  manifests and numerical samples, summaries, existing-test output, and drafts.
  Historical reports/manifests retain the original remote paths. Their large
  referenced artifacts remain in Modal or the local ignored results directory.
  evidence/initial_harness_sha256.json records the historical harness, before
  prepare_inputs.py was repaired for two-file source reconstruction. Current
  archive hashes live in EVIDENCE.json. Benchmark execution scripts are unchanged.

What stays local / on Modal
  inputs/: generated source tarball and overlaid source files.
  results/: full logs, generated images, profiler traces and extracted traces.
  *.log, __pycache__/: launcher logs and Python caches.
  These paths are ignored. No model weights, credentials or full source tree
  are committed. Unrelated experiments are not part of this archive.

Recreate exact input sources without changing any SGLang checkout
  python prepare_inputs.py /path/to/sglang

This reads only the pinned Git object, applies the archived final patch in a
scratch directory, and verifies both source hashes and the archive hash. It
creates the inputs needed by the original and final experiment drivers. The
source repository may be on another branch or have unrelated working changes.

Execution (allocates Modal resources; not needed to inspect this archive)
  modal run --detach anima_experiment.py
  modal run --detach kernel_screen_modal.py
  modal run --detach dispatch_abccba.py

Each launcher prints a unique run ID. The first driver also downloads the pinned
model snapshot through its CPU download function before allocating the GPU.
The latter two reuse the exact completed Modal image im-rRXDzAdc13HCyQHStCXqwE.
That image contains the baseline source and initial candidate; the dispatch
driver additionally uploads the final backend file from inputs/.

Modal resources are account-scoped. The image must remain available, and the
model snapshot must exist in the ming-image-hf-cache volume for the final run.
Results volume: anima-modulate-experiments. Rebuilding the original driver uses
the mutable lmsysorg/sglang:dev image plus dependency resolution; it does NOT
promise the historical software environment. A rebuilt environment requires
fresh matched baselines. Historical package versions are in each manifest.

Download and analyze (no new GPU experiment)
  python download_results.py RUN_ID
  python analyze_results.py results/INITIAL_RUN_ID
  python analyze_dispatch.py DISPATCH_RUN_ID

Actual archived run IDs
  initial: anima-20261003T224511Z-017ed58d
  geometry: kernel-screen-20261003T231957Z-36d5d3ca
  dispatch: dispatch-abccba-20261003T232600Z-e9defb4f

Archive validation performed locally
  Python AST and JSON parsing; archived-file SHA256 validation; source input
  reconstruction from pinned Git objects; final source/patch consistency;
  numerical sample, image-hash and completed-run checks; secret-pattern scan.
  No new GPU run was started during archival. Existing GPU results are preserved.
