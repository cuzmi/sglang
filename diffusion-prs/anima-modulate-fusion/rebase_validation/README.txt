Anima PR #2: resolution against upstream dispatch changes

Base: affa261e3d289fe4f907c9b2e8d773fef0d36dba
Candidate: 05bee50b2f4fef4484bab3db3bff7ef4495e98e5

The PR now contains two commits:
  068508bad4: cache the exported modulation FFI callable.
  05bee50b2f: integrate existing modulation fusion into Anima AdaLN.

Upstream bc5a055dd7 (#42391) already removed the duplicate eligibility check,
made modulate_scale_shift_cuda the custom-op entrypoint, and removed catch/
disable behavior. Resolution preserves that current upstream contract. The
candidate adds only callable caching and the Anima call-site integration.

provenance.json and candidate.patch identify the exact measured source.
The new complete baseline is archived locally as inputs/rebased_baseline.tar.gz
and overlaid into a fresh source directory in the existing software image.
All tests import that directory via PYTHONPATH, avoiding mixed old/new source.
Modal image im-rRXDzAdc13HCyQHStCXqwE supplies the historical dependency runtime;
package versions and GPU are recorded in the run manifest.

Correctness-only execution:
  modal run --detach rebase_validation/run_modal.py
Download from parent anima directory:
  python download_results.py RUN_ID
  python rebase_validation/analyze.py RUN_ID

Inputs can be reconstructed from the named Git commits using git archive for
the base and git show for the two candidate files. The provenance lists SHA256
values; run_modal.py checks the archive and overlaid source before execution.
The Modal image/volume require access to the original account.

check_callers.py exercises the same real component entrypoints as cross_model/,
but spies on the current CUDA entrypoint and checks error propagation instead
of the deleted disabled-runtime registry. verify_anima.py checks full AdaLN
and graph correctness without the old microbenchmark loop.

No repository unit tests, model weights, full image/video generation, or latency
measurements are added by this validation. The original Anima performance and
pixel-comparison results refer to the old base and must remain historical.
