# Ming RMSNorm unit-test coverage review

This review follows archival of the completed experiment. No new GPU run or production-code change was made for this review. The recorded result remains 44 kernel tests + 11 model wrapper tests, plus the standalone 42-case numerical sweep.

## Already covered

- Kernel: BF16/FP16, widths 128/2560/3840, long sequences, batch 2, input scales/epsilon, storage offset, rejected layouts/dtypes/devices, compiled custom op, CUDA graph replay.
- Ming wrapper: actual fused dispatch, native equality, verified/non-disabled gate, FP32/strided/residual fallback, mismatch returning native and disabling later attempts, compile tracing retaining native, warmed graph replay.
- Existing CPU model test `test_ming_norm_and_swiglu_preserve_reference_rounding` already checks Ming's CPU reference semantics. It was inspected, not rerun in the H200 experiment.
- Full-model diagnostics verified all seven live shapes, including `(1,256,2560)`, `(122880,128)` and `(9600,128)`, with no fallback.

## Two focused additions recommended

1. **A shared gate must verify each new signature.** Current parameterized wrapper tests create a new gate for every shape; they would not detect accidentally changing the implementation to verify only once globally. With one gate and one norm, call shape A twice and shape B twice. Spy on `NativeRMSNorm.forward_native` to assert exactly one native comparison per shape, two verified signatures, and native-exact outputs. A B input must not inherit A's verification merely because width/dtype match. This directly covers the reason row count is in the signature.
2. **An unseen shape during CUDA graph capture must stay native.** The current graph test warms and verifies the same shape first. Prewarm native kernels/JIT as needed, start with a fresh gate (or a new shape on an already verified gate), capture the wrapper, and assert the fused function was not called and no new verification was recorded. Replay with changed input and compare to the native result. After capture, a normal eager call should verify and enable the candidate. This tests the guard against a host synchronization during capture.

These are dispatch/lifecycle regressions, not new random-shape sweeps. They can live beside the existing Ming wrapper tests in `test/registered/kernels/ops/diffusion/test_model_fast_paths.py` without creating another test file.

## Not required for this patch

- More arbitrary widths or standalone copies of the reference formula: the shared kernel is unchanged and Ming's actual widths are already covered.
- Adding all seven live shapes to the large kernel matrix: real diagnostics cover them; targeted `(1,256,2560)` or `(132480,128)` CI cases may be substituted later if maintainers prefer exact workload shapes, but no demonstrated numerical gap requires it now.
- Treating more prompts/resolutions as unit tests: that would broaden E2E quality/performance evidence, which is distinct from these two wrapper boundaries.

The wrapper's `torch.compile(..., backend="eager")` test validates tracing behavior; it does not establish full-model Inductor or BCG performance. Keep that limitation explicit.
