<!-- Thank you for your contribution! Please follow these guidelines to enhance your pull request. If anything is unclear, submit your PR and reach out to maintainers for assistance. Join our Slack community at https://slack.sglang.io to discuss further. -->

## Motivation

Ming-Image forces native FP32 RMSNorm, expanding each norm into eight kernels. Reuse the existing reduction-preserving fusion to reduce this overhead while retaining native output rounding. On the measured H200 eager workload, median denoise latency decreased 20.49% and request latency decreased 18.81%.

## Modifications

- Route supported MingRMSNorm calls through `rmsnorm_preserve_reduction`; retain the same-shape native FP32 mean and cast-before-weight semantics. No shared kernel changes.
- Verify each device/dtype/shape/epsilon signature on first use with `BitExactFusionGate`. Preserve native fallback for unsupported inputs, residuals, compile tracing and unseen signatures during graph capture.
- Extend kernel tests to Ming widths and add focused model dispatch/fallback/compile/graph tests.

## Accuracy Tests

H200, PyTorch 2.13.0+cu130, Triton 3.7.1:

- 44 kernel tests and 11 Ming wrapper tests passed. An additional 42-case BF16/FP16 shape/scale sweep matched native exactly and asserted actual fused dispatch.
- Eight timed requests and two separate diagnostic requests produced identical output pixels, SHA256 `b04fe457e7a38a1007c8f23a1d0dd3c42ccd57afdb8c547d6670da661f82448f`.
- Candidate diagnostic gate remained enabled; corresponding native/candidate outputs matched. This is parity for one fixed generation workload, not a broad quality evaluation.

```sh
pytest -q test/registered/kernels/ops/diffusion/test_rmsnorm_preserve_reduction.py
pytest -q test/registered/kernels/ops/diffusion/test_model_fast_paths.py -k ming_rmsnorm
```

The wrapper compile test checks tracing using backend=eager; the kernel test exercises default Inductor. Full-model compiled/BCG performance and other GPU platforms were not measured.

## Speed Tests and Profiling

Baseline `6fa3fe69e2e5e19b75cadd9fc285b72634551992` versus that snapshot with only the candidate Ming model source. One H200, TP1, eager, FlashAttention, compile/BCG/offload disabled. `inclusionAI/Ming-Image-0.1-Design` at `208087ada1486931692c1896f38d4cd16ff3df82`; 1024×1024, 12 steps, CFG 1, seed 42. Prompt: `A minimalist exhibition poster with a red geometric sculpture`.

Same-GPU fresh-process A B B A A B B A; one 1-step request warmup per process, excluded from measurements. Four samples per variant, no failed requests.

| Metric (median) | Native | Candidate | Reduction |
| --- | ---: | ---: | ---: |
| Denoise step sum | 1692.129 ms | 1345.372 ms | 20.49% |
| Total request | 1802.948 ms | 1463.758 ms | 18.81% |

Native/candidate denoise ranges were 1691.221–1694.449 / 1343.422–1347.240 ms. Separate profiler runs confirmed 205 MingRMSNorm calls per step and 1640 → 615 correlated kernels. Profiler timings are not used for the speed claims.

Small-shape call microbenchmarks regressed about 2–4%; large representative shapes improved about 65–68%. These microbenchmarks include Python dispatch gaps. Results do not establish gains on other GPUs, resolutions, prompts, or compiled execution.

Reproduction CLI: `sglang generate --model-path inclusionAI/Ming-Image-0.1-Design --revision 208087ada1486931692c1896f38d4cd16ff3df82 --num-gpus 1 --enable-torch-compile=false --enable-breakable-cuda-graph=false --performance-mode=speed --width 1024 --height 1024 --num-inference-steps 12 --guidance-scale 1 --seed 42 --prompt "A minimalist exhibition poster with a red geometric sculpture" --quality lossless --save-output --warmup-mode request --warmup-steps 1 --perf-dump-path perf.json`.

Before publishing, attach the local experiment report/raw logs or a reviewer-accessible artifact bundle; no local filesystem paths are relied on as public evidence.

## Checklist

- [x] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit).
- [x] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests).
- [ ] Update documentation according to [Write documentations](https://docs.sglang.io/developer_guide/contribution_guide.html#write-documentations).
- [x] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.io/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.io/developer_guide/contribution_guide.html#benchmark-the-speed).
- [x] Follow the SGLang code style [guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance).

## Review and Merge Process

1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process).
2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers.
3. Trigger CI tests with [comments](https://docs.sglang.io/developer_guide/contribution_guide.html#how-to-trigger-ci-tests) or contact authorized users to do so.
   - Common commands include `/tag-and-rerun-ci`, `/tag-run-ci-label`, `/rerun-failed-ci`
4. After green CI and required approvals, ask Merge Oncalls or people with Write permission to merge the PR.
