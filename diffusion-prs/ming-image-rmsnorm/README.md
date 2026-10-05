# Ming-Image RMSNorm experiment archive

Evidence-only archive. [Reproduction scripts and source patches](https://github.com/cuzmi/sglang/tree/893495236a5ba25667313768b5e791fc51e355d8/diffusion-prs/ming-image-rmsnorm) are retained in a fixed snapshot; commands and source-file paths below refer to that snapshot.

This archive records the H200 experiment completed on 2026-10-03. It is separate from committing the SGLang production patch or publishing a PR.

- Read [REPORT.md](REPORT.md) for measured results and limits, and [EXPERIMENT.md](EXPERIMENT.md) for controls.
- [candidate.patch](https://github.com/cuzmi/sglang/tree/893495236a5ba25667313768b5e791fc51e355d8/diffusion-prs/ming-image-rmsnorm/candidate.patch) contains exactly one SGLang model change and two test-file changes. [source_manifest.json](source_manifest.json) identifies the baseline and GPU-tested overlay hashes. The fixed reproduction snapshot preserves those exact overlays.
- `evidence/full-manifest.json` retains all eight timing records and both diagnostic records, model revision/configuration, versions and source hashes. `evidence/analysis.json` contains CUPTI-correlated kernel counts; `evidence/micro.json` retains the numerical sweep and microbench samples. Test and pre-commit output are preserved as small text files.
- [TEST_REVIEW.md](TEST_REVIEW.md) records the follow-up test coverage review. Its proposed tests are not part of the archived 55-test result.
- `EVIDENCE.json` hashes the current tracked archive. `evidence/original_artifact_hashes.json` is the historical post-experiment inventory, including raw files retained outside Git. Reports have since gained archive instructions; historical hashes are not current report hashes.

## Data

The E2E workload is one manually selected prompt, `A minimalist exhibition poster with a red geometric sculpture`, repeated with seed 42. It is not a multi-prompt dataset. Operator checks use `torch.randn`, seed 42, random weights, representative Ming shapes and multiple input scales. The fixed model is `inclusionAI/Ming-Image-0.1-Design` at `208087ada1486931692c1896f38d4cd16ff3df82`.

## Rebuild and run

Requires Python 3.9+, Git, a local SGLang repository containing baseline `6fa3fe69e2e5e19b75cadd9fc285b72634551992`, and an authenticated Modal installation. Run from this directory:

```sh
python prepare_inputs.py --sglang /path/to/sglang
modal run --detach experiment.py --mode micro
modal run --detach experiment.py --mode full
```

The preparation command validates overlay hashes and recreates the measured `git archive` without changing SGLang's branch or including its uncommitted files. The full driver requires the pinned checkpoint in the existing `ming-image-hf-cache` Volume; offline mode is intentional. These Modal commands allocate a paid H200. The driver uses a moving container tag; a future rebuild can differ from the recorded software environment.

## Retrieve existing results without running a GPU

Modal Volume: `ming-rmsnorm-experiments`.

- Numerical tests: `micro-20261003T231823Z`.
- ABBA and traces: `full-20261003T232158Z`.
- Full application: https://modal.com/apps/xinyuj2/main/ap-jTzjvHkBKw61haxoAVtxcm

```sh
mkdir -p retrieved
modal volume get ming-rmsnorm-experiments micro-20261003T231823Z/ retrieved/
modal volume get ming-rmsnorm-experiments full-20261003T232158Z/ retrieved/
python analyze_results.py retrieved/full-20261003T232158Z
```

The destination must already be a directory. Selected raw traces, PNGs, launch logs and timing records are tracked in this evidence branch. Model weights, source tarballs, evidence tarballs and caches are excluded.
