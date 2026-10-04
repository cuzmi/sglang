# Anima modulation fusion: published evidence

Code: [cuzmi/sglang#2](https://github.com/cuzmi/sglang/pull/2), head `05bee50b2f4fef4484bab3db3bff7ef4495e98e5`, base `affa261e3d289fe4f907c9b2e8d773fef0d36dba`.

## Current-head evidence

- [Synchronized worker-latency report](rebased_perf/REPORT-synchronized.txt): four fresh process samples per version, ABBA + BAAB, 3185.73 to 3065.52 ms mean (-3.77%) on one H200 and one fixed workload. Identical timing-only synchronization is applied to both arms. This excludes loading, warmup, transport, and PNG saving; it is not client E2E.
- [Performance controls and reproduction](rebased_perf/README.txt), [timing patch](rebased_perf/evidence/synchronized/timing.patch), [summary](rebased_perf/evidence/synchronized/summary.json).
- [Raw synchronized run, logs, timings and all eight PNGs](results/anima-rebased-sync-perf-20261004T164048Z-a9b075f6/).
- [Separate native-timer report](rebased_perf/REPORT-native.txt) and [raw native run](results/anima-rebased-perf-20261004T163234Z-b2ba18da/). This timer may omit trailing CUDA work; the two runs use separate instances and are not pooled.
- [Rebased correctness report](rebase_validation/REPORT.txt), [source provenance](rebase_validation/provenance.json), [validation scripts](rebase_validation/README.txt), and [raw component results](results/anima-rebased-correctness-20261004T161931Z-54a6bdc1/).

The two production source hashes were checked against the current PR commit when publishing. The 16 full-model outputs from the two performance runs have matching bytes/pixels per the archived validation. There is no independent A/A control or cross-instance replication of the synchronized experiment. No new GPU measurements were made for publication.

## Historical evidence

[Original archive](README.txt), [initial regression](evidence/initial/REPORT.txt), [older combined experiment](evidence/dispatch/REPORT.txt), and [cross-model checks](cross_model/README.txt) use the older base `6fa3fe69e2`. They must not be presented as measurements of the rebased PR. The old duplicate-check removal and error-fallback contract no longer describe the current delta.

[Original raw runs and traces](results/anima-20261003T224511Z-017ed58d/) include both full traces and middle-step slices. The rebased implementation was not reprofiled. Other historical runs are retained under `results/` with their original run IDs.

The original archive files are unchanged. Statements in older READMEs/drafts that results were ignored, tests were uncommitted, or a PR had not been created are historical. This branch publishes the selected raw results alongside those reports. Checkpoint/source tarballs and caches remain excluded.
