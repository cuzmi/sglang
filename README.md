# SGLang PR results

Shared evidence branch: `codex/pr-results` in `cuzmi/sglang`.

This branch stores experiment evidence separately from production changes. It is not intended for merging into upstream `main`. Each experiment has its own directory; PRs should link to a fixed commit of this branch.

| Experiment | Review entry point | Code PR | Measured source |
| --- | --- | --- | --- |
| Anima modulation fusion | [Published evidence](diffusion-prs/anima-modulate-fusion/PUBLISHED.md) | [cuzmi/sglang#2](https://github.com/cuzmi/sglang/pull/2) | Base `affa261e3d289fe4f907c9b2e8d773fef0d36dba`, candidate `05bee50b2f4fef4484bab3db3bff7ef4495e98e5` |
| Ming-Image RMSNorm | [Published evidence](diffusion-prs/ming-image-rmsnorm/PUBLISHED.md) | [cuzmi/sglang#1](https://github.com/cuzmi/sglang/pull/1) | Base `6fa3fe69e2e5e19b75cadd9fc285b72634551992`; tested production file matches candidate `8adb4c267d4f2b38d463817ed6c51b937ae89266` |
| Anima FP32 RoPE | [Published evidence](diffusion-prs/anima-fp32-rope/PUBLISHED.md) | [sgl-project/sglang#42598](https://github.com/sgl-project/sglang/pull/42598) | Base `6fa3fe69e2`; candidate `6143cce2d2` |

## Evidence contents

Reports and limitations, raw measurements, correctness results, logs, generated images, compressed profiler traces, environment configuration, source revisions and hashes. Historical and negative results remain labelled in each experiment's entry point.

[Reproduction scripts, source copies and patches](https://github.com/cuzmi/sglang/tree/893495236a5ba25667313768b5e791fc51e355d8/diffusion-prs) are retained in this fixed snapshot and in the original `RemoteFiles` experiment archives. Historical commands and source paths in reports/manifests refer to that snapshot. They are not duplicated on this branch.

`SHA256SUMS.json` covers all current files except itself. `PUBLICATION.json` records original import provenance; its source hashes describe the imported archive before documentation cleanup. No measurements were rerun or changed.
