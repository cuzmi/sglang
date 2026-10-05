# Ming-Image RMSNorm: published evidence

Code: [cuzmi/sglang#1](https://github.com/cuzmi/sglang/pull/1), head `8adb4c267d4f2b38d463817ed6c51b937ae89266`.

Measured baseline: `6fa3fe69e2e5e19b75cadd9fc285b72634551992`. The tested `ming_image.py` overlay has SHA256 `1bc9d2b61589a5ef5354ed5d637d9cf72a440c6d11087859f7de59fcdeb1ffc2`; publication verified this matches the model file at the PR head. Experimental test extensions in the archived patch/inputs were used for validation and are not part of the one-file production PR. The current PR target branch is newer than the measured baseline; no new-base rerun is claimed.

## Review entry points

- [Report with all eight timing rows, correctness and limitations](REPORT.md).
- [Controls and reproduction](EXPERIMENT.md), [archive/retrieval instructions](README.md), [source manifest](source_manifest.json).
- [Full raw experiment](results/full-20261003T232158Z/): manifests, per-process logs, per-request perf dumps, all generated PNGs, paired full profiler traces and middle-step slices.
- [Numerical and microbenchmark run](results/micro-20261003T231823Z/), [compact trace analysis](evidence/analysis.json), [test evidence](evidence/).

On the fixed H200 eager workload, four process samples per version gave median denoise 1692.13 to 1345.37 ms (-20.49%) and native total-request metric 1802.95 to 1463.76 ms (-18.81%). The native metric is preserved as originally measured; this publication does not add a synchronized request-boundary audit or reinterpret it as client/network E2E. The workload uses one prompt/seed, 1024x1024 and 12 steps. Small-shape call microbenchmarks regress around 2-4%; no universal gain or broad image-quality claim is made.

Raw measurements and test logs are preserved byte-for-byte. Scripts, source copies, patches, submission drafts and titles are excluded from this evidence branch. Their statements that code was not yet committed/pushed or raw results were ignored reflect the original archive date. Selected raw results are now published here. Preparation failure logs are retained; they are not timing samples. Source-tree/evidence tarballs and model weights are excluded. No new GPU run was started for this publication.
