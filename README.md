# SGLang PR results

Shared evidence branch: `codex/pr-results` in `cuzmi/sglang`.

This branch stores experiment evidence separately from production changes. It is not intended for merging into upstream `main`. Each experiment has its own directory; PRs should link to a fixed commit of this branch.

| Experiment | Review entry point | Code PR | Measured source |
| --- | --- | --- | --- |
| Anima modulation fusion | [Published evidence](diffusion-prs/anima-modulate-fusion/PUBLISHED.md) | [cuzmi/sglang#2](https://github.com/cuzmi/sglang/pull/2) | Base `affa261e3d289fe4f907c9b2e8d773fef0d36dba`, candidate `05bee50b2f4fef4484bab3db3bff7ef4495e98e5` |
| Ming-Image RMSNorm | [Published evidence](diffusion-prs/ming-image-rmsnorm/PUBLISHED.md) | [cuzmi/sglang#1](https://github.com/cuzmi/sglang/pull/1) | Base `6fa3fe69e2e5e19b75cadd9fc285b72634551992`; tested production file matches candidate `8adb4c267d4f2b38d463817ed6c51b937ae89266` |
| Anima FP32 RoPE | [Published evidence](diffusion-prs/anima-fp32-rope/PUBLISHED.md) | [sgl-project/sglang#42598](https://github.com/sgl-project/sglang/pull/42598) | Base `6fa3fe69e2`; candidate `6143cce2d2` |

## Archive contents

Reports, reproduction/analysis scripts, numerical checks, source patches, environment manifests, raw run logs and timings, generated PNGs, and compressed profiler traces. Historical experiments, failed preparation logs, and negative results are retained and labelled in the experiment entry points. No new GPU job was run for this publication.

The original compact archives came from `cuzmi/RemoteFiles`; raw measurements, source inputs, scripts and test logs are preserved byte-for-byte. PR submission drafts, titles and template preparation notes are excluded; documentation and evidence indexes are maintained for the retained files. Their older README files may describe raw results as ignored or stored only on Modal. In this published branch the selected raw `results/` directories are also tracked explicitly. The experiment `PUBLISHED.md` files explain current applicability.

`PUBLICATION.json` records source paths and original import checksums; those import hashes describe the source archive before documentation cleanup. `SHA256SUMS.json` covers every published file other than itself. Model checkpoints, source-tree tarballs, redundant evidence tarballs, and caches are excluded. Some original reproduction scripts use account-scoped Modal images/volumes; publishing their results does not make those services publicly accessible.
