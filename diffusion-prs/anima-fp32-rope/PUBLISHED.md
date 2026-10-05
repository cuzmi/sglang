# Anima FP32 RoPE experiment evidence

Code PR: [sgl-project/sglang#42598](https://github.com/sgl-project/sglang/pull/42598).

Measured runtime: `6143cce2d2f5701b07ae155ad652705cabe7b646`, baseline `6fa3fe69e2e5e19b75cadd9fc285b72634551992`.
Run: `rope-20261004T161949Z-d5819700`. No new GPU runs were needed for publication.

On H200, 1024×1024, 30 steps, seed 42, CFG 4, FlashAttention, compile/BCG off:

| Metric | Baseline | Fused | Reduction |
| --- | ---: | ---: | ---: |
| Denoise | 2843.80 ms | 2250.42 ms | 20.87% |
| Framework request duration | 3110.29 ms | 2520.63 ms | 18.96% |

Same-container A/A + ABBA, two independent processes per variant; A/A denoise difference 0.07%. This is one configuration, not a universal speed claim. Request time excludes model loading/startup.

50 standalone numerical cases and 17 existing Anima tests passed. All eight PNGs are byte-identical. The paired operator launches 16 → 1 kernels; full-model trace confirms the fused path.

- [Full report and reproduction](README.txt)
- [Raw manifest](results/rope-20261004T161949Z-d5819700/manifest.json)
- [Analyzed summary](results/rope-20261004T161949Z-d5819700/summary.json)
- [Numerical checks](results/rope-20261004T161949Z-d5819700/operator.json)
- [Experiment runner](https://github.com/cuzmi/sglang/tree/893495236a5ba25667313768b5e791fc51e355d8/diffusion-prs/anima-fp32-rope/experiment.py) and [standalone validation](https://github.com/cuzmi/sglang/tree/893495236a5ba25667313768b5e791fc51e355d8/diffusion-prs/anima-fp32-rope/verify_rope.py)
- [Baseline image](pr_assets/baseline.png) / [fused image](pr_assets/fused.png)
- [Original logs, images and compressed traces](results/rope-20261004T161949Z-d5819700)

Raw measurements and validation logs retain their original bytes. `EVIDENCE.json` indexes the retained evidence; [image evidence](IMAGE_EVIDENCE.txt) records the output hashes and generation settings. `ARCHIVE_MANIFEST.json` records this directory's current file hashes; the root `SHA256SUMS.json` covers the complete results branch. Account-scoped Modal image/volume references are preserved and are not made publicly accessible by this archive.
