# Ming-Image RMSNorm 实验结果

完成日期：2026-10-03（America/New_York）。分支 `perf/ming-image-rmsnorm`，基线 `6fa3fe69e2e5e19b75cadd9fc285b72634551992`。

## 结论

在本次固定 H200 eager 工作负载中，复用现有 `rmsnorm_preserve_reduction` 有明确收益：Denoise 中位耗时 **1692.129 → 1345.372 ms（-20.49%）**；请求中位耗时 **1802.948 → 1463.758 ms（-18.81%）**。这是本轮独立运行结果，不引用旧 trace 推算收益。

8 个计时请求和 2 个诊断请求全部完成 12 步，输出 1024×1024 PNG 像素完全一致，SHA256 `b04fe457e7a38a1007c8f23a1d0dd3c42ccd57afdb8c547d6670da661f82448f`。这说明该固定样例与基线输出一致，不是广泛的生成质量评测。

## 改动

只接入 MingRMSNorm：保留原生 FP32 mean reduction 的形状和算术顺序，融合转换/平方及归一化/cast/weight 乘法。现有共享 kernel 未修改。

每个 `(device, dtype, shape, eps)` 首次校验 `torch.equal`。不匹配时返回原生结果并禁用融合；非支持 dtype/layout/platform、residual、compile tracing 及未验证 shape 的 CUDA graph capture 使用原生路径。已验证的 shape 可参与 graph replay。

源码补丁包含一个模型文件、两个测试文件。未纳入工作区的 Anima 改动。候选模型 SHA256：`1bc9d2b61589a5ef5354ed5d637d9cf72a440c6d11087859f7de59fcdeb1ffc2`。

## 环境和计时

- GPU：NVIDIA H200，单 GPU、TP1，8 CPU cores / 64 GiB host RAM。
- Torch 2.13.0+cu130；Triton 3.7.1；Diffusers 0.37.0；Transformers 5.12.1。
- Checkpoint：`inclusionAI/Ming-Image-0.1-Design`，revision `208087ada1486931692c1896f38d4cd16ff3df82`；hidden 3840、cap feature 2560、head dim 128。
- Eager / FlashAttention，compile、BCG、offload 均关闭；1024×1024、12 步、CFG 1、seed 42、固定 prompt。
- 同卡 A B B A A B B A，每次新进程、一次 1-step request warmup，排除加载/预热。每组 4 个样本；不是用 profiler 的耗时作速度比较。
- Native denoise 极差/中位数：0.19%；candidate：0.28%。

| 组 | Denoise 合计 ms | 请求耗时 ms |
| --- | ---: | ---: |
| 01-A | 1692.787 | 1804.439 |
| 02-B | 1343.422 | 1461.649 |
| 03-B | 1343.768 | 1462.306 |
| 04-A | 1691.221 | 1799.795 |
| 05-A | 1691.472 | 1802.955 |
| 06-B | 1346.977 | 1465.211 |
| 07-B | 1347.240 | 1466.207 |
| 08-A | 1694.449 | 1802.942 |

## 正确性与测试

- 42 组独立数值测试：BF16/FP16，128/2560/3840 宽度，实际长序列和 batch 2，输入量级 1e-4/1/100，零不一致元素；同时验证 wrapper 实际调用融合。
- `test_rmsnorm_preserve_reduction.py`：44 passed，包含 dtype/shape/scale、layout guards、compile 和 graph replay。
- `test_model_fast_paths.py -k ming_rmsnorm`：11 passed；确认融合被调用、gate 已验证且未禁用，以及 FP32/strided/residual/mismatch fallback、wrapper compile tracing 和 warmed graph replay。wrapper compile 测试使用 backend=eager；共享 kernel 的 compile 测试使用默认 Inductor。
- 对三个改动文件运行完整 pre-commit：全部适用检查通过。未运行上游 CI、AMD、其他 GPU 或完整模型 compile/BCG。

## 独立 trace 证据

各选取完整的中间 profiler step，只选择 `cat=user_annotation` 的 CPU marker（Torch 还生成同名 GPU annotation，不能重复计数），利用 MingRMSNorm CPU marker 内 CUDA launch 的 CUPTI correlation ID 关联 GPU kernel。

| 指标 | Native | Candidate |
| --- | ---: | ---: |
| 每步 MingRMSNorm 调用 | 205 | 205 |
| 关联 GPU kernel 数 | 1640 | 615 |
| 关联 kernel duration 累加 ms | 46.641 | 14.744 |

候选诊断 dispatch：`{"calls": 2665, "fused": 2665, "shapes": {"((1, 256, 2560), (655360, 2560, 1), 'torch.bfloat16')": 13, "((1, 4096, 3840), (15728640, 3840, 1), 'torch.bfloat16')": 104, "((122880, 128), (128, 1), 'torch.bfloat16')": 52, "((1, 320, 3840), (1228800, 3840, 1), 'torch.bfloat16')": 104, "((9600, 128), (128, 1), 'torch.bfloat16')": 52, "((1, 4416, 3840), (16957440, 3840, 1), 'torch.bfloat16')": 1560, "((132480, 128), (128, 1), 'torch.bfloat16')": 780}, "gate_disabled": false, "verified_signatures": ["(device(type='cuda', index=0), torch.bfloat16, (1, 4416, 3840), 1e-05)", "(device(type='cuda', index=0), torch.bfloat16, (122880, 128), 1e-05)", "(device(type='cuda', index=0), torch.bfloat16, (9600, 128), 1e-05)", "(device(type='cuda', index=0), torch.bfloat16, (1, 4096, 3840), 1e-05)", "(device(type='cuda', index=0), torch.bfloat16, (1, 256, 2560), 1e-05)", "(device(type='cuda', index=0), torch.bfloat16, (132480, 128), 1e-05)", "(device(type='cuda', index=0), torch.bfloat16, (1, 320, 3840), 1e-05)"]}`。

Kernel duration 累加仅作机制诊断，不能直接当成请求节省时间；速度结论来自前面的 unprofiled ABBA。

## 微基准及限制

| 输入 shape | Native us/call | Candidate us/call | 耗时下降 |
| --- | ---: | ---: | ---: |
| `[1, 17, 128]` | 63.98 | 65.45 | -2.30% |
| `[1, 320, 2560]` | 64.46 | 67.02 | -3.97% |
| `[1, 320, 3840]` | 64.87 | 66.74 | -2.89% |
| `[1, 4096, 3840]` | 224.54 | 71.73 | 68.05% |
| `[1, 4416, 3840]` | 241.18 | 76.66 | 68.21% |
| `[132480, 128]` | 253.20 | 88.54 | 65.03% |
| `[2, 257, 3840]` | 64.47 | 66.15 | -2.59% |

微基准以 CUDA events 包住连续 Python 调用，包含发射间隙和分配，不是纯 kernel 吞吐。小张量有约 2%–4% 回退，大张量收益较大；没有掩盖或剔除小 shape。当前不外推其他分辨率、prompt、硬件、Design-Layer、多 GPU 或编译模式。

## 可审查材料

- `candidate.patch` / `source_manifest.json`：精确代码与哈希。
- `experiment.py` / `verify_rmsnorm.py` / `EXPERIMENT.md`：实验入口、固定条件和复现说明。
- `results/`：完整 manifest、原始日志、计时、图片、trace；`analysis.json` 与 `middle-step.trace.json.gz` 是分析产物。
- `pre-commit.log`：格式及静态检查日志。
- `PR_BODY_DRAFT.md`：依照仓库当前模板填写的英文 PR 草稿。

参考 [LTX-2 接线 PR #34315](https://github.com/sgl-project/sglang/pull/34315) 和 [Qwen-Image 2.1 PR #39983](https://github.com/sgl-project/sglang/pull/39983)：提交重点是数值契约、平台/shape 边界、真实融合证据、固定环境下的 accuracy/speed 结果和可复现命令。没有把这两篇 PR 的作者性能数据当作本轮证据。

完整 GitHub 重复选题检索未完成；公开检索结果/API 访问不完整，不能据此认定无人进行中。SGLang 生产代码仍是本地未提交改动，未 push 或创建 GitHub PR；RemoteFiles 实验材料单独归档提交。

## 运行记录

两次初始启动分别在读取不存在的 `sgl-kernel` 包元数据、假定 checkpoint 根目录有 `config.json` 时退出，均发生于正式计时之前。修正记录逻辑后，有效 full run 的全部请求成功；失败启动日志也保留在本目录。CLI 的 worker 强制清理提示存在于各组原始日志，所有组退出码为 0。

Modal 完整实验：[ap-jTzjvHkBKw61haxoAVtxcm](https://modal.com/apps/xinyuj2/main/ap-jTzjvHkBKw61haxoAVtxcm)。

## Git 归档

精简的原始计时 manifest、trace 分析、数值数据及测试日志位于受版本控制的 `evidence/`。完整原始结果、图片、trace 和旧证据压缩包保持本地忽略；Modal Volume 中的副本可按 README.md 的命令取回。`evidence/original_artifact_hashes.json` 保留实验结束时的哈希清单，`EVIDENCE.json` 则校验本次归档文件。
