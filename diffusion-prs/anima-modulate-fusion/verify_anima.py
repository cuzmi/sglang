"""Standalone experiment validation, kept outside the SGLang test suite."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from sglang.kernels.ops.diffusion import modulate_scale_shift_cuda
from sglang.multimodal_gen.runtime.models.dits.anima import AnimaAdaLayerNorm
import sglang.kernels.ops.diffusion.modulate.modulate_scale_shift_jit as backend


def reference(layer, x, embedded, temb):
    modulation = layer.linear_1(torch.nn.functional.silu(embedded))[0]
    modulation = layer.linear_2(modulation)[0]
    modulation = modulation + temb[..., :modulation.shape[-1]]
    values = modulation.unsqueeze(1).chunk(3 if layer.gated else 2, dim=-1)
    out = layer.norm(x) * (1 + values[1]) + values[0]
    return (out, values[2]) if layer.gated else out


def equal(a, b):
    if isinstance(a, tuple):
        return all(torch.equal(x, y) for x, y in zip(a, b))
    return torch.equal(a, b)


def bench(fn, repetitions=100):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / repetitions


@torch.inference_mode()
def main():
    torch.manual_seed(42)
    report = {"correctness": [], "microbenchmark": []}
    original = backend._modulate_scale_shift_custom_op
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        for batch in [1, 2]:
            for tokens in [17, 1024, 4096]:
                for gated in [False, True]:
                    layer = AnimaAdaLayerNorm(2048, 256, gated=gated).cuda().to(dtype)
                    for p in layer.parameters():
                        p.normal_(std=0.02)
                    x = torch.randn(batch, tokens, 2048, device="cuda", dtype=dtype)
                    embedded = torch.randn(batch, 2048, device="cuda", dtype=dtype)
                    temb = torch.randn(batch, 6144, device="cuda", dtype=dtype)
                    expected = reference(layer, x, embedded, temb)
                    with patch.object(backend, "_modulate_scale_shift_custom_op", wraps=original) as spy:
                        actual = layer(x, embedded, temb)
                    assert equal(actual, expected), (dtype, batch, tokens, gated)
                    assert spy.call_count == (0 if dtype == torch.float32 else 1), "Unexpected dispatch/fallback"
                    report["correctness"].append(dict(dtype=str(dtype), batch=batch, tokens=tokens, gated=gated, bit_exact=True, fused_calls=spy.call_count))
    # Warmed CUDA graph replay of the production forward, including batched copies.
    layer = AnimaAdaLayerNorm(2048, 256).cuda().bfloat16()
    for p in layer.parameters():
        p.normal_(std=0.02)
    x = torch.randn(2, 17, 2048, device="cuda", dtype=torch.bfloat16)
    embedded = torch.randn(2, 2048, device="cuda", dtype=torch.bfloat16)
    temb = torch.randn(2, 6144, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            layer(x, embedded, temb)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = layer(x, embedded, temb)
    graph.replay()
    torch.cuda.synchronize()
    assert equal(captured, reference(layer, x, embedded, temb))
    report["cuda_graph_replay_bit_exact"] = True
    for dtype in [torch.bfloat16, torch.float16]:
        for batch in [1, 2]:
            for tokens in [1024, 4096]:
                x = torch.randn(batch, tokens, 2048, device="cuda", dtype=dtype)
                shift, scale, _ = torch.randn(batch, 1, 6144, device="cuda", dtype=dtype).chunk(3, -1)
                def eager():
                    return x * (1 + scale) + shift
                def fused():
                    return modulate_scale_shift_cuda(x, scale.squeeze(1).contiguous(), shift.squeeze(1).contiguous())
                assert torch.equal(eager(), fused())
                values = [bench(fn) for fn in [eager, fused, fused, eager]]
                report["microbenchmark"].append(dict(dtype=str(dtype), batch=batch, tokens=tokens, hidden=2048, abba_us=values, includes_row_copies=True))
    Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
