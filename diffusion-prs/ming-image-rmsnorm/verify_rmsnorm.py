"""GPU correctness and ABBA microbench for the pinned Ming RMSNorm experiment."""
import json
import sys
from pathlib import Path
from unittest.mock import patch
import torch
import sglang.multimodal_gen.runtime.models.dits.ming_image as ming
from sglang.srt.layers.layernorm import RMSNorm
from sglang.kernels.ops.diffusion import BitExactFusionGate, rmsnorm_preserve_reduction


def bench(fn):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 10)
    return values


@torch.inference_mode()
def main():
    report = {"correctness": [], "microbenchmark": []}
    torch.manual_seed(42)
    shapes = [(1, 17, 128), (1, 320, 2560), (1, 320, 3840), (1, 4096, 3840), (1, 4416, 3840), (132480, 128), (2, 257, 3840)]
    for dtype in [torch.bfloat16, torch.float16]:
        for shape in shapes:
            for scale in [1e-4, 1., 100.]:
                layer = ming.MingRMSNorm(shape[-1]).cuda().to(dtype)
                layer.weight.normal_()
                x = (torch.randn(shape, device='cuda') * scale).to(dtype)
                ref = RMSNorm.forward_native(layer, x)
                raw = rmsnorm_preserve_reduction(x, layer.weight, layer.variance_epsilon)
                entry = dict(shape=list(shape), dtype=str(dtype), scale=scale, mismatch=int((raw != ref).sum()), max_abs=float((raw.float()-ref.float()).abs().max()))
                report['correctness'].append(entry)
                if entry['mismatch']:
                    Path(sys.argv[1]).write_text(json.dumps(report, indent=2))
                    raise AssertionError(entry)
                with patch.object(ming, '_MING_RMSNORM_FUSION', BitExactFusionGate('test', per_signature=True)):
                    with patch.object(ming, 'rmsnorm_preserve_reduction', wraps=rmsnorm_preserve_reduction) as spy:
                        for _ in range(2):
                            assert torch.equal(layer(x), ref)
                        assert spy.call_count == 2
                        assert ming._MING_RMSNORM_FUSION.verified and not ming._MING_RMSNORM_FUSION.disabled
    for shape in shapes:
        layer = ming.MingRMSNorm(shape[-1]).cuda().bfloat16()
        layer.weight.normal_()
        x = torch.randn(shape, device='cuda', dtype=torch.bfloat16)
        eager = lambda: RMSNorm.forward_native(layer, x)
        fused = lambda: layer(x)
        assert torch.equal(eager(), fused())
        timings = [bench(fn) for fn in [eager, fused, fused, eager]]
        report['microbenchmark'].append(dict(shape=list(shape), abba_us=timings))
    Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + '\n')
    print('PASS', len(report['correctness']), 'exact cases; microbench saved', flush=True)

if __name__ == '__main__':
    main()
