"""Standalone correctness/measurement experiment, not a repository unit test."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch
import triton
from sglang.kernels.ops.diffusion import (
    BitExactFusionGate, can_use_fused_rope_rotate_half_fp32,
    fused_rope_rotate_half_fp32, tensors_equal,
)
import sglang.multimodal_gen.runtime.models.dits.anima as model


def bits_equal(a, b):
    return all(torch.equal(x.contiguous().view(torch.int16), y.contiguous().view(torch.int16)) for x, y in zip(a, b))


@torch.inference_mode()
def main():
    torch.manual_seed(20261004)
    report = dict(correctness=[], fallback=[], microbenchmark=[])
    for dtype in (torch.bfloat16, torch.float16):
        for batch in (1, 2):
            for seq in (1, 17, 1024, 4096):
                for heads in (1, 8, 16):
                    q, k = [torch.randn(batch, seq, heads, 128, device='cuda', dtype=dtype) for _ in range(2)]
                    # Independent half rows also catch an accidental shared-half assumption.
                    angles = torch.randn(seq, 128, device='cuda') * 50
                    cos, sin = angles.cos(), angles.sin()
                    saved = [t.clone() for t in (q, k, cos, sin)]
                    ref = model._anima_rope_eager(q, k, cos, sin)
                    direct = fused_rope_rotate_half_fp32(q, k, cos, sin)
                    assert bits_equal(ref, direct), (dtype, batch, seq, heads)
                    mounted = model._anima_rope(q, k, cos, sin)
                    assert bits_equal(ref, mounted)
                    assert model._ANIMA_ROPE.is_verified((q.device, q.dtype, q.shape))
                    assert not model._ANIMA_ROPE.disabled
                    assert all(torch.equal(a,b) for a,b in zip(saved,(q,k,cos,sin)))
                    report['correctness'].append(dict(dtype=str(dtype), shape=list(q.shape), bit_exact=True, mounted=True, nonmutation=True))
        # Signed zero, extremes, and small finite values, with no overflow to NaN.
        vals = torch.tensor([0., -0., 1., -1., torch.finfo(dtype).tiny, -torch.finfo(dtype).tiny,
                             torch.finfo(dtype).max, -torch.finfo(dtype).max],device='cuda',dtype=dtype)
        q = vals.repeat(16).view(1,1,1,128)
        k = q.flip(-1).contiguous()
        cos = torch.full((1,128),0.5,device='cuda'); sin=cos.clone()
        assert bits_equal(fused_rope_rotate_half_fp32(q,k,cos,sin), model._anima_rope_eager(q,k,cos,sin))
        report['correctness'].append(dict(dtype=str(dtype),case='finite_edges',bit_exact=True))

    q,k=[torch.randn(2,17,16,128,device='cuda',dtype=torch.bfloat16) for _ in range(2)]
    a=torch.randn(17,128,device='cuda');cos,sin=a.cos(),a.sin()
    for label, args in [('fp32',(q.float(),k.float(),cos,sin)),('strided',(q[:,:,::2],k[:,:,::2],cos,sin)),
                        ('small_dim',(q[...,:64],k[...,:64],cos[:,:64],sin[:,:64])),
                        ('cos_bf16',(q,k,cos.bfloat16(),sin.bfloat16())),
                        ('cpu',tuple(t.cpu() for t in (q,k,cos,sin)))]:
        assert not can_use_fused_rope_rotate_half_fp32(*args)
        assert tensors_equal(model._anima_rope(*args),model._anima_rope_eager(*args))
        report['fallback'].append(label)
    with torch.inference_mode(False), torch.enable_grad():
        args=[t.clone().requires_grad_() for t in (q,k,cos,sin)]
        assert not can_use_fused_rope_rotate_half_fp32(*args)
        out=model._anima_rope(*args)
        sum(t.float().sum() for t in out).backward()
        grads=[t.grad.clone() for t in args]
        args2=[t.detach().clone().requires_grad_() for t in args]
        sum(t.float().sum() for t in model._anima_rope_eager(*args2)).backward()
        assert all(torch.equal(g,t.grad) for g,t in zip(grads,args2))
        report['fallback'].append('autograd_exact')
    with patch('sglang.kernels.ops.diffusion.fused_rope_rotate_half_fp32',side_effect=RuntimeError('injected')) as spy:
        model._ANIMA_ROPE=BitExactFusionGate('injected',per_signature=True)
        for _ in range(2):
            assert tensors_equal(model._anima_rope(q,k,cos,sin),model._anima_rope_eager(q,k,cos,sin))
        assert spy.call_count==1 and model._ANIMA_ROPE.disabled
    report['fallback'].append('exception_no_retry')
    model._ANIMA_ROPE=BitExactFusionGate('mismatch',per_signature=True)
    with patch('sglang.kernels.ops.diffusion.fused_rope_rotate_half_fp32',return_value=(torch.zeros_like(q),torch.zeros_like(k))):
        assert tensors_equal(model._anima_rope(q,k,cos,sin),model._anima_rope_eager(q,k,cos,sin))
        assert model._ANIMA_ROPE.disabled
    report['fallback'].append('mismatch')
    model._ANIMA_ROPE=BitExactFusionGate('restored',per_signature=True)
    # First capture must stay eager; warmed capture must use the verified kernel.
    for warmed in (False,True):
        model._ANIMA_ROPE=BitExactFusionGate('capture',per_signature=True)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                (model._anima_rope if warmed else model._anima_rope_eager)(q,k,cos,sin)
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result=model._anima_rope(q,k,cos,sin)
        graph.replay();torch.cuda.synchronize()
        assert bits_equal(result,model._anima_rope_eager(q,k,cos,sin))
        assert model._ANIMA_ROPE.is_verified((q.device,q.dtype,q.shape)) == warmed
    report['cuda_graph_first_and_warmed']=True
    compiled=torch.compile(model._anima_rope,fullgraph=True)
    assert tensors_equal(compiled(q,k,cos,sin),torch.compile(model._anima_rope_eager,fullgraph=True)(q,k,cos,sin))
    report['compile_preserves_compiled_eager']=True

    for seq in (1024,4096,16384):
        q,k=[torch.randn(1,seq,16,128,device='cuda',dtype=torch.bfloat16) for _ in range(2)]
        a=torch.randn(seq,128,device='cuda');cos,sin=a.cos(),a.sin()
        eager=lambda:model._anima_rope_eager(q,k,cos,sin)
        fused=lambda:model._anima_rope(q,k,cos,sin)
        assert bits_equal(eager(),fused())
        vals=[triton.testing.do_bench(fn,warmup=100,rep=300) * 1000 for fn in (eager,eager,eager,fused,fused,eager)]
        report['microbenchmark'].append(dict(shape=list(q.shape),order=['A0','A1','A','B','B','A'],us=vals))
        if seq==4096:
            for label,fn in [('eager',eager),('fused',fused)]:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
                    for _ in range(10):fn()
                    torch.cuda.synchronize()
                p.export_chrome_trace(str(Path(sys.argv[1]).parent/f'operator-{label}.json'))
    Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+'\n')
    print('OPERATOR_PASS',len(report['correctness']),json.dumps(report['microbenchmark']),flush=True)

if __name__=='__main__':main()
