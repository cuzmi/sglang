"""Controlled H200 geometry and host-dispatch screening for Anima modulation.

No production source edits. Kernel variants only change launch geometry.
CUDA-graph throughput isolates repeated device execution from host submission.
Host measurements deliberately include submission cost and are not kernel time.
"""
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import random
import statistics
import time

import torch
from sglang.kernels.jit.utils import load_jit, make_cpp_args
import sglang.kernels.ops.diffusion.modulate.modulate_scale_shift_jit as backend
from sglang.multimodal_gen.runtime.models.dits import anima


def median(values):
    return statistics.median(values)


def calls(fn, count=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    t = time.perf_counter_ns()
    for _ in range(count):
        fn()
    enqueued = (time.perf_counter_ns() - t) / count / 1000
    end.record()
    end.synchronize()
    return {"host_enqueue_us": enqueued, "event_elapsed_us": start.elapsed_time(end) * 1000 / count}


def capture(fn, count=64):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(10):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(count):
            fn()
    return graph, count


def replay(spec, count=20):
    graph, inner = spec
    for _ in range(3):
        graph.replay()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(count):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / count / inner


@torch.inference_mode()
def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
              "packages": {p: md.version(p) for p in ["torch", "triton", "apache-tvm-ffi"]},
              "geometry": [], "dispatch": [], "adaln": []}
    def save():
        (out / "screen.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.manual_seed(42)
    rng = random.Random(42)
    dtype = torch.bfloat16
    module = backend._jit_modulate_scale_shift_module(dtype)
    ffi = module.modulate_scale_shift
    kernels = {"r4_t256": ffi}
    text = Path('/workspace/sglang/python/sglang/kernels/jit/csrc/diffusion/modulate_scale_shift.cuh').read_text()
    report["original_kernel_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    args = make_cpp_args(dtype)
    for rows, threads in [(1,128),(1,256),(2,128),(2,256),(4,128),(8,128),(8,256),(16,256)]:
        name = f"r{rows}_t{threads}"
        src = text.replace('kRowsPerBlock = 4;', f'kRowsPerBlock = {rows};').replace('kColsVecPerBlock = 256;', f'kColsVecPerBlock = {threads};')
        path = out / (name + '.cuh')
        path.write_text(src)
        print('COMPILE', name, flush=True)
        mod = load_jit('anima_geometry', name, *args, cuda_files=[str(path)],
                       cuda_wrappers=[('modulate_scale_shift', f'modulate_scale_shift::ModulateScaleShiftKernel<{args}>::run')])
        kernels[name] = mod.modulate_scale_shift

    # Same CUDA implementation; only the Python dispatch differs.
    original_public = backend.modulate_scale_shift
    def cached_unchecked(x, scale, shift):
        output = torch.empty_like(x)
        ffi(output, x, scale, shift)
        return output
    def checked_cached(x, scale, shift):
        if backend.can_use_modulate_scale_shift_cuda(x, scale, shift):
            return cached_unchecked(x, scale, shift)
        return x * (1 + scale[:, None]) + shift[:, None]
    functions = {
        'eager': lambda x,s,t: x * (1 + s[:, None]) + t[:, None],
        'public_double_guard': original_public,
        'single_guard': backend.modulate_scale_shift_cuda,
        'op_without_python_guard': backend._modulate_scale_shift_custom_op,
        'checked_cached_ffi': checked_cached,
        'cached_ffi_without_python_guard': cached_unchecked,
    }
    for batch, seq in [(1,4096),(1,1024),(2,4096),(2,1024)]:
        shape = (batch,seq,2048)
        print('SHAPE',shape,flush=True)
        x = torch.randn(shape, device='cuda', dtype=dtype)
        shift3, scale3, _ = torch.randn(batch,1,6144,device='cuda',dtype=dtype).chunk(3,-1)
        scale, shift = scale3.squeeze(1).contiguous(), shift3.squeeze(1).contiguous()
        expected = x*(1+scale3)+shift3
        output = torch.empty_like(x)
        graphs = {}
        for name, kernel in kernels.items():
            kernel(output,x,scale,shift)
            assert torch.equal(output,expected), (name,shape)
            graphs[name] = capture(lambda kernel=kernel: kernel(output,x,scale,shift))
        values = {n: [] for n in graphs}
        for _ in range(7):
            order = list(graphs); rng.shuffle(order)
            for name in order:
                values[name].append(replay(graphs[name]))
        for name, samples in values.items():
            report['geometry'].append(dict(shape=shape,variant=name,bit_exact=True,graph_us=samples,median_graph_us=median(samples)))
        del graphs
        # Real Anima view preparation is included in every fused path here.
        wrapped = {name: (lambda fn=fn: fn(x,scale3.squeeze(1).contiguous(),shift3.squeeze(1).contiguous())) for name,fn in functions.items()}
        wrapped['eager'] = lambda: x*(1+scale3)+shift3
        samples = {n:[] for n in wrapped}
        for name, fn in wrapped.items():
            assert torch.equal(fn(),expected), (name,shape)
        for _ in range(7):
            order=list(wrapped);rng.shuffle(order)
            for name in order:
                samples[name].append(calls(wrapped[name]))
        for name, data in samples.items():
            report['dispatch'].append(dict(shape=shape,variant=name,bit_exact=True,samples=data,
                                            median_enqueue_us=median([s['host_enqueue_us'] for s in data]),
                                            median_event_elapsed_us=median([s['event_elapsed_us'] for s in data])))
        save()
    # Target-sized full AdaLN, all GEMMs and LayerNorm retained.
    layer = anima.AnimaAdaLayerNorm(2048,256).cuda().bfloat16()
    for p in layer.parameters(): p.normal_(std=0.02)
    x=torch.randn(1,4096,2048,device='cuda',dtype=dtype)
    emb=torch.randn(1,2048,device='cuda',dtype=dtype)
    temb=torch.randn(1,6144,device='cuda',dtype=dtype)
    original_forward = layer.forward
    def eager_forward(x, embedded_timestep, temb):
        modulation = layer.linear_1(torch.nn.functional.silu(embedded_timestep))[0]
        modulation = layer.linear_2(modulation)[0]
        modulation = modulation + temb[..., :modulation.shape[-1]]
        values = modulation.unsqueeze(1).chunk(3, dim=-1)
        return layer.norm(x) * (1 + values[1]) + values[0], values[2]
    expected=eager_forward(x,emb,temb)
    selected=['eager','public_double_guard','single_guard','checked_cached_ffi','cached_ffi_without_python_guard']
    data={name:[] for name in selected}
    for _ in range(7):
        order=selected.copy();rng.shuffle(order)
        for name in order:
            anima.modulate_scale_shift=functions[name]
            layer.forward = eager_forward if name == 'eager' else original_forward
            actual=layer(x,emb,temb)
            assert all(torch.equal(a,b) for a,b in zip(expected,actual)),name
            data[name].append(calls(lambda: layer(x,emb,temb)))
    anima.modulate_scale_shift=original_public
    layer.forward=original_forward
    for name,samples in data.items():
        report['adaln'].append(dict(variant=name,bit_exact=True,samples=samples,
                                   median_enqueue_us=median([s['host_enqueue_us'] for s in samples]),
                                   median_event_elapsed_us=median([s['event_elapsed_us'] for s in samples])))
    report['status']='complete'
    save()
    for section in ['geometry','dispatch','adaln']:
        print(section.upper(),json.dumps(report[section]),flush=True)

if __name__=='__main__':
    import sys
    main(sys.argv[1])
