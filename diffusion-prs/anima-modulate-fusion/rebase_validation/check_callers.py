"""Run real model modulation entrypoints without downloading checkpoints."""
import hashlib
import json
from pathlib import Path
import sys
from contextlib import ExitStack
from unittest.mock import patch

import torch
from torch import nn
import sglang.kernels.ops.diffusion.modulate.modulate_scale_shift_jit as backend
from sglang.kernels.ops.diffusion import BitExactFusionGate
from sglang.multimodal_gen.runtime.models.dits import flux, ideogram, lingbot_world, ltx_2


def bits_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def digest(t):
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def build(model, batch, seq, dtype, lane, stack):
    width = {'flux':3072, 'ideogram':4608, 'lingbot':5120, 'ltx_video':4096, 'ltx_audio':2048}[model]
    torch.manual_seed(42)
    x=torch.randn(batch,seq,width,device='cuda',dtype=dtype)
    table=torch.randn(batch,1,6*width,device='cuda',dtype=dtype)*0.1
    shift,scale=table.chunk(6,-1)[:2]
    if model=='flux':
        gate=BitExactFusionGate('cross-model flux',per_signature=True)
        gate.disabled=lane=='fallback'
        stack.enter_context(patch.object(flux,'_FLUX_LN_MOD',gate))
        stack.enter_context(patch.object(flux,'_FLUX_LN_MOD_SIGS',set()))
        norm=nn.LayerNorm(width,eps=1e-6,elementwise_affine=False).cuda()
        site=nn.Module()
        fn=lambda:flux._flux_norm_modulate(site,norm,x,scale.squeeze(1),shift.squeeze(1))
        ref=lambda:norm(x)*(1+scale)+shift
    elif model=='ideogram':
        stack.enter_context(patch.object(ideogram,'_IDEOGRAM_ZERO_SHIFTS',{}))
        norm=ideogram.Ideogram4RMSNorm(width).cuda().to(dtype)
        fn=lambda:ideogram._norm_scale(x,scale,norm,False)
        ref=lambda:norm(x)*(1+scale)
    elif model=='lingbot':
        # Actual forward with precomputed camera scale/shift, a supported public
        # argument. Unrelated camera-projection weights are never accessed.
        module=lingbot_world.LingBotWorldCamConditioner.__new__(lingbot_world.LingBotWorldCamConditioner)
        nn.Module.__init__(module)
        cs=torch.randn_like(x)*0.1;ct=torch.randn_like(x)*0.1
        if lane=='noncontiguous':
            x=x.transpose(1,2).contiguous().transpose(1,2)
        fn=lambda:module(x,x,(cs,ct))
        ref=lambda:(1+cs)*x+ct
    else:
        stack.enter_context(patch.object(ltx_2,'_LTX2_MODULATE',BitExactFusionGate('cross-model ltx')))
        norm=ltx_2.RMSNormNoWeight()
        block=nn.Module()
        if lane=='per_token':
            scale=torch.randn_like(x)*0.1;shift=torch.randn_like(x)*0.1
        fn=lambda:ltx_2._ltx2_rms_norm_modulate(block,norm,x,scale,shift,1e-6)
        ref=lambda:norm(x,1e-6)*(1+scale)+shift
    return x,fn,ref


@torch.inference_mode()
def main(destination):
    out=Path(destination);out.mkdir(exist_ok=True,parents=True)
    report={'status':'running','scope':'real model modulation components, no checkpoints or full generation','cases':[],'graphs':[],'errors':[]}
    def save(): (out/'callers.json').write_text(json.dumps(report,indent=2)+'\n')
    models=['flux','ideogram','lingbot','ltx_video','ltx_audio']
    for model in models:
        lanes={'flux':['normal','fallback'],'lingbot':['normal','noncontiguous'],'ltx_video':['normal','per_token'],'ltx_audio':['normal']} .get(model,['normal'])
        for lane in lanes:
            for dtype in [torch.bfloat16,torch.float16,torch.float32]:
                for batch in [1,2]:
                    label=f'{model}/{lane}/{dtype}/{batch}'
                    try:
                        with ExitStack() as stack:
                            pass  # Upstream no longer keeps a disabled-runtime registry.
                            x,fn,ref=build(model,batch,17,dtype,lane,stack)
                            original=backend.modulate_scale_shift_cuda
                            with patch.object(backend,'modulate_scale_shift_cuda',wraps=original) as spy, patch.object(ltx_2,'modulate_scale_shift_cuda',spy):
                                actual=fn();first=spy.call_count
                                again=fn();repeat=spy.call_count-first
                            assert bits_equal(actual,ref()) and bits_equal(again,ref()),'bit mismatch'
                            low=dtype!=torch.float32
                            # Mandatory dispatch where this specific shared path is eligible.
                            mandatory=low and ((model=='flux' and batch==1) or (model=='ideogram' and batch==1) or (model=='lingbot' and lane=='normal') or (model.startswith('ltx') and lane=='normal'))
                            if mandatory: assert first>0,'expected shared CUDA call missing'
                            if not low or lane in ['noncontiguous','per_token'] or (model=='ideogram' and batch==2):
                                assert first==repeat==0,'unexpected shared CUDA dispatch'
                            report['cases'].append({'label':label,'shape':list(x.shape),'bit_exact':True,'first_fused_calls':first,'repeat_fused_calls':repeat,'output_sha256':digest(actual)})
                    except Exception as exc:
                        report['errors'].append({'label':label,'error':repr(exc)})
                    save()
    # Current main propagates supported-input kernel errors; it no longer
    # swallows them or disables future attempts. Check that caching preserves it.
    try:
        x=torch.randn(1,17,2048,device='cuda',dtype=torch.bfloat16);s=torch.randn(1,2048,device='cuda',dtype=x.dtype);t=torch.randn_like(s)
        with patch.object(backend,'modulate_scale_shift_cuda',side_effect=RuntimeError('intentional regression probe')) as failed:
            for _ in range(2):
                try: backend.modulate_scale_shift(x,s,t)
                except RuntimeError as exc: assert str(exc)=='intentional regression probe'
                else: raise AssertionError('Expected kernel error propagation')
            assert failed.call_count==2
        report['shared_error_propagation']='passed'
    except Exception as exc: report['errors'].append({'label':'shared_error_propagation','error':repr(exc)})
    for model in models:
        try:
            with ExitStack() as stack:
                pass  # Upstream no longer keeps a disabled-runtime registry.
                lane='fallback' if model=='flux' else 'normal'
                x,fn,ref=build(model,1,17,torch.bfloat16,lane,stack)
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):fn()
                torch.cuda.current_stream().wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph): actual=fn()
                x.add_(0.03125);graph.replay();torch.cuda.synchronize()
                assert bits_equal(actual,ref()),'graph replay mismatch'
                report['graphs'].append({'model':model,'bit_exact':True,'input_changed_before_replay':True})
        except Exception as exc: report['errors'].append({'label':model+'/graph','error':repr(exc)})
        save()
    report['status']='complete' if not report['errors'] else 'failed'
    save();print(json.dumps(report,indent=2))
    if report['errors']:raise RuntimeError(report['errors'])

if __name__=='__main__':main(sys.argv[1])
