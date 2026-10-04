"""Correctness-only regression of shared modulation callers on one H200."""
from datetime import datetime,timezone
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4
import modal
HERE=Path(__file__).resolve().parent
BASE='affa261e3d289fe4f907c9b2e8d773fef0d36dba'
SOURCE=Path('/workspace/sglang-rebased')
REL=Path('python/sglang/kernels/ops/diffusion/modulate/modulate_scale_shift_jit.py')
EXPECTED='9fd588001fb3b7b9bd0abbc68c4f1210278cb19f40eea45ed74f649e2daa190e'
app=modal.App('anima-rebased-correctness')
image=(modal.Image.from_id('im-rRXDzAdc13HCyQHStCXqwE')
       .add_local_file(HERE.parent/'inputs/rebased_baseline.tar.gz','/workspace/rebased_baseline.tar.gz',copy=True)
       .run_commands('mkdir -p /workspace/sglang-rebased && tar -xzf /workspace/rebased_baseline.tar.gz -C /workspace/sglang-rebased')
       .add_local_file(HERE/'check_callers.py','/workspace/check_callers.py',copy=True)
       .add_local_file(HERE/'verify_anima.py','/workspace/verify_rebased_anima.py',copy=True)
       .add_local_file(HERE.parent/'inputs/rebased_backend.py','/workspace/candidate_backend.py',copy=True)
       .add_local_file(HERE.parent/'inputs/rebased_anima.py','/workspace/rebased_anima.py',copy=True)
       .env({'PYTHONPATH':'/workspace/sglang-rebased/python'}))
results=modal.Volume.from_name('anima-modulate-experiments')

def sha(data):return hashlib.sha256(data).hexdigest()

def run(command,path):
    with path.open('w') as log:
        process=subprocess.Popen(command,cwd=SOURCE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in process.stdout:
            log.write(line);log.flush()
            print(line,end='',flush=True)
        return process.wait()

@app.function(image=image,gpu='H200',cpu=8,memory=65536,timeout=1800,volumes={'/results':results})
def check():
    import torch
    run_id=datetime.now(timezone.utc).strftime('anima-rebased-correctness-%Y%m%dT%H%M%SZ-')+uuid4().hex[:8]
    out=Path('/results')/run_id;out.mkdir()
    baseline=(SOURCE/REL).read_bytes();candidate=Path('/workspace/candidate_backend.py').read_bytes()
    assert sha(candidate)==EXPECTED
    assert sha(Path('/workspace/rebased_baseline.tar.gz').read_bytes())=='bc6052c159cb5f1eef2236a7e8e82e06c9b0377ea79a7d1beb35c55ea664ac68'
    model_rel=Path('python/sglang/multimodal_gen/runtime/models/dits/anima.py')
    base_model=(SOURCE/model_rel).read_bytes();new_model=Path('/workspace/rebased_anima.py').read_bytes()
    assert sha(new_model)=='ae62fe965c2facdf224aa164a5002dc5bd2e403a48fb51cf6094179b4d254013'
    manifest={'status':'running','baseline_commit':BASE,'scope':'checkpoint-free real modulation entrypoints; no timing or full image/video generation','gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda,'packages':{p:md.version(p) for p in ['torch','triton','diffusers','transformers','apache-tvm-ffi']},'baseline_sha256':sha(baseline),'candidate_sha256':sha(candidate),'candidate_commit':'05bee50b2f4fef4484bab3db3bff7ef4495e98e5','candidate_model_sha256':sha(new_model),'arms':[],'errors':[]}
    def save():(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');results.commit()
    print('RUN_ID='+run_id,flush=True)
    commands=[('existing-kernel',[sys.executable,'-m','pytest','-q','test/registered/kernels/ops/diffusion/test_modulate.py','-k','modulate_scale_shift']),
              ('existing-model-sites',[sys.executable,'-m','pytest','-q','test/registered/kernels/ops/diffusion/test_model_fast_paths.py','-k','flux_fused_ln_modulate or flux_norm_modulate or ltx2_lossless']),
              ('existing-ltx-layout',[sys.executable,'-m','pytest','-q','python/sglang/multimodal_gen/test/unit/test_ltx2_modulate_mount.py']),
              ('existing-anima',[sys.executable,'-m','pytest','-q','python/sglang/multimodal_gen/test/unit/test_anima.py','python/sglang/multimodal_gen/test/unit/test_anima_cuda.py'])]
    try:
        for arm,data in [('baseline',baseline),('candidate',candidate)]:
            (SOURCE/REL).write_bytes(data)
            (SOURCE/model_rel).write_bytes(base_model if arm=='baseline' else new_model)
            folder=out/arm;folder.mkdir()
            info={'variant':arm,'backend_sha256':sha(data),'tests':{}}
            for label,cmd in [('caller-checks',[sys.executable,'/workspace/check_callers.py',str(folder)]),*commands]:
                print('RUN',arm,label,flush=True)
                code=run(cmd,folder/(label+'.log'));info['tests'][label]={'command':cmd,'exit_code':code}
                if code:manifest['errors'].append(arm+'/'+label)
                save()
            if arm=='candidate':
                cmd=[sys.executable,'/workspace/verify_rebased_anima.py',str(folder/'anima.json')]
                code=run(cmd,folder/'anima.log');info['tests']['full-adaln']={'command':cmd,'exit_code':code}
                if code:manifest['errors'].append('candidate/full-adaln')
            manifest['arms'].append(info);save()
        a=json.loads((out/'baseline/callers.json').read_text());b=json.loads((out/'candidate/callers.json').read_text())
        ah={r['label']:r['output_sha256'] for r in a['cases']};bh={r['label']:r['output_sha256'] for r in b['cases']}
        manifest['matching_case_outputs']=ah==bh
        if ah!=bh:manifest['errors'].append('cross-version output mismatch')
        manifest['status']='complete' if not manifest['errors'] else 'failed'
    except Exception as exc:
        manifest['status']='failed';manifest['errors'].append(repr(exc));raise
    finally:save()
    return run_id

@app.local_entrypoint()
def main():print('RESULTS',check.remote())
