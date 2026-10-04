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
BASE='6fa3fe69e2e5e19b75cadd9fc285b72634551992'
SOURCE=Path('/workspace/sglang')
REL=Path('python/sglang/kernels/ops/diffusion/modulate/modulate_scale_shift_jit.py')
EXPECTED='5705525464fedef85e83631e44aa53ce87bd7a7b1b83d7294a25c22621bcc8b0'
app=modal.App('anima-shared-modulation-correctness')
image=(modal.Image.from_id('im-rRXDzAdc13HCyQHStCXqwE')
       .add_local_file(HERE/'check_callers.py','/workspace/check_callers.py',copy=True)
       .add_local_file(HERE.parent/'inputs/dispatch_modulate_scale_shift_jit.py','/workspace/candidate_backend.py',copy=True))
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
    run_id=datetime.now(timezone.utc).strftime('cross-model-correctness-%Y%m%dT%H%M%SZ-')+uuid4().hex[:8]
    out=Path('/results')/run_id;out.mkdir()
    baseline=(SOURCE/REL).read_bytes();candidate=Path('/workspace/candidate_backend.py').read_bytes()
    assert sha(candidate)==EXPECTED
    manifest={'status':'running','baseline_commit':BASE,'scope':'checkpoint-free real modulation entrypoints; no timing or full image/video generation','gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda,'packages':{p:md.version(p) for p in ['torch','triton','diffusers','transformers','apache-tvm-ffi']},'baseline_sha256':sha(baseline),'candidate_sha256':sha(candidate),'arms':[],'errors':[]}
    def save():(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');results.commit()
    print('RUN_ID='+run_id,flush=True)
    commands=[('existing-kernel',[sys.executable,'-m','pytest','-q','test/registered/kernels/ops/diffusion/test_modulate.py','-k','modulate_scale_shift']),
              ('existing-model-sites',[sys.executable,'-m','pytest','-q','test/registered/kernels/ops/diffusion/test_model_fast_paths.py','-k','flux_fused_ln_modulate or flux_norm_modulate or ltx2_lossless']),
              ('existing-ltx-layout',[sys.executable,'-m','pytest','-q','python/sglang/multimodal_gen/test/unit/test_ltx2_modulate_mount.py'])]
    try:
        for arm,data in [('baseline',baseline),('candidate',candidate)]:
            (SOURCE/REL).write_bytes(data)
            folder=out/arm;folder.mkdir()
            info={'variant':arm,'backend_sha256':sha(data),'tests':{}}
            for label,cmd in [('caller-checks',[sys.executable,'/workspace/check_callers.py',str(folder)]),*commands]:
                print('RUN',arm,label,flush=True)
                code=run(cmd,folder/(label+'.log'));info['tests'][label]={'command':cmd,'exit_code':code}
                if code:manifest['errors'].append(arm+'/'+label)
                save()
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
