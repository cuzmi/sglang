"""Pinned Anima post-rebase performance: ABBA + BAAB on one H200.

Fresh CLI process per arm; same-resolution one-step request warmup; 30-step
unprofiled timed request. Any failure or pixel mismatch invalidates the run.
The exact candidate already passed rebase_validation correctness checks.
"""
from datetime import datetime, timezone
import hashlib
import importlib.metadata as md
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import modal

BASE='affa261e3d289fe4f907c9b2e8d773fef0d36dba'
HEAD='05bee50b2f4fef4484bab3db3bff7ef4495e98e5'
IMAGE='im-USSXj25TgwLcB1dg8jFGSS'
MODEL='circlestone-labs/Anima-Base-v1.0-Diffusers'
REVISION='073c3a9db359c31ad0e8aa268d15775473c2176c'
HF_HOME='/vol/hf-cache'
MODEL_PATH=f'{HF_HOME}/hub/models--circlestone-labs--Anima-Base-v1.0-Diffusers/snapshots/{REVISION}'
PROMPT='masterpiece, best quality, safe, watercolor landscape, a quiet seaside village at sunset'
SOURCE=Path('/workspace/sglang-rebased')
REL=Path('python/sglang/multimodal_gen/runtime/models/dits/anima.py')
BACKEND=Path('python/sglang/kernels/ops/diffusion/modulate/modulate_scale_shift_jit.py')
CUDA=Path('python/sglang/kernels/jit/csrc/diffusion/modulate_scale_shift.cuh')
BASE_HASHES={str(REL):'a21fae70ad5b9164a4804821339ff2c1a94d826488d8de4de4cf467e1ff88cd6',str(BACKEND):'dcbce5b2978f460ba98f0b6e6c647b9432c055e3c1221efbde7029e6341e3bec'}
HEAD_HASHES={str(REL):'ae62fe965c2facdf224aa164a5002dc5bd2e403a48fb51cf6094179b4d254013',str(BACKEND):'9fd588001fb3b7b9bd0abbc68c4f1210278cb19f40eea45ed74f649e2daa190e'}
ORDER=['baseline','candidate','candidate','baseline','candidate','baseline','baseline','candidate']
app=modal.App('anima-rebased-performance')
image=modal.Image.from_id(IMAGE)
cache=modal.Volume.from_name('ming-image-hf-cache')
results=modal.Volume.from_name('anima-modulate-experiments')


def sha(data):return hashlib.sha256(data).hexdigest()


def gpu_snapshot():
    command=['nvidia-smi','--query-gpu=name,uuid,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,utilization.gpu,memory.used','--format=csv,noheader']
    p=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    q=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    return {'utc':datetime.now(timezone.utc).isoformat(),'gpu':p.stdout.strip(),'gpu_exit_code':p.returncode,'processes':q.stdout.strip(),'process_exit_code':q.returncode}


def logged(command,path,env):
    print('RUN',command,flush=True)
    with path.open('w') as log:
        proc=subprocess.Popen(command,cwd=SOURCE,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in proc.stdout:
            log.write(line);log.flush();print(line,end='',flush=True)
        code=proc.wait()
    if code:raise RuntimeError(f'Exit {code}: {path}')


def generate(out,label,variant):
    from PIL import Image
    arm=out/label;arm.mkdir()
    command=['sglang','generate','--model-path',MODEL_PATH,'--model-id',MODEL,'--revision',REVISION,
             '--num-gpus','1','--enable-torch-compile=false','--enable-breakable-cuda-graph=false',
             '--performance-mode=speed','--attention-backend','fa','--width','1024','--height','1024',
             '--num-inference-steps','30','--guidance-scale','4','--negative-prompt','','--prompt',PROMPT,
             '--seed','42','--generator-device','cpu','--quality','lossless','--save-output',
             '--warmup-mode','request','--warmup-steps','1','--perf-dump-path',str(arm/'perf.json'),
             '--output-path',str(arm/'outputs'),'--output-file-name','sample']
    env={**os.environ,'PYTHONPATH':str(SOURCE/'python'),'CUDA_VISIBLE_DEVICES':'0','HF_HUB_OFFLINE':'1',
         'TRANSFORMERS_OFFLINE':'1','SGLANG_DIFFUSION_SYNC_STAGE_PROFILING':'0'}
    before=gpu_snapshot()
    logged(command,arm/'generate.log',env)
    perf=json.loads((arm/'perf.json').read_text())
    assert len(perf['denoise_steps_ms'])==30
    assert math.isfinite(perf['total_duration_ms']) and perf['total_duration_ms']>0
    png=arm/'outputs/sample.png'
    with Image.open(png) as img:
        assert img.size==(1024,1024) and img.mode=='RGB'
        pixels=sha(img.tobytes())
    r={'label':label,'variant':variant,'command':command,'total_duration_ms':perf['total_duration_ms'],
       'denoise_steps_ms':perf['denoise_steps_ms'],'pixel_sha256':pixels,'png_sha256':sha(png.read_bytes()),
       'source_sha256':{str(p):sha((SOURCE/p).read_bytes()) for p in [REL,BACKEND]},
       'gpu_before':before,'gpu_after':gpu_snapshot()}
    print('RESULT',json.dumps({k:r[k] for k in ['label','variant','total_duration_ms','pixel_sha256']}),flush=True)
    return r


@app.function(image=image,gpu='H200',cpu=8,memory=65536,timeout=3600,volumes={HF_HOME:cache,'/results':results})
def benchmark():
    import torch
    import sglang
    cache.reload()
    assert Path(MODEL_PATH).is_dir(), 'Pinned model snapshot missing'
    assert str(Path(sglang.__file__).resolve()).startswith(str(SOURCE/'python'))
    assert sha(Path('/workspace/rebased_baseline.tar.gz').read_bytes())=='bc6052c159cb5f1eef2236a7e8e82e06c9b0377ea79a7d1beb35c55ea664ac68'
    baseline={p:(SOURCE/p).read_bytes() for p in [REL,BACKEND]}
    candidate={REL:Path('/workspace/rebased_anima.py').read_bytes(),BACKEND:Path('/workspace/candidate_backend.py').read_bytes()}
    assert {str(p):sha(v) for p,v in baseline.items()}==BASE_HASHES
    assert {str(p):sha(v) for p,v in candidate.items()}==HEAD_HASHES
    run_id=datetime.now(timezone.utc).strftime('anima-rebased-perf-%Y%m%dT%H%M%SZ-')+uuid4().hex[:8]
    out=Path('/results')/run_id;out.mkdir()
    cpu_models=sorted({l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')})
    m={'status':'running','baseline_commit':BASE,'candidate_commit':HEAD,'image_id':IMAGE,'model_id':MODEL,
       'model_revision':REVISION,'gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda,
       'cpu_models':cpu_models,'visible_cpu_count':os.cpu_count(),'sglang_import_path':sglang.__file__,
       'python':sys.version,'packages':{p:md.version(p) for p in ['torch','triton','diffusers','transformers','apache-tvm-ffi']},
       'baseline_source_sha256':BASE_HASHES,'candidate_source_sha256':HEAD_HASHES,'cuda_source_sha256':sha((SOURCE/CUDA).read_bytes()),
       'order':ORDER,'workload':{'width':1024,'height':1024,'steps':30,'cfg':4,'seed':42,'prompt':PROMPT,'negative_prompt':'','dtype':'bfloat16','tp':1,'attention':'fa','compile':False,'cuda_graph':False,'warmup_mode':'request','warmup_steps':1},
       'metric':'native worker total_duration_ms, excludes load/warmup/PNG saving','validity':'All 8 arms must complete with matched work and identical output pixels. No failed-arm exclusion.','runs':[]}
    def save():(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');results.commit()
    print('RUN_ID='+run_id,flush=True);save()
    try:
        reference=None
        for i,variant in enumerate(ORDER,1):
            data=baseline if variant=='baseline' else candidate
            for p,v in data.items():(SOURCE/p).write_bytes(v)
            r=generate(out,f'{i:02d}-{variant}',variant)
            assert r['source_sha256']==(BASE_HASHES if variant=='baseline' else HEAD_HASHES)
            assert sha((SOURCE/CUDA).read_bytes())==m['cuda_source_sha256']
            m['runs'].append(r);save()
            if reference is None:reference=r['pixel_sha256']
            assert r['pixel_sha256']==reference,'Pixel mismatch'
        m['status']='complete'
    except Exception as exc:
        m['status']='failed';m['error']=repr(exc);raise
    finally:save()
    return run_id

@app.local_entrypoint()
def main():print('RESULTS',benchmark.remote())
