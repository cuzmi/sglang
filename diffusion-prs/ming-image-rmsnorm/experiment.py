"""Pinned Ming RMSNorm correctness + unprofiled ABBA + separate diagnostics."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
import modal

HERE = Path(__file__).resolve().parent
BASE = '6fa3fe69e2e5e19b75cadd9fc285b72634551992'
SOURCE = Path('/workspace/sglang')
REL = Path('python/sglang/multimodal_gen/runtime/models/dits/ming_image.py')
MODEL = 'inclusionAI/Ming-Image-0.1-Design'
REVISION = '208087ada1486931692c1896f38d4cd16ff3df82'
PROMPT = 'A minimalist exhibition poster with a red geometric sculpture'
HF_HOME = '/vol/hf-cache'
VOLUME = 'ming-rmsnorm-experiments'
app = modal.App('ming-rmsnorm-experiment')
image = (modal.Image.from_registry('lmsysorg/sglang:dev').entrypoint([])
    .add_local_file(HERE/'inputs/baseline.tar.gz', '/workspace/baseline.tar.gz', copy=True)
    .run_commands('mkdir -p /workspace/sglang && tar -xzf /workspace/baseline.tar.gz -C /workspace/sglang', "SGLANG_BUILD_RUST_EXTS=none python -m pip install -e '/workspace/sglang/python[diffusion]'")
    .add_local_file(HERE/'inputs/candidate_ming_image.py', '/workspace/candidate_ming_image.py', copy=True)
    .add_local_file(HERE/'verify_rmsnorm.py', '/workspace/verify_rmsnorm.py', copy=True)
    .add_local_file(HERE/'inputs/test_rmsnorm_preserve_reduction.py', str(SOURCE/'test/registered/kernels/ops/diffusion/test_rmsnorm_preserve_reduction.py'), copy=True)
    .add_local_file(HERE/'inputs/test_model_fast_paths.py', str(SOURCE/'test/registered/kernels/ops/diffusion/test_model_fast_paths.py'), copy=True)
    .env({'HF_HOME': HF_HOME, 'PYTHONPATH': str(SOURCE/'python'), 'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '8'}))
cache = modal.Volume.from_name('ming-image-hf-cache', create_if_missing=True)
results = modal.Volume.from_name(VOLUME, create_if_missing=True)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def logged(cmd, path, env):
    print('RUN', cmd, flush=True)
    with path.open('w') as log:
        p = subprocess.Popen(cmd, cwd=SOURCE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            log.write(line)
            log.flush()
            print(line, end='', flush=True)
        rc = p.wait()
    if rc:
        raise RuntimeError(f'exit {rc}: {path}')


PROFILE_WRAPPER = '''
# Diagnostic only; absent from timed arms.
import json as _json
import os as _os
from pathlib import Path as _Path
from torch.profiler import record_function as _rf
_ming_norm_original = MingRMSNorm.forward_native
_ming_kernel_original = globals().get("rmsnorm_preserve_reduction")
_ming_counts = {"calls": 0, "fused": 0, "shapes": {}, "gate_disabled": None}
def _ming_save():
    gate = globals().get("_MING_RMSNORM_FUSION")
    if gate is not None:
        _ming_counts["gate_disabled"] = gate.disabled
        _ming_counts["verified_signatures"] = [str(s) for s in gate.verified_sigs]
    _Path(_os.environ["MING_COUNTS"]).write_text(_json.dumps(_ming_counts, indent=2))
def _ming_norm_diagnostic(self, x, *args, **kwargs):
    _ming_counts["calls"] += 1
    key = str((tuple(x.shape), tuple(x.stride()), str(x.dtype)))
    _ming_counts["shapes"][key] = _ming_counts["shapes"].get(key, 0) + 1
    with _rf("MingRMSNorm"):
        out = _ming_norm_original(self, x, *args, **kwargs)
    if _ming_counts["calls"] % 205 == 0:
        _ming_save()
    return out
MingRMSNorm.forward_native = _ming_norm_diagnostic
if _ming_kernel_original is not None:
    def rmsnorm_preserve_reduction(*args):
        _ming_counts["fused"] += 1
        return _ming_kernel_original(*args)
'''


@app.function(image=image, gpu='H200', cpu=8, memory=65536, timeout=3600, volumes={HF_HOME:cache, '/results':results})
def run(mode: str):
    import torch
    import triton
    import importlib.metadata
    from PIL import Image
    run_id = mode + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = Path('/results')/run_id
    out.mkdir()
    baseline = (SOURCE/REL).read_bytes()
    candidate = Path('/workspace/candidate_ming_image.py').read_bytes()
    manifest = dict(status='running', base=BASE, baseline_sha256=sha(baseline), candidate_sha256=sha(candidate), archive_sha256=sha(Path('/workspace/baseline.tar.gz').read_bytes()), model=MODEL, revision=REVISION, gpu=torch.cuda.get_device_name(), versions={p:importlib.metadata.version(p) for p in ['torch','triton','diffusers','transformers']}, nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True), arms=[])
    env = {**os.environ, 'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1', 'CUDA_VISIBLE_DEVICES':'0', 'PYTHONDONTWRITEBYTECODE':'1'}
    def save():
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        results.commit()
    try:
        (SOURCE/REL).write_bytes(candidate)
        if mode == 'micro':
            logged([sys.executable, '/workspace/verify_rmsnorm.py', str(out/'micro.json')], out/'tests.log', env)
            logged([sys.executable, '-m', 'pytest', '-q', 'test/registered/kernels/ops/diffusion/test_rmsnorm_preserve_reduction.py'], out/'kernel-tests.log', env)
            logged([sys.executable, '-m', 'pytest', '-q', 'test/registered/kernels/ops/diffusion/test_model_fast_paths.py', '-k', 'ming_rmsnorm'], out/'wrapper-tests.log', env)
        else:
            modelpath = Path(HF_HOME)/'hub/models--inclusionAI--Ming-Image-0.1-Design/snapshots'/REVISION
            assert modelpath.is_dir(), modelpath
            manifest['model_configs'] = {str(p.relative_to(modelpath)): json.loads(p.read_text()) for p in modelpath.rglob('*config.json')}
            for label, variant, profile in [(f'{i+1:02d}-{v}',v,False) for i,v in enumerate(['A','B','B','A','A','B','B','A'])] + [('profile-A','A',True),('profile-B','B',True)]:
                arm = out/label
                arm.mkdir()
                (SOURCE/REL).write_bytes((baseline if variant=='A' else candidate) + (PROFILE_WRAPPER.encode() if profile else b''))
                # Do not let a same-size same-second source replacement reuse bytecode.
                for pyc in (SOURCE/REL).parent.glob('__pycache__/ming_image.*.pyc'):
                    pyc.unlink()
                cmd = ['sglang','generate','--model-path',MODEL,'--revision',REVISION,'--num-gpus','1','--enable-torch-compile=false','--enable-breakable-cuda-graph=false','--performance-mode=speed','--width','1024','--height','1024','--num-inference-steps','12','--guidance-scale','1','--prompt',PROMPT,'--seed','42','--quality','lossless','--save-output','--warmup-mode','request','--warmup-steps','1','--perf-dump-path',str(arm/'perf.json'),'--output-path',str(arm/'outputs'),'--output-file-name','sample']
                if profile:
                    cmd += ['--profile','--num-profiled-timesteps','5']
                arm_env={**env,'SGLANG_DIFFUSION_SYNC_STAGE_PROFILING':'1' if profile else '0','SGLANG_DIFFUSION_TORCH_PROFILER_DIR':str(arm/'traces'),'MING_COUNTS':str(arm/'counts.json')}
                logged(cmd, arm/'generate.log',arm_env)
                perf=json.loads((arm/'perf.json').read_text())
                assert len(perf['denoise_steps_ms'])==12
                with Image.open(arm/'outputs/sample.png') as im:
                    assert im.size==(1024,1024)
                    digest=sha(im.tobytes())
                record=dict(label=label,variant=variant,profile=profile,command=cmd,pixel_sha256=digest,total_duration_ms=perf['total_duration_ms'],denoise_steps_ms=perf['denoise_steps_ms'])
                manifest['arms'].append(record)
                save()
                print('ARM_RESULT',json.dumps(record),flush=True)
            assert len({a['pixel_sha256'] for a in manifest['arms']})==1, 'output mismatch'
            counts=json.loads((out/'profile-B/counts.json').read_text())
            assert counts['fused']>0 and counts['gate_disabled'] is False, counts
        manifest['status']='complete'
    except Exception as exc:
        manifest['status']='failed'
        manifest['error']=repr(exc)
        raise
    finally:
        (SOURCE/REL).write_bytes(baseline)
        save()
    return run_id

@app.local_entrypoint()
def main(mode: str = 'micro'):
    print('RESULT',run.remote(mode))
