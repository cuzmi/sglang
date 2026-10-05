"""Self-contained pinned Anima FP32 RoPE experiment. Modal H200, maximum 1 hour.
Run with: python -m modal run --detach experiment.py
Warmup is excluded, profiling is separate, any output mismatch invalidates the run.
"""
from datetime import datetime, timezone
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
BASE="6fa3fe69e2e5e19b75cadd9fc285b72634551992"
MODEL="circlestone-labs/Anima-Base-v1.0-Diffusers"
REVISION="073c3a9db359c31ad0e8aa268d15775473c2176c"
HF_HOME="/vol/hf-cache"
MODEL_PATH=f"{HF_HOME}/hub/models--circlestone-labs--Anima-Base-v1.0-Diffusers/snapshots/{REVISION}"
PROMPT="masterpiece, best quality, safe, watercolor landscape, a quiet seaside village at sunset"
SOURCE=Path("/workspace/sglang")
REL=Path("python/sglang/multimodal_gen/runtime/models/dits/anima.py")
app=modal.App("anima-fp32-rope")
image=(modal.Image.from_id("im-rRXDzAdc13HCyQHStCXqwE")
       .add_local_dir(HERE/"inputs","/workspace/rope-inputs",copy=True)
       .add_local_file(HERE/"verify_rope.py","/workspace/verify_rope.py",copy=True))
cache=modal.Volume.from_name("ming-image-hf-cache")
results=modal.Volume.from_name("anima-rope-experiments",create_if_missing=True)

def sha(data):
    return hashlib.sha256(data).hexdigest()


def logged(command, path, env):
    print("RUN", command, flush=True)
    with path.open("w") as log:
        proc = subprocess.Popen(command, cwd=SOURCE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        code = proc.wait()
    if code:
        raise RuntimeError(f"Exit {code}: {path}")


def generate(run_dir, label, variant, profile=False, seed=42, size=1024, steps=30):
    from PIL import Image
    arm = run_dir / label
    arm.mkdir()
    command = ["sglang", "generate", "--model-path", MODEL_PATH, "--model-id", MODEL, "--revision", REVISION,
               "--num-gpus", "1", "--enable-torch-compile=false", "--enable-breakable-cuda-graph=false",
               "--performance-mode=speed", "--attention-backend", "fa", "--width", str(size), "--height", str(size),
               "--num-inference-steps", str(steps), "--guidance-scale", "4", "--negative-prompt", "",
               "--prompt", PROMPT, "--seed", str(seed), "--generator-device", "cpu", "--quality", "lossless", "--save-output",
               "--warmup-mode", "request", "--warmup-steps", "1", "--perf-dump-path", str(arm / "perf.json"),
               "--output-path", str(arm / "outputs"), "--output-file-name", "sample"]
    if profile:
        command += ["--profile", "--num-profiled-timesteps", "5"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "SGLANG_DIFFUSION_SYNC_STAGE_PROFILING": "1" if profile else "0",
           "SGLANG_DIFFUSION_TORCH_PROFILER_DIR": str(arm / "traces"), "ANIMA_DIAGNOSTIC_DIR": str(arm)}
    logged(command, arm / "generate.log", env)
    perf = json.loads((arm / "perf.json").read_text())
    assert len(perf["denoise_steps_ms"]) == steps, perf
    png = arm / "outputs/sample.png"
    with Image.open(png) as img:
        assert img.size == (size, size)
        pixels = sha(img.tobytes())
        mode = img.mode
    result = dict(label=label, variant=variant, profile=profile, seed=seed, size=size, steps=steps, command=command,
                  total_duration_ms=perf["total_duration_ms"], denoise_steps_ms=perf["denoise_steps_ms"],
                  denoise_ms=sum(s["duration_ms"] for s in perf["denoise_steps_ms"]), pixel_sha256=pixels, mode=mode,
                  png_sha256=sha(png.read_bytes()), source_sha256=sha((SOURCE / REL).read_bytes()))
    print("RESULT", json.dumps(result), flush=True)
    return result



# Appended only for diagnostic profile, never for unprofiled timing arms.
DIAGNOSTIC = r"""
import json as _json
import os as _os
from pathlib import Path as _Path
from torch.profiler import record_function as _rf
_rope_original = _anima_rope
_rope_count = 0
_rope_signatures = set()
def _anima_rope(q,k,cos,sin):
    global _rope_count
    with _rf("AnimaFP32RoPE"):
        out = _rope_original(q,k,cos,sin)
    assert not _ANIMA_ROPE.disabled
    assert _ANIMA_ROPE.is_verified((q.device,q.dtype,q.shape))
    sig = (str(q.dtype), tuple(q.shape), tuple(q.stride()), tuple(cos.shape))
    if sig not in _rope_signatures:
        assert tensors_equal(out, _anima_rope_eager(q,k,cos,sin))
        _rope_signatures.add(sig)
    _rope_count += 1
    if _rope_count % 28 == 0:
        (_Path(_os.environ["ANIMA_DIAGNOSTIC_DIR"])/f"dispatch-{_os.getpid()}.json").write_text(
            _json.dumps(dict(calls=_rope_count,signatures=list(_rope_signatures),disabled=_ANIMA_ROPE.disabled)))
    return out
"""

@app.function(image=image,gpu="H200",cpu=8,memory=65536,timeout=3600,volumes={HF_HOME:cache,"/results":results})
def experiment():
    import torch
    cache.reload()
    inputs=Path('/workspace/rope-inputs')
    frozen=json.loads((inputs/'manifest.json').read_text())
    base_model=(inputs/'baseline-anima.py').read_bytes()
    assert sha(base_model)==frozen['baseline_model_sha256']
    # Cached image contains the public baseline archive; restore it before overlay.
    archive=Path('/workspace/baseline.tar.gz')
    assert sha(archive.read_bytes())=='349f9021b4d7bcd7d3c42425d1691fccc42f8d75f2ac5054882407d621ebb233'
    subprocess.run(['tar','-xzf',str(archive),'-C',str(SOURCE)],check=True)
    assert (SOURCE/REL).read_bytes()==base_model
    for rel,info in frozen['files'].items():
        data=(inputs/info['file']).read_bytes();assert sha(data)==info['sha256']
        (SOURCE/rel).write_bytes(data)
    fused_model=(SOURCE/REL).read_bytes()
    run_id=datetime.now(timezone.utc).strftime('rope-%Y%m%dT%H%M%SZ-')+uuid4().hex[:8]
    out=Path('/results')/run_id;out.mkdir()
    m=dict(status='running',baseline_commit=BASE,sources=frozen,model_id=MODEL,model_revision=REVISION,
           gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,
           packages={p:md.version(p) for p in ['torch','triton','diffusers','transformers']},
           input_sha256=sha(json.dumps(dict(prompt=PROMPT,seed=42,size=1024,steps=30,guidance=4),sort_keys=True).encode()),
           validity='All arms complete; identical logical work and pixels; no concurrent GPU work from this campaign.',runs=[])
    def save():
        (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');results.commit()
    print('RUN_ID='+run_id,flush=True)
    try:
        logged([sys.executable,'/workspace/verify_rope.py',str(out/'operator.json')],out/'operator.log',os.environ.copy())
        m['operator']='passed';save()
        logged([sys.executable,'-m','pytest','-q','python/sglang/multimodal_gen/test/unit/test_anima.py',
                'python/sglang/multimodal_gen/test/unit/test_anima_cuda.py'],out/'existing-tests.log',os.environ.copy())
        m['existing_tests']='passed';save()
        reference=None
        # Two baseline-only control processes, then ABBA; same container and GPU.
        for i,variant in enumerate(['eager','eager','eager','fused','fused','eager'],1):
            (SOURCE/REL).write_bytes(base_model if variant=='eager' else fused_model)
            r=generate(out,f'{i:02d}-{variant}',variant)
            r['phase']='AA' if i<=2 else 'ABBA'
            m['runs'].append(r);save()
            if reference is None: reference=r['pixel_sha256']
            assert r['pixel_sha256']==reference,'Pixel mismatch'
        for variant in ['eager','fused']:
            (SOURCE/REL).write_bytes(base_model if variant=='eager' else fused_model+DIAGNOSTIC.encode())
            r=generate(out,'profile-'+variant,variant,profile=True)
            m['runs'].append(r);save()
            assert r['pixel_sha256']==reference,'Profile pixel mismatch'
        m['status']='complete'
    except Exception as exc:
        m['status']='failed';m['error']=repr(exc);raise
    finally:
        save()
    return run_id

@app.local_entrypoint()
def main():
    print('RESULTS',experiment.remote())
