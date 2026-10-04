"""Same-GPU A/B/C-C/B/A: original eager, first fusion mount, cached dispatch.

Uses the prior exact image and pinned model. The CUDA source stays unchanged.
All subprocesses are fresh and use identical request warmup and workload.
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
BACKEND=Path("python/sglang/kernels/ops/diffusion/modulate/modulate_scale_shift_jit.py")
CUDA_REL=Path("python/sglang/kernels/jit/csrc/diffusion/modulate_scale_shift.cuh")
EXPECTED_BACKEND_SHA="5705525464fedef85e83631e44aa53ce87bd7a7b1b83d7294a25c22621bcc8b0"
app=modal.App("anima-dispatch-abccba")
image=(modal.Image.from_id("im-rRXDzAdc13HCyQHStCXqwE")
       .add_local_file(HERE/"inputs/dispatch_modulate_scale_shift_jit.py","/workspace/dispatch_modulate_scale_shift_jit.py",copy=True))
cache=modal.Volume.from_name("ming-image-hf-cache")
results=modal.Volume.from_name("anima-modulate-experiments")


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


@app.function(image=image,gpu="H200",cpu=8,memory=65536,timeout=3600,volumes={HF_HOME:cache,"/results":results})
def experiment():
    import torch
    cache.reload()
    run_id=datetime.now(timezone.utc).strftime("dispatch-abccba-%Y%m%dT%H%M%SZ-")+uuid4().hex[:8]
    out=Path("/results")/run_id;out.mkdir()
    base_model=(SOURCE/REL).read_bytes()
    fused_model=Path("/workspace/candidate_anima.py").read_bytes()
    base_backend=(SOURCE/BACKEND).read_bytes()
    fixed_backend=Path("/workspace/dispatch_modulate_scale_shift_jit.py").read_bytes()
    assert sha(fixed_backend)==EXPECTED_BACKEND_SHA
    m=dict(status="running",baseline_commit=BASE,model_id=MODEL,model_revision=REVISION,gpu=torch.cuda.get_device_name(),
           cuda=torch.version.cuda,packages={p:md.version(p) for p in ["torch","triton","diffusers","transformers","apache-tvm-ffi"]},
           cuda_source_sha256=sha((SOURCE/CUDA_REL).read_bytes()),baseline_model_sha256=sha(base_model),fused_model_sha256=sha(fused_model),
           original_backend_sha256=sha(base_backend),fixed_backend_sha256=sha(fixed_backend),runs=[])
    def save():
        (out/"manifest.json").write_text(json.dumps(m,indent=2)+"\n");results.commit()
    print("RUN_ID="+run_id,flush=True)
    try:
        (SOURCE/REL).write_bytes(fused_model);(SOURCE/BACKEND).write_bytes(fixed_backend)
        logged([sys.executable,"/workspace/verify_anima.py",str(out/"operator.json")],out/"operator.log",os.environ.copy())
        m['operator']=json.loads((out/'operator.json').read_text());save()
        commands=[("existing-modulate",[sys.executable,"-m","pytest","-q","test/registered/kernels/ops/diffusion/test_modulate.py","-k","modulate_scale_shift"]),
                  ("existing-anima",[sys.executable,"-m","pytest","-q","python/sglang/multimodal_gen/test/unit/test_anima.py","python/sglang/multimodal_gen/test/unit/test_anima_cuda.py"])]
        for label,command in commands:
            logged(command,out/(label+'.log'),os.environ.copy());m[label]='passed';save()
        reference=None
        for i,variant in enumerate(["eager","initial_mount","cached_dispatch","cached_dispatch","initial_mount","eager"],1):
            model=base_model if variant=="eager" else fused_model
            backend=fixed_backend if variant=="cached_dispatch" else base_backend
            (SOURCE/REL).write_bytes(model);(SOURCE/BACKEND).write_bytes(backend)
            r=generate(out,f"{i:02d}-{variant}",variant)
            r['backend_sha256']=sha(backend)
            assert sha((SOURCE/CUDA_REL).read_bytes())==m['cuda_source_sha256']
            m['runs'].append(r);save()
            if reference is None: reference=r['pixel_sha256']
            assert r['pixel_sha256']==reference,"Pixel mismatch"
        m['status']='complete'
    except Exception as exc:
        m['status']='failed';m['error']=repr(exc);raise
    finally:
        save()
    return run_id

@app.local_entrypoint()
def main():
    print("RESULTS",experiment.remote())
