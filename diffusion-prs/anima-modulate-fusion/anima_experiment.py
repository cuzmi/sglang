"""Pinned local-source Anima AdaLN correctness, ABBA, and profiling experiment.

Run: modal run --detach anima_experiment.py
Only the archived public baseline and the one candidate model file are uploaded.
Timed subprocesses use untouched source files; diagnostics run separately.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from uuid import uuid4

import modal

HERE = Path(__file__).resolve().parent
BASE = "6fa3fe69e2e5e19b75cadd9fc285b72634551992"
MODEL = "circlestone-labs/Anima-Base-v1.0-Diffusers"
REVISION = "073c3a9db359c31ad0e8aa268d15775473c2176c"
PROMPT = "masterpiece, best quality, safe, watercolor landscape, a quiet seaside village at sunset"
SOURCE = Path("/workspace/sglang")
REL = Path("python/sglang/multimodal_gen/runtime/models/dits/anima.py")
HF_HOME = "/vol/hf-cache"
VOLUME = "anima-modulate-experiments"
MODEL_PATH = f"{HF_HOME}/hub/models--circlestone-labs--Anima-Base-v1.0-Diffusers/snapshots/{REVISION}"

def sha(data):
    return hashlib.sha256(data).hexdigest()

SOURCE_SHA = "349f9021b4d7bcd7d3c42425d1691fccc42f8d75f2ac5054882407d621ebb233"
CANDIDATE_SHA = "ae62fe965c2facdf224aa164a5002dc5bd2e403a48fb51cf6094179b4d254013"
app = modal.App("anima-modulate-abba")
image = (
    modal.Image.from_registry("lmsysorg/sglang:dev")
    .entrypoint([])
    .add_local_file(HERE / "inputs/baseline.tar.gz", "/workspace/baseline.tar.gz", copy=True)
    .run_commands(
        "mkdir -p /workspace/sglang && tar -xzf /workspace/baseline.tar.gz -C /workspace/sglang",
        "SGLANG_BUILD_RUST_EXTS=none python -m pip install -e '/workspace/sglang/python[diffusion]'",
    )
    .add_local_file(HERE / "inputs/candidate_anima.py", "/workspace/candidate_anima.py", copy=True)
    .add_local_file(HERE / "verify_anima.py", "/workspace/verify_anima.py", copy=True)
    .env({"HF_HOME": HF_HOME, "PYTHONPATH": str(SOURCE / "python"), "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "8"})
)
cache = modal.Volume.from_name("ming-image-hf-cache", create_if_missing=True)
results = modal.Volume.from_name(VOLUME, create_if_missing=True)

@app.function(image=image, cpu=4, memory=16384, timeout=1800, volumes={HF_HOME: cache})
def download():
    from huggingface_hub import snapshot_download
    path = snapshot_download(MODEL, revision=REVISION, ignore_patterns=["*.md", ".gitattributes"])
    cache.commit()
    return path


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


PROFILE_WRAPPER = '''
# Diagnostic only. Never appended to unprofiled timing arms.
import atexit as _ae
import json as _json
import os as _os
from pathlib import Path as _Path
from torch.profiler import record_function as _rf
from sglang.kernels.ops.diffusion import modulate_scale_shift
_anima_original_forward = AnimaAdaLayerNorm.forward
_anima_original_modulate = modulate_scale_shift
_anima_counts = {"adaln_calls": 0, "modulate_calls": 0, "checked_signatures": []}
_anima_seen = set()
def _anima_profile_forward(self, *args, **kwargs):
    _anima_counts["adaln_calls"] += 1
    with _rf("AnimaAdaLayerNorm"):
        out = _anima_original_forward(self, *args, **kwargs)
    if _anima_counts["adaln_calls"] % 85 == 0:
        _anima_save_counts()
    return out
AnimaAdaLayerNorm.forward = _anima_profile_forward

def _anima_checked_modulate(x, scale, shift):
    from sglang.kernels.ops.diffusion import can_use_modulate_scale_shift_cuda
    from sglang.kernels.ops.diffusion.modulate.modulate_scale_shift_jit import _FAILED_RUNTIME_KEYS
    assert can_use_modulate_scale_shift_cuda(x, scale, shift)
    with _rf("AnimaModulate"):
        out = _anima_original_modulate(x, scale, shift)
    assert (x.device.index, x.dtype) not in _FAILED_RUNTIME_KEYS
    _anima_counts["modulate_calls"] += 1
    sig = (tuple(x.shape), str(x.dtype), tuple(scale.stride()))
    if sig not in _anima_seen:
        assert torch.equal(out, x * (1 + scale[:, None]) + shift[:, None])
        _anima_seen.add(sig)
        _anima_counts["checked_signatures"].append(sig)
    return out
modulate_scale_shift = _anima_checked_modulate

def _anima_save_counts():
    if _anima_counts["adaln_calls"]:
        p = _Path(_os.environ["ANIMA_DIAGNOSTIC_DIR"]) / f"counts-{_os.getpid()}.json"
        p.write_text(_json.dumps(_anima_counts, indent=2))
_ae.register(_anima_save_counts)
'''


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


@app.function(image=image, gpu="H200", cpu=8, memory=65536, timeout=7200, volumes={HF_HOME: cache, "/results": results})
def experiment():
    import torch
    import importlib.metadata as md
    cache.reload()
    assert sha(Path("/workspace/baseline.tar.gz").read_bytes()) == SOURCE_SHA
    candidate = Path("/workspace/candidate_anima.py").read_bytes()
    assert sha(candidate) == CANDIDATE_SHA
    baseline = (SOURCE / REL).read_bytes()
    run_id = datetime.now(timezone.utc).strftime("anima-%Y%m%dT%H%M%SZ-") + uuid4().hex[:8]
    out = Path("/results") / run_id
    out.mkdir()
    print("RUN_ID=" + run_id, flush=True)
    manifest = dict(status="running", baseline_commit=BASE, archive_sha256=SOURCE_SHA,
                    baseline_source_sha256=sha(baseline), candidate_source_sha256=CANDIDATE_SHA,
                    model_id=MODEL, model_revision=REVISION, prompt=PROMPT,
                    gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda, runs=[],
                    packages={p: md.version(p) for p in ["torch", "triton", "diffusers", "transformers", "sglang", "apache-tvm-ffi"]},
                    validity="All subprocesses succeed; equal step counts and output dimensions; bit-identical pixels for matched inputs; no failed CUDA fast path.")
    def save():
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        results.commit()
    save()
    try:
        (SOURCE / REL).write_bytes(candidate)
        logged([sys.executable, "/workspace/verify_anima.py", str(out / "operator.json")], out / "operator.log", os.environ.copy())
        manifest["operator"] = json.loads((out / "operator.json").read_text())
        save()
        for label, command in [
            ("existing-modulate", [sys.executable, "-m", "pytest", "-q", "test/registered/kernels/ops/diffusion/test_modulate.py", "-k", "modulate_scale_shift"]),
            ("existing-anima", [sys.executable, "-m", "pytest", "-q", "python/sglang/multimodal_gen/test/unit/test_anima.py", "python/sglang/multimodal_gen/test/unit/test_anima_cuda.py"]),
        ]:
            logged(command, out / (label + ".log"), os.environ.copy())
            manifest[label] = "passed"
            save()
        # Two ABBA blocks; each CLI starts a fresh process and performs request warmup.
        reference = None
        for i, variant in enumerate(["baseline", "candidate", "candidate", "baseline"] * 2, 1):
            (SOURCE / REL).write_bytes(baseline if variant == "baseline" else candidate)
            r = generate(out, f"{i:02d}-{variant}", variant)
            manifest["runs"].append(r)
            save()
            if reference is None:
                reference = r["pixel_sha256"]
            assert r["pixel_sha256"] == reference, "Matched output pixel mismatch"
        # A different seed/resolution provides a second full-pipeline correctness case.
        paired = []
        for variant in ["baseline", "candidate"]:
            (SOURCE / REL).write_bytes(baseline if variant == "baseline" else candidate)
            r = generate(out, f"quality-{variant}", variant, seed=123, size=512, steps=30)
            manifest["runs"].append(r)
            paired.append(r["pixel_sha256"])
            save()
        assert paired[0] == paired[1], "Secondary workload pixel mismatch"
        for variant in ["baseline", "candidate"]:
            (SOURCE / REL).write_bytes((baseline if variant == "baseline" else candidate) + PROFILE_WRAPPER.encode())
            r = generate(out, f"profile-{variant}", variant, profile=True)
            manifest["runs"].append(r)
            assert r["pixel_sha256"] == reference, "Diagnostic output mismatch"
            counts = [json.loads(p.read_text()) for p in (out / f"profile-{variant}").glob("counts-*.json")]
            manifest[f"profile_{variant}_counts"] = counts
            assert counts, "Missing diagnostic call counts"
            if variant == "candidate":
                assert sum(c["modulate_calls"] for c in counts) > 0
            save()
        manifest["status"] = "complete"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        raise
    finally:
        (SOURCE / REL).write_bytes(candidate)
        save()
    return run_id

@app.local_entrypoint()
def main():
    print("Model cache:", download.remote())
    print("Results:", VOLUME, experiment.remote())
