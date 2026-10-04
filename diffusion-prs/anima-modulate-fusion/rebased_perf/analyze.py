"""Validate all eight arms and archive a compact, reproducible performance report."""
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys

HERE = Path(__file__).resolve().parent
BASE = "affa261e3d289fe4f907c9b2e8d773fef0d36dba"
HEAD = "05bee50b2f4fef4484bab3db3bff7ef4495e98e5"
ORDER = ["baseline", "candidate", "candidate", "baseline",
         "candidate", "baseline", "baseline", "candidate"]
run = HERE.parent / "results" / sys.argv[1]
m = json.loads((run / "manifest.json").read_text())
assert m["status"] == "complete" and "error" not in m
assert m["baseline_commit"] == BASE and m["candidate_commit"] == HEAD
assert m["order"] == ORDER and len(m["runs"]) == 8
assert [r["variant"] for r in m["runs"]] == ORDER
assert len({r["pixel_sha256"] for r in m["runs"]}) == 1
assert len({r["png_sha256"] for r in m["runs"]}) == 1
assert m["workload"] == {
    "width": 1024, "height": 1024, "steps": 30, "cfg": 4, "seed": 42,
    "prompt": "masterpiece, best quality, safe, watercolor landscape, a quiet seaside village at sunset",
    "negative_prompt": "", "dtype": "bfloat16", "tp": 1, "attention": "fa",
    "compile": False, "cuda_graph": False, "warmup_mode": "request", "warmup_steps": 1,
}
mode = "synchronized" if "timing_instrumentation" in m else "native"
dst = HERE / "evidence" / mode
dst.mkdir(parents=True, exist_ok=True)
shutil.copyfile(run / "manifest.json", dst / "manifest.json")
if mode == "synchronized":
    timing = m["timing_instrumentation"]
    assert timing["baseline_sha256"] == "4431f15ceaf926d8b1afde8ab15d1c6bca66fe06a63690daf0d1da83f2b616db"
    assert timing["patch"].count("+            torch.cuda.synchronize()") == 2
    (dst / "timing.patch").write_text(timing["patch"])
snapshots = []
request_ids = set()
for i, r in enumerate(m["runs"], 1):
    assert r["label"] == f"{i:02d}-{r['variant']}"
    assert r["source_sha256"] == m[r["variant"] + "_source_sha256"]
    folder = run / r["label"]
    perf = json.loads((folder / "perf.json").read_text())
    assert perf["total_duration_ms"] == r["total_duration_ms"] > 0
    assert perf["denoise_steps_ms"] == r["denoise_steps_ms"]
    assert len(perf["denoise_steps_ms"]) == 30
    assert perf["request_id"] not in request_ids
    request_ids.add(perf["request_id"])
    assert perf["meta"]["prompt"] == [m["workload"]["prompt"]]
    assert perf["meta"]["model"].endswith("/" + m["model_revision"])
    if mode == "synchronized":
        assert r["boundary_sync_markers"] == {"warmup": 1, "measured": 1}
        log = (folder / "generate.log").read_text()
        assert log.count("ANIMA_BENCHMARK_END_SYNC warmup=True") == 1
        assert log.count("ANIMA_BENCHMARK_END_SYNC warmup=False") == 1
    assert hashlib.sha256((folder / "outputs/sample.png").read_bytes()).hexdigest() == r["png_sha256"]
    shutil.copyfile(folder / "perf.json", dst / (r["label"] + ".json"))
    for when in ["before", "after"]:
        snap = r["gpu_" + when]
        assert snap["gpu_exit_code"] == snap["process_exit_code"] == 0
        snapshots.append({"arm": r["label"], "when": when, **snap})
assert len({s["gpu"].split(",")[1].strip() for s in snapshots}) == 1


def summarize(runs):
    out = {}
    for variant in ["baseline", "candidate"]:
        values = [r["total_duration_ms"] for r in runs if r["variant"] == variant]
        out[variant] = {"n": len(values), "values_ms": values,
                        "mean_ms": statistics.mean(values),
                        "median_ms": statistics.median(values),
                        "min_ms": min(values), "max_ms": max(values),
                        "sample_stdev_ms": statistics.stdev(values)}
    out["latency_change_percent"] = (out["candidate"]["mean_ms"] / out["baseline"]["mean_ms"] - 1) * 100
    return out


summary = {"run_id": run.name, "mode": mode, "metric": m["metric"], "baseline_commit": BASE, "candidate_commit": HEAD,
           "overall": summarize(m["runs"]), "abba": summarize(m["runs"][:4]),
           "baab": summarize(m["runs"][4:]),
           "all_output_pixels_equal": True, "all_png_bytes_equal": True,
           "gpu_snapshots": snapshots}
(dst / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
lines = [f"Anima post-rebase performance ({mode})", "", f"Run: {run.name}",
         f"Base: {BASE}", f"Head: {HEAD}", f"Image: {m['image_id']}",
         f"GPU: {m['gpu']}; CUDA: {m['cuda']}; CPU: {m['cpu_models']}",
         f"Packages: {m['packages']}", "",
         "1024x1024, BF16, TP1, FA, 30 steps, CFG4, seed42, no compile/CUDA graph.",
         "One container/GPU; ABBA + BAAB; fresh CLI process for every arm.",
         "One-step same-resolution request warmup per process; one timed 30-step request.",
         f"Metric: {m['metric']}.",
         "No profiler during timing. Four process samples per variant.", "",
         "Arm          Variant    Worker request (ms)"]
for r in m["runs"]:
    lines.append(f"{r['label']:<12} {r['variant']:<10} {r['total_duration_ms']:.6f}")
for name in ["overall", "abba", "baab"]:
    s = summary[name]
    lines.extend(["", name.upper()])
    for variant in ["baseline", "candidate"]:
        a = s[variant]
        lines.append(f"{variant}: n={a['n']}, mean={a['mean_ms']:.3f} ms, "
                     f"range={a['min_ms']:.3f}..{a['max_ms']:.3f} ms, "
                     f"sample SD={a['sample_stdev_ms']:.3f} ms")
    lines.append(f"Candidate mean latency change: {s['latency_change_percent']:+.3f}%")
lines.extend(["", "Validity: all eight arms completed 30 steps with exact expected source hashes.",
              "All eight decoded pixel hashes and PNG byte hashes match.",
              f"Pixel SHA256: {m['runs'][0]['pixel_sha256']}", "",
              "Limitations: one GPU instance, one prompt/seed/resolution, four process samples per version.",
              "No same-version A/A run or cross-instance replication; this does not establish a universal speedup.",
              "ABBA/BAAB balance simple ordering effects but cannot remove all runtime noise.",
              "GPU boundary snapshots are archived; they do not prove isolation throughout each timed request.",
              "Earlier measurements used a different base and instance; they are not pooled here."])
if mode == "native":
    lines.extend(["", "Timing caveat: the native timer stops before output transport without an explicit end sync.",
                  "This is a host-side worker metric and may omit a trailing CUDA tail.",
                  "Use the separately archived synchronized experiment for device-completed request timing."])
else:
    lines.extend(["", "Identical timing-only worker instrumentation synchronizes before timer start and after forward.",
                  "The marker is printed after the timer stops; one warmup and one measured marker per arm are verified.",
                  "Only this benchmark image is instrumented; the production branch is unchanged."])
(HERE / f"REPORT-{mode}.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
