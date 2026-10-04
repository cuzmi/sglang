"""Summarize the complete, same-instance ABC-CBA dispatch experiment."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent
run = ROOT / 'results' / sys.argv[1]
m = json.loads((run / 'manifest.json').read_text())
assert m['status'] == 'complete', m['status']
assert [r['variant'] for r in m['runs']] == [
    'eager', 'initial_mount', 'cached_dispatch',
    'cached_dispatch', 'initial_mount', 'eager',
]
assert m['existing-modulate'] == m['existing-anima'] == 'passed'
assert len({r['pixel_sha256'] for r in m['runs']}) == 1
assert len({r['png_sha256'] for r in m['runs']}) == 1
summary = {'run_id': run.name, 'variants': {}}
for name in ['eager', 'initial_mount', 'cached_dispatch']:
    samples = [r['total_duration_ms'] for r in m['runs'] if r['variant'] == name]
    summary['variants'][name] = dict(samples_ms=samples, mean_ms=statistics.mean(samples),
                                     min_ms=min(samples), max_ms=max(samples))
base = summary['variants']['eager']['mean_ms']
initial = summary['variants']['initial_mount']['mean_ms']
fixed = summary['variants']['cached_dispatch']['mean_ms']
summary['initial_vs_eager_pct'] = (initial / base - 1) * 100
summary['fixed_vs_eager_pct'] = (fixed / base - 1) * 100
summary['fixed_vs_initial_pct'] = (fixed / initial - 1) * 100
summary['pixel_sha256'] = m['runs'][0]['pixel_sha256']
(run / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
lines = ['Anima cached dispatch: controlled full-model ABC-CBA', '',
         f"Run: {run.name}", f"Baseline: {m['baseline_commit']}",
         f"Model: {m['model_id']} @ {m['model_revision']}",
         f"GPU: {m['gpu']}; CUDA: {m['cuda']}", f"Packages: {m['packages']}", '',
         'One container/GPU, fresh process per arm, same-resolution one-step request warmup.',
         '1024x1024, BF16, 30 steps, CFG 4, seed 42, TP1, FA, compile and CUDA graph off.',
         'Native worker total_duration_ms excludes model loading, warmup and PNG saving.',
         'All six requests completed with identical decoded pixels and PNG bytes.', '',
         '| Order | Variant | Worker request (ms) |', '| --- | --- | ---: |']
for i, r in enumerate(m['runs'], 1):
    lines.append(f"| {i} | {r['variant']} | {r['total_duration_ms']:.3f} |")
lines += ['', '| Variant | Mean (ms) | Min–max (ms) |', '| --- | ---: | ---: |']
for name, r in summary['variants'].items():
    lines.append(f"| {name} | {r['mean_ms']:.3f} | {r['min_ms']:.3f}–{r['max_ms']:.3f} |")
lines += ['', f"Initial mount vs eager: {summary['initial_vs_eager_pct']:+.3f}% latency.",
          f"Cached dispatch vs eager: {summary['fixed_vs_eager_pct']:+.3f}% latency.",
          f"Cached dispatch vs initial mount: {summary['fixed_vs_initial_pct']:+.3f}% latency.", '',
          'Only two samples per variant: a bounded workload comparison, not a latency distribution.',
          'Do not pool these absolute times with the earlier Modal instance.',
          'The earlier initial-mount ABBA regressed 4.93%; retain that negative result.', '',
          'Correctness: standalone full AdaLN checks, CUDA graph replay, existing modulate (12)',
          'and Anima (17) tests all completed before the model comparison. See operator.json and test logs.',
          'No new repository unit tests were added.', '',
          'The final patch caches the exported FFI callable and avoids a repeated eligibility check.',
          'CUDA arithmetic and launch geometry are identical across all three arms.',
          f"CUDA source SHA256: {m['cuda_source_sha256']}",
          f"Fused model SHA256: {m['fused_model_sha256']}",
          f"Fixed backend SHA256: {m['fixed_backend_sha256']}",
          f"Pixel SHA256: {summary['pixel_sha256']}", '',
          'Geometry/host-path screening is a separate experiment:',
          '../kernel-screen-20261003T231957Z-36d5d3ca/REPORT.txt',
          'Screening prototypes are not the exact production wrapper. This run measures the exact final patch.']
(run / 'REPORT.txt').write_text('\n'.join(lines) + '\n')
hashes = {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(run.rglob('*')) if p.is_file() and p.name != 'SHA256.json'}
(run / 'SHA256.json').write_text(json.dumps(hashes, indent=2) + '\n')
print(json.dumps(summary, indent=2))
