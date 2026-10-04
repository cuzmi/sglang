"""Summarize unprofiled ABBA and correlate AdaLN CUDA launches in diagnostics."""
import collections
import gzip
import json
from pathlib import Path
import re
import statistics
import sys


def inside(outer, inner):
    return (outer.get('pid') == inner.get('pid') and outer.get('tid') == inner.get('tid')
            and outer['ts'] <= inner['ts'] < outer['ts'] + outer['dur'])


def trace_summary(path, variant):
    with gzip.open(path, 'rt') as f:
        data = json.load(f)
    events = data['traceEvents']
    spans = [e for e in events if e.get('ph') == 'X' and e.get('dur', 0) > 0]
    steps = sorted([e for e in spans if re.match(r'^ProfilerStep#\d+$', e.get('name', ''))], key=lambda e:e['ts'])
    assert len(steps) >= 3, f'Not enough profiler steps in {path}'
    step = steps[len(steps)//2]
    ranges = [e for e in spans if e.get('name') == 'AnimaAdaLayerNorm' and inside(step, e)]
    assert ranges, 'No AdaLN annotations in middle profiler step'
    ops = [e for e in spans if e.get('cat') == 'cpu_op']
    launches = [e for e in spans if e.get('cat') in ('cuda_runtime', 'cuda_driver') and 'Launch' in e['name']]
    kernels = [e for e in spans if e.get('cat') == 'kernel']
    correlations = {e.get('args', {}).get('correlation') for e in launches if any(inside(r, e) for r in ranges)}
    correlations.discard(None)
    ada_kernels = [e for e in kernels if e.get('args', {}).get('correlation') in correlations]
    targets = []
    for r in ranges:
        contained = [e for e in ops if inside(r, e)]
        if variant == 'baseline':
            norms = [e for e in contained if e['name'] == 'aten::layer_norm']
            assert len(norms) == 1
            boundary = norms[0]['ts'] + norms[0]['dur']
            targets += [e for e in contained if e['ts'] >= boundary and e['name'] in ('aten::add', 'aten::mul')]
        else:
            targets += [e for e in spans if e['name'] == 'AnimaModulate' and inside(r, e)]
    corr = {e.get('args', {}).get('correlation') for e in launches if any(inside(t, e) for t in targets)}
    corr.discard(None)
    target_kernels = [e for e in kernels if e.get('args', {}).get('correlation') in corr]
    assert target_kernels
    expected = len(ranges) * (3 if variant == 'baseline' else 1)
    assert len(target_kernels) == expected, (variant, len(ranges), len(target_kernels), expected)
    summary = dict(path=str(path), step=step['name'], adaln_calls=len(ranges),
                   adaln_kernel_count=len(ada_kernels), modulation_kernel_count=len(target_kernels),
                   modulation_kernel_time_ms=sum(e['dur'] for e in target_kernels)/1000,
                   modulation_kernel_names=dict(collections.Counter(e['name'] for e in target_kernels)),
                   adaln_kernel_names=dict(collections.Counter(e['name'] for e in ada_kernels)),
                   note='Diagnostic durations only; not end-to-end performance evidence.')
    selected_corr = correlations
    selected = [e for e in events if e.get('ph') == 'M' or
                (e.get('ts') is not None and step['ts'] <= e['ts'] < step['ts']+step['dur']) or
                (e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset') and e.get('args',{}).get('correlation') in selected_corr)]
    out = path.with_name('middle-step.trace.json.gz')
    with gzip.open(out, 'wt', compresslevel=1) as f:
        json.dump({**data, 'traceEvents': selected}, f)
    return summary


def main():
    root = Path(sys.argv[1])
    manifest = json.loads((root/'manifest.json').read_text())
    assert manifest['status'] == 'complete', manifest.get('error')
    runs = [r for r in manifest['runs'] if re.match(r'^\d\d-', r['label'])]
    assert len(runs) == 8 and len({r['pixel_sha256'] for r in runs}) == 1
    report = {'runs': [], 'timing': {}, 'profile': {}}
    for r in runs:
        report['runs'].append({k: r[k] for k in ['label', 'total_duration_ms', 'denoise_ms', 'pixel_sha256']})
    for variant in ['baseline','candidate']:
        values = [r['total_duration_ms'] for r in runs if r['variant']==variant]
        report['timing'][variant] = dict(n=len(values), mean_ms=statistics.mean(values), median_ms=statistics.median(values),
                                         min_ms=min(values), max_ms=max(values), spread_ms=max(values)-min(values), stdev_ms=statistics.stdev(values))
        traces = [p for p in (root/f'profile-{variant}/traces').rglob('*.trace.json.gz') if p.name!='middle-step.trace.json.gz']
        assert len(traces) == 1, traces
        report['profile'][variant] = trace_summary(traces[0], variant)
    a, b = (report['timing'][v]['mean_ms'] for v in ['baseline','candidate'])
    report['delta'] = dict(saved_ms=a-b, reduction_percent=100*(a-b)/a)
    (root/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
