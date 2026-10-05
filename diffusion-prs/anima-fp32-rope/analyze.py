"""Summarize controlled request timing and trace evidence without pooling arms."""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys

run=Path(sys.argv[1]);m=json.loads((run/'manifest.json').read_text())
summary={'status':m['status'],'baseline_commit':m['baseline_commit'],'gpu':m['gpu'],'packages':m['packages']}
if (run/'operator.json').exists():
    op=json.loads((run/'operator.json').read_text())
    summary['correctness_cases']=len(op['correctness'])
    summary['fallback']=op['fallback']
    summary['operator']=[]
    for case in op['microbenchmark']:
        aa0,aa1,a1,b1,b2,a2=case['us']
        a=statistics.mean([a1,a2]);b=statistics.mean([b1,b2])
        summary['operator'].append(dict(shape=case['shape'],eager_us=a,fused_us=b,speedup=a/b,
            aa_diff_pct=(aa1/aa0-1)*100,raw_us=case['us']))
for phase in ['AA','ABBA']:
    runs=[r for r in m['runs'] if r.get('phase')==phase]
    stats={}
    for variant in ['eager','fused']:
        rows=[r for r in runs if r['variant']==variant]
        if rows:
            stats[variant]={metric:dict(mean=statistics.mean(r[metric] for r in rows),raw=[r[metric] for r in rows],
                range_pct=(max(r[metric] for r in rows)/min(r[metric] for r in rows)-1)*100)
                for metric in ['denoise_ms','total_duration_ms']}
    if 'eager' in stats and 'fused' in stats:
        stats['reduction_pct']={metric:(1-stats['fused'][metric]['mean']/stats['eager'][metric]['mean'])*100
                                for metric in ['denoise_ms','total_duration_ms']}
    summary[phase]=stats
summary['output_pixel_hashes']=sorted(set(r['pixel_sha256'] for r in m['runs']))
summary['all_output_pixels_equal']=len(summary['output_pixel_hashes'])==1
summary['traces']=[]
for p in sorted(run.rglob('*')):
    if not (p.name.endswith('.json') or p.name.endswith('.json.gz')):continue
    if p.name in ['manifest.json','operator.json','perf.json','summary.json']:continue
    data=gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes()
    try:trace=json.loads(data)
    except (ValueError,UnicodeDecodeError):continue
    if not isinstance(trace,dict) or 'traceEvents' not in trace:continue
    events=trace['traceEvents'];kernels=[e for e in events if e.get('cat')=='kernel']
    counts=collections.Counter(e['name'] for e in kernels)
    ranges=[e for e in events if e.get('name')=='AnimaFP32RoPE' and e.get('ph')=='X' and e.get('cat')=='user_annotation']
    fused=[e for e in kernels if '_rope_rotate_half_fp32_kernel' in e['name']]
    summary['traces'].append(dict(path=str(p.relative_to(run)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        profiler_steps=[e['name'] for e in events if e.get('cat')=='user_annotation' and e.get('name','').startswith('ProfilerStep#')],
        kernels=len(kernels),summed_kernel_us=sum(e['dur'] for e in kernels),rope_ranges=len(ranges),
        fused_rope_kernels=len(fused),fused_rope_us=sum(e['dur'] for e in fused),
        top_kernels=counts.most_common(12)))
(run/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
