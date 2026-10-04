"""Analyze downloaded manifests and CUPTI-correlated RMSNorm trace ranges."""
import bisect
import collections
import gzip
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root/'manifest.json').read_text())
assert manifest['status'] == 'complete', manifest.get('error')
report = {'source': manifest['base'], 'versions': manifest['versions'], 'timing': {}, 'traces': {}}
for variant in ['A', 'B']:
    arms = [a for a in manifest['arms'] if a['variant'] == variant and not a['profile']]
    denoise = [sum(s['duration_ms'] for s in a['denoise_steps_ms']) for a in arms]
    total = [a['total_duration_ms'] for a in arms]
    report['timing'][variant] = {'denoise_ms': denoise, 'total_ms': total, 'median_denoise_ms': statistics.median(denoise), 'median_total_ms': statistics.median(total), 'denoise_range_pct_of_median': (max(denoise)-min(denoise))/statistics.median(denoise)*100}
a,b = [report['timing'][v] for v in ['A','B']]
report['denoise_reduction_pct'] = (1-b['median_denoise_ms']/a['median_denoise_ms'])*100
report['total_reduction_pct'] = (1-b['median_total_ms']/a['median_total_ms'])*100
report['pixel_hashes'] = sorted({a['pixel_sha256'] for a in manifest['arms']})
assert len(report['pixel_hashes']) == 1
for variant in ['A','B']:
    folder = root/f'profile-{variant}'
    report['traces'][variant] = {'dispatch': json.loads((folder/'counts.json').read_text())}
    paths = list((folder/'traces').rglob('*.trace.json.gz'))
    assert len(paths)==1, paths
    with gzip.open(paths[0],'rt') as f:
        data=json.load(f)
    events=data['traceEvents']
    steps=sorted([e for e in events if e.get('ph')=='X' and e.get('cat')=='user_annotation' and e.get('name','').startswith('ProfilerStep#')],key=lambda e:e['ts'])
    step=steps[len(steps)//2]
    norms=sorted([e for e in events if e.get('ph')=='X' and e.get('cat')=='user_annotation' and e.get('name')=='MingRMSNorm' and step['ts']<=e['ts']<step['ts']+step['dur']],key=lambda e:e['ts'])
    assert len(norms)==205, (step['name'],len(norms))
    starts=[e['ts'] for e in norms]
    corr=set()
    for e in events:
        if e.get('cat') not in ['cuda_runtime','cuda_driver'] or 'Launch' not in e.get('name',''):
            continue
        idx=bisect.bisect_right(starts,e['ts'])-1
        if idx>=0 and e['ts']<norms[idx]['ts']+norms[idx]['dur']:
            corr.add(e.get('args',{}).get('correlation'))
    kernels=[e for e in events if e.get('cat')=='kernel' and e.get('args',{}).get('correlation') in corr]
    assert kernels
    names=collections.Counter(e['name'] for e in kernels)
    durations=collections.defaultdict(float)
    for e in kernels:
        durations[e['name']]+=e['dur']
    report['traces'][variant].update(step=step['name'],norm_calls=len(norms),kernel_count=len(kernels),kernel_sum_ms=sum(e['dur'] for e in kernels)/1000,kernels=[{'name':n,'count':c,'sum_ms':durations[n]/1000} for n,c in names.most_common()])
    selected=[e for e in events if e.get('ph')=='M' or step['ts']<=e.get('ts',-1)<step['ts']+step['dur'] or (e.get('cat')=='kernel' and e.get('args',{}).get('correlation') in corr)]
    with gzip.open(folder/'middle-step.trace.json.gz','wt') as f:
        json.dump({**data,'traceEvents':selected},f,separators=(',',':'))
(root/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
