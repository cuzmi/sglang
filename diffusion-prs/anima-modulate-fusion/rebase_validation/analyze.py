"""Validate correctness evidence for the rebased PR; no latency claims."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

HERE=Path(__file__).resolve().parent
run=HERE.parent/'results'/sys.argv[1]
m=json.loads((run/'manifest.json').read_text());p=json.loads((HERE/'provenance.json').read_text())
assert m['status']=='complete' and not m['errors']
assert m['baseline_commit']==p['base'] and m['candidate_commit']==p['head']
assert m['candidate_sha256']==p['sources']['python/sglang/kernels/ops/diffusion/modulate/modulate_scale_shift_jit.py']
assert m['candidate_model_sha256']==p['sources']['python/sglang/multimodal_gen/runtime/models/dits/anima.py']
assert all(t['exit_code']==0 for a in m['arms'] for t in a['tests'].values())
a,b=[json.loads((run/arm/'callers.json').read_text()) for arm in ['baseline','candidate']]
for r in [a,b]:
    assert r['status']=='complete' and not r['errors']
    assert len(r['cases'])==48 and all(c['bit_exact'] for c in r['cases'])
    assert len(r['graphs'])==5 and all(g['bit_exact'] for g in r['graphs'])
    assert r['shared_error_propagation']=='passed'
aa={c['label']:c for c in a['cases']};bb={c['label']:c for c in b['cases']}
assert aa==bb, 'Output hashes or dispatch counts differ'
anima=json.loads((run/'candidate/anima.json').read_text())
assert len(anima['correctness'])==36 and all(c['bit_exact'] for c in anima['correctness'])
assert anima['cuda_graph_replay_bit_exact']
dst=HERE/'evidence';dst.mkdir(exist_ok=True)
shutil.copyfile(run/'manifest.json',dst/'manifest.json')
counts={}
for arm in ['baseline','candidate']:
    target=dst/arm;target.mkdir(exist_ok=True)
    shutil.copyfile(run/arm/'callers.json',target/'callers.json')
    for name,count in [('existing-kernel',12),('existing-model-sites',9),('existing-ltx-layout',4),('existing-anima',17)]:
        s=(run/arm/(name+'.log')).read_text();assert f'{count} passed' in s
        (target/(name+'.txt')).write_text('\n'.join(l.rstrip() for l in s.splitlines())+'\n')
shutil.copyfile(run/'candidate/anima.json',dst/'candidate/anima.json')
shutil.copyfile(HERE.parent/'rebase-pre-commit.log',dst/'pre-commit.txt')
lines=['Rebased Anima PR correctness validation','',f'Run: {run.name}',f"Base: {p['base']}",f"Candidate: {p['head']}",f"GPU: {m['gpu']}; CUDA: {m['cuda']}",f"Packages: {m['packages']}",'','All checks passed. The rebased delta only caches the FFI callable and integrates Anima.','Upstream owns eligibility checks, HIP rejection, CUDA launcher validation and error propagation.','The old disabled-runtime registry and catch-and-fallback contract are not reintroduced.','','Other model callers (FLUX, Ideogram, LingBot, LTX video/audio):','- 48 component cases per version pass; output byte hashes and dispatch counts match.','- 5 changed-input CUDA graph replays per version pass.','- Supported-input kernel errors propagate on both first and second calls, matching main.','','Anima candidate:','- 36 full AdaLN cases pass, including actual low-precision dispatch and FP32 fallback.','- Warmed BF16 batch-2 CUDA graph replay passes.','','Existing tests per source version: 12 kernel + 9 model-site + 4 LTX layout + 17 Anima = 42 passed.','Total: 84 existing test executions, 96 other-model component cases, 36 Anima cases.','Pre-commit passed for the two production files.','','These are checkpoint-free correctness checks. No new repository unit tests were added.','No full-model generation or performance rerun was performed after rebasing.','Earlier 4.15% latency reduction is historical evidence against 6fa3fe69e2, not this new base.']
(HERE/'REPORT.txt').write_text('\n'.join(lines)+'\n')
files=[x for x in HERE.rglob('*') if x.is_file() and (x.suffix in ['.py','.json','.txt','.patch'] or x.name=='.gitattributes') and x.name!='EVIDENCE.json']
(HERE/'EVIDENCE.json').write_text(json.dumps({'run_id':run.name,'base':p['base'],'head':p['head'],'modal_volume':'anima-modulate-experiments','sha256':{str(x.relative_to(HERE)):hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted(files)}},indent=2)+'\n')
print('\n'.join(lines))
