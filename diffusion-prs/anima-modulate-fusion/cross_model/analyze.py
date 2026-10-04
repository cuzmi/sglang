"""Validate and retain compact correctness-only evidence for other callers."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

HERE=Path(__file__).resolve().parent
run=HERE.parent/'results'/sys.argv[1]
m=json.loads((run/'manifest.json').read_text())
assert m['status']=='complete',m
assert not m['errors'] and m['matching_case_outputs']
a,b=[json.loads((run/arm/'callers.json').read_text()) for arm in ['baseline','candidate']]
for r in [a,b]:
    assert r['status']=='complete' and not r['errors'] and len(r['cases'])==48
    assert all(c['bit_exact'] for c in r['cases'])
    assert len(r['graphs'])==5 and all(g['bit_exact'] and g['input_changed_before_replay'] for g in r['graphs'])
    assert r['shared_error_fallback']=='passed'
assert all(test['exit_code']==0 for arm in m['arms'] for test in arm['tests'].values())
aa={c['label']:c for c in a['cases']};bb={c['label']:c for c in b['cases']}
assert aa.keys()==bb.keys()
for label in aa:
    for field in ['output_sha256','first_fused_calls','repeat_fused_calls']:
        assert aa[label][field]==bb[label][field],(label,field)
lines=['Shared modulation caller correctness regression','',f'Run: {run.name}',f"GPU: {m['gpu']}; CUDA: {m['cuda']}",f"Packages: {m['packages']}",f"Baseline: {m['baseline_commit']}",f"Candidate backend SHA256: {m['candidate_sha256']}",'','Result: all checks passed in both source versions; no new output or dispatch differences.','No performance measurements or full checkpoint-based generation were run.','','| Model | Cases per version | Baseline/candidate outputs |','| --- | ---: | --- |']
for model in ['flux','ideogram','lingbot','ltx']:
    count=sum(c['label'].startswith(model) for c in a['cases'])
    lines.append(f'| {model} | {count} | identical byte hashes |')
lines += ['','48/48 eager-reference byte comparisons per version, 96 total.','All first-call and repeated-call shared-kernel invocation counts match across versions.','FP16/BF16/FP32, batch 1/2, odd sequence length 17, model widths 2048–5120.','5/5 changed-input CUDA graph replays per version, 10 total.','Shared wrapper error fallback/disabled retries pass in both versions.','','Existing tests, per version:','- Shared modulation kernel: 12 passed.','- FLUX and LTX model sites: 9 passed.','- LTX row layout and CPU fallback: 4 passed.','Total existing test executions: 50 passed (25 baseline + 25 candidate).','','Scope details: README.txt. Model entrypoints are real; tensors are synthetic.','LingBot uses its supported precomputed camera scale/shift branch.','FLUX covers normal priority and a controlled fallback to the shared kernel.','LTX uses its lossless path; Ideogram uses enable_fused=False.','The tests also cover paths that intentionally bypass the shared kernel.','Results do not establish full-model generation quality, performance, distributed','execution, other GPUs, or torch.compile compatibility. No repository tests were changed.']
(HERE/'REPORT.txt').write_text('\n'.join(lines)+'\n')
evidence=HERE/'evidence';evidence.mkdir(exist_ok=True)
shutil.copyfile(run/'manifest.json',evidence/'manifest.json')
for arm in ['baseline','candidate']:
    dst=evidence/arm;dst.mkdir(exist_ok=True)
    shutil.copyfile(run/arm/'callers.json',dst/'callers.json')
    for name in ['existing-kernel','existing-model-sites','existing-ltx-layout']:
        text=(run/arm/(name+'.log')).read_text()
        assert f'{dict(zip(["existing-kernel","existing-model-sites","existing-ltx-layout"],[12,9,4]))[name]} passed' in text
        (dst/(name+'.txt')).write_text('\n'.join(l.rstrip() for l in text.splitlines())+'\n')
files=sorted(p for p in HERE.rglob('*') if p.is_file() and p.suffix in ['.py','.txt','.json'] and p.name!='EVIDENCE.json')
(HERE/'EVIDENCE.json').write_text(json.dumps({'run_id':run.name,'modal_results_volume':'anima-modulate-experiments','scope':m['scope'],'baseline_commit':m['baseline_commit'],'candidate_backend_sha256':m['candidate_sha256'],'correctness_cases_per_version':48,'graph_replays_per_version':5,'existing_tests_per_version':25,'cross_version_byte_hashes_equal':True,'cross_version_dispatch_counts_equal':True,'sha256':{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},indent=2)+'\n')
print('\n'.join(lines))
