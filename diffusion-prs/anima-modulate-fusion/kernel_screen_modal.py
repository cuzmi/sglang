"""Run only the bounded geometry/dispatch screen in the prior exact image."""
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from uuid import uuid4
import modal

app=modal.App('anima-modulate-kernel-screen')
image=(modal.Image.from_id('im-rRXDzAdc13HCyQHStCXqwE')
       .add_local_file(Path(__file__).with_name('kernel_screen.py'),'/workspace/kernel_screen.py',copy=True))
results=modal.Volume.from_name('anima-modulate-experiments')

@app.function(image=image,gpu='H200',cpu=8,memory=65536,timeout=2400,volumes={'/results':results})
def screen():
    run_id=datetime.now(timezone.utc).strftime('kernel-screen-%Y%m%dT%H%M%SZ-')+uuid4().hex[:8]
    out=Path('/results')/run_id
    out.mkdir()
    source=Path('/workspace/sglang/python/sglang/multimodal_gen/runtime/models/dits/anima.py')
    source.write_bytes(Path('/workspace/candidate_anima.py').read_bytes())
    print('RUN_ID='+run_id,flush=True)
    try:
        with (out/'screen.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,'/workspace/kernel_screen.py',str(out)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,cwd='/workspace/sglang')
            for line in proc.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
            code=proc.wait()
        if code: raise RuntimeError(f'Screen exited {code}: {out}')
    finally:
        results.commit()
    return run_id

@app.local_entrypoint()
def main():
    print('RESULTS',screen.remote())
