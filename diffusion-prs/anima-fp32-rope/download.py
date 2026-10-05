"""Download run artifacts from the experiment-owned volume, preserving paths."""
from pathlib import Path
import sys
import modal
from modal.volume import FileEntryType
volume=modal.Volume.from_name('anima-rope-experiments')
root=Path(__file__).resolve().parent/'results'
for entry in volume.iterdir(sys.argv[1],recursive=True):
    if entry.type!=FileEntryType.FILE:continue
    dest=root/entry.path.lstrip('/')
    if dest.exists() and dest.stat().st_size==entry.size and dest.name!='manifest.json':continue
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('wb') as f:
        for chunk in volume.read_file(entry.path):f.write(chunk)
    print(dest.relative_to(root),entry.size,flush=True)
