"""Download completed Modal result directories (skip directory entries)."""
from pathlib import Path
import sys
import modal
from modal.volume import FileEntryType

volume = modal.Volume.from_name('anima-modulate-experiments')
run_id = sys.argv[1]
remote = sys.argv[2] if len(sys.argv) > 2 else run_id
root = Path(__file__).resolve().parent / 'results'
for entry in volume.iterdir(remote, recursive=True):
    if entry.type != FileEntryType.FILE:
        continue
    dest = root / entry.path.lstrip('/')
    if dest.exists() and dest.stat().st_size == entry.size and dest.name != 'manifest.json':
        continue
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('wb') as stream:
        for chunk in volume.read_file(entry.path):
            stream.write(chunk)
    print(dest.relative_to(root), entry.size, flush=True)
