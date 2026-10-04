"""Rebuild the exact baseline archive without copying a dirty working tree."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sglang", required=True, type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    source = json.loads((here / "source_manifest.json").read_text())
    measured = json.loads((here / "evidence/full-manifest.json").read_text())
    for relative, expected in source["files"].items():
        name = Path(relative).name
        if name == "ming_image.py":
            name = "candidate_ming_image.py"
        path = here / "inputs" / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"GPU-tested overlay changed: {path}")
    destination = here / "inputs/baseline.tar.gz"
    temporary = destination.with_suffix(".tmp")
    try:
        subprocess.run(
            ["git", "-C", str(args.sglang.resolve()), "archive", "--format=tar.gz",
             f"--output={temporary}", source["base"]], check=True
        )
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if digest != measured["archive_sha256"]:
            raise RuntimeError(f"Baseline archive hash differs: {digest}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Verified baseline and three overlays: {destination}")


if __name__ == "__main__":
    main()
