"""Recreate measured inputs from a baseline Git object and archived patches.

Does not depend on, or modify, the source repository's working tree.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile

BASE = "6fa3fe69e2e5e19b75cadd9fc285b72634551992"
REL = "python/sglang/multimodal_gen/runtime/models/dits/anima.py"
EXPECTED = "ae62fe965c2facdf224aa164a5002dc5bd2e403a48fb51cf6094179b4d254013"
BACKEND = "python/sglang/kernels/ops/diffusion/modulate/modulate_scale_shift_jit.py"
EXPECTED_BACKEND = "5705525464fedef85e83631e44aa53ce87bd7a7b1b83d7294a25c22621bcc8b0"
ARCHIVE_SHA = "349f9021b4d7bcd7d3c42425d1691fccc42f8d75f2ac5054882407d621ebb233"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path, help="SGLang clone containing the baseline commit")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "inputs")
    args = parser.parse_args()
    repo, out = args.repo.resolve(), args.output.resolve()
    patches = Path(__file__).resolve().parent / "patches"
    with tempfile.TemporaryDirectory(prefix="anima-inputs-") as temporary:
        scratch = Path(temporary)
        for rel in [REL, BACKEND]:
            path = scratch / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(subprocess.check_output(["git", "show", f"{BASE}:{rel}"], cwd=repo))
        subprocess.run(["git", "apply", "--check", str(patches / "dispatch_candidate.patch")], cwd=scratch, check=True)
        subprocess.run(["git", "apply", str(patches / "dispatch_candidate.patch")], cwd=scratch, check=True)
        candidate, backend = (scratch / REL).read_bytes(), (scratch / BACKEND).read_bytes()
        assert hashlib.sha256(candidate).hexdigest() == EXPECTED
        assert hashlib.sha256(backend).hexdigest() == EXPECTED_BACKEND
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "baseline.tar.gz"
        subprocess.run(["git", "archive", "--format=tar.gz", f"--output={archive}", BASE], cwd=repo, check=True)
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == ARCHIVE_SHA
        (out / "candidate_anima.py").write_bytes(candidate)
        (out / "dispatch_modulate_scale_shift_jit.py").write_bytes(backend)
        for name in ["candidate.patch", "dispatch_candidate.patch"]:
            (out / name).write_bytes((patches / name).read_bytes())
    print("Prepared hash-verified baseline and both measured variants in", out)


if __name__ == "__main__":
    main()
