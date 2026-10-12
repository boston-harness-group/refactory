"""OS-enforced read separation for local paired lanes (macOS Seatbelt)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def profile(lane: Path, peers: list[Path], readonly: list[Path], hidden: list[Path],
            *, allowed_reads: list[Path] | None = None) -> str:
    """Read separation is enforced in every descendant, not an agent instruction."""
    def subpath(path: Path) -> str:
        return f"(subpath {json.dumps(str(path.resolve()))})"
    if allowed_reads is None:
        rules = ["(version 1)", "(allow default)"]
    else:
        # Deny content reads outside the allowlist in user/session/cache roots.
        # Metadata traversal must remain possible to execute an allowed binary
        # through a venv symlink. Peer/evidence roots below deny all read access.
        rules = ["(version 1)", "(allow default)"]
        # Seatbelt deny rules dominate allow rules. Express the complement of
        # the allowlist in one deny filter instead of a blanket deny+allow.
        excludes = " ".join(f"(require-not {subpath(p)})" for p in [lane, *allowed_reads])
        boundaries = [Path.home(), Path("/private/tmp"), Path("/private/var/folders")]
        rules += [f"(deny file-read-data (require-all {subpath(boundary)} {excludes}))" for boundary in boundaries]
    forbidden = [p for p in peers + hidden if p.resolve() != lane.resolve()]
    rules.extend(f"(deny file-read* {subpath(p)})" for p in forbidden)
    rules.extend(f"(deny file-write* {subpath(p)})" for p in peers + readonly + hidden)
    return "\n".join(rules) + "\n"


def runtime_probe(lane: Path, policy: Path, python: str, forbidden: list[Path]) -> dict:
    """Prove configured private runtime state is writable and shared state unreadable."""
    runtime = lane / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    code = """
import json,sys
from pathlib import Path
own, forbidden=json.loads(sys.argv[1]); own=Path(own)
p=own/'runtime-state-control.txt'; p.write_text('private'); assert p.read_text()=='private'
checks=[]
for name in forbidden:
    try: Path(name).read_bytes(); denied=False
    except PermissionError: denied=True
    checks.append({'path':name,'denied':denied})
print(json.dumps(checks)); sys.exit(0 if checks and all(c['denied'] for c in checks) else 1)
"""
    try:
        cp = subprocess.run(["/usr/bin/sandbox-exec", "-f", str(policy), python, "-c", code,
                             json.dumps([str(runtime), [str(p) for p in forbidden]])],
                            cwd=lane, capture_output=True, text=True, timeout=20)
        return {"status": "passed" if cp.returncode == 0 else "unavailable",
                "stdout": cp.stdout, "stderr": cp.stderr, "returncode": cp.returncode,
                "boundary": "Allowlisted content reads, private runtime stores, shared-state negative controls"}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "unavailable", "reason": str(exc)}


def probe(lane: Path, peers: list[Path], policy: Path, python: str = sys.executable) -> dict:
    """Positive own-read control and direct + subprocess peer-denial controls."""
    own = lane / "isolation-canary.txt"
    own.write_text("own lane\n")
    canaries = []
    for peer in peers:
        peer.mkdir(parents=True, exist_ok=True)
        canary = peer / "isolation-canary.txt"
        canary.write_text("peer lane\n")
        canaries.append(str(canary.resolve()))
    code = """
import json,subprocess,sys
from pathlib import Path
own, peers=json.loads(sys.argv[1])
assert Path(own).read_text() == 'own lane\\n'
checks=[]
for p in peers:
    try: Path(p).read_bytes(); denied=False
    except PermissionError: denied=True
    child=subprocess.run([sys.executable,'-c','from pathlib import Path; import sys; Path(sys.argv[1]).read_bytes()',p],capture_output=True)
    checks.append({'path':p,'direct_denied':denied,'child_denied':child.returncode!=0 and b'PermissionError' in child.stderr})
print(json.dumps(checks))
sys.exit(0 if checks and all(c['direct_denied'] and c['child_denied'] for c in checks) else 1)
"""
    argv = ["/usr/bin/sandbox-exec", "-f", str(policy), python, "-c", code,
            json.dumps([str(own.resolve()), canaries])]
    try:
        result = subprocess.run(argv, cwd=lane, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "unavailable", "reason": str(exc), "argv": argv}
    return {"status": "passed" if result.returncode == 0 else "unavailable",
            "argv": argv, "stdout": result.stdout, "stderr": result.stderr,
            "returncode": result.returncode,
            "boundary": "Seatbelt inherited by descendants; listed peer/evidence roots unreadable"}
