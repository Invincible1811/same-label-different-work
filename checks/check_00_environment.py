"""Environment guard: every library pinned in requirements.txt must be installed at
exactly that version. Run this first; if a later install (e.g. for an extra tool)
upgrades a pinned library, this check fails before any result can silently change.

Run from the repository root:  python checks/check_00_environment.py
"""
import importlib.metadata as md
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
results = []
for line in (ROOT / "requirements.txt").read_text().splitlines():
    line = line.split("#")[0].strip()
    m = re.fullmatch(r"([A-Za-z0-9_.\-]+)==([A-Za-z0-9_.\-]+)", line)
    if not m:
        continue
    name, want = m.groups()
    try:
        have = md.version(name)
    except md.PackageNotFoundError:
        have = "not installed"
    ok = have == want
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: pinned {want}, installed {have}")
print(f"\nPython {sys.version.split()[0]} (results produced with 3.11.8)")
print(f"{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
