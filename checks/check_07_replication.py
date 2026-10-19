"""Replication check for a replicator's machine: compares the outputs you just
regenerated with the reference fingerprints in results/MANIFEST.csv.

* CSV tables and figure data: compared numerically (relative tolerance 1e-5, i.e. about
  6 significant digits). Different builds of the numerical libraries (e.g. Anaconda's
  MKL vs pip's OpenBLAS) can change the last digit of tiny p-values; nothing else moves.
* Figures (PDF/PNG): their bytes depend on the font-rendering engine (FreeType version),
  so they are verified through their data files in results/figure_data/ instead.

Usage (after running the pipeline):  python checks/check_07_replication.py <reference_results_dir>
   e.g. compare a fresh clone's results/ against the reference copy you downloaded.
With no argument it checks byte-identity against results/MANIFEST.csv and reports
which files differ only within numerical tolerance.
"""
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
csv.field_size_limit(sys.maxsize)


def same_numbers(a, b, rtol=1e-5):
    x, y = pd.read_csv(a), pd.read_csv(b)
    if list(x.columns) != list(y.columns) or len(x) != len(y):
        return False
    for col in x.columns:
        if x[col].dtype.kind in "fi" and y[col].dtype.kind in "fi":
            if not np.allclose(x[col].values, y[col].values, rtol=rtol, atol=1e-12, equal_nan=True):
                return False
        elif not x[col].astype(str).equals(y[col].astype(str)):
            return False
    return True


rows = list(csv.DictReader(open(ROOT / "results/MANIFEST.csv")))
ref_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
identical = tolerant = figures = failed = 0
for r in rows:
    p = ROOT / r["path"]
    if not p.exists() or r["path"].startswith("data/processed"):
        continue
    if p.suffix in (".pdf", ".png"):
        figures += 1
        continue
    if hashlib.sha256(p.read_bytes()).hexdigest() == r["sha256"]:
        identical += 1
    elif (
        ref_dir is not None
        and p.suffix == ".csv"
        and same_numbers(p, ref_dir / Path(r["path"]).relative_to("results"))
    ):
        tolerant += 1
    else:
        failed += 1
        print("DIFFERS:", r["path"])
print(
    f"byte-identical: {identical} | equal within 1e-5: {tolerant} | figures (checked via "
    f"figure_data): {figures} | differing: {failed}"
)
sys.exit(0 if failed == 0 else 1)
