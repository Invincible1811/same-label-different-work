"""Seed-robustness of the Monte Carlo steps: results that rely on random draws
(permutation p-values in RQ1, bootstrap median CIs in RQ2) are recomputed with two
other seeds; the conclusions must not change.

Run from the repository root:  python checks/check_06_montecarlo.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stats_utils import boot_median_ci, median_ci_exact, permutation_chi2  # noqa: E402

T = ROOT / "results/tables"
results = []


def check(name, ok, detail):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


d = pd.read_csv(ROOT / "data/processed/issues_clean.csv")
closed_n = d[d.is_closed].agent_type.value_counts()
tested = closed_n[closed_n >= 30].index
dt = d[d.agent_type.isin(tested)]
ct = dt[dt.is_closed]
tests = {
    "State": (dt.agent_type, dt.api_state),
    "Closure reason": (ct.agent_type, ct.closure_reason),
    "Labelled": (dt.agent_type, np.where(dt.n_labels > 0, "y", "n")),
    "Language": (dt.agent_type, dt.lang_group),
    "Attribution": (dt.agent_type, dt.attribution_signal),
}
t3 = pd.read_csv(T / "rq1_T3_association_tests.csv")

print("— RQ1 permutation χ² p-values with other seeds (10,000 permutations each) —")
for seed in (1, 2):
    ps_ = [permutation_chi2(np.asarray(g), np.asarray(c), seed=seed)["p"] for g, c in tests.values()]
    holm = multipletests(ps_, method="holm")[1]
    check(
        f"seed {seed}: all 5 associations significant after Holm (p < 0.001)",
        all(h < 0.001 for h in holm),
        f"max Holm p = {max(holm):.4f} (seed 42: {t3.p_holm.max():.4f})",
    )

print("\n— RQ2 median CIs: why the exact method replaced the bootstrap —")
summ = pd.read_csv(T / "rq2_T2_ttc_summary.csv").set_index("agent").drop("All")
c = d[d.rq2_eligible]
worst = 0.0
for a in summ.index:
    x = c.loc[c.agent_type == a, "ttc_hours"].values
    cis = [boot_median_ci(x, np.random.default_rng(s)) for s in (42, 1, 2)]
    width = max(ci[1] for ci in cis) - min(ci[0] for ci in cis)
    worst = max(worst, (max(ci[1] for ci in cis) - min(ci[1] for ci in cis)) / width)
check(
    "bootstrap median CIs ARE seed-sensitive (documented reason for the change)",
    worst > 0.25,
    f"upper bound moves by up to {100 * worst:.0f}% of the interval across seeds 42/1/2",
)
again = [median_ci_exact(c.loc[c.agent_type == a, "ttc_hours"].values) for a in summ.index]
check(
    "exact order-statistic CIs reproduce the reported values (deterministic)",
    all(
        abs(lo - summ.loc[a, "median_ci_lo"]) < 1e-3 * max(1, abs(lo))
        and abs(hi - summ.loc[a, "median_ci_hi"]) < 1e-3 * max(1, abs(hi))
        for a, (lo, hi, _) in zip(summ.index, again)
    ),
    "identical on recomputation; no seed involved",
)
check(
    "exact CIs achieve >= 95% coverage for every agent",
    all(cov >= 0.95 for _, _, cov in again),
    f"min achieved coverage = {min(cov for _, _, cov in again):.4f}",
)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
