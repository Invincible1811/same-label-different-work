"""Independent check of RQ1 (src/02_rq1.py).

Recomputes the headline RQ1 numbers with different code than the pipeline:
counts with the csv module, Wilson intervals and the two-proportion z-test from
their textbook formulas, χ²/Cramér's V with scipy's standard chi2_contingency
(instead of our permutation routine), and compares them with results/tables/rq1_*.csv.

Run from the repository root:  python checks/check_02_rq1.py
Exit code 0 = all checks pass.
"""
import csv
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency

ROOT = Path(__file__).resolve().parents[1]
csv.field_size_limit(sys.maxsize)
T = ROOT / "results/tables"
results = []


def close(name, expected, actual, tol=0.05):
    ok = abs(float(expected) - float(actual)) <= tol
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: expected {float(expected):.3f}, got {float(actual):.3f}")


def wilson(k, n, z=1.959964):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (c - h), 100 * (c + h)


rows = list(csv.DictReader(open(ROOT / "data/processed/issues_clean.csv", newline="", encoding="utf-8")))
labels = list(csv.DictReader(open(ROOT / "data/processed/labels_clean.csv", newline="", encoding="utf-8")))
t1 = {r["agent"]: r for r in csv.DictReader(open(T / "rq1_T1_agent_profile.csv"))}
t3 = {r["test"]: r for r in csv.DictReader(open(T / "rq1_T3_association_tests.csv"))}
fa = {(r["group"], r["min_age_days"]): r for r in csv.DictReader(open(T / "rq1_FA_own_bot_closure.csv"))}
fb = {r["agent"]: r for r in csv.DictReader(open(T / "rq1_FB_labelling_illusion.csv"))}

by_agent = defaultdict(list)
for r in rows:
    by_agent[r["agent_type"]].append(r)

print("— T1 agent profile —")
for a in ["CodeRabbit", "Claude_Code", "Sweep", "Devin", "Amazon_Q"]:
    g = by_agent[a]
    n, k = len(g), sum(r["is_closed"] == "True" for r in g)
    close(f"{a}: n", n, t1[a]["n"], 0)
    close(f"{a}: closed %", 100 * k / n, t1[a]["closed_pct"])
    lo, hi = wilson(k, n)
    close(f"{a}: closed % CI low", lo, t1[a]["closed_lo"])
    close(f"{a}: closed % CI high", hi, t1[a]["closed_hi"])
    close(f"{a}: labelled %", 100 * sum(int(r["n_labels"]) > 0 for r in g) / n, t1[a]["labelled_pct"])
    close(f"{a}: own-bot %", 100 * sum(r["own_bot_author"] == "True" for r in g) / n, t1[a]["own_bot_pct"])
    per_repo = list(Counter(r["repo_name"] for r in g).values())
    close(f"{a}: issues per repo SD", statistics.stdev(per_repo), t1[a]["issues_per_repo_sd"], 0.001)

print("\n— T3 association tests (scipy chi2_contingency, 11 tested agents) —")
closed_n = Counter(r["agent_type"] for r in rows if r["is_closed"] == "True")
tested = {a for a, n in closed_n.items() if n >= 30}
sub = [r for r in rows if r["agent_type"] in tested]
for name, key in [
    ("agent × State (open vs closed)", lambda r: r["api_state"]),
    ("agent × Labelled (yes/no)", lambda r: int(r["n_labels"]) > 0),
    ("agent × Language group", lambda r: r["lang_group"]),
    ("agent × Attribution signal", lambda r: r["attribution_signal"]),
]:
    agents = sorted(tested)
    cats = sorted({key(r) for r in sub}, key=str)
    table = np.array([[sum(1 for r in by_agent[a] if key(r) == c) for c in cats] for a in agents])
    chi2, _, _, _ = chi2_contingency(table, correction=False)
    v = math.sqrt(chi2 / (table.sum() * (min(table.shape) - 1)))
    close(f"{name}: χ²", chi2, t3[name]["chi2"], 0.01)
    close(f"{name}: Cramér's V", v, t3[name]["cramers_v"], 0.0005)

print("\n— Finding A: own-bot vs other closure (textbook two-proportion z-test) —")
for a in ["CodeRabbit", "Devin"]:
    g = by_agent[a]
    own = [r for r in g if r["own_bot_author"] == "True"]
    oth = [r for r in g if r["own_bot_author"] != "True"]
    k1, n1 = sum(r["is_closed"] == "True" for r in own), len(own)
    k2, n2 = sum(r["is_closed"] == "True" for r in oth), len(oth)
    pp = (k1 + k2) / (n1 + n2)
    z = (k1 / n1 - k2 / n2) / math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    row = fa[(a, "0")]
    close(f"{a}: own-bot closed %", 100 * k1 / n1, row["own_bot_closed_pct"])
    close(f"{a}: other closed %", 100 * k2 / n2, row["other_closed_pct"])
    close(f"{a}: difference (pp)", 100 * (k1 / n1 - k2 / n2), row["diff_pp"])
    close(f"{a}: z", z, row["z"], 0.001)

print("\n— Finding B: labelling illusion (Sweep) —")
triage = {lab["issue_id"] for lab in labels if lab["category"] != "agent_workflow"}
g = by_agent["Sweep"]
close(
    "Sweep: any label %", 100 * sum(int(r["n_labels"]) > 0 for r in g) / len(g), fb["Sweep"]["any_label_pct"]
)
close(
    "Sweep: triage-label %",
    100 * sum(r["issue_id"] in triage for r in g) / len(g),
    fb["Sweep"]["triage_label_pct"],
)
close("Sweep: rank on triage labels (13 = last)", 13, fb["Sweep"]["rank_triage"], 0)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
