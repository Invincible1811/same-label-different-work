"""Independent check of RQ2 (src/03_rq2.py).

Every test is re-implemented from its textbook definition, without scipy's test
functions, scikit-posthocs or lifelines:
  * Kruskal–Wallis H from average ranks with the tie correction (Kruskal & Wallis, 1952)
  * Dunn's z for all pairs + Holm's step-down adjustment (Dunn, 1964; Holm, 1979)
  * Cliff's δ by brute force over all pairs of observations (Cliff, 1993)
  * Kaplan–Meier product-limit estimator and its median (Kaplan & Meier, 1958)
  * Mann–Whitney U from ranks
and compared with results/tables/rq2_*.csv.

Run from the repository root:  python checks/check_03_rq2.py
"""
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
csv.field_size_limit(sys.maxsize)
T = ROOT / "results/tables"
results = []


def close(name, expected, actual, tol):
    # result CSVs store 6 significant digits, so allow that relative precision as well
    ok = abs(float(expected) - float(actual)) <= max(tol, 1e-5 * abs(float(actual)))
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: expected {float(expected):.6g}, got {float(actual):.6g}")


def avg_ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def tie_sum(values):
    counts = defaultdict(int)
    for v in values:
        counts[v] += 1
    return sum(t**3 - t for t in counts.values())


def normal_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def kruskal(groups):
    allv = [v for g in groups.values() for v in g]
    r = avg_ranks(allv)
    n, pos, H = len(allv), 0, 0.0
    mean_ranks = {}
    for a, g in groups.items():
        rs = r[pos : pos + len(g)]
        pos += len(g)
        mean_ranks[a] = sum(rs) / len(g)
        H += sum(rs) ** 2 / len(g)
    H = 12 / (n * (n + 1)) * H - 3 * (n + 1)
    H /= 1 - tie_sum(allv) / (n**3 - n)
    return H, n, mean_ranks, allv


def km_median(durations, events):
    """Product-limit estimator; returns the first time S(t) <= 0.5 (inf if never)."""
    times = sorted(set(t for t, e in zip(durations, events) if e))
    s = 1.0
    for t in times:
        at_risk = sum(1 for d in durations if d >= t)
        died = sum(1 for d, e in zip(durations, events) if e and d == t)
        s *= 1 - died / at_risk
        if s <= 0.5 + 1e-12:  # tolerate floating-point error when S(t) is exactly 0.5
            return t
    return math.inf


rows = list(csv.DictReader(open(ROOT / "data/processed/issues_clean.csv", newline="", encoding="utf-8")))
closed = [r for r in rows if r["rq2_eligible"] == "True"]
groups = defaultdict(list)
for r in closed:
    groups[r["agent_type"]].append(float(r["ttc_hours"]))
groups = dict(sorted(groups.items()))

summ = {r["agent"]: r for r in csv.DictReader(open(T / "rq2_T2_ttc_summary.csv"))}
tests = {r["test"]: r for r in csv.DictReader(open(T / "rq2_T6_tests.csv"))}
pairs = list(csv.DictReader(open(T / "rq2_T3_pairwise_dunn_cliff.csv")))
km = {r["agent"]: r for r in csv.DictReader(open(T / "rq2_T4_kaplan_meier.csv"))}
sens = {r["subset"]: r for r in csv.DictReader(open(T / "rq2_T7_sensitivity.csv"))}
y1 = {r["agent"]: r for r in csv.DictReader(open(T / "rq2_Y1_own_bot_ttc.csv"))}

print("— Summary (medians, statistics module) —")
close("agents in RQ2", 11, len(groups), 0)
close("closed issues in RQ2", 3269, sum(len(g) for g in groups.values()), 0)
for a, g in groups.items():
    close(f"{a}: median TTC (h)", statistics.median(g), summ[a]["median_h"], 1e-6)
    ok = float(summ[a]["median_ci_lo"]) <= float(summ[a]["median_h"]) <= float(summ[a]["median_ci_hi"])
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {a}: exact CI contains the median")

print("\n— Kruskal–Wallis by hand —")
H, n, mean_ranks, allv = kruskal(groups)
main = tests["Kruskal–Wallis, all closed"]
close("H", H, main["H"], 1e-3)
close("ε² = H(n+1)/(n²−1)", H * (n + 1) / (n**2 - 1), main["epsilon2"], 1e-6)

print("\n— Dunn's test (all 55 pairs) + Holm by hand —")
agents = list(groups)
C = n * (n + 1) / 12 - tie_sum(allv) / (12 * (n - 1))
raw = {}
for i, a in enumerate(agents):
    for b in agents[i + 1 :]:
        z = (mean_ranks[a] - mean_ranks[b]) / math.sqrt(C * (1 / len(groups[a]) + 1 / len(groups[b])))
        raw[(a, b)] = 2 * normal_sf(abs(z))
m = len(raw)
holm, running = {}, 0.0
for k, (pair, p) in enumerate(sorted(raw.items(), key=lambda kv: kv[1])):
    running = max(running, min(1.0, (m - k) * p))
    holm[pair] = running
close("number of pairs", 55, m, 0)
close(
    "pairs significant after Holm (α = 0.05)",
    sum(p < 0.05 for p in holm.values()),
    sum(r["significant"] == "True" for r in pairs),
    0,
)
for r in pairs[:5] + [p for p in pairs if p["agent_b"] == "CodeRabbit"][:3]:
    key = (
        (r["agent_a"], r["agent_b"]) if (r["agent_a"], r["agent_b"]) in holm else (r["agent_b"], r["agent_a"])
    )
    close(f"Dunn Holm p {r['agent_a']} vs {r['agent_b']}", holm[key], r["dunn_p_holm"], 1e-6)

print("\n— Cliff's δ by brute force —")
for r in pairs:
    if (r["agent_a"], r["agent_b"]) in [
        ("Claude_Code", "CodeRabbit"),
        ("PR_Agent", "CodeRabbit"),
        ("Sweep", "Devin"),
        ("Claude_Code", "Sweep"),
        ("Kiro", "Jules"),
    ]:
        x, y = np.array(groups[r["agent_a"]]), np.array(groups[r["agent_b"]])
        close(
            f"δ {r['agent_a']} vs {r['agent_b']}",
            np.sign(x[:, None] - y[None, :]).mean(),
            r["cliffs_delta"],
            1e-9,
        )

print("\n— Kaplan–Meier medians by hand (open issues censored) —")
all_rows = [r for r in rows if r["agent_type"] in groups]
for a in ["Claude_Code", "SWE_agent", "Jules", "Sweep", "CodeRabbit", "Amazon_Q"]:
    g = [r for r in all_rows if r["agent_type"] == a]
    med = km_median([float(r["duration_hours"]) for r in g], [r["event_closed"] == "1" for r in g])
    exp = float(km[a]["km_median_h"])
    if math.isinf(med) and math.isinf(exp):
        results.append(True)
        print(f"[PASS] {a}: KM median not reached (both)")
    else:
        close(f"{a}: KM median (h)", med, exp, 1e-6)
    close(
        f"{a}: censored (still open)", sum(r["event_closed"] == "0" for r in g), km[a]["n_censored_open"], 0
    )

print("\n— Sensitivity S2 (human-authored only), Kruskal–Wallis by hand —")
g2 = defaultdict(list)
for r in closed:
    if r["author_is_bot"] != "True":
        g2[r["agent_type"]].append(float(r["ttc_hours"]))
g2 = {a: v for a, v in sorted(g2.items()) if len(v) >= 30}
H2, n2, _, _ = kruskal(g2)
s2 = sens["S2: human-authored only (author_type = User)"]
close("S2 agents", len(g2), s2["agents"], 0)
close("S2 H", H2, s2["H"], 1e-3)
close("S2 ε²", H2 * (n2 + 1) / (n2**2 - 1), s2["epsilon2"], 1e-6)

print("\n— Y1: Mann–Whitney U from ranks and Cliff's δ —")
for a in ["CodeRabbit", "Devin"]:
    own = [float(r["ttc_hours"]) for r in closed if r["agent_type"] == a and r["own_bot_author"] == "True"]
    oth = [float(r["ttc_hours"]) for r in closed if r["agent_type"] == a and r["own_bot_author"] != "True"]
    rk = avg_ranks(own + oth)
    U = sum(rk[: len(own)]) - len(own) * (len(own) + 1) / 2
    close(f"{a}: U", U, y1[a]["mannwhitney_U"], 1e-6)
    close(f"{a}: own-bot median", statistics.median(own), y1[a]["own_bot_median_h"], 1e-6)
    close(f"{a}: other median", statistics.median(oth), y1[a]["other_median_h"], 1e-6)
    close(
        f"{a}: Cliff's δ",
        np.sign(np.array(own)[:, None] - np.array(oth)[None, :]).mean(),
        y1[a]["cliffs_delta"],
        1e-9,
    )

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
