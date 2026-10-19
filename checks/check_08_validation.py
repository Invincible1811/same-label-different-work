"""Independent check of the manual-validation scores (src/10_validation.py).

Re-computes every measure in results/tables/validation_summary.csv from the author's answers
(docs/validation/VALIDATION_FILLED_BY_AUTHOR.xlsx) and our classifications
(results/tables/validation_*_key.csv), with a hand-written Wilson interval, and compares.

With an argument, it instead scores another rater's copy of the workbook (for example the
professor's) and prints the scores next to the author's, without comparing:
    python checks/check_08_validation.py <other rater's workbook.xlsx>

Run from the repository root:  python checks/check_08_validation.py
"""
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results/tables"
WORKBOOK = (
    Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/validation/VALIDATION_FILLED_BY_AUTHOR.xlsx"
)
INV, ROLE, FIT, LANG = (
    "agent_involved (Y/N/U)",
    "role (W/T/M/O)",
    "fits_label (Y/N)",
    "language (EN/NOT_EN/MIXED)",
)


def wilson(k, n, z=1.959964):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * p, 100 * (centre - half), 100 * (centre + half)


def tab(name, column):
    df = pd.read_excel(WORKBOOK, sheet_name=name, dtype=str).fillna("")
    df = df.apply(lambda c: c.str.strip())
    df[column] = df[column].str.upper()
    return df[df[column] != ""]


mine = {}  # measure_id -> (k, n)
v1 = tab("V1_attribution", INV)
if len(v1):
    key = pd.read_csv(T / "validation_V1_key.csv")
    v1 = v1.astype({"item": int}).merge(key, on="item")
    v1[ROLE] = v1[ROLE].str.upper()
    yes, no = v1[INV] == "Y", v1[INV] == "N"
    mine["V1_correct"] = (yes.sum(), (yes | no).sum())
    mine["V1_undecidable"] = ((v1[INV] == "U").sum(), len(v1))
    for agent in sorted(v1.agent_type.unique()):
        a = v1.agent_type == agent
        mine[f"V1_correct_{agent}"] = ((a & yes).sum(), (a & (yes | no)).sum())
    for signal in sorted(v1.attribution_signal.unique()):
        s = v1.attribution_signal == signal
        mine[f"V1_correct_signal_{signal}"] = ((s & yes).sum(), (s & (yes | no)).sum())
    for code in "WTMO":
        mine[f"V1_role_{code}"] = ((yes & (v1[ROLE] == code)).sum(), yes.sum())
    flagged = v1.own_bot_author & (v1[INV] != "U")
    mine["V1_ownbot_precision"] = ((flagged & (v1[ROLE] == "W")).sum(), flagged.sum())
    wrote = yes & (v1[ROLE] == "W")
    mine["V1_ownbot_recall"] = ((wrote & v1.own_bot_author).sum(), wrote.sum())
    mine["V1_wrong_other_signal"] = ((no & (v1.attribution_signal == "other_signal")).sum(), no.sum())

v2 = tab("V2_topic_labels", FIT)
v2 = v2[v2.proposed_label != "Topic 2"]
if len(v2):
    mine["V2_fit"] = ((v2[FIT] == "Y").sum(), len(v2))
    for topic in sorted(v2.topic.unique()):
        g = v2[v2.topic == topic]
        mine[f"V2_fit_T{topic}"] = ((g[FIT] == "Y").sum(), len(g))

v3 = tab("V3_language", LANG)
if len(v3):
    v3 = v3.astype({"item": int}).merge(pd.read_csv(T / "validation_V3_key.csv"), on="item")
    non, en = v3[v3.lang_group != "english"], v3[v3.lang_group == "english"]
    mine["V3_nonEN_correct"] = ((non[LANG] != "EN").sum(), len(non))
    mine["V3_EN_correct"] = ((en[LANG] == "EN").sum(), len(en))

if len(sys.argv) > 1:  # another rater: report only
    for m, (k, n) in mine.items():
        print(f"{m:42s} {k:3d}/{n:<3d} {wilson(k, n)[0]:5.1f}%")
    sys.exit(0)

ref = pd.read_csv(T / "validation_summary.csv").set_index("measure_id")
results = [set(mine) == set(ref.index)]
print(
    f"[{'PASS' if results[0] else 'FAIL'}] same set of measures ({len(mine)} recomputed, {len(ref)} in table)"
)
for m, (k, n) in mine.items():
    if m not in ref.index:
        continue
    r = ref.loc[m]
    pct, lo, hi = wilson(k, n)
    ok = (k, n) == (r.k, r.n) and max(abs(pct - r.pct), abs(lo - r.ci_lo), abs(hi - r.ci_hi)) < 1e-3
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {m}: {k}/{n} = {pct:.1f}% [{lo:.1f}, {hi:.1f}]")
print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
