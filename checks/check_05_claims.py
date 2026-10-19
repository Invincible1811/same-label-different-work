"""Claims audit: every number written by hand in the documents (docs/DECISION_LOG.md,
docs/HOW_TO_CHECK.md) and the key results and threats noted during the study are recomputed
from the data here, so no
typed figure can drift from the results.

Run from the repository root:  python checks/check_05_claims.py
Exit code 0 = every claim matches.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T, FD = ROOT / "results/tables", ROOT / "results/figure_data"
raw = pd.read_csv(ROOT / "data/raw/issues.csv")
raw_labels = pd.read_csv(ROOT / "data/raw/issue_labels.csv")
d = pd.read_csv(ROOT / "data/processed/issues_clean.csv")
lab = pd.read_csv(ROOT / "data/processed/labels_clean.csv")
t1 = pd.read_csv(T / "rq1_T1_agent_profile.csv").set_index("agent")
t3 = pd.read_csv(T / "rq1_T3_association_tests.csv")
fa = pd.read_csv(T / "rq1_FA_own_bot_closure.csv")
fb = pd.read_csv(T / "rq1_FB_labelling_illusion.csv").set_index("agent")
results = []


def claim(where, text, written, actual, tol=0.0):
    ok = abs(float(written) - float(actual)) <= tol + 1e-9
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {where}: {text} — written {written}, data {round(float(actual), 4)}")


pct = lambda x: 100 * x
nopr = raw[~raw.is_pull_request.astype(bool)]

# ---------------- raw-data facts (decision log, section 1)
P0 = "Raw data"
claim(P0, "raw issues", 5000, len(raw))
claim(P0, "raw columns", 39, raw.shape[1])
claim(P0, "raw repos", 3477, raw.repo_name.nunique())
for a, n in {
    "CodeRabbit": 1960,
    "Claude_Code": 880,
    "Sweep": 696,
    "Devin": 547,
    "SWE_agent": 310,
    "Gemini_Code_Assist": 257,
    "Amazon_Q": 108,
    "Aider": 68,
    "Kiro": 64,
    "PR_Agent": 55,
    "Jules": 49,
    "Copilot": 3,
    "Cursor": 3,
}.items():
    claim(P0, f"raw {a}", n, (raw.agent_type == a).sum())
claim(P0, "raw closed", 3309, (raw.api_state == "closed").sum())
claim(P0, "raw open", 1691, (raw.api_state == "open").sum())
claim(P0, "state_final ≠ api_state", 1959, (raw.state_final != raw.api_state).sum())
claim(P0, "pull requests", 18, raw.is_pull_request.astype(bool).sum())
claim(P0, "empty bodies (raw)", 129, raw.body.fillna("").str.strip().eq("").sum())
claim(
    P0,
    "CodeRabbit % Bot (raw)",
    85,
    pct((raw[raw.agent_type == "CodeRabbit"].author_type == "Bot").mean()),
    0.5,
)
claim(P0, "Devin % Bot (raw)", 86, pct((raw[raw.agent_type == "Devin"].author_type == "Bot").mean()), 0.5)
for k, v in {"completed": 2888, "not_planned": 372, "duplicate": 34, "reopened": 34}.items():
    claim(P0, f"state_reason {k}", v, (raw.state_reason == k).sum())
claim(P0, "state_reason empty", 1672, raw.state_reason.isna().sum())

# ---------------- data processing table (decision log, section 4)
P2 = "Cleaning D1-D15"
claim(P2, "issues after cleaning", 4952, len(d))
claim(P2, "repos after cleaning", 3466, d.repo_name.nunique())
claim(P2, "closed after cleaning", 3275, d.is_closed.sum())
claim(P2, "RQ2-eligible", 3269, d.rq2_eligible.sum())
claim(P2, "D3 duplicates removed", 30, len(nopr) - len(d))
claim(P2, "D5 instant closes < 1 min", 47, (d.ttc_hours < 1 / 60).sum())
claim(P2, "D5 TTC > 1 year", 119, (d.ttc_hours > 24 * 365).sum())
claim(P2, "D6 open", 1677, (~d.is_closed).sum())
claim(P2, "D6 open %", 33.9, pct((~d.is_closed).mean()), 0.05)
claim(P2, "D8 bot-authored", 2278, d.author_is_bot.sum())
claim(P2, "D8 bot-authored %", 46.0, pct(d.author_is_bot.mean()), 0.05)
claim(P2, "D8 pre-release", 10, d.pre_release.sum())
for c, v in {"feature": 767, "agent_workflow": 664, "bug": 406, "documentation": 157, "other": 2734}.items():
    claim(P2, f"D9 label pairs {c}", v, (lab.category == c).sum())
claim(P2, "D10 empty bodies", 128, d.body_empty.sum())
claim(P2, "D13 non-Latin", 695, (d.lang_group == "non_latin").sum())
claim(P2, "D13 non-Latin %", 14.0, pct((d.lang_group == "non_latin").mean()), 0.05)
claim(P2, "D13 Latin non-English", 48, (d.lang_group == "latin_non_english").sum())
claim(P2, "D14 renamed", 533, d.repo_renamed.sum())
claim(P2, "D14 transferred", 13, d.issue_transferred.sum())
for k, v in {
    "agent_own_bot": 2091,
    "automation_echo": 134,
    "self_referential_repo": 189,
    "other_signal": 2538,
}.items():
    claim(P2, f"D15 {k}", v, (d.attribution_signal == k).sum())
claim(P2, "D15 S3 subset", 4591, d.s3_precision.sum())
claim(P2, "D15 S3 RQ2-eligible", 3049, (d.s3_precision & d.rq2_eligible).sum())

# ---------------- RQ1 results
R = "RQ1 results"
claim(R, "max % cells expected < 5", 36, t3.pct_cells_expected_lt5.max(), 0.5)
for name, v in {
    "State": 0.21,
    "Closure reason": 0.11,
    "Labelled": 0.33,
    "Language": 0.21,
    "Attribution": 0.54,
}.items():
    claim(R, f"V {name}", v, t3[t3.test.str.contains(name)].cramers_v.iloc[0], 0.005)
claim(R, "all Holm-adjusted p < 0.001 (1 = true)", 1, int(t3.p_holm.max() < 0.001))
row = lambda g, age=0: fa[(fa.group == g) & (fa.min_age_days == age)].iloc[0]
cr, dv, pool = row("CodeRabbit"), row("Devin"), fa[fa.group.str.startswith("All")].iloc[0]
for txt, w, a in [
    ("CR own-bot closed %", 56.9, cr.own_bot_closed_pct),
    ("CR own CI lo", 54.5, cr.own_bot_lo),
    ("CR own CI hi", 59.3, cr.own_bot_hi),
    ("CR other closed %", 75.3, cr.other_closed_pct),
    ("CR other CI lo", 70.1, cr.other_lo),
    ("CR other CI hi", 79.8, cr.other_hi),
    ("CR diff pp", -18.3, cr.diff_pp),
    ("CR diff lo", -23.5, cr.diff_lo),
    ("CR diff hi", -12.6, cr.diff_hi),
    ("Devin own-bot closed %", 52.8, dv.own_bot_closed_pct),
    ("Devin own CI lo", 48.2, dv.own_bot_lo),
    ("Devin own CI hi", 57.5, dv.own_bot_hi),
    ("Devin other closed %", 76.8, dv.other_closed_pct),
    ("Devin other CI lo", 67.5, dv.other_lo),
    ("Devin other CI hi", 84.0, dv.other_hi),
    ("Devin diff pp", -23.9, dv.diff_pp),
    ("Devin diff lo", -32.5, dv.diff_lo),
    ("Devin diff hi", -13.6, dv.diff_hi),
    ("pooled own %", 56.0, pool.own_bot_closed_pct),
    ("pooled own lo", 53.9, pool.own_bot_lo),
    ("pooled own hi", 58.2, pool.own_bot_hi),
    ("pooled other %", 73.5, pool.other_closed_pct),
    ("pooled other lo", 71.9, pool.other_lo),
    ("pooled other hi", 75.1, pool.other_hi),
    ("CR ≥180d diff", -18.1, row("CodeRabbit", 180).diff_pp),
    ("Devin ≥180d diff", -24.8, row("Devin", 180).diff_pp),
]:
    claim(R, txt, w, a, 0.05)
claim(R, "CR z", -5.95, cr.z, 0.005)
claim(R, "Devin z", -4.35, dv.z, 0.005)
claim(R, "CR p (×1e-9)", 2.7, cr.p * 1e9, 0.05)
claim(R, "Devin p (×1e-5)", 1.4, dv.p * 1e5, 0.05)
claim(R, "CR Cohen's h", 0.39, abs(cr.cohens_h), 0.005)
claim(R, "Devin Cohen's h", 0.51, abs(dv.cohens_h), 0.005)
claim(R, "CR own-bot median age (days)", 403, cr.own_bot_median_age_days)
claim(R, "CR other median age (days)", 233, cr.other_median_age_days)
claim(R, "Devin own-bot median age (days)", 503, dv.own_bot_median_age_days)
claim(R, "Devin other median age (days)", 348, dv.other_median_age_days)
claim(R, "Sweep any label %", 82.0, fb.loc["Sweep", "any_label_pct"], 0.05)
claim(R, "Sweep any label CI lo", 79.0, fb.loc["Sweep", "any_lo"], 0.05)
claim(R, "Sweep any label CI hi", 84.7, fb.loc["Sweep", "any_hi"], 0.05)
claim(R, "Sweep triage label %", 19.1, fb.loc["Sweep", "triage_label_pct"], 0.05)
claim(R, "Sweep drop pp", 62.9, fb.loc["Sweep", "drop_pp"], 0.05)
claim(R, "Sweep rank (any label, all 13 agents)", 2, fb.loc["Sweep", "rank_any"])
claim(
    R,
    "Sweep rank (any label, tested agents)",
    1,
    fb[fb.tested].any_label_pct.rank(ascending=False).loc["Sweep"],
)
claim(R, "Sweep rank (triage label, all 13)", 13, fb.loc["Sweep", "rank_triage"])
claim(R, "Amazon Q drop pp", 9.3, fb.loc["Amazon_Q", "drop_pp"], 0.05)
claim(R, "Jules drop pp", 10.2, fb.loc["Jules", "drop_pp"], 0.05)
others = fb.drop(["Sweep", "Amazon_Q", "Jules"])
claim(R, "max drop among other agents (pp)", 1.5, others.drop_pp.max(), 0.05)

# ---------------- facts behind the threats to validity
P5 = "Threats"
tested = t1[t1.tested & (t1.index != "All")]
claim(
    P5,
    "largest repo share among tested agents ≤ 13% (1 = true)",
    1,
    int(tested.top_repo_share_pct.max() <= 13),
)
claim(
    P5,
    "renovate[bot] issues labelled CodeRabbit",
    3,
    ((d.author_login == "renovate[bot]") & (d.agent_type == "CodeRabbit")).sum(),
)
for a, w in {"CodeRabbit": 85, "Devin": 82}.items():
    claim(P5, f"{a} own-bot %", w, t1.loc[a, "own_bot_pct"], 0.5)
for a in ["Claude_Code", "Sweep", "Jules"]:
    claim(
        P5,
        f"{a} other-signal % > 90",
        1,
        int(pct((d[d.agent_type == a].attribution_signal == "other_signal").mean()) > 90),
    )
claim(P5, "SWE-agent self-ref %", 21, t1.loc["SWE_agent", "self_ref_repo_pct"], 0.5)
claim(P5, "PR-Agent self-ref %", 20, t1.loc["PR_Agent", "self_ref_repo_pct"], 0.5)
claim(P5, "Amazon Q actions-echo %", 34, t1.loc["Amazon_Q", "automation_echo_pct"], 0.5)
closed_n = d[d.is_closed].agent_type.value_counts()
claim(P5, "agents dropped by ≥ 50 closed rule", 5, ((closed_n >= 30) & (closed_n < 50)).sum())

# ---------------- numbers in HOW_TO_CHECK and the decision log
E = "Guide and log"
med = d[d.is_closed].groupby("agent_type").ttc_hours.median()
claim(E, "Claude Code median TTC (h)", 11.1, med["Claude_Code"], 0.05)
claim(E, "CodeRabbit median TTC (h)", 180.3, med["CodeRabbit"], 0.05)
claim(E, "median TTC all (h)", 51.13, d.ttc_hours.median(), 0.005)
claim(E, "max TTC (years)", 6.6, d.ttc_hours.max() / 8760, 0.05)
claim(E, "English documents", 4209, (d.lang_group == "english").sum())
claim(E, "English median words", 145, d[d.lang_group == "english"].text.str.split().str.len().median())
claim(E, "distinct labels (retained)", 1345, lab.label_norm.nunique())
for a, w in {
    "Devin": 33.8,
    "Gemini_Code_Assist": 33.1,
    "Claude_Code": 20.3,
    "CodeRabbit": 12.0,
    "Aider": 4.4,
    "Sweep": 2.2,
}.items():
    claim(E, f"{a} non-English %", w, t1.loc[a, "non_english_pct"], 0.05)
claim(E, "Amazon Q raw non-PR rows", 108, (nopr.agent_type == "Amazon_Q").sum())
claim(E, "Amazon Q raw closed", 39, ((nopr.agent_type == "Amazon_Q") & (nopr.api_state == "closed")).sum())
claim(E, "Amazon Q closed % (clean)", 36.4, t1.loc["Amazon_Q", "closed_pct"], 0.05)
claim(
    E,
    "issues carrying label 'sweep' (raw)",
    454,
    raw_labels[raw_labels.label.str.lower() == "sweep"].issue_id.nunique(),
)
claim(
    E,
    "GH Archive latest event = 2026-04-15",
    20260415,
    int(pd.to_datetime(raw.created_at, utc=True).max().strftime("%Y%m%d")),
)
claim(
    E,
    "issues with GH Archive event before 2024",
    54,
    (pd.to_datetime(raw.created_at, utc=True) < "2024-01-01").sum(),
)

dt = pd.read_csv(ROOT / "data/processed/rq3_doc_topics.csv")
sh = pd.read_csv(T / "rq3_T2_agent_topic_shares.csv")
claim(E, "RQ3 documents dominated by fine topic 4", 1140, (dt.dominant == 4).sum())
claim(E, "of which CodeRabbit", 771, ((dt.dominant == 4) & (dt.agent_type == "CodeRabbit")).sum())
claim(
    E,
    "CodeRabbit mean share of fine topic 4 (%)",
    41.7,
    sh[(sh.level == "fine") & (sh.agent == "CodeRabbit") & (sh.topic == 4)].mean_share_pct.iloc[0],
    0.05,
)

# ---------------- figures show what their data says
F = "Figures"
f1 = pd.read_csv(FD / "rq1_F1_volume_state.csv", index_col=0)
re = pd.crosstab(d.agent_type, d.closure_reason)
claim(
    F,
    "F1 counts = recomputed crosstab (max abs diff)",
    0,
    (f1[re.columns] - re.reindex(f1.index)).abs().values.max(),
)
claim(F, "F1 n = agent totals", 0, (f1.n - d.agent_type.value_counts().reindex(f1.index)).abs().max())
f2 = pd.read_csv(FD / "rq1_F2_labels_heatmap.csv", index_col=0)
claim(
    F,
    "F2 'Any label' = T1 labelled % (max abs diff)",
    0,
    (f2["Any label"] - t1.loc[f2.index, "labelled_pct"]).abs().max(),
    0.01,
)
f3 = pd.read_csv(FD / "rq1_F3_attribution_language.csv")
claim(F, "F3 issues sum to dataset (each panel)", 2 * len(d), f3.issues.sum())
x3 = pd.read_csv(FD / "rq1_X3_monthly_volume.csv", index_col=0)
created = pd.to_datetime(d.api_created_at, utc=True)
claim(
    F,
    "X3 plotted issues = issues created Jan 2024–Apr 2026",
    ((created >= "2024-01-01") & (created < "2026-05-01")).sum(),
    x3.values.sum(),
)
x2 = pd.read_csv(FD / "rq1_X2_residuals.csv")
claim(F, "X2 residual cells (3 panels × 11 agents)", 11 * (3 + 3 + 4), len(x2))


# ---------------- RQ2 numbers written in docs/DECISION_LOG.md
Q = "RQ2 results"
r2 = lambda f: pd.read_csv(T / f)
dist, summ2, pairs2, km2 = (
    r2("rq2_T1_distribution_check.csv").set_index("agent"),
    r2("rq2_T2_ttc_summary.csv").set_index("agent"),
    r2("rq2_T3_pairwise_dunn_cliff.csv"),
    r2("rq2_T4_kaplan_meier.csv").set_index("agent"),
)
tests2, sens2, rank2, y1 = (
    r2("rq2_T6_tests.csv"),
    r2("rq2_T7_sensitivity.csv"),
    r2("rq2_T8_ranking_closed_vs_km.csv").set_index("agent"),
    r2("rq2_Y1_own_bot_ttc.csv").set_index("agent"),
)
ag = dist.drop("All")
claim(Q, "closed issues in RQ2", 3269, summ2.loc["All", "n_closed"])
claim(Q, "KM issues (11 agents)", 4946, km2.loc["All", "n_all"])
claim(
    Q, "raw TTC non-normal for all 11 (max Shapiro p < 1e-6; 1 = true)", 1, int(ag.shapiro_p_ttc.max() < 1e-6)
)
claim(Q, "min raw skewness", 2.0, ag.skew_ttc.min(), 0.05)
claim(Q, "max raw skewness", 13.0, ag.skew_ttc.max(), 0.05)
claim(Q, "log TTC non-normal agents", 9, (ag.shapiro_p_log1p < 0.05).sum())
claim(Q, "Amazon Q log Shapiro p", 0.055, ag.loc["Amazon_Q", "shapiro_p_log1p"], 0.0005)
claim(Q, "Kiro log Shapiro p", 0.067, ag.loc["Kiro", "shapiro_p_log1p"], 0.0005)
claim(Q, "Lilliefors floor 0.001 (min p)", 0.001, ag.lilliefors_p_log1p.min(), 1e-9)
kw_all = tests2[tests2.test == "Kruskal–Wallis, all closed"].iloc[0]
claim(Q, "KW H", 266.4, kw_all.H, 0.05)
claim(Q, "KW df", 10, kw_all.df)
claim(Q, "KW p (×1e-51)", 2.0, kw_all.p * 1e51, 0.05)
claim(Q, "KW ε²", 0.082, kw_all.epsilon2, 0.0005)
for a, (m, lo, hi) in {"Claude_Code": (11.1, 7.4, 15.7), "CodeRabbit": (180.3, 152.2, 252.2)}.items():
    claim(Q, f"{a} median h", m, summ2.loc[a, "median_h"], 0.05)
    claim(Q, f"{a} median CI lo", lo, summ2.loc[a, "median_ci_lo"], 0.05)
    claim(Q, f"{a} median CI hi", hi, summ2.loc[a, "median_ci_hi"], 0.05)
claim(Q, "all-agent median h", 50.6, summ2.loc["All", "median_h"], 0.05)
claim(Q, "significant pairs", 13, pairs2.significant.sum())
for k, v in {"negligible": 28, "small": 22, "medium": 5, "large": 0}.items():
    claim(Q, f"pairs with {k} δ", v, (pairs2.delta_effect == k).sum())
med = pairs2[pairs2.delta_effect == "medium"]
claim(
    Q,
    "medium pairs involving Claude Code",
    4,
    ((med.agent_a == "Claude_Code") | (med.agent_b == "Claude_Code")).sum(),
)
claim(
    Q,
    "largest |δ| = Claude vs CodeRabbit",
    -0.44,
    pairs2.loc[pairs2.cliffs_delta.abs().idxmax(), "cliffs_delta"],
    0.005,
)
claim(
    Q,
    "largest |δ| pair is Claude–CodeRabbit (1 = true)",
    1,
    int(
        set(pairs2.loc[pairs2.cliffs_delta.abs().idxmax(), ["agent_a", "agent_b"]])
        == {"Claude_Code", "CodeRabbit"}
    ),
)
comp = tests2[tests2.test.str.contains("completed")].iloc[0]
np_ = tests2[tests2.test.str.contains("not_planned")].iloc[0]
claim(Q, "completed-only ε²", 0.097, comp.epsilon2, 0.0005)
claim(Q, "not-planned agents", 6, np_.agents)
claim(Q, "not-planned ε²", 0.062, np_.epsilon2, 0.0005)
claim(Q, "not-planned p (×1e-4)", 6.6, np_.p * 1e4, 0.05)
sv = sens2.set_index(sens2.subset.str[:2])
for k, v in {"S1": 0.083, "S3": 0.092, "S4": 0.082, "S2": 0.045}.items():
    claim(Q, f"{k} ε²", v, sv.loc[k, "epsilon2"], 0.0005)
claim(Q, "min rank ρ across S1–S4 ≥ 0.90 (1 = true)", 1, int(sens2.spearman_rank_vs_main.min() >= 0.90))
claim(Q, "log-rank χ²", 394.0, kw_all.logrank_chi2_all_issues, 0.05)
claim(Q, "log-rank p (×1e-78)", 1.8, kw_all.logrank_p_all_issues * 1e78, 0.05)
for a, v in {
    "Claude_Code": 1.1,
    "SWE_agent": 4.8,
    "Gemini_Code_Assist": 4.8,
    "CodeRabbit": 138.9,
    "Devin": 181.2,
}.items():
    claim(Q, f"{a} KM median days", v, km2.loc[a, "km_median_h"] / 24, 0.05)
claim(Q, "Amazon Q KM median not reached (1 = true)", 1, int(np.isinf(km2.loc["Amazon_Q", "km_median_h"])))
claim(Q, "Amazon Q % still open", 63.6, rank2.loc["Amazon_Q", "pct_still_open"], 0.05)
claim(
    Q,
    "ranking ρ closed-only vs KM",
    0.22,
    rank2.closed_only_rank.corr(rank2.km_rank, method="spearman"),
    0.005,
)
for a, (r1, rk) in {"PR_Agent": (1, 9), "Amazon_Q": (6, 11), "Jules": (10, 4)}.items():
    claim(Q, f"{a} closed-only rank", r1, rank2.loc[a, "closed_only_rank"])
    claim(Q, f"{a} KM rank", rk, rank2.loc[a, "km_rank"])
claim(Q, "PR-Agent % still open", 45.5, rank2.loc["PR_Agent", "pct_still_open"], 0.05)
for a, (o, t, dl, ko, kt) in {
    "CodeRabbit": (261.3, 88.6, 0.19, 202.8, 11.1),
    "Devin": (54.0, 361.6, -0.29, 189.7, 58.1),
}.items():
    claim(Q, f"Y1 {a} own-bot median h", o, y1.loc[a, "own_bot_median_h"], 0.05)
    claim(Q, f"Y1 {a} other median h", t, y1.loc[a, "other_median_h"], 0.05)
    claim(Q, f"Y1 {a} δ", dl, y1.loc[a, "cliffs_delta"], 0.005)
    claim(Q, f"Y1 {a} KM own-bot days", ko, y1.loc[a, "km_median_own_bot_h"] / 24, 0.05)
    claim(Q, f"Y1 {a} KM other days", kt, y1.loc[a, "km_median_other_h"] / 24, 0.05)
claim(Q, "Y1 CodeRabbit log-rank Holm p (×1e-17)", 5.2, y1.loc["CodeRabbit", "logrank_p_holm"] * 1e17, 0.05)
claim(Q, "Y1 Devin log-rank Holm p (×1e-4)", 5.1, y1.loc["Devin", "logrank_p_holm"] * 1e4, 0.05)


# ---------------- RQ2 statements (qualitative wording, encoded as true/false)
S = "RQ2 statement"
yes = lambda b: int(bool(b))
claim(
    S,
    "log TTC not rejected exactly for Amazon Q and Kiro",
    1,
    yes(set(ag[ag.shapiro_p_log1p >= 0.05].index) == {"Amazon_Q", "Kiro"}),
)
claim(
    S,
    "Amazon Q n = 39, Kiro n = 46 (two of the smaller groups, not the smallest)",
    1,
    yes(
        summ2.loc["Amazon_Q", "n_closed"] == 39
        and summ2.loc["Kiro", "n_closed"] == 46
        and summ2.drop("All").n_closed.min() == 30
    ),
)
claim(S, "lowest closed-only median = PR-Agent", 1, yes(summ2.drop("All").median_h.idxmin() == "PR_Agent"))
claim(
    S, "highest closed-only median = CodeRabbit", 1, yes(summ2.drop("All").median_h.idxmax() == "CodeRabbit")
)
claim(
    S,
    "PR-Agent median 9.5 h [1.1, 50.1]",
    1,
    yes(
        abs(summ2.loc["PR_Agent", "median_h"] - 9.5) < 0.05
        and abs(summ2.loc["PR_Agent", "median_ci_lo"] - 1.1) < 0.05
        and abs(summ2.loc["PR_Agent", "median_ci_hi"] - 50.1) < 0.05
    ),
)
claim(
    S,
    "51.13 h = median of all 3,275 closed issues",
    1,
    yes(abs(d[d.is_closed].ttc_hours.median() - 51.13) < 0.005 and d.is_closed.sum() == 3275),
)
claim(
    S,
    "Claude Code 2nd closed-only, 1st Kaplan–Meier",
    1,
    yes(rank2.loc["Claude_Code", "closed_only_rank"] == 2 and rank2.loc["Claude_Code", "km_rank"] == 1),
)
claim(S, "slowest closed-only = CodeRabbit", 1, yes(rank2.closed_only_rank.idxmax() == "CodeRabbit"))
claim(
    S,
    "3 slowest by Kaplan–Meier = Amazon Q, Devin, PR-Agent",
    1,
    yes(set(rank2.km_rank.nlargest(3).index) == {"Amazon_Q", "Devin", "PR_Agent"}),
)
claim(S, "50 of 55 pairs negligible or small", 50, pairs2.delta_effect.isin(["negligible", "small"]).sum())
claim(
    S,
    "S2 drops effect to small (0.082 → 0.045)",
    1,
    yes(sv.loc["S2", "effect"] == "small" and sv.loc["S2", "epsilon2"] < kw_all.epsilon2),
)
claim(S, "S1, S3, S4 stay medium", 1, yes(all(sv.loc[k, "effect"] == "medium" for k in ["S1", "S3", "S4"])))
claim(
    S,
    "bot-written issues slower by Kaplan–Meier for both CodeRabbit and Devin (Holm p < 0.001)",
    1,
    yes(all(y1.km_median_own_bot_h > y1.km_median_other_h) and all(y1.logrank_p_holm < 0.001)),
)
claim(
    S,
    "closed bot-written issues slower for CodeRabbit, faster for Devin",
    1,
    yes(
        y1.loc["CodeRabbit", "own_bot_median_h"] > y1.loc["CodeRabbit", "other_median_h"]
        and y1.loc["Devin", "own_bot_median_h"] < y1.loc["Devin", "other_median_h"]
    ),
)


# ---------------- RQ1 and raw-data statements (qualitative wording, encoded as true/false)
S1_ = "RQ1 statement"
claim(
    S1_,
    "own-bot issues are older than other issues (both agents)",
    1,
    yes(
        cr.own_bot_median_age_days > cr.other_median_age_days
        and dv.own_bot_median_age_days > dv.other_median_age_days
    ),
)
claim(
    S1_,
    "≥180-day restriction leaves the gap essentially unchanged (within 1 pp)",
    1,
    yes(
        abs(row("CodeRabbit", 180).diff_pp - cr.diff_pp) < 1
        and abs(row("Devin", 180).diff_pp - dv.diff_pp) < 1
    ),
)
claim(S1_, "own-bot closure lower within both agents", 1, yes(cr.diff_pp < 0 and dv.diff_pp < 0))
top_sweep = (
    pd.read_csv(T / "rq1_T2_top_labels.csv")
    .query("agent_type == 'Sweep'")
    .sort_values("n_issues", ascending=False)
    .iloc[0]
)
claim(
    S1_, "Sweep's most frequent label is its trigger label 'sweep'", 1, yes(top_sweep.label_norm == "sweep")
)
claim(S1_, "Sweep is the most-labelled tested agent", 1, yes(fb[fb.tested].any_label_pct.idxmax() == "Sweep"))
for a in ["Claude_Code", "Sweep", "Gemini_Code_Assist"]:
    claim(
        "Raw-data statement",
        f"{a} mostly User accounts (raw)",
        1,
        yes((raw[raw.agent_type == a].author_type == "User").mean() > 0.5),
    )


# ---------------- RQ2 figures show what their data says
G = "Figures RQ2"
r1 = pd.read_csv(FD / "rq2_R1_ttc_boxplots.csv")
claim(G, "R1 rows = closed issues of 11 agents", 3269, len(r1))
claim(
    G,
    "R1 per-agent counts = summary n (max abs diff)",
    0,
    (r1.agent.value_counts() - summ2.drop("All").n_closed).abs().max(),
)
claim(
    G,
    "R1 per-agent medians = summary medians (max abs diff)",
    0,
    (r1.groupby("agent").ttc_hours.median() - summ2.drop("All").median_h).abs().max(),
    1e-3,
)
r2m = pd.read_csv(FD / "rq2_R2_cliffs_delta.csv", index_col=0)
diffs = [
    abs(r2m.loc[r.agent_a, r.agent_b] - r.cliffs_delta) + abs(r2m.loc[r.agent_b, r.agent_a] + r.cliffs_delta)
    for r in pairs2.itertuples()
]
claim(G, "R2 every cell = ±Cliff's δ of the pairwise table (max abs diff)", 0, max(diffs), 1e-3)
r3 = pd.read_csv(FD / "rq2_R3_kaplan_meier.csv")
for a in ["Claude_Code", "CodeRabbit", "Devin", "SWE_agent"]:
    cur = r3[r3.agent == a].sort_values("hours")
    first = cur.loc[cur.pct_still_open <= 50 + 1e-9, "hours"].min()
    claim(
        G,
        f"R3 {a} curve crosses 50% at the reported KM median",
        0,
        abs(first - km2.loc[a, "km_median_h"]),
        1e-2,
    )
claim(
    G,
    "R3 Amazon Q curve never reaches 50% (1 = true)",
    1,
    yes(r3[r3.agent == "Amazon_Q"].pct_still_open.min() > 50),
)
r4 = pd.read_csv(FD / "rq2_R4_qq_plots.csv")
claim(G, "R4 points per agent = closed issues (11 agents + 'All')", 2 * 3269, len(r4))

print(f"\n{sum(results)}/{len(results)} claims verified")
sys.exit(0 if all(results) else 1)
