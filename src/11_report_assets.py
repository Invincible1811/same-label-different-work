"""Step 11: generate every number and table used in the report from the result files.

Writes report/generated/numbers.tex (one LaTeX macro per number) and
report/generated/tab_*.tex. The report text uses only these macros, so no number in
the PDF is typed by hand. Run after the analysis scripts:  python src/11_report_assets.py
"""
import re

import numpy as np
import pandas as pd
from scipy.stats import norm

from config import PROCESSED, ROOT, TABLES
from rq3_common import build

OUT = ROOT / "report" / "generated"
OUT.mkdir(parents=True, exist_ok=True)
macros = {}
AGENT_NAME = {
    "CodeRabbit": "CodeRabbit",
    "Claude_Code": "Claude Code",
    "Sweep": "Sweep",
    "Devin": "Devin",
    "SWE_agent": "SWE-agent",
    "Gemini_Code_Assist": "Gemini Code Assist",
    "Amazon_Q": "Amazon Q",
    "Aider": "Aider",
    "Kiro": "Kiro",
    "PR_Agent": "PR-Agent",
    "Jules": "Jules",
    "Copilot": "Copilot",
    "Cursor": "Cursor",
    "All": "All",
}


def fmt(x, d=0):
    """Thousands separator {,} and fixed decimals."""
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    s = f"{x:,.{d}f}"
    return s.replace(",", "{,}").replace("-", "\\ensuremath{-}")  # typographic minus


def sci(p):
    """p-value as LaTeX scientific notation, e.g. 2.0\\times10^{-51}."""
    if p == 0:
        return "<10^{-300}"
    m, e = f"{p:.1e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def put(name, value):
    assert re.fullmatch(r"[A-Za-z]+", name), name  # LaTeX macro names: letters only
    assert name not in macros, f"duplicate macro {name}"
    macros[name] = value


def camel(agent):
    return re.sub(r"[^A-Za-z]", "", AGENT_NAME[agent].title())


# -------------------- data
d = pd.read_csv(PROCESSED / "issues_clean.csv")
summary = pd.read_csv(TABLES / "dataset_summary_before_after.csv").set_index("measure")
log = pd.read_csv(TABLES / "cleaning_log.csv")
put("NRaw", fmt(5000))
put("NIssues", fmt(len(d)))
put("NRepos", fmt(d.repo_name.nunique()))
put("NReposRaw", fmt(int(summary.loc["repositories", "before_cleaning"])))
put("NAgents", fmt(d.agent_type.nunique()))
put("NPR", fmt(int(log.loc[log.step == "1 drop pull requests", "removed"].iloc[0])))
put("NDup", fmt(int(log.loc[log.step == "2 drop re-submitted duplicates", "removed"].iloc[0])))
put("NClosed", fmt(int(d.is_closed.sum())))
put("NOpen", fmt(int((~d.is_closed).sum())))
put("PctOpen", fmt(100 * (~d.is_closed).mean(), 1))
put("NBot", fmt(int(d.author_is_bot.sum())))
put("PctBot", fmt(100 * d.author_is_bot.mean(), 1))
put("NNonLatin", fmt(int((d.lang_group == "non_latin").sum())))
put("NLatinNonEn", fmt(int((d.lang_group == "latin_non_english").sum())))
put("NEnglish", fmt(int((d.lang_group == "english").sum())))
put("NOwnBot", fmt(int(d.own_bot_author.sum())))
put("PctOwnBot", fmt(100 * d.own_bot_author.mean(), 1))
put("NActionsEcho", fmt(int(d.automation_author.sum())))
put("NSelfRef", fmt(int((d.attribution_signal == "self_referential_repo").sum())))
put("NSThree", fmt(int(d.s3_precision.sum())))
put("NPreRelease", fmt(int(d.pre_release.sum())))
put("NRQTwo", fmt(int(d.rq2_eligible.sum())))
put("NInstant", fmt(int((d.ttc_hours < 1 / 60).sum())))
put("NOverYear", fmt(int((d.ttc_hours > 24 * 365).sum())))
put("NLabelsDistinct", fmt(pd.read_csv(TABLES / "label_category_map.csv").label_norm.nunique()))
put("MedianTTCAll", fmt(d.ttc_hours.median(), 1))

# -------------------- RQ1
t1 = pd.read_csv(TABLES / "rq1_T1_agent_profile.csv").set_index("agent")
t3 = pd.read_csv(TABLES / "rq1_T3_association_tests.csv")
fa = pd.read_csv(TABLES / "rq1_FA_own_bot_closure.csv")
fb = pd.read_csv(TABLES / "rq1_FB_labelling_illusion.csv").set_index("agent")
for key, name in [
    ("State", "State"),
    ("Closure reason", "Reason"),
    ("Labelled", "Labelled"),
    ("Language", "Language"),
    ("Attribution", "Attribution"),
]:
    r = t3[t3.test.str.contains(key)].iloc[0]
    put(f"V{name}", fmt(r.cramers_v, 2))
    put(f"Eff{name}", r.effect)
    put(f"Chi{name}", fmt(r.chi2, 1))
put("MaxPctSparse", fmt(t3.pct_cells_expected_lt5.max(), 0))
lab = pd.read_csv(PROCESSED / "labels_clean.csv").merge(d[["issue_id", "agent_type"]], on="issue_id")
put("SweepLabelShare", fmt((lab[lab.agent_type == "Sweep"].label_norm == "sweep").mean() * 100, 1))
tested = t1[t1.tested & (t1.index != "All")]
put("TopRepoMax", fmt(tested.top_repo_share_pct.max(), 1))
put("TopThreeShare", fmt(tested.n.nlargest(3).sum() / t1.loc["All", "n"] * 100, 1))
for a in ["CodeRabbit", "Devin"]:
    g = fa[(fa.group == a) & (fa.min_age_days == 0)].iloc[0]
    g180 = fa[(fa.group == a) & (fa.min_age_days == 180)].iloc[0]
    c = camel(a)
    put(f"OwnClosed{c}", fmt(g.own_bot_closed_pct, 1))
    put(f"OtherClosed{c}", fmt(g.other_closed_pct, 1))
    put(f"OwnCI{c}", f"[{fmt(g.own_bot_lo, 1)}, {fmt(g.own_bot_hi, 1)}]")
    put(f"OtherCI{c}", f"[{fmt(g.other_lo, 1)}, {fmt(g.other_hi, 1)}]")
    put(f"Diff{c}", fmt(g.diff_pp, 1))
    put(f"DiffCI{c}", f"[{fmt(g.diff_lo, 1)}, {fmt(g.diff_hi, 1)}]")
    put(f"Z{c}", fmt(g.z, 2))
    put(f"P{c}", sci(g.p))
    put(f"H{c}", fmt(abs(g.cohens_h), 2))
    put(f"HEff{c}", g.h_effect)
    put(f"AgeOwn{c}", fmt(g.own_bot_median_age_days))
    put(f"AgeOther{c}", fmt(g.other_median_age_days))
    put(f"DiffOld{c}", fmt(g180.diff_pp, 1))
    put(f"NOwn{c}", fmt(int(g.own_bot_n)))
    put(f"NOther{c}", fmt(int(g.other_n)))
pool = fa[fa.group.str.startswith("All")].iloc[0]
put("OwnClosedAll", fmt(pool.own_bot_closed_pct, 1))
put("OtherClosedAll", fmt(pool.other_closed_pct, 1))
put("SweepAnyLabel", fmt(fb.loc["Sweep", "any_label_pct"], 1))
put("SweepAnyLabelCI", f"[{fmt(fb.loc['Sweep', 'any_lo'], 1)}, {fmt(fb.loc['Sweep', 'any_hi'], 1)}]")
put("SweepTriageLabel", fmt(fb.loc["Sweep", "triage_label_pct"], 1))
put("SweepDrop", fmt(fb.loc["Sweep", "drop_pp"], 1))
put("AmazonQDrop", fmt(fb.loc["Amazon_Q", "drop_pp"], 1))
put("JulesDrop", fmt(fb.loc["Jules", "drop_pp"], 1))
put("OtherMaxDrop", fmt(fb.drop(["Sweep", "Amazon_Q", "Jules"]).drop_pp.max(), 1))
for a in ["Devin", "Gemini_Code_Assist", "Claude_Code", "CodeRabbit", "Sweep", "Aider"]:
    put(f"NonEn{camel(a)}", fmt(t1.loc[a, "non_english_pct"], 1))
for a in ["CodeRabbit", "Devin"]:
    put(f"OwnBotShare{camel(a)}", fmt(t1.loc[a, "own_bot_pct"], 1))
put("SelfRefSweagent", fmt(t1.loc["SWE_agent", "self_ref_repo_pct"], 1))
put("EchoAmazonq", fmt(t1.loc["Amazon_Q", "automation_echo_pct"], 1))
put("ClosedMin", fmt(t1.drop("All")[t1.drop("All").tested].closed_pct.min(), 1))
put("ClosedMinAgent", AGENT_NAME[t1.drop("All")[t1.drop("All").tested].closed_pct.idxmin()])
put("ClosedMax", fmt(t1.drop("All")[t1.drop("All").tested].closed_pct.max(), 1))
put("ClosedMaxAgent", AGENT_NAME[t1.drop("All")[t1.drop("All").tested].closed_pct.idxmax()])

# RQ1 table (compact agent profile)
rows = []
for a in [x for x in t1.index if x != "All"] + ["All"]:
    r = t1.loc[a]
    dag = "" if r.tested else "$^\\dagger$"
    rows.append(
        f"{AGENT_NAME[a]}{dag} & {fmt(r.n)} & {fmt(r.repos)} & {fmt(r.closed_pct, 1)} "
        f"& {fmt(r.not_planned_pct_of_closed, 1)} & {fmt(r.labelled_pct, 1)} & {fmt(r.non_english_pct, 1)} "
        f"& {fmt(r.own_bot_pct, 1)} \\\\"
    )
(OUT / "tab_rq1_profile.tex").write_text("\n".join(rows[:-1]) + "\n\\midrule\n" + rows[-1] + "\n")

# -------------------- RQ2
summ = pd.read_csv(TABLES / "rq2_T2_ttc_summary.csv").set_index("agent")
tests = pd.read_csv(TABLES / "rq2_T6_tests.csv")
pairs = pd.read_csv(TABLES / "rq2_T3_pairwise_dunn_cliff.csv")
km = pd.read_csv(TABLES / "rq2_T4_kaplan_meier.csv").set_index("agent")
sens = pd.read_csv(TABLES / "rq2_T7_sensitivity.csv")
rank = pd.read_csv(TABLES / "rq2_T8_ranking_closed_vs_km.csv").set_index("agent")
y1 = pd.read_csv(TABLES / "rq2_Y1_own_bot_ttc.csv").set_index("agent")
dist = pd.read_csv(TABLES / "rq2_T1_distribution_check.csv").set_index("agent").drop("All")
kw = tests[tests.test == "Kruskal–Wallis, all closed"].iloc[0]
put("KWH", fmt(kw.H, 1))
put("KWdf", fmt(int(kw.df)))
put("KWp", sci(kw.p))
put("KWeps", fmt(kw.epsilon2, 3))
put("KWeff", kw.effect)
put("LRchi", fmt(kw.logrank_chi2_all_issues, 1))
put("LRp", sci(kw.logrank_p_all_issues))
put("SkewMin", fmt(dist.skew_ttc.min(), 1))
put("SkewMax", fmt(dist.skew_ttc.max(), 1))
put("NNonNormalLog", fmt(int((dist.shapiro_p_log1p < 0.05).sum())))
put("ShapiroAmazonq", fmt(dist.loc["Amazon_Q", "shapiro_p_log1p"], 3))
put("ShapiroKiro", fmt(dist.loc["Kiro", "shapiro_p_log1p"], 3))
put("MedianTTCTested", fmt(summ.loc["All", "median_h"], 1))
for a in summ.index.drop("All"):
    c = camel(a)
    put(f"Med{c}", fmt(summ.loc[a, "median_h"], 1))
    put(f"MedCI{c}", f"[{fmt(summ.loc[a, 'median_ci_lo'], 1)}, {fmt(summ.loc[a, 'median_ci_hi'], 1)}]")
    km_h = km.loc[a, "km_median_h"]
    put(f"KM{c}", "not reached" if np.isinf(km_h) else fmt(km_h / 24, 1))
    put(f"KMRank{c}", fmt(int(rank.loc[a, "km_rank"])))
    put(f"ClosedRank{c}", fmt(int(rank.loc[a, "closed_only_rank"])))
    put(f"PctOpen{c}", fmt(rank.loc[a, "pct_still_open"], 1))
put("NSig", fmt(int(pairs.significant.sum())))
for e in ["negligible", "small", "medium", "large"]:
    put(f"NPairs{e.title()}", fmt(int((pairs.delta_effect == e).sum())))
big = pairs.loc[pairs.cliffs_delta.abs().idxmax()]
put("DeltaMax", fmt(big.cliffs_delta, 2))
comp = tests[tests.test.str.contains("completed")].iloc[0]
npl = tests[tests.test.str.contains("not_planned")].iloc[0]
put("EpsCompleted", fmt(comp.epsilon2, 3))
put("EpsNotPlanned", fmt(npl.epsilon2, 3))
put("NAgentsNotPlanned", fmt(int(npl.agents)))
put("PNotPlanned", sci(npl.p))
sv = sens.set_index(sens.subset.str[:2])
for k, n in [("S1", "One"), ("S2", "Two"), ("S3", "Three"), ("S4", "Four")]:
    put(f"EpsS{n}", fmt(sv.loc[k, "epsilon2"], 3))
    put(f"RhoS{n}", fmt(sv.loc[k, "spearman_rank_vs_main"], 2))
put("NReposSFour", fmt(int(sv.loc["S4", "n"])))
put("RankRho", fmt(rank.closed_only_rank.corr(rank.km_rank, method="spearman"), 2))
for a in ["CodeRabbit", "Devin"]:
    c = camel(a)
    put(f"YOwn{c}", fmt(y1.loc[a, "own_bot_median_h"], 1))
    put(f"YOther{c}", fmt(y1.loc[a, "other_median_h"], 1))
    put(f"YDelta{c}", fmt(y1.loc[a, "cliffs_delta"], 2))
    put(f"YKMOwn{c}", fmt(y1.loc[a, "km_median_own_bot_h"] / 24, 1))
    put(f"YKMOther{c}", fmt(y1.loc[a, "km_median_other_h"] / 24, 1))
    put(f"YLRp{c}", sci(y1.loc[a, "logrank_p_holm"]))

# RQ2 table
rows = []
for a in rank.index:
    s = summ.loc[a]
    kmv = "n.r." if np.isinf(km.loc[a, "km_median_h"]) else fmt(km.loc[a, "km_median_h"] / 24, 1)
    rows.append(
        f"{AGENT_NAME[a]} & {fmt(s.n_closed)} & {fmt(s.median_h, 1)} & [{fmt(s.median_ci_lo, 1)}, "
        f"{fmt(s.median_ci_hi, 1)}] "
        f"& {fmt(rank.loc[a, 'pct_still_open'], 1)} & {kmv} & "
        f"{int(rank.loc[a, 'closed_only_rank'])} & {int(rank.loc[a, 'km_rank'])} \\\\"
    )
(OUT / "tab_rq2_ttc.tex").write_text("\n".join(rows) + "\n")

# -------------------- RQ3
pre = pd.read_csv(TABLES / "rq3_preprocessing_log.csv").set_index("stage").value
put("NEnglishTested", fmt(int(pre["English issues (D13)"])))
put("NDocsRQThree", fmt(int(pre["documents with >= 10 tokens after cleaning (modelled)"])))
put("MedTokens", fmt(int(pre["median tokens per modelled document"])))
put("NTemplateLines", fmt(int(pre["template lines removed (masked line in >= 10 issues)"])))
put("NVocab", fmt(len(build()[2])))
ms = pd.read_csv(TABLES / "rq3_model_selection.csv")
stage1 = ms[ms.stage == 1].groupby("k").c_v.mean()
put("CvMin", fmt(stage1.min(), 2))
put("CvMax", fmt(stage1.max(), 2))
chosen = pd.read_csv(TABLES / "rq3_selected_model.csv").iloc[0]
single = pd.read_csv(TABLES / "rq3_single_lda_stability.csv").set_index("k")
put("SelK", fmt(int(chosen.k)))
put("SelCv", fmt(chosen.c_v, 3))
put("SelRepro", fmt(single.loc[int(chosen.k), "share_reproduced_J_ge_0.5"] * 100))
put("SingleReproMax", fmt(single["share_reproduced_J_ge_0.5"].max() * 100))
ens_sel = pd.read_csv(TABLES / "rq3_ensemble_selection.csv")
for lvl, name in [("headline", "Headline"), ("fine", "Fine")]:
    put(f"Eps{name}", fmt(ens_sel.loc[ens_sel.level == lvl, "eps"].iloc[0], 2))
t1 = pd.read_csv(TABLES / "rq3_T1_topics.csv")
for lvl, name in [("headline", "H"), ("fine", "F")]:
    tl = t1[t1.level == lvl]
    put(f"NTopics{name}", fmt(len(tl)))
    for r in tl.itertuples():
        num = ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight"][r.topic - 1]
        put(f"Shared{name}{num}", fmt(r.shared_top10_words_seed7))
        put(f"Prev{name}{num}", fmt(r.prevalence_pct, 1))
        put(f"Dom{name}{num}", fmt(r.dominant_in_n_docs))
assoc3 = pd.read_csv(TABLES / "rq3_T3_association.csv")
for lvl, name in [("headline", "H"), ("fine", "F")]:
    a3 = assoc3[assoc3.level == lvl].set_index("analysis")
    for key, tag in [
        ("Main", "Main"),
        ("S2", "STwo"),
        ("S3", "SThree"),
        ("non-reproduced", "Stable"),
        (">= 5 tokens", "Five"),
    ]:
        hit = [i for i in a3.index if key in i]
        if hit:
            put(f"VT{name}{tag}", fmt(a3.loc[hit[0], "cramers_v"], 2))
            put(f"EffT{name}{tag}", a3.loc[hit[0], "effect"])
res_f = pd.read_csv(TABLES / "rq3_T3b_adjusted_residuals_fine.csv", index_col=0)
res_s3 = pd.read_csv(TABLES / "rq3_T3c_adjusted_residuals_S3_fine.csv", index_col=0)
code_topic = int(t1[(t1.level == "fine") & t1.top_words.str.contains("function")].topic.iloc[0])
put("ResCodeRabbitCode", fmt(res_f.loc["CodeRabbit", str(code_topic)], 1))
put("ResSThreeCodeRabbitCode", fmt(res_s3.loc["CodeRabbit", str(code_topic)], 1))
roles = pd.read_csv(TABLES / "rq3_T6_topic_roles.csv")
cr = roles[(roles.agent == "CodeRabbit") & (roles.topic == code_topic)].iloc[0]
put("CodeShareOwnCodeRabbit", fmt(cr.own_bot_mean_share_pct, 1))
put("CodeShareOtherCodeRabbit", fmt(cr.other_mean_share_pct, 1))
cr_test = roles[(roles.agent == "CodeRabbit") & roles.chi2.notna()].iloc[0]
put("VRoleCodeRabbit", fmt(cr_test.cramers_v, 2))
ttc3 = pd.read_csv(TABLES / "rq3_T5_topic_ttc.csv").set_index("topic")
ttc3_t = pd.read_csv(TABLES / "rq3_T5b_topic_ttc_tests.csv")
put("TopicKWeps", fmt(ttc3_t.epsilon2.iloc[0], 3))
put("TopicKWeff", ttc3_t.effect.iloc[0])
put("TopicLRp", sci(ttc3_t.p.iloc[1]))
put("KMDaysCode", fmt(ttc3.loc[code_topic, "km_median_h"] / 24, 1))
work_topic = int(t1[(t1.level == "fine") & t1.top_words.str.contains("workflow")].topic.iloc[0])
put("KMDaysWorkflow", fmt(ttc3.loc[work_topic, "km_median_h"] / 24, 1))
tri = pd.read_csv(TABLES / "rq3_T8_triangulation.csv").iloc[0]
bst = pd.read_csv(TABLES / "rq3_T8b_bertopic_stability.csv")
put("BTAriMin", fmt(bst.ARI_assignments.min(), 2))
put("BTAriMax", fmt(bst.ARI_assignments.max(), 2))
put("BTReproMin", fmt(int(bst["topics_reproduced_J_ge_0.5"].min())))
put("BTV", fmt(tri.cramers_v_agent_x_topic_BERTopic, 2))
put("BTEff", tri.effect_BERTopic)
put("BTVMin", fmt(tri.cramers_v_BERTopic_min_over_seeds, 2))
put("BTVMax", fmt(tri.cramers_v_BERTopic_max_over_seeds, 2))
put("TriARI", fmt(tri.ARI_LDA_vs_BERTopic, 2))
put("TriAMI", fmt(tri.AMI_LDA_vs_BERTopic, 2))
put("TriSame", fmt(tri.share_docs_same_matched_topic * 100))
put("TriWordJ", fmt(tri.word_overlap_mean_jaccard_matched, 2))
put("TriRho", fmt(tri.spearman_agent_topic_residuals_LDA_vs_BERTopic, 2))
bt_topics = pd.read_csv(TABLES / "rq3_T7_bertopic_topics.csv")
bt_res = pd.read_csv(TABLES / "rq3_T8d_bertopic_adjusted_residuals.csv", index_col=0)
bt_code = int(bt_topics[bt_topics.top_words.str.contains("function")].bertopic_topic.iloc[0])
put("BTResCodeRabbitCode", fmt(bt_res.loc["CodeRabbit", str(bt_code)], 1))

# claims behind the RQ3 wording: who writes code-level issues; CodeRabbit's own issues refer to PRs/reviews
dt = pd.read_csv(PROCESSED / "rq3_doc_topics.csv").join(
    d.set_index("issue_id")[["own_bot_author", "title", "body"]], on="issue_id"
)
put(
    "CodeTopicShareCodeRabbit",
    fmt((dt[dt.dominant == code_topic].agent_type == "CodeRabbit").mean() * 100, 1),
)
put("DocShareCodeRabbit", fmt((dt.agent_type == "CodeRabbit").mean() * 100, 1))
cr_own = dt[(dt.agent_type == "CodeRabbit") & dt.own_bot_author]
PR_REF = r"(?i)follow[- ]?up|from (?:pr|pull request)|review comment|code review|pull/\d+|PR #?\d+"
put(
    "CodeRabbitOwnPRRef",
    fmt((cr_own.title.fillna("") + " " + cr_own.body.fillna("")).str.contains(PR_REF).mean() * 100, 1),
)
digest = d[d.title.fillna("").str.contains(r"Latest \d+ Papers|Reading List")]  # automated paper digests
put("NDigest", fmt(len(digest)))
put("NDigestInSThree", fmt(int(digest.s3_precision.sum())))
for a in ["CodeRabbit", "Devin"]:  # all bot-written issues of the agent (not only the RQ3 corpus)
    own_all = d[(d.agent_type == a) & d.own_bot_author]
    text_all = own_all.title.fillna("") + " " + own_all.body.fillna("")
    put(f"OwnPRRefAll{camel(a)}", fmt(text_all.str.contains(PR_REF).mean() * 100, 1))
put("ZCritFine", fmt(norm.ppf(1 - 0.025 / res_f.size), 2))

# RQ3 topic table: both levels, label, stability (shared top-10 words with the other ensemble), prevalence
rows = []
for lvl, tag in [("headline", "H"), ("fine", "T")]:
    for r in t1[t1.level == lvl].itertuples():
        lab = r.label if r.interpreted else "\\emph{not interpreted}"
        words = ", ".join(r.top_words.split(", ")[:6])
        rows.append(
            f"{tag}{r.topic} & {lab} & {r.shared_top10_words_seed7}/10 & {fmt(r.prevalence_pct, 1)} "
            f"& \\texttt{{{words}}} \\\\"
        )
    if lvl == "headline":
        rows.append("\\midrule")
(OUT / "tab_rq3_topics.tex").write_text("\n".join(rows).replace("_", "\\_") + "\n")

# -------------------- metrics promised in the GQM table, reported compactly (M1.2, M1.3, M1.6, M2.2, M3.3)
prof = pd.read_csv(TABLES / "rq1_T1_agent_profile.csv")
prof = prof[prof.tested & (prof.agent != "All")].set_index("agent")
for col, name, d in [("gini_repos", "Gini", 2), ("labels_per_issue_mean", "LabelsMean", 2)]:
    put(f"{name}Min", fmt(prof[col].min(), d))
    put(f"{name}MinAgent", AGENT_NAME[prof[col].idxmin()])
    put(f"{name}Max", fmt(prof[col].max(), d))
    put(f"{name}MaxAgent", AGENT_NAME[prof[col].idxmax()])
lo_agent, hi_agent = prof.closed_pct.idxmin(), prof.closed_pct.idxmax()
put("ClosedMinCI", f"[{fmt(prof.loc[lo_agent, 'closed_lo'], 1)}, {fmt(prof.loc[lo_agent, 'closed_hi'], 1)}]")
put("ClosedMaxCI", f"[{fmt(prof.loc[hi_agent, 'closed_lo'], 1)}, {fmt(prof.loc[hi_agent, 'closed_hi'], 1)}]")
ttc_sum = pd.read_csv(TABLES / "rq2_T2_ttc_summary.csv").set_index("agent")
put("TTCMeanAll", fmt(ttc_sum.loc["All", "mean_h"], 1))
put("TTCMedianAll", fmt(ttc_sum.loc["All", "median_h"], 1))
put("TTCIQRAll", fmt(ttc_sum.loc["All", "iqr_h"], 1))
put("TTCSDAll", fmt(ttc_sum.loc["All", "sd_h"], 1))
shares3 = pd.read_csv(TABLES / "rq3_T2_agent_topic_shares.csv")
cr4 = shares3[
    (shares3.level == "fine") & (shares3.agent == "CodeRabbit") & (shares3.topic == code_topic)
].iloc[0]
put("ShareCodeCodeRabbit", fmt(cr4.mean_share_pct, 1))
put("ShareCodeCodeRabbitCI", f"[{fmt(cr4.ci_lo, 1)}, {fmt(cr4.ci_hi, 1)}]")

# -------------------- manual validation (src/10_validation.py)
val = pd.read_csv(TABLES / "validation_summary.csv").set_index("measure_id")
VALIDATION_MACROS = {
    "V1_correct": "ValCorrect",
    "V1_correct_Aider": "ValCorrectAider",
    "V1_correct_CodeRabbit": "ValCorrectCodeRabbit",
    "V1_correct_Sweep": "ValCorrectSweep",
    "V1_correct_signal_other_signal": "ValCorrectOtherSignal",
    "V1_role_W": "ValRoleW",
    "V1_role_T": "ValRoleT",
    "V1_role_M": "ValRoleM",
    "V1_ownbot_precision": "ValOwnBotPrecision",
    "V1_ownbot_recall": "ValOwnBotRecall",
    "V1_wrong_other_signal": "ValWrongOther",
    "V2_fit": "ValFit",
    "V2_fit_T1": "ValFitOne",
    "V2_fit_T3": "ValFitThree",
    "V2_fit_T4": "ValFitFour",
    "V2_fit_T5": "ValFitFive",
    "V3_nonEN_correct": "ValNonEn",
    "V3_EN_correct": "ValEn",
}
for measure, name in VALIDATION_MACROS.items():
    r = val.loc[measure]
    put(f"{name}Pct", fmt(r.pct, 1))
    put(f"{name}K", fmt(int(r.k)))
    put(f"{name}N", fmt(int(r.n)))
    put(f"{name}CI", f"[{fmt(r.ci_lo, 1)}, {fmt(r.ci_hi, 1)}]")

# -------------------- write
(OUT / "numbers.tex").write_text(
    "% generated by src/11_report_assets.py -- do not edit by hand\n"
    + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items()))
    + "\n"
)
print(f"{len(macros)} macros, tables: {sorted(p.name for p in OUT.glob('tab_*.tex'))}")
