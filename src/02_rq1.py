"""Step 2: RQ1 — how do issues attributed to different AI coding agents differ in
volume, current state, and labelling practice?  (metrics M1.1–M1.10 + extras X1–X4)

Reads data/processed/issues_clean.csv and labels_clean.csv (run 01_clean.py first).
Writes tables to results/tables/rq1_*.csv (+ .tex) and figures to results/figures/rq1_*.{pdf,png}.

Run:  python src/02_rq1.py
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

import plotstyle as ps
from stats_utils import fmt_ci, gini, permutation_chi2, v_label, wilson
from config import FIGURE_DATA, FIGURES, MIN_CLOSED_RQ2, PROCESSED, RELEASE_DATES, TABLES

ps.apply()


# -------------------- data
d = pd.read_csv(PROCESSED / "issues_clean.csv", parse_dates=["api_created_at"])
labels = pd.read_csv(PROCESSED / "labels_clean.csv")
order = d.agent_type.value_counts().index.tolist()  # by volume, descending
closed_counts = d[d.is_closed].agent_type.value_counts()
tested = [a for a in order if closed_counts.get(a, 0) >= MIN_CLOSED_RQ2]  # D7
mark = lambda a: a if a in tested else f"{a}†"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

# -------------------- T1 agent profile
rows = []
for a in order + ["All"]:
    g = d if a == "All" else d[d.agent_type == a]
    n = len(g)
    closed = g[g.is_closed]
    repo_counts = g.repo_name.value_counts()
    cl = wilson(int(g.is_closed.sum()), n)
    lab = wilson(int((g.n_labels > 0).sum()), n)
    ne = wilson(int((g.lang_group != "english").sum()), n)
    ob = wilson(int(g.own_bot_author.sum()), n)
    nc = max(len(closed), 1)
    rows.append(
        {
            "agent": a,
            "tested": a in tested or a == "All",
            "n": n,
            "pct_dataset": 100 * n / len(d),
            "repos": g.repo_name.nunique(),
            "issues_per_repo_median": repo_counts.median(),
            "issues_per_repo_sd": repo_counts.std(ddof=1),
            "top_repo_share_pct": 100 * repo_counts.iloc[0] / n,
            "gini_repos": gini(repo_counts.values),
            "closed_pct": cl[0],
            "closed_lo": cl[1],
            "closed_hi": cl[2],
            "completed_pct_of_closed": 100 * (closed.closure_reason == "completed").sum() / nc,
            "not_planned_pct_of_closed": 100 * (closed.closure_reason == "not_planned").sum() / nc,
            "duplicate_pct_of_closed": 100 * (closed.closure_reason == "duplicate").sum() / nc,
            "labelled_pct": lab[0],
            "labelled_lo": lab[1],
            "labelled_hi": lab[2],
            "labels_per_issue_median": g.n_labels.median(),
            "labels_per_issue_mean": g.n_labels.mean(),
            "labels_per_issue_sd": g.n_labels.std(),
            "non_english_pct": ne[0],
            "non_english_lo": ne[1],
            "non_english_hi": ne[2],
            "own_bot_pct": ob[0],
            "own_bot_lo": ob[1],
            "own_bot_hi": ob[2],
            "automation_echo_pct": 100 * (g.attribution_signal == "automation_echo").mean(),
            "self_ref_repo_pct": 100 * (g.attribution_signal == "self_referential_repo").mean(),
        }
    )
t1 = pd.DataFrame(rows)
t1.round(3).to_csv(TABLES / "rq1_T1_agent_profile.csv", index=False)

t1_view = pd.DataFrame(  # readable version of T1 for the console summary
    {
        "Agent": [mark(a) if a != "All" else "All" for a in t1.agent],
        "n": t1.n,
        "%": t1.pct_dataset.round(1),
        "Repos": t1.repos,
        "Gini": t1.gini_repos.round(2),
        "Closed % [95% CI]": [fmt_ci(*r) for r in t1[["closed_pct", "closed_lo", "closed_hi"]].values],
        "Not planned %": t1.not_planned_pct_of_closed.round(1),
        "Labelled % [95% CI]": [
            fmt_ci(*r) for r in t1[["labelled_pct", "labelled_lo", "labelled_hi"]].values
        ],
        "Labels/issue": [
            f"{m:.2f} ({s:.2f})" for m, s in t1[["labels_per_issue_mean", "labels_per_issue_sd"]].values
        ],
        "Non-English %": t1.non_english_pct.round(1),
        "Own-bot %": t1.own_bot_pct.round(1),
    }
)

# -------------------- T2 labels
cats = ["bug", "feature", "documentation", "agent_workflow", "other"]
t2 = pd.DataFrame({"labelled_pct": d.groupby("agent_type").n_labels.apply(lambda s: 100 * (s > 0).mean())})
for c in cats:
    t2[f"{c}_pct"] = d.groupby("agent_type")[f"has_{c}_label"].mean() * 100
t2 = t2.loc[order]
t2.round(2).to_csv(TABLES / "rq1_T2_label_categories.csv")
lab = labels.merge(d[["issue_id", "agent_type"]], on="issue_id")
top = (
    lab.groupby(["agent_type", "label_norm"])
    .size()
    .rename("n_issues")
    .reset_index()
    .sort_values(["agent_type", "n_issues"], ascending=[True, False])
    .groupby("agent_type")
    .head(5)
)
top["category"] = top.label_norm.map(labels.drop_duplicates("label_norm").set_index("label_norm").category)
top.to_csv(TABLES / "rq1_T2_top_labels.csv", index=False)

# -------------------- T3 association tests (M1.8–M1.10)
dt = d[d.agent_type.isin(tested)]
closed_t = dt[dt.is_closed]
tests = {
    "State (open vs closed)": (dt.agent_type, dt.api_state),
    "Closure reason (closed issues)": (closed_t.agent_type, closed_t.closure_reason),
    "Labelled (yes/no)": (dt.agent_type, np.where(dt.n_labels > 0, "labelled", "unlabelled")),
    "Language group": (dt.agent_type, dt.lang_group),
    "Attribution signal": (dt.agent_type, dt.attribution_signal),
}
res, resid = [], {}
for name, (g, c) in tests.items():
    r = permutation_chi2(np.asarray(g), np.asarray(c))
    resid[name] = r
    res.append(
        {
            "test": f"agent × {name}",
            "n": r["n"],
            "chi2": r["stat"],
            "df": r["df"],
            "p_perm": r["p"],
            "cramers_v": r["v"],
            "df_star": r["df_star"],
            "effect": v_label(r["v"], r["df_star"]),
            "pct_cells_expected_lt5": r["pct_expected_lt5"],
            "min_expected": r["min_expected"],
        }
    )
t3 = pd.DataFrame(res)
t3["p_holm"] = multipletests(t3.p_perm, method="holm")[1]
t3.round(4).to_csv(TABLES / "rq1_T3_association_tests.csv", index=False)

# X2: adjusted standardised residuals with a Bonferroni cell-wise critical value
rows = []
for name, r in resid.items():
    z_crit = norm.ppf(1 - 0.05 / (2 * r["residuals"].size))
    for agent, row in r["residuals"].iterrows():
        for cat, z in row.items():
            rows.append(
                {
                    "test": name,
                    "agent": agent,
                    "category": cat,
                    "observed": int(r["observed"].loc[agent, cat]),
                    "adj_residual": z,
                    "z_crit_bonferroni": z_crit,
                    "significant": abs(z) > z_crit,
                }
            )
pd.DataFrame(rows).round(3).to_csv(TABLES / "rq1_X2_adjusted_residuals.csv", index=False)

# X3: monthly volume
d["month"] = d.api_created_at.dt.tz_localize(None).dt.to_period("M")
monthly = d.groupby(["agent_type", "month"]).size().rename("issues").reset_index()
monthly.to_csv(TABLES / "rq1_X3_monthly_volume.csv", index=False)

# -------------------- figures
# F1 volume and state
outcome_cols = [
    ("completed", ps.SLOTS[0], "Closed: completed"),
    ("not_planned", ps.SLOTS[1], "Closed: not planned"),
    ("duplicate", ps.SLOTS[2], "Closed: duplicate"),
    ("open", ps.NEUTRAL, "Open (censored)"),
]
shares = pd.crosstab(d.agent_type, d.closure_reason, normalize="index").reindex(order) * 100
counts = d.agent_type.value_counts().reindex(order)
fig, (a1, a2) = ps.plt.subplots(
    1, 2, figsize=(9, 4.6), sharey=True, gridspec_kw={"width_ratios": [1, 1.6], "wspace": 0.08}
)
y = np.arange(len(order))[::-1]
a1.barh(y, counts.values, height=0.62, color=ps.SLOTS[0])
for yi, v in zip(y, counts.values):
    a1.text(v + counts.max() * 0.015, yi, f"{v:,}", va="center", fontsize=7.5, color=ps.INK_2)
a1.set_yticks(y, [ps.name(mark(a)) for a in order])
a1.set_xlim(0, counts.max() * 1.18)
a1.set_title("Volume (issues)", loc="left")
ps.clean_axes(a1)
left = np.zeros(len(order))
for key, col, lab_ in outcome_cols:
    vals = shares.get(key, pd.Series(0, index=order)).fillna(0).values
    a2.barh(y, vals, left=left, height=0.62, color=col, edgecolor=ps.SURFACE, linewidth=1.2, label=lab_)
    for yi, l, v in zip(y, left, vals):
        if v >= 9:
            a2.text(
                l + v / 2, yi, f"{v:.0f}%", ha="center", va="center", fontsize=7, color=ps.text_color_on(col)
            )
    left += vals
a2.set_xlim(0, 100)
a2.set_title("State and closure reason (% of agent's issues)", loc="left")
a2.legend(ncol=4, loc="lower left", bbox_to_anchor=(0, -0.16), handlelength=1, columnspacing=1)
ps.clean_axes(a2)
fig.text(0.01, -0.06, "† fewer than 30 closed issues: shown, not tested.", fontsize=7, color=ps.MUTED)
ps.save(fig, FIGURES / "rq1_F1_volume_state")
f1 = pd.crosstab(d.agent_type, d.closure_reason).reindex(order).fillna(0).astype(int)
f1 = f1.join((f1.div(f1.sum(1), axis=0) * 100).round(2).add_suffix("_pct"))
f1.insert(0, "n", counts)
f1.to_csv(FIGURE_DATA / "rq1_F1_volume_state.csv")

# F2 labelling heatmap
hm = t2[["labelled_pct"] + [f"{c}_pct" for c in cats]]
hm.columns = ["Any label", "Bug", "Feature", "Docs", "Agent-workflow", "Other"]
fig, ax = ps.plt.subplots(figsize=(6.4, 4.6))
im = ax.imshow(hm.values, cmap=ps.SEQ, vmin=0, vmax=100, aspect="auto")
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
        v = hm.values[i, j]
        ax.text(
            j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color=ps.text_color_on(ps.SEQ(v / 100))
        )
ax.set_xticks(range(hm.shape[1]), hm.columns)
ax.set_yticks(range(hm.shape[0]), [ps.name(mark(a)) for a in hm.index])
ax.tick_params(axis="x", length=0, labelsize=7.5, colors=ps.INK_2)
for s in ax.spines.values():
    s.set_visible(False)
ax.axvline(0.5, color=ps.SURFACE, linewidth=3)
ax.set_title("Labelling: % of each agent's issues with a label of each category", loc="left")
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=7)
cb.set_label("% of the agent's issues", fontsize=7, color=ps.INK_2)
ps.save(fig, FIGURES / "rq1_F2_labels_heatmap")
hm.round(2).to_csv(FIGURE_DATA / "rq1_F2_labels_heatmap.csv")


# F3 attribution signal and language
def stacked(ax, series, spec, title):
    sh = pd.crosstab(d.agent_type, series, normalize="index").reindex(order) * 100
    left = np.zeros(len(order))
    for key, col, lab_ in spec:
        vals = sh.get(key, pd.Series(0, index=order)).fillna(0).values
        ax.barh(y, vals, left=left, height=0.62, color=col, edgecolor=ps.SURFACE, linewidth=1.2, label=lab_)
        for yi, l, v in zip(y, left, vals):
            if v >= 12:
                ax.text(
                    l + v / 2,
                    yi,
                    f"{v:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=ps.text_color_on(col),
                )
        left += vals
    ax.set_xlim(0, 100)
    ax.set_title(title, loc="left")
    ps.clean_axes(ax)
    ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, -0.22), handlelength=1, columnspacing=1)


fig, (a1, a2) = ps.plt.subplots(1, 2, figsize=(9, 4.8), sharey=True, gridspec_kw={"wspace": 0.08})
stacked(
    a1,
    d.attribution_signal,
    [
        ("agent_own_bot", ps.SLOTS[0], "Agent's own bot account"),
        ("automation_echo", ps.SLOTS[1], "github-actions[bot] echo"),
        ("self_referential_repo", ps.SLOTS[2], "Agent's own repository"),
        ("other_signal", ps.NEUTRAL, "Other signal (text/branch/human)"),
    ],
    "How the issue is linked to the agent (%)",
)
stacked(
    a2,
    d.lang_group,
    [
        ("non_latin", ps.SLOTS[0], "Non-Latin script (CJK, Cyrillic…)"),
        ("latin_non_english", ps.SLOTS[1], "Latin, non-English"),
        ("english", ps.NEUTRAL, "English"),
    ],
    "Language of the issue (%)",
)
a1.set_yticks(y, [ps.name(mark(a)) for a in order])
ps.save(fig, FIGURES / "rq1_F3_attribution_language")
f3 = []
for col in ["attribution_signal", "lang_group"]:
    ct = pd.crosstab(d.agent_type, d[col]).reindex(order).fillna(0).astype(int)
    pc = (ct.div(ct.sum(1), axis=0) * 100).round(2)
    for a in order:
        for cat in ct.columns:
            f3.append(
                {"panel": col, "agent": a, "category": cat, "issues": ct.loc[a, cat], "pct": pc.loc[a, cat]}
            )
pd.DataFrame(f3).to_csv(FIGURE_DATA / "rq1_F3_attribution_language.csv", index=False)

# X2 residual heatmaps
panels = ["Closure reason (closed issues)", "Language group", "Attribution signal"]
nice = {
    "completed": "Completed",
    "not_planned": "Not planned",
    "duplicate": "Duplicate",
    "english": "English",
    "latin_non_english": "Latin non-En",
    "non_latin": "Non-Latin",
    "agent_own_bot": "Own bot",
    "automation_echo": "Actions echo",
    "self_referential_repo": "Own repo",
    "other_signal": "Other",
}
fig, axes = ps.plt.subplots(
    1, 3, figsize=(10, 4.4), sharey=True, gridspec_kw={"width_ratios": [3, 3, 4], "wspace": 0.06}
)
tested_order = [a for a in order if a in tested]
for ax, name in zip(axes, panels):
    r = resid[name]["residuals"].reindex(tested_order)
    z_crit = norm.ppf(1 - 0.05 / (2 * r.size))
    im = ax.imshow(r.clip(-10, 10).values, cmap=ps.DIVERGING, vmin=-10, vmax=10, aspect="auto")
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            z = r.values[i, j]
            ax.text(
                j,
                i,
                f"{z:+.1f}" + ("*" if abs(z) > z_crit else ""),
                ha="center",
                va="center",
                fontsize=6.6,
                color=ps.text_color_on(ps.DIVERGING((np.clip(z, -10, 10) + 10) / 20)),
                fontweight="bold" if abs(z) > z_crit else "normal",
            )
    ax.set_xticks(range(r.shape[1]), [nice.get(c, c) for c in r.columns], rotation=30, ha="right")
    ax.tick_params(axis="x", length=0, colors=ps.INK_2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(name, loc="left", fontsize=8.5)
axes[0].set_yticks(range(len(tested_order)), [ps.name(a) for a in tested_order])
cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=7)
cb.set_label("adjusted residual (blue = fewer than expected, red = more)", fontsize=7, color=ps.INK_2)
fig.text(
    0.01,
    -0.08,
    "* |residual| above the Bonferroni-adjusted critical value for the panel (family-wise α = 0.05). "
    "Values clipped to ±10 for colour only.",
    fontsize=7,
    color=ps.MUTED,
)
ps.save(fig, FIGURES / "rq1_X2_residuals")
pd.concat({k: resid[k]["residuals"].reindex(tested_order).stack() for k in panels}).rename(
    "adj_residual"
).rename_axis(["panel", "agent", "category"]).reset_index().round(3).to_csv(
    FIGURE_DATA / "rq1_X2_residuals.csv", index=False
)

# X3 monthly volume, small multiples
months = pd.period_range("2024-01", "2026-04", freq="M")
fig, axes = ps.plt.subplots(4, 4, figsize=(10, 7.2), sharex=True)
for ax, a in zip(axes.flat, order):
    s = monthly[monthly.agent_type == a].set_index("month").issues.reindex(months, fill_value=0)
    colors = [ps.SLOTS[0]] * (len(months) - 1) + [ps.BLUE_RAMP[1]]  # last month partial
    ax.bar(range(len(months)), s.values, width=0.8, color=colors)
    rel = pd.Period(RELEASE_DATES[a][0][:7], freq="M")
    if months[0] <= rel <= months[-1]:
        x = months.get_loc(rel)
        ax.axvline(x, color=ps.INK_2, linewidth=0.7)
        ax.text(x + 0.4, s.max() * 0.92 if s.max() else 0.9, "release", fontsize=6, color=ps.INK_2)
    ax.set_title(f"{ps.name(mark(a))} (n = {counts[a]:,})", loc="left", fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=6.5, colors=ps.MUTED)
    ax.yaxis.set_major_locator(ps.plt.MaxNLocator(integer=True, nbins=4))
for ax in list(axes.flat)[len(order) :]:
    ax.axis("off")
ticks = [i for i, m in enumerate(months) if m.month in (1, 7)]
ncols = axes.shape[1]
for k, ax in enumerate(list(axes.flat)[: len(order)]):
    if k + ncols >= len(order):  # lowest visible panel in its column
        ax.tick_params(labelbottom=True)
        ax.set_xticks(ticks, [months[i].strftime("%b %y") for i in ticks], rotation=0)
fig.suptitle(
    "Issues created per month by agent (independent y-axes)",
    x=0.01,
    ha="left",
    fontsize=9.5,
    fontweight="bold",
)
fig.text(
    0.01,
    -0.01,
    f"Jan 2024 – Apr 2026 (CodAGE window starts 2024-01-01; April 2026 partial, lighter). "
    f"{int((d.api_created_at < '2024-01-01').sum())} issues created before 2024 not shown. "
    "Vertical line = agent's public release (release_dates.csv).",
    fontsize=7,
    color=ps.MUTED,
)
fig.tight_layout()
ps.save(fig, FIGURES / "rq1_X3_monthly_volume")
(
    monthly.assign(month=monthly.month.astype(str))
    .pivot(index="month", columns="agent_type", values="issues")
    .reindex([str(m) for m in months])
    .fillna(0)
    .astype(int)[order]
    .to_csv(FIGURE_DATA / "rq1_X3_monthly_volume.csv")
)

# -------------------- Finding A: agent-written issues stay open
# Within-agent comparison (agents whose issues include both own-bot and other signals),
# so the contrast is the authorship role, not the agent. Age check: own-bot issues
# could be newer (less time to close); we report median age and an age-restricted rerun.
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest

snapshot = pd.Timestamp("2026-10-09T17:48:25Z")
d["age_days"] = (snapshot - d.api_created_at).dt.days


def two_prop(g, label, min_age=0):
    g = g[g.age_days >= min_age]
    own, oth = g[g.own_bot_author], g[~g.own_bot_author]
    k1, n1, k2, n2 = int(own.is_closed.sum()), len(own), int(oth.is_closed.sum()), len(oth)
    z, p = proportions_ztest([k1, k2], [n1, n2])
    lo, hi = confint_proportions_2indep(k1, n1, k2, n2, method="newcomb")
    p1, p2 = k1 / n1, k2 / n2
    h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))  # Cohen's h
    w1, w2 = wilson(k1, n1), wilson(k2, n2)
    return {
        "group": label,
        "min_age_days": min_age,
        "own_bot_n": n1,
        "own_bot_closed_pct": w1[0],
        "own_bot_lo": w1[1],
        "own_bot_hi": w1[2],
        "own_bot_median_age_days": own.age_days.median(),
        "other_n": n2,
        "other_closed_pct": w2[0],
        "other_lo": w2[1],
        "other_hi": w2[2],
        "other_median_age_days": oth.age_days.median(),
        "diff_pp": 100 * (p1 - p2),
        "diff_lo": 100 * lo,
        "diff_hi": 100 * hi,
        "z": z,
        "p": p,
        "cohens_h": h,
    }


fa = pd.DataFrame(
    [
        two_prop(d[d.agent_type == "CodeRabbit"], "CodeRabbit"),
        two_prop(d[d.agent_type == "Devin"], "Devin"),
        two_prop(d[d.agent_type == "CodeRabbit"], "CodeRabbit", 180),
        two_prop(d[d.agent_type == "Devin"], "Devin", 180),
        two_prop(d, "All agents (pooled, descriptive)"),
    ]
)
fa["p_holm"] = np.nan
fa.loc[:1, "p_holm"] = multipletests(fa.p[:2], method="holm")[1]  # family: the two agents
fa["h_effect"] = pd.cut(
    fa.cohens_h.abs(),
    [0, 0.2, 0.5, 0.8, np.inf],
    labels=["negligible", "small", "medium", "large"],
    right=False,
)
fa.to_csv(TABLES / "rq1_FA_own_bot_closure.csv", index=False, float_format="%.6g")

# -------------------- Finding B: the labelling illusion
triage_ids = set(labels.loc[labels.category != "agent_workflow", "issue_id"])
d["triage_labelled"] = d.issue_id.isin(triage_ids)
fb = []
for a in order:
    g = d[d.agent_type == a]
    anyl, tri = wilson(int((g.n_labels > 0).sum()), len(g)), wilson(int(g.triage_labelled.sum()), len(g))
    fb.append(
        {
            "agent": a,
            "tested": a in tested,
            "n": len(g),
            "any_label_pct": anyl[0],
            "any_lo": anyl[1],
            "any_hi": anyl[2],
            "triage_label_pct": tri[0],
            "triage_lo": tri[1],
            "triage_hi": tri[2],
            "agent_workflow_label_pct": 100 * g.has_agent_workflow_label.mean(),
            "drop_pp": anyl[0] - tri[0],
        }
    )
fb = pd.DataFrame(fb)
fb["rank_any"] = fb.any_label_pct.rank(ascending=False, method="min").astype(int)
fb["rank_triage"] = fb.triage_label_pct.rank(ascending=False, method="min").astype(int)
fb.round(3).to_csv(TABLES / "rq1_FB_labelling_illusion.csv", index=False)

# -------------------- console summary
with pd.option_context("display.width", 250, "display.max_columns", 40):
    print(t1_view.to_string(index=False))
    print()
    print(t3.drop(columns=["df_star"]).round(4).to_string(index=False))
    print()
    print(
        fa[
            [
                "group",
                "min_age_days",
                "own_bot_n",
                "own_bot_closed_pct",
                "other_n",
                "other_closed_pct",
                "diff_pp",
                "diff_lo",
                "diff_hi",
                "p",
                "p_holm",
                "cohens_h",
                "h_effect",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    print()
    print(
        fb[["agent", "any_label_pct", "triage_label_pct", "drop_pp", "rank_any", "rank_triage"]]
        .round(1)
        .to_string(index=False)
    )
