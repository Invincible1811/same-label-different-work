"""Step 3: RQ2 — does time-to-close (TTC) differ across AI coding agents, and how
large are the differences?  (metrics M2.1–M2.7, sensitivity checks S1–S3, extra Y1)

Reads data/processed/issues_clean.csv (run 01_clean.py first).
Writes tables to results/tables/rq2_*.csv, figure data to results/figure_data/rq2_*.csv
and figures to results/figures/rq2_*.{pdf,png}.

Run:  python src/03_rq2.py
"""
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.utils import median_survival_times
from scipy.stats import kruskal, mannwhitneyu, probplot, shapiro, skew
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.stats.multitest import multipletests

import plotstyle as ps
from stats_utils import cliffs_delta, delta_label, eps2, eps2_label, median_ci_exact
from config import FIGURE_DATA, FIGURES, MIN_CLOSED_RQ2, PROCESSED, SEED, TABLES

ps.apply()
MINUTE = 1 / 60


# -------------------- helpers
def kw(df, min_n=MIN_CLOSED_RQ2):
    counts = df.agent_type.value_counts()
    agents = sorted(counts[counts >= min_n].index)
    groups = [df.loc[df.agent_type == a, "ttc_hours"].values for a in agents]
    H, p = kruskal(*groups)
    n = sum(len(g) for g in groups)
    e = eps2(H, n)
    return dict(agents=len(agents), n=n, H=H, df=len(agents) - 1, p=p, epsilon2=e, effect=eps2_label(e))


# -------------------- data
d = pd.read_csv(PROCESSED / "issues_clean.csv")
c = d[d.rq2_eligible].copy()  # closed issues, 11 agents (D7)
agents = c.agent_type.unique().tolist()
order = c.groupby("agent_type").ttc_hours.median().sort_values().index.tolist()  # fastest first
rng = np.random.default_rng(SEED)
for p_ in (TABLES, FIGURES, FIGURE_DATA):
    p_.mkdir(parents=True, exist_ok=True)

# -------------------- M2.3 distribution check
rows = []
for a in order + ["All"]:
    x = c.ttc_hours.values if a == "All" else c.loc[c.agent_type == a, "ttc_hours"].values
    lx = np.log1p(x)
    rows.append(
        {
            "agent": a,
            "n": len(x),
            "skew_ttc": skew(x),
            "skew_log1p_ttc": skew(lx),
            "shapiro_W_ttc": shapiro(x).statistic,
            "shapiro_p_ttc": shapiro(x).pvalue,
            "shapiro_W_log1p": shapiro(lx).statistic,
            "shapiro_p_log1p": shapiro(lx).pvalue,
            "lilliefors_KS_log1p": lilliefors(lx, dist="norm")[0],
            "lilliefors_p_log1p": lilliefors(lx, dist="norm")[1],
        }
    )
dist = pd.DataFrame(rows)
dist["normal_rejected_log1p_at_0.05"] = dist.shapiro_p_log1p < 0.05
dist.to_csv(TABLES / "rq2_T1_distribution_check.csv", index=False, float_format="%.6g")

# -------------------- M2.1–M2.2 summary
rows = []
for a in order + ["All"]:
    x = c.ttc_hours.values if a == "All" else c.loc[c.agent_type == a, "ttc_hours"].values
    lo, hi, cov = median_ci_exact(
        x
    )  # exact order-statistic CI (deterministic; replaces the seed-sensitive bootstrap)
    rows.append(
        {
            "agent": a,
            "n_closed": len(x),
            "median_h": np.median(x),
            "median_ci_lo": lo,
            "median_ci_hi": hi,
            "median_ci_coverage": cov,
            "q1_h": np.percentile(x, 25),
            "q3_h": np.percentile(x, 75),
            "iqr_h": np.percentile(x, 75) - np.percentile(x, 25),
            "p90_h": np.percentile(x, 90),
            "mean_h": x.mean(),
            "sd_h": x.std(ddof=1),
            "median_days": np.median(x) / 24,
        }
    )
summ = pd.DataFrame(rows)
summ.to_csv(TABLES / "rq2_T2_ttc_summary.csv", index=False, float_format="%.6g")

# -------------------- M2.4 Kruskal–Wallis
main = kw(c)
# -------------------- M2.5 Dunn (Holm) + Cliff's δ
dunn = sp.posthoc_dunn(c, val_col="ttc_hours", group_col="agent_type", p_adjust="holm")
pairs = []
for i, a in enumerate(order):
    for b in order[i + 1 :]:
        x, y = c.loc[c.agent_type == a, "ttc_hours"].values, c.loc[c.agent_type == b, "ttc_hours"].values
        dl = cliffs_delta(x, y)
        pairs.append(
            {
                "agent_a": a,
                "agent_b": b,
                "median_a_h": np.median(x),
                "median_b_h": np.median(y),
                "dunn_p_holm": dunn.loc[a, b],
                "significant": dunn.loc[a, b] < 0.05,
                "cliffs_delta": dl,
                "delta_effect": delta_label(dl),
            }
        )
pairs = pd.DataFrame(pairs)
pairs.to_csv(TABLES / "rq2_T3_pairwise_dunn_cliff.csv", index=False, float_format="%.6g")

# -------------------- M2.6 Kaplan–Meier with censored open issues
s = d[d.agent_type.isin(agents)].copy()
km_rows, km_curves = [], []
for a in order + ["All"]:
    g = s if a == "All" else s[s.agent_type == a]
    kmf = KaplanMeierFitter().fit(g.duration_hours, g.event_closed, label=a)
    ci = median_survival_times(kmf.confidence_interval_)
    km_rows.append(
        {
            "agent": a,
            "n_all": len(g),
            "n_closed": int(g.event_closed.sum()),
            "n_censored_open": int((g.event_closed == 0).sum()),
            "km_median_h": kmf.median_survival_time_,
            "km_median_ci_lo": ci.iloc[0, 0],
            "km_median_ci_hi": ci.iloc[0, 1],
            "pct_closed_within_7d": 100 * (1 - kmf.predict(24 * 7)),
            "pct_closed_within_90d": 100 * (1 - kmf.predict(24 * 90)),
        }
    )
    sf = kmf.survival_function_.reset_index()
    sf.columns = ["hours", "pct_still_open"]
    sf["pct_still_open"] *= 100
    sf.insert(0, "agent", a)
    km_curves.append(sf)
lr = multivariate_logrank_test(s.duration_hours, s.agent_type, s.event_closed)
km = pd.DataFrame(km_rows)
km.to_csv(TABLES / "rq2_T4_kaplan_meier.csv", index=False, float_format="%.6g")
km_curves = pd.concat(km_curves)

# -------------------- M2.7 TTC by closure outcome
out_rows, out_tests = [], []
for outcome in ["completed", "not_planned", "duplicate"]:
    sub = c[c.closure_reason == outcome]
    for a in order:
        x = sub.loc[sub.agent_type == a, "ttc_hours"]
        out_rows.append(
            {
                "outcome": outcome,
                "agent": a,
                "n": len(x),
                "median_h": x.median() if len(x) else np.nan,
                "q1_h": x.quantile(0.25) if len(x) else np.nan,
                "q3_h": x.quantile(0.75) if len(x) else np.nan,
            }
        )
    if outcome != "duplicate":  # duplicate: 31 issues, descriptive only
        r = kw(sub, min_n=10)
        r.update(outcome=outcome, min_n_per_agent=10)
        out_tests.append(r)
pd.DataFrame(out_rows).to_csv(TABLES / "rq2_T5_ttc_by_outcome.csv", index=False, float_format="%.6g")

# -------------------- S1–S3 sensitivity + significance summary
sens = []
main_medians = c.groupby("agent_type").ttc_hours.median()
for name, sub in [
    ("Main (all closed, 11 agents)", c),
    ("S1: created after agent release", c[~c.pre_release]),
    ("S2: human-authored only (author_type = User)", c[~c.author_is_bot]),
    ("S3: CodAGE precision subset", c[c.s3_precision]),
]:
    r = kw(sub)
    med = sub.groupby("agent_type").ttc_hours.median()
    cnt = sub.agent_type.value_counts()
    keep = cnt[cnt >= MIN_CLOSED_RQ2].index
    r["spearman_rank_vs_main"] = (
        med[keep].rank().corr(main_medians[keep].rank(), method="spearman") if len(keep) > 2 else np.nan
    )
    r["fastest_agent"], r["slowest_agent"] = med[keep].idxmin(), med[keep].idxmax()
    r["subset"] = name
    sens.append(r)
tests = pd.DataFrame(
    [dict(test="Kruskal–Wallis, all closed", **main)]
    + [
        dict(
            test=f"Kruskal–Wallis, {r['outcome']} only",
            **{k: v for k, v in r.items() if k not in ("outcome", "min_n_per_agent")},
        )
        for r in out_tests
    ]
)
tests["logrank_chi2_all_issues"] = [lr.test_statistic, np.nan, np.nan]
tests["logrank_p_all_issues"] = [lr.p_value, np.nan, np.nan]
tests.to_csv(TABLES / "rq2_T6_tests.csv", index=False, float_format="%.6g")
# S4: repository clustering — one observation per (agent, repository) = the repository's median TTC
rm = c.groupby(["agent_type", "repo_name"], as_index=False).ttc_hours.median()
r4 = kw(rm, min_n=1)
med4 = rm.groupby("agent_type").ttc_hours.median()
r4["spearman_rank_vs_main"] = med4.rank().corr(main_medians.reindex(med4.index).rank(), method="spearman")
r4["fastest_agent"], r4["slowest_agent"] = med4.idxmin(), med4.idxmax()
r4["subset"] = "S4: one value per agent-repository (repository median)"
sens.append(r4)
cols = [
    "subset",
    "agents",
    "n",
    "H",
    "df",
    "p",
    "epsilon2",
    "effect",
    "spearman_rank_vs_main",
    "fastest_agent",
    "slowest_agent",
]
pd.DataFrame(sens)[cols].to_csv(TABLES / "rq2_T7_sensitivity.csv", index=False, float_format="%.6g")

# -------------------- selection bias: closed-only vs censoring-aware ranking
kmi = km.set_index("agent").drop("All")
rank_tab = pd.DataFrame(
    {
        "closed_only_median_h": main_medians,
        "km_median_h": kmi.km_median_h,
        "pct_still_open": 100 * kmi.n_censored_open / kmi.n_all,
    }
).reindex(order)
rank_tab["closed_only_rank"] = rank_tab.closed_only_median_h.rank().astype(int)
rank_tab["km_rank"] = rank_tab.km_median_h.rank().astype(int)  # inf (median not reached) ranks last
rank_tab["rank_change"] = rank_tab.km_rank - rank_tab.closed_only_rank
rank_rho = rank_tab.closed_only_rank.corr(rank_tab.km_rank, method="spearman")
rank_tab.rename_axis("agent").to_csv(TABLES / "rq2_T8_ranking_closed_vs_km.csv", float_format="%.6g")

# -------------------- Y1: does "agent-written" also mean "slower"?
y1 = []
for a in ["CodeRabbit", "Devin"]:
    g_all = d[d.agent_type == a]
    g = g_all[g_all.is_closed]
    own, oth = g[g.own_bot_author].ttc_hours.values, g[~g.own_bot_author].ttc_hours.values
    mw = mannwhitneyu(own, oth, alternative="two-sided")
    lrt = logrank_test(
        g_all[g_all.own_bot_author].duration_hours,
        g_all[~g_all.own_bot_author].duration_hours,
        g_all[g_all.own_bot_author].event_closed,
        g_all[~g_all.own_bot_author].event_closed,
    )
    km_own = KaplanMeierFitter().fit(
        g_all[g_all.own_bot_author].duration_hours, g_all[g_all.own_bot_author].event_closed
    )
    km_oth = KaplanMeierFitter().fit(
        g_all[~g_all.own_bot_author].duration_hours, g_all[~g_all.own_bot_author].event_closed
    )
    dl = cliffs_delta(own, oth)
    y1.append(
        {
            "agent": a,
            "own_bot_closed_n": len(own),
            "other_closed_n": len(oth),
            "own_bot_median_h": np.median(own),
            "other_median_h": np.median(oth),
            "mannwhitney_U": mw.statistic,
            "mannwhitney_p": mw.pvalue,
            "cliffs_delta": dl,
            "delta_effect": delta_label(dl),
            "km_median_own_bot_h": km_own.median_survival_time_,
            "km_median_other_h": km_oth.median_survival_time_,
            "logrank_chi2": lrt.test_statistic,
            "logrank_p": lrt.p_value,
        }
    )
y1 = pd.DataFrame(y1)
y1["mannwhitney_p_holm"] = multipletests(y1.mannwhitney_p, method="holm")[1]
y1["logrank_p_holm"] = multipletests(y1.logrank_p, method="holm")[1]
y1.to_csv(TABLES / "rq2_Y1_own_bot_ttc.csv", index=False, float_format="%.6g")

# -------------------- figures
ylab = {a: f"{ps.name(a)}  (n = {(c.agent_type == a).sum():,})" for a in order}

# R1 box plots, log scale
fig, ax = ps.plt.subplots(figsize=(8.5, 5))
data = [np.clip(c.loc[c.agent_type == a, "ttc_hours"].values, MINUTE, None) for a in order]
bp = ax.boxplot(
    data,
    vert=False,
    widths=0.55,
    patch_artist=True,
    showfliers=True,
    medianprops=dict(color=ps.INK, linewidth=1.4),
    whiskerprops=dict(color=ps.MUTED, linewidth=0.8),
    capprops=dict(color=ps.MUTED, linewidth=0.8),
    flierprops=dict(marker="o", markersize=2, markerfacecolor=ps.NEUTRAL, markeredgecolor="none", alpha=0.6),
)
for b in bp["boxes"]:
    b.set(facecolor=ps.BLUE_RAMP[1], edgecolor=ps.SLOTS[0], linewidth=0.8)
ax.set_xscale("log")
ticks = [MINUTE, 1, 24, 24 * 7, 24 * 30, 24 * 365, 24 * 365 * 5]
ax.set_xticks(ticks, ["1 min", "1 h", "1 day", "1 week", "1 month", "1 year", "5 years"])
med_h = summ.set_index("agent").median_h
ax.set_yticks(range(1, len(order) + 1), [f"{ylab[a]}   median {med_h[a]:.1f} h" for a in order])
ax.invert_yaxis()
ps.clean_axes(ax)
ax.set_title("Time-to-close of closed issues, by agent (sorted by median; log scale)", loc="left")
fig.text(
    0.01,
    -0.03,
    f"Kruskal–Wallis H = {main['H']:.1f}, df = {main['df']}, p = {main['p']:.1e}, "
    f"ε² = {main['epsilon2']:.3f} ({main['effect']}). Values below 1 minute drawn at 1 minute.",
    fontsize=7,
    color=ps.MUTED,
)
ps.save(fig, FIGURES / "rq2_R1_ttc_boxplots")
pd.concat(
    [pd.DataFrame({"agent": a, "ttc_hours": c.loc[c.agent_type == a, "ttc_hours"].values}) for a in order]
).to_csv(FIGURE_DATA / "rq2_R1_ttc_boxplots.csv", index=False, float_format="%.6g")

# R2 Cliff's δ heatmap (row agent vs column agent; positive = row agent slower)
mat = pd.DataFrame(np.nan, index=order, columns=order)
sig = pd.DataFrame(False, index=order, columns=order)
for _, r in pairs.iterrows():
    mat.loc[r.agent_a, r.agent_b], mat.loc[r.agent_b, r.agent_a] = r.cliffs_delta, -r.cliffs_delta
    sig.loc[r.agent_a, r.agent_b] = sig.loc[r.agent_b, r.agent_a] = r.significant
fig, ax = ps.plt.subplots(figsize=(8, 6.6))
im = ax.imshow(mat.values, cmap=ps.DIVERGING, vmin=-0.6, vmax=0.6)
for i in range(len(order)):
    for j in range(len(order)):
        v = mat.values[i, j]
        if np.isnan(v):
            continue
        ax.text(
            j,
            i,
            f"{v:+.2f}" + ("*" if sig.values[i, j] else ""),
            ha="center",
            va="center",
            fontsize=6.3,
            color=ps.text_color_on(ps.DIVERGING((np.clip(v, -0.6, 0.6) + 0.6) / 1.2)),
            fontweight="bold" if sig.values[i, j] else "normal",
        )
ax.set_xticks(range(len(order)), [ps.name(a) for a in order], rotation=40, ha="right")
ax.set_yticks(range(len(order)), [ps.name(a) for a in order])
ax.tick_params(length=0, colors=ps.INK_2)
for sp_ in ax.spines.values():
    sp_.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=7)
cb.set_label("Cliff's δ (red = row agent's issues take longer)", fontsize=7, color=ps.INK_2)
ax.set_title("Pairwise differences in time-to-close (Cliff's δ)", loc="left")
fig.text(
    0.01,
    -0.02,
    "* Dunn's test significant after Holm correction (α = 0.05). |δ| < 0.147 negligible, < 0.33 small, "
    "< 0.474 medium, otherwise large (Romano et al., 2006). Agents sorted by median time-to-close.",
    fontsize=7,
    color=ps.MUTED,
)
ps.save(fig, FIGURES / "rq2_R2_cliffs_delta")
mat.round(4).to_csv(FIGURE_DATA / "rq2_R2_cliffs_delta.csv")

# R3 Kaplan–Meier small multiples
fig, axes = ps.plt.subplots(3, 4, figsize=(10, 6.8), sharex=True, sharey=True)
ref = km_curves[km_curves.agent == "All"]
for ax, a in zip(axes.flat, order):
    cur = km_curves[km_curves.agent == a]
    ax.step(
        np.clip(ref.hours, MINUTE, None),
        ref.pct_still_open,
        where="post",
        color=ps.NEUTRAL,
        linewidth=1.2,
        label="All agents",
    )
    ax.step(
        np.clip(cur.hours, MINUTE, None),
        cur.pct_still_open,
        where="post",
        color=ps.SLOTS[0],
        linewidth=1.6,
        label=a,
    )
    kmm = km.set_index("agent").loc[a, "km_median_h"]
    ax.set_title(
        f"{ps.name(a)}  (median {kmm / 24:.1f} d)"
        if np.isfinite(kmm)
        else f"{ps.name(a)}  (median not reached)",
        loc="left",
        fontsize=7.8,
    )
    ax.set_xscale("log")
    ax.grid(True)
    ax.set_axisbelow(True)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
axes.flat[-1].axis("off")
axes.flat[-1].legend(
    axes.flat[0].get_legend_handles_labels()[0], ["All agents (reference)", "This agent"], loc="center"
)
for ax in axes[-1]:
    ax.set_xticks([1, 24, 24 * 7, 24 * 365], ["1 h", "1 d", "1 w", "1 y"])
axes[-1, -2].tick_params(labelbottom=True)
for ax in axes[:, 0]:
    ax.set_ylabel("% still open")
fig.suptitle(
    "Kaplan–Meier: share of issues still open over time (open issues censored at 2026-10-09)",
    x=0.01,
    ha="left",
    fontsize=9.5,
    fontweight="bold",
)
fig.text(
    0.01,
    -0.01,
    f"Log-rank test across 11 agents: χ² = {lr.test_statistic:.1f}, df = 10, p = {lr.p_value:.1e}. "
    "Median = time by which half of the agent's issues are closed.",
    fontsize=7,
    color=ps.MUTED,
)
fig.tight_layout()
ps.save(fig, FIGURES / "rq2_R3_kaplan_meier")
km_curves.to_csv(FIGURE_DATA / "rq2_R3_kaplan_meier.csv", index=False, float_format="%.6g")

# R4 distribution check: Q–Q plots of log(1 + TTC)
fig, axes = ps.plt.subplots(3, 4, figsize=(10, 6.8))
qq = []
for ax, a in zip(axes.flat, order + ["All"]):
    lx = np.log1p(c.ttc_hours.values if a == "All" else c.loc[c.agent_type == a, "ttc_hours"].values)
    (osm, osr), (slope, inter, _) = probplot(lx, dist="norm")
    ax.scatter(osm, osr, s=3, color=ps.SLOTS[0], linewidths=0)
    ax.plot(osm, slope * osm + inter, color=ps.INK_2, linewidth=0.8)
    row = dist.set_index("agent").loc[a]
    ax.set_title(
        f"{ps.name(a)}\nShapiro–Wilk W = {row.shapiro_W_log1p:.3f}, p = {row.shapiro_p_log1p:.1e}",
        loc="left",
        fontsize=7,
    )
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    qq.append(pd.DataFrame({"agent": a, "theoretical_quantile": osm, "log1p_ttc": osr}))
fig.suptitle(
    "Normal Q–Q plots of log(1 + time-to-close in hours): points off the line = not normal",
    x=0.01,
    ha="left",
    fontsize=9.5,
    fontweight="bold",
)
fig.text(
    0.01,
    -0.01,
    "Shapiro–Wilk test on log-transformed values; normality is rejected (p < 0.05) for the agents "
    "shown with small p. "
    "Raw time-to-close is far more skewed (see rq2_T1_distribution_check.csv).",
    fontsize=7,
    color=ps.MUTED,
)
fig.tight_layout()
ps.save(fig, FIGURES / "rq2_R4_qq_plots")
pd.concat(qq).to_csv(FIGURE_DATA / "rq2_R4_qq_plots.csv", index=False, float_format="%.6g")

# -------------------- console summary
with pd.option_context("display.width", 250, "display.max_columns", 30):
    print(
        summ[["agent", "n_closed", "median_h", "median_ci_lo", "median_ci_hi", "q1_h", "q3_h", "mean_h"]]
        .round(1)
        .to_string(index=False)
    )
    print("\nMain KW:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in main.items()})
    print("Log-rank (all issues incl. open): chi2 =", round(lr.test_statistic, 2), "p =", lr.p_value)
    print(
        "\nPairs significant:",
        int(pairs.significant.sum()),
        "of",
        len(pairs),
        "| effect sizes:",
        pairs.delta_effect.value_counts().to_dict(),
    )
    print(
        "\n",
        km[["agent", "n_all", "n_censored_open", "km_median_h", "km_median_ci_lo", "km_median_ci_hi"]]
        .round(1)
        .to_string(index=False),
    )
    print("\n", tests.round(4).to_string(index=False))
    print("\n", pd.DataFrame(sens)[cols].round(4).to_string(index=False))
    print("\n", y1.round(4).to_string(index=False))
    print("\nRanking closed-only vs KM: Spearman rho =", round(rank_rho, 3))
    print(rank_tab.round(1).to_string())
    print(
        "\nNormality rejected (log1p) for",
        int(dist.iloc[:-1]["normal_rejected_log1p_at_0.05"].sum()),
        "of",
        len(dist) - 1,
        "agents",
    )
