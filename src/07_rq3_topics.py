"""Step 7: RQ3 — what topics do agent-attributed issues raise, and how does topic
prevalence differ across agents?  (M3.2–M3.4, extras Z1–Z3, Q3, Q4, sensitivity checks)

Two levels from the Ensemble LDA of 06_rq3_ensemble.py (see its docstring):
  * headline: topics reproduced across two independent ensembles (pre-set rule);
  * fine:     finer view; each topic is graded by its cross-seed overlap (shared top-10 words
              with the matched topic of the seed-7 ensemble): reproduced (>= 7, the pre-set rule),
              partly reproduced (5-6, at least half), not reproduced (< 5; flagged, not interpreted).
Core analysis (topics, agent shares, association, stability) at both levels; the extras
(validation sheet, topics × time-to-close, topics × authorship role, figures) on the fine view.

Reads data/processed/rq3_corpus.csv, rq3_ensemble_seed42.model, results/tables/rq3_ensemble_selection.csv
and rq3_stable_topics.csv (run 04, 05 and 06 first).
Run:  python src/07_rq3_topics.py      (about 15 minutes, mostly the >= 5-token ensemble refit)
"""
import importlib

import numpy as np
import pandas as pd
from gensim.models import EnsembleLda
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from scipy.optimize import linear_sum_assignment
from scipy.stats import kruskal

import plotstyle as ps
from config import (
    FIGURE_DATA,
    FIGURES,
    PROCESSED,
    ROOT,
    RQ3_ITERATIONS,
    RQ3_PASSES,
    RQ3_INTERPRETED,
    RQ3_TOPIC_LABELS,
    SEED,
    TABLES,
)
from rq3_common import build
from stats_utils import eps2, eps2_label, permutation_chi2, v_label

ps.apply()
N_BOOT = 1000
VALIDATION_PER_TOPIC = 5
LEVELS = ("headline", "fine")
PARTLY_MIN_SHARED = 5  # at least half of the top-10 words recur in the other ensemble
rng = np.random.default_rng(SEED)

c, texts, dictionary, corpus = build()
issues = pd.read_csv(PROCESSED / "issues_clean.csv")
meta = issues.set_index("issue_id")
sel = pd.read_csv(TABLES / "rq3_ensemble_selection.csv")
stable = pd.read_csv(TABLES / "rq3_stable_topics.csv")
EPS = {lvl: float(sel.loc[sel.level == lvl, "eps"].iloc[0]) for lvl in LEVELS}


def theta_of(lda, bows):
    """Document-topic proportions. gensim builds the ensemble's LdaModel without a seed, so inference
    would start from numpy's unseeded global generator; a fixed generator makes it reproducible."""
    lda.random_state = np.random.RandomState(SEED)
    th = np.zeros((len(bows), lda.num_topics))
    for i, bow in enumerate(bows):
        for t, p in lda.get_document_topics(bow, minimum_probability=0.0):
            th[i, t] = p
    return th


def topword_sets(lda, n=10):
    return [set(w for w, _ in lda.show_topic(t, topn=n)) for t in range(lda.num_topics)]


def assoc(frame, level, name):
    r = permutation_chi2(frame.agent_type.values, frame.dominant.values)
    return r, {
        "level": level,
        "analysis": name,
        "documents": r["n"],
        "agents": frame.agent_type.nunique(),
        "chi2": r["stat"],
        "df": r["df"],
        "p_perm": r["p"],
        "cramers_v": r["v"],
        "df_star": r["df_star"],
        "effect": v_label(r["v"], r["df_star"]),
        "pct_cells_expected_lt5": r["pct_expected_lt5"],
    }


# -------------------- final models (seed-42 ensemble at each eps)
model, theta, info = {}, {}, {}
for lvl in LEVELS:
    ens = EnsembleLda.load(str(PROCESSED / "rq3_ensemble_seed42.model"))
    ens.recluster(eps=EPS[lvl], min_samples=None, min_cores=None)
    model[lvl] = ens.generate_gensim_representation()
    theta[lvl] = theta_of(model[lvl], corpus)
    info[lvl] = stable[stable.level == lvl].set_index("topic")
    # the topic order must match 06's table (same ensemble, same eps)
    for t in range(model[lvl].num_topics):
        assert [w for w, _ in model[lvl].show_topic(t, 10)] == info[lvl].loc[t + 1, "top_words"].split(
            ", "
        ), (lvl, t)

K = {lvl: model[lvl].num_topics for lvl in LEVELS}
names = {lvl: [f"T{t + 1}" for t in range(K[lvl])] for lvl in LEVELS}
label = {
    lvl: {t: RQ3_TOPIC_LABELS[lvl].get(t + 1, f"Topic {t + 1}") for t in range(K[lvl])} for lvl in LEVELS
}
reproduced = {lvl: {t: bool(info[lvl].loc[t + 1, "reproduced"]) for t in range(K[lvl])} for lvl in LEVELS}
stability = {
    lvl: {
        t: "reproduced"
        if reproduced[lvl][t]
        else "partly reproduced"
        if info[lvl].loc[t + 1, "shared_top10_words"] >= PARTLY_MIN_SHARED
        else "not reproduced"
        for t in range(K[lvl])
    }
    for lvl in LEVELS
}

interpreted = {lvl: {t: (t + 1) in RQ3_INTERPRETED[lvl] for t in range(K[lvl])} for lvl in LEVELS}
docs = {}
for lvl in LEVELS:
    d = c[["issue_id", "agent_type", "n_tokens"]].copy()
    for t in range(K[lvl]):
        d[names[lvl][t]] = theta[lvl][:, t]
    d["dominant"] = theta[lvl].argmax(1) + 1
    d["dominant_theta"] = theta[lvl].max(1)
    docs[lvl] = d
out = docs["fine"].copy()
out["dominant_headline"] = docs["headline"].dominant
out.to_csv(PROCESSED / "rq3_doc_topics.csv", index=False, float_format="%.6g")
agents = c.agent_type.value_counts().index.tolist()  # by volume

# -------------------- M3.2 topic description
top_rows, rep_rows = [], []
for lvl in LEVELS:
    for t in range(K[lvl]):
        words = model[lvl].show_topic(t, topn=10)
        top_rows.append(
            {
                "level": lvl,
                "topic": t + 1,
                "label": label[lvl][t],
                "stability": stability[lvl][t],
                "interpreted": interpreted[lvl][t],
                "shared_top10_words_seed7": int(info[lvl].loc[t + 1, "shared_top10_words"]),
                "top_words": ", ".join(w for w, _ in words),
                "top_word_weights": ", ".join(f"{p:.3f}" for _, p in words),
                "prevalence_pct": 100 * theta[lvl][:, t].mean(),
                "dominant_in_n_docs": int((docs[lvl].dominant == t + 1).sum()),
            }
        )
        for r, i in enumerate(np.argsort(-theta[lvl][:, t])[:3], start=1):
            iid = c.issue_id.iloc[i]
            rep_rows.append(
                {
                    "level": lvl,
                    "topic": t + 1,
                    "rank": r,
                    "issue_id": iid,
                    "agent": c.agent_type.iloc[i],
                    "theta": theta[lvl][i, t],
                    "title": meta.loc[iid, "title"],
                }
            )
topics = pd.DataFrame(top_rows)
topics.to_csv(TABLES / "rq3_T1_topics.csv", index=False, float_format="%.6g")
pd.DataFrame(rep_rows).to_csv(TABLES / "rq3_T1b_representative_issues.csv", index=False, float_format="%.6g")
fine_topics = topics[topics.level == "fine"].reset_index(drop=True)

# -------------------- Q4 validation sheet, fine view (filled in by the author)
val = []
for t in range(K["fine"]):
    pool = docs["fine"][docs["fine"].dominant == t + 1]
    pick = pool.sample(min(VALIDATION_PER_TOPIC, len(pool)), random_state=SEED)
    for _, r in pick.iterrows():
        body = str(meta.loc[r.issue_id, "body"]) if pd.notna(meta.loc[r.issue_id, "body"]) else ""
        val.append(
            {
                "topic": t + 1,
                "proposed_label": label["fine"][t],
                "top_words": fine_topics.top_words[t],
                "issue_id": r.issue_id,
                "url": meta.loc[r.issue_id, "html_url"],
                "title": meta.loc[r.issue_id, "title"],
                "body_excerpt": " ".join(body.split())[:400],
                "theta": round(r[names["fine"][t]], 3),
                "fits_label (Y/N)": "",
                "notes": "",
            }
        )
vdir = ROOT / "docs" / "validation"
vdir.mkdir(parents=True, exist_ok=True)
pd.DataFrame(val).to_csv(vdir / "V2_topic_labels_sheet.csv", index=False)

# -------------------- M3.3 topic prevalence per agent (Q3 bootstrap CIs)
share_rows = []
for lvl in LEVELS:
    for a in agents:
        sub = docs[lvl][docs[lvl].agent_type == a]
        th = sub[names[lvl]].values
        boots = np.array([th[rng.integers(0, len(th), len(th))].mean(0) for _ in range(N_BOOT)])
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        for t in range(K[lvl]):
            share_rows.append(
                {
                    "level": lvl,
                    "agent": a,
                    "n_docs": len(th),
                    "topic": t + 1,
                    "label": label[lvl][t],
                    "mean_share_pct": 100 * th[:, t].mean(),
                    "ci_lo": 100 * lo[t],
                    "ci_hi": 100 * hi[t],
                    "dominant_pct": 100 * (sub.dominant == t + 1).mean(),
                }
            )
shares = pd.DataFrame(share_rows)
shares.to_csv(TABLES / "rq3_T2_agent_topic_shares.csv", index=False, float_format="%.6g")

# -------------------- sensitivity refit: documents with >= 5 tokens
pre = importlib.import_module("04_rq3_preprocess")
c5, texts5, dict5, corpus5 = build(pre.run(min_tokens=5, write=False))
ens5 = EnsembleLda(
    corpus=corpus5,
    id2word=dict5,
    num_topics=12,
    num_models=16,
    passes=RQ3_PASSES,
    iterations=RQ3_ITERATIONS,
    random_state=SEED,
    ensemble_workers=1,
    distance_workers=1,
    topic_model_class="lda",
)
lda5 = {}
for lvl in LEVELS:
    ens5.recluster(eps=EPS[lvl], min_samples=None, min_cores=None)
    lda5[lvl] = ens5.generate_gensim_representation()

# -------------------- M3.4 association + sensitivity
flags = issues.set_index("issue_id")[["author_is_bot", "s3_precision"]]
assoc_rows, main_r = [], {}
for lvl in LEVELS:
    dd = docs[lvl].join(flags, on="issue_id")
    main_r[lvl], row = assoc(dd, lvl, "Main: agent × dominant topic")
    s3_r, s3_row = assoc(dd[dd.s3_precision], lvl, "S3: CodAGE precision subset")
    assoc_rows += [row, assoc(dd[~dd.author_is_bot], lvl, "S2: human-authored only")[1], s3_row]
    s3_r["residuals"].round(3).to_csv(TABLES / f"rq3_T3c_adjusted_residuals_S3_{lvl}.csv")
    unstable = [t + 1 for t in range(K[lvl]) if stability[lvl][t] == "not reproduced"]
    if unstable:
        assoc_rows.append(
            assoc(
                dd[~dd.dominant.isin(unstable)], lvl, "Without documents dominated by a non-reproduced topic"
            )[1]
        )
    d5 = c5[["issue_id", "agent_type"]].copy()
    d5["dominant"] = theta_of(lda5[lvl], corpus5).argmax(1) + 1
    assoc_rows.append(assoc(d5, lvl, "Documents with >= 5 tokens (ensemble refit)")[1])
    main_r[lvl]["residuals"].round(3).to_csv(TABLES / f"rq3_T3b_adjusted_residuals_{lvl}.csv")
assoc_tab = pd.DataFrame(assoc_rows)
assoc_tab.to_csv(TABLES / "rq3_T3_association.csv", index=False, float_format="%.6g")

# -------------------- Z1 stability: independent ensemble + >= 5-token refit
stab_rows = []
for lvl in LEVELS:
    ref, other = topword_sets(model[lvl]), topword_sets(lda5[lvl])
    jac = np.array([[len(a & b) / len(a | b) for b in other] for a in ref])
    pairs = dict(zip(*linear_sum_assignment(-jac)))
    for t in range(K[lvl]):
        stab_rows.append(
            {
                "level": lvl,
                "topic": t + 1,
                "label": label[lvl][t],
                "stability": stability[lvl][t],
                "interpreted": interpreted[lvl][t],
                "shared_top10_words_seed7": int(info[lvl].loc[t + 1, "shared_top10_words"]),
                "jaccard_seed7": info[lvl].loc[t + 1, "jaccard_with_seed7_match"],
                "jaccard_5token_refit": jac[t, pairs[t]] if t in pairs else np.nan,
            }
        )
stab = pd.DataFrame(stab_rows)
stab.to_csv(TABLES / "rq3_T4_stability.csv", index=False, float_format="%.6g")

# -------------------- Z2 topics × time-to-close, fine view (RQ2 link)
fd = docs["fine"]
Kf = K["fine"]
tt = fd.join(
    issues.set_index("issue_id")[["is_closed", "ttc_hours", "duration_hours", "event_closed"]], on="issue_id"
)
closed = tt[tt.is_closed]
H, p_kw = kruskal(*[closed.loc[closed.dominant == t + 1, "ttc_hours"].values for t in range(Kf)])
lr = multivariate_logrank_test(tt.duration_hours, tt.dominant, tt.event_closed)
z2 = []
for t in range(Kf):
    sub = tt[tt.dominant == t + 1]
    kmf = KaplanMeierFitter().fit(sub.duration_hours, sub.event_closed)
    z2.append(
        {
            "topic": t + 1,
            "label": label["fine"][t],
            "stability": stability["fine"][t],
            "interpreted": interpreted["fine"][t],
            "n_docs": len(sub),
            "n_closed": int(sub.is_closed.sum()),
            "pct_closed": 100 * sub.is_closed.mean(),
            "median_ttc_closed_h": sub.loc[sub.is_closed, "ttc_hours"].median(),
            "km_median_h": kmf.median_survival_time_,
        }
    )
z2 = pd.DataFrame(z2)
z2.to_csv(TABLES / "rq3_T5_topic_ttc.csv", index=False, float_format="%.6g")
e2 = eps2(H, len(closed))
z2_test = pd.DataFrame(
    [
        {
            "test": "Kruskal–Wallis TTC by dominant topic (closed)",
            "n": len(closed),
            "stat": H,
            "df": Kf - 1,
            "p": p_kw,
            "epsilon2": e2,
            "effect": eps2_label(e2),
        },
        {
            "test": "Log-rank by dominant topic (all, open censored)",
            "n": len(tt),
            "stat": lr.test_statistic,
            "df": Kf - 1,
            "p": lr.p_value,
            "epsilon2": np.nan,
            "effect": "",
        },
    ]
)
z2_test.to_csv(TABLES / "rq3_T5b_topic_ttc_tests.csv", index=False, float_format="%.6g")

# -------------------- Z3 topics × attribution role within agent, fine view
roles = fd.join(issues.set_index("issue_id")[["own_bot_author"]], on="issue_id")
z3 = []
for a in ["CodeRabbit", "Devin"]:
    sub = roles[roles.agent_type == a]
    r = permutation_chi2(np.where(sub.own_bot_author, "own_bot", "other"), sub.dominant.values)
    z3.append(
        {
            "agent": a,
            "own_bot_docs": int(sub.own_bot_author.sum()),
            "other_docs": int((~sub.own_bot_author).sum()),
            "chi2": r["stat"],
            "df": r["df"],
            "p_perm": r["p"],
            "cramers_v": r["v"],
            "effect": v_label(r["v"], r["df_star"]),
        }
    )
    for t in range(Kf):
        z3.append(
            {
                "agent": a,
                "topic": t + 1,
                "label": label["fine"][t],
                "own_bot_mean_share_pct": 100 * sub.loc[sub.own_bot_author, names["fine"][t]].mean(),
                "other_mean_share_pct": 100 * sub.loc[~sub.own_bot_author, names["fine"][t]].mean(),
            }
        )
pd.DataFrame(z3).to_csv(TABLES / "rq3_T6_topic_roles.csv", index=False, float_format="%.6g")

# -------------------- figures (fine view; topics we do not interpret are greyed out and marked)
short = []
for t in range(Kf):
    name = f"T{t + 1}" if label["fine"][t] == f"Topic {t + 1}" else f"T{t + 1} {label['fine'][t]}"
    short.append(name if interpreted["fine"][t] else f"T{t + 1} (not interpreted)")

# R1 topic top words (small multiples of horizontal bars)
fig, axes = ps.plt.subplots(1, Kf, figsize=(2.3 * Kf, 2.6))
fw = []
for t, ax in enumerate(np.atleast_1d(axes)):
    words = model["fine"].show_topic(t, topn=10)[::-1]
    color = ps.SLOTS[0] if interpreted["fine"][t] else ps.MUTED
    ax.barh([w for w, _ in words], [p for _, p in words], color=color, height=0.7)
    ax.set_title(
        f"{short[t]}\n{fine_topics.prevalence_pct[t]:.1f}% of text, "
        f"{fine_topics.shared_top10_words_seed7[t]}/10 top words shared across seeds",
        loc="left",
        fontsize=7.5,
    )
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=6.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fw += [{"topic": t + 1, "word": w, "weight": p} for w, p in words[::-1]]
fig.suptitle(
    f"Fine view: {Kf} Ensemble-LDA topics, top-10 words (grey = not interpreted)",
    x=0.01,
    ha="left",
    fontsize=9.5,
    fontweight="bold",
)
fig.tight_layout()
ps.save(fig, FIGURES / "rq3_R1_topic_words")
pd.DataFrame(fw).to_csv(FIGURE_DATA / "rq3_R1_topic_words.csv", index=False, float_format="%.6g")

# R2 agent × topic heatmap (mean share %)
fine_shares = shares[shares.level == "fine"]
hm = fine_shares.pivot(index="agent", columns="topic", values="mean_share_pct").reindex(agents)
mr = main_r["fine"]
fig, ax = ps.plt.subplots(figsize=(1.6 + 0.9 * Kf, 5.2))
im = ax.imshow(
    hm.values, cmap=ps.SEQ, vmin=0, vmax=max(40, np.ceil(hm.values.max() / 10) * 10), aspect="auto"
)
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
        v = hm.values[i, j]
        ax.text(
            j,
            i,
            f"{v:.0f}",
            ha="center",
            va="center",
            fontsize=6.8,
            color=ps.text_color_on(ps.SEQ(v / im.norm.vmax)),
        )
ax.set_xticks(range(Kf), short, rotation=35, ha="right", fontsize=7)
ax.set_yticks(range(len(agents)), [f"{ps.name(a)} (n = {(fd.agent_type == a).sum()})" for a in agents])
ax.tick_params(length=0, colors=ps.INK_2)
for spine in ax.spines.values():
    spine.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=7)
cb.set_label("mean topic share (%)", fontsize=7, color=ps.INK_2)
ax.set_title("Topic profile of each agent (fine view, mean document–topic share, %)", loc="left")
fig.text(
    0.01,
    -0.03,
    f"χ² (agent × dominant topic) = {mr['stat']:.0f}, permutation p = {mr['p']:.4f}, "
    f"Cramér's V = {mr['v']:.2f} ({v_label(mr['v'], mr['df_star'])}). "
    "95% bootstrap CIs in rq3_T2_agent_topic_shares.csv.",
    fontsize=7,
    color=ps.MUTED,
)
ps.save(fig, FIGURES / "rq3_R2_agent_topic_heatmap")
hm.round(3).to_csv(FIGURE_DATA / "rq3_R2_agent_topic_heatmap.csv")

# R3 topics × time-to-close (KM medians, days, with % closed)
z2s = z2.sort_values("km_median_h")
fig, ax = ps.plt.subplots(figsize=(7.5, 0.35 * Kf + 1.2))
yy = np.arange(len(z2s))[::-1]
vals = np.where(np.isfinite(z2s.km_median_h), z2s.km_median_h / 24, np.nan)
ax.barh(yy, np.nan_to_num(vals), color=ps.SLOTS[0], height=0.6)
for yi, v, pc in zip(yy, vals, z2s.pct_closed):
    text = "median not reached" if np.isnan(v) else f"{v:.1f} d"
    ax.text(
        (0 if np.isnan(v) else v) + 1,
        yi,
        f"{text}  · {pc:.0f}% closed",
        va="center",
        fontsize=7,
        color=ps.INK_2,
    )
ax.set_yticks(yy, [short[t - 1] for t in z2s.topic], fontsize=7.5)
ps.clean_axes(ax)
ax.set_xlabel("Kaplan–Meier median time-to-close (days)")
ax.set_title("Time-to-close by dominant topic (fine view, open issues censored)", loc="left")
fig.text(
    0.01,
    -0.04,
    f"Log-rank χ² = {lr.test_statistic:.1f}, df = {Kf - 1}, p = {lr.p_value:.1e}; "
    f"closed only: Kruskal–Wallis H = {H:.1f}, ε² = {e2:.3f} ({eps2_label(e2)}).",
    fontsize=7,
    color=ps.MUTED,
)
ps.save(fig, FIGURES / "rq3_R3_topic_ttc")
z2s.to_csv(FIGURE_DATA / "rq3_R3_topic_ttc.csv", index=False, float_format="%.6g")

# -------------------- console summary
with pd.option_context("display.width", 250, "display.max_colwidth", 110, "display.max_columns", 20):
    print(f"documents = {len(corpus)}, vocabulary = {len(dictionary)}, eps = {EPS}\n")
    print(
        topics[
            [
                "level",
                "topic",
                "stability",
                "shared_top10_words_seed7",
                "prevalence_pct",
                "dominant_in_n_docs",
                "top_words",
            ]
        ]
        .round(1)
        .to_string(index=False)
    )
    print(
        "\n",
        pd.DataFrame(rep_rows)[["level", "topic", "rank", "agent", "theta", "title"]]
        .round(2)
        .to_string(index=False),
    )
    print("\n", assoc_tab.round(4).to_string(index=False))
    print("\n", stab.round(2).to_string(index=False))
    print("\n", z2.round(1).to_string(index=False))
    print(z2_test.round(4).to_string(index=False))
    print("\n", pd.DataFrame([r for r in z3 if "chi2" in r]).round(4).to_string(index=False))
