"""Step 5: RQ3 model selection for LDA (M3.1, extra Q2).

Stage 1: number of topics k = 5..20 with symmetric priors, each fitted with 3 seeds;
         C_v coherence (Röder et al., 2015) averaged over seeds (UMass reported too).
Stage 2: for the 3 best k, tune the document-topic prior α ∈ {symmetric, asymmetric, auto}
         and the topic-word prior η ∈ {symmetric (= 1/k), 0.01, 0.1} (Agrawal et al., 2018: tuned
         LDA is more stable than off-the-shelf settings), seed 42.
         Note: η = "auto" is NOT used. In gensim 4.3.0 the learned η is stored on the model but the
         topics are computed from LdaState.eta, which keeps its initial value, so η = "auto" leaves
         the topics unchanged (verified: identical topic-word matrices). Fixed η values do take effect.
Rule (fixed before looking at topics): the configuration with the highest C_v; if several
are within 0.005 of the best, the one with the smallest k.

Reads data/processed/rq3_corpus.csv (run 04_rq3_preprocess.py first).
Writes results/tables/rq3_model_selection.csv, rq3_selected_model.csv,
results/figure_data/rq3_S1_coherence.csv and results/figures/rq3_S1_coherence.{pdf,png}.
Run:  python src/05_rq3_select.py      (about 10–15 minutes)
"""
import time

import numpy as np
import pandas as pd
from gensim.models import CoherenceModel
from scipy.optimize import linear_sum_assignment

import plotstyle as ps
from config import FIGURE_DATA, FIGURES, RQ3_K_RANGE, SEED, TABLES
from rq3_common import build, fit

ps.apply()
SEEDS = [SEED, SEED + 1, SEED + 2]
TIE = 0.005


def coherence(model, texts, dictionary, corpus):
    cv = CoherenceModel(
        model=model, texts=texts, dictionary=dictionary, coherence="c_v", topn=10, processes=1
    ).get_coherence()
    um = CoherenceModel(
        model=model, corpus=corpus, dictionary=dictionary, coherence="u_mass", topn=10
    ).get_coherence()
    return cv, um


def main():
    c, texts, dictionary, corpus = build()
    print(f"documents {len(corpus)}, vocabulary {len(dictionary)} terms")
    rows, t0 = [], time.time()
    stab_rows = []
    for k in RQ3_K_RANGE:
        models = {}
        for s in SEEDS:
            models[s] = fit(corpus, dictionary, k, s)
            cv, um = coherence(models[s], texts, dictionary, corpus)
            rows.append(
                {
                    "stage": 1,
                    "k": k,
                    "alpha": "symmetric",
                    "eta": "symmetric",
                    "seed": s,
                    "c_v": cv,
                    "u_mass": um,
                }
            )
        # stability (Greene et al., 2014): topics of seed 42 matched to each other seed by top-10-word Jaccard
        ref = [set(w for w, _ in models[SEEDS[0]].show_topic(t, 10)) for t in range(k)]
        js = []
        for s in SEEDS[1:]:
            oth = [set(w for w, _ in models[s].show_topic(t, 10)) for t in range(k)]
            J = np.array([[len(a & b) / len(a | b) for b in oth] for a in ref])
            r, cc = linear_sum_assignment(-J)
            js.extend(J[r, cc])
        js = np.array(js)
        stab_rows.append({"k": k, "mean_jaccard": js.mean(), "share_reproduced_J_ge_0.5": (js >= 0.5).mean()})
        print(
            f"stage 1  k={k:2d}  mean C_v={np.mean([r['c_v'] for r in rows if r['k'] == k]):.4f}  "
            f"stability J={js.mean():.2f}, reproduced={100 * (js >= 0.5).mean():.0f}%  "
            f"({time.time() - t0:.0f}s)"
        )
    pd.DataFrame(stab_rows).to_csv(TABLES / "rq3_single_lda_stability.csv", index=False, float_format="%.6g")
    s1 = pd.DataFrame(rows)
    top_k = s1.groupby("k").c_v.mean().sort_values(ascending=False).index[:3].tolist()
    for k in top_k:
        for alpha in ["symmetric", "asymmetric", "auto"]:
            for eta in ["symmetric", 0.01, 0.1]:
                cv, um = coherence(fit(corpus, dictionary, k, SEED, alpha, eta), texts, dictionary, corpus)
                rows.append(
                    {"stage": 2, "k": k, "alpha": alpha, "eta": eta, "seed": SEED, "c_v": cv, "u_mass": um}
                )
                print(f"stage 2  k={k:2d}  alpha={alpha:10s} eta={str(eta):9s} C_v={cv:.4f}")
    res = pd.DataFrame(rows)
    res.to_csv(TABLES / "rq3_model_selection.csv", index=False, float_format="%.6g")

    # selection rule over stage 2 (stage-1 symmetric/symmetric at seed 42 is part of stage 2 too)
    s2 = res[res.stage == 2]
    best = s2.c_v.max()
    cand = s2[s2.c_v >= best - TIE].sort_values(["k", "c_v"], ascending=[True, False])
    chosen = cand.iloc[0]
    sel = pd.DataFrame(
        [
            {
                "k": int(chosen.k),
                "alpha": chosen.alpha,
                "eta": str(chosen.eta),
                "seed": SEED,
                "c_v": chosen.c_v,
                "best_c_v": best,
                "tie_tolerance": TIE,
                "n_documents": len(corpus),
                "vocabulary": len(dictionary),
                "top3_k_stage1": " ".join(map(str, top_k)),
            }
        ]
    )
    sel.to_csv(TABLES / "rq3_selected_model.csv", index=False, float_format="%.6g")
    print("\nselected:", sel.to_dict("records")[0])

    # figure: coherence vs k (mean and range over seeds), stage-2 best per k marked
    agg = s1.groupby("k").c_v.agg(["mean", "min", "max"]).reset_index()
    agg.to_csv(FIGURE_DATA / "rq3_S1_coherence.csv", index=False, float_format="%.6g")
    fig, ax = ps.plt.subplots(figsize=(7.5, 3.8))
    ax.fill_between(agg.k, agg["min"], agg["max"], color=ps.BLUE_RAMP[0], label="range over 3 seeds")
    ax.plot(
        agg.k,
        agg["mean"],
        color=ps.SLOTS[0],
        linewidth=2,
        marker="o",
        markersize=4,
        label="mean C_v (symmetric priors)",
    )
    b2 = s2.groupby("k").c_v.max()
    ax.scatter(
        b2.index, b2.values, color=ps.SLOTS[1], s=36, zorder=3, label="best after α/η tuning (stage 2)"
    )
    ax.annotate(
        f"chosen: k = {int(chosen.k)}\nα = {chosen.alpha}, η = {chosen.eta}",
        xy=(chosen.k, chosen.c_v),
        xytext=(10, -34),
        textcoords="offset points",
        fontsize=7.5,
        color=ps.INK_2,
        arrowprops=dict(arrowstyle="-", color=ps.MUTED, linewidth=0.7),
    )
    ax.set_xticks(list(RQ3_K_RANGE))
    ax.set_xlabel("number of topics k")
    ax.set_ylabel("C_v coherence")
    ps.clean_axes(ax, grid_axis="y")
    ax.spines["left"].set_visible(True)
    ax.legend(loc="lower right")
    ax.set_title("LDA model selection: topic coherence by number of topics", loc="left")
    ps.save(fig, FIGURES / "rq3_S1_coherence")


if __name__ == "__main__":
    main()
