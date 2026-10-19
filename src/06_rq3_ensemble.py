"""Step 6: RQ3 stable topics with Ensemble LDA (Brigl, 2019; gensim EnsembleLda).

Why: single LDA runs are not reproducible on this corpus. Refitting the selected model
with other seeds reproduces only a minority of topics (05_rq3_select.py, stability columns;
Greene et al., 2014; Agrawal et al., 2018). Ensemble LDA fits many models and keeps only
topics that recur across them.

Design:
  * two independent ensembles (seeds 42 and 7), each of 16 LDA models with 12 topics
    (more than the best single-model k, so recurring topics are not forced together);
  * clustering threshold eps in {0.10, 0.15, ..., 0.35};
  * a topic is "reproduced" if its top-10 words have Jaccard >= 0.5 (>= 7 shared words)
    with its optimally matched topic in the other ensemble.
Two levels are reported (decision log, 2026-10-18):
  * HEADLINE (rule fixed before looking at topics): the eps that maximises the number of
    reproduced topics; ties -> higher mean Jaccard -> smaller eps.
  * FINE VIEW (rule added after the headline gave only 2 topics): the eps with the most
    topics found by both ensembles (min over the two seeds); ties -> smaller eps. Every
    fine-view topic is reported with its cross-seed overlap, and topics that are not
    reproduced are flagged and not interpreted.
The seed-42 ensemble is the final model at both levels.

Reads data/processed/rq3_corpus.csv. Writes data/processed/rq3_ensemble_seed42.model and
rq3_ensemble_seed7.model (not redistributed), results/tables/rq3_ensemble_selection.csv
and rq3_stable_topics.csv.
Run:  python src/06_rq3_ensemble.py      (about 15-20 minutes)
"""
import time

import numpy as np
import pandas as pd
from gensim.models import EnsembleLda
from scipy.optimize import linear_sum_assignment

from config import PROCESSED, RQ3_ITERATIONS, RQ3_PASSES, TABLES
from rq3_common import build

SEEDS = (42, 7)
EPS_GRID = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35)
NUM_MODELS, NUM_TOPICS = 16, 12
J_REPRODUCED = 0.5


def top_words(ensemble, dictionary, n=10):
    return [[dictionary[j] for j in np.argsort(-row)[:n]] for row in ensemble.get_topics()]


def match(a, b):
    """Optimal one-to-one matching of topics a to topics b by top-word Jaccard."""
    jac = np.array([[len(set(x) & set(y)) / len(set(x) | set(y)) for y in b] for x in a])
    rows, cols = linear_sum_assignment(-jac)
    return jac, dict(zip(rows, cols))


def main():
    c, texts, dictionary, corpus = build()
    ens = {}
    for s in SEEDS:
        t0 = time.time()
        ens[s] = EnsembleLda(
            corpus=corpus,
            id2word=dictionary,
            num_topics=NUM_TOPICS,
            num_models=NUM_MODELS,
            passes=RQ3_PASSES,
            iterations=RQ3_ITERATIONS,
            random_state=s,
            ensemble_workers=1,
            distance_workers=1,
            topic_model_class="lda",
        )
        ens[s].save(str(PROCESSED / f"rq3_ensemble_seed{s}.model"))
        print(f"ensemble seed {s}: {NUM_MODELS} models fitted ({time.time() - t0:.0f}s)", flush=True)

    rows, topic_rows = [], []
    for eps in EPS_GRID:
        words = {}
        for s in SEEDS:
            ens[s].recluster(eps=eps, min_samples=None, min_cores=None)
            words[s] = top_words(ens[s], dictionary)
        a, b = words[SEEDS[0]], words[SEEDS[1]]
        jac, pairs = match(a, b) if a and b else (np.empty((0, 0)), {})
        matched = np.array([jac[i, j] for i, j in pairs.items()])
        rows.append(
            {
                "eps": eps,
                "stable_topics_seed42": len(a),
                "stable_topics_seed7": len(b),
                "reproduced_J_ge_0.5": int((matched >= J_REPRODUCED).sum()),
                "mean_jaccard": matched.mean() if len(matched) else np.nan,
            }
        )
        for i, words_i in enumerate(a):
            j = pairs.get(i)
            topic_rows.append(
                {
                    "eps": eps,
                    "topic": i + 1,
                    "top_words": ", ".join(words_i),
                    "seed7_match_top_words": ", ".join(b[j]) if j is not None else "",
                    "shared_top10_words": len(set(words_i) & set(b[j])) if j is not None else 0,
                    "jaccard_with_seed7_match": jac[i, j] if j is not None else 0.0,
                }
            )
    sel = pd.DataFrame(rows)
    sel["min_topics_both_seeds"] = sel[["stable_topics_seed42", "stable_topics_seed7"]].min(axis=1)
    headline = (
        sel.sort_values(["reproduced_J_ge_0.5", "mean_jaccard", "eps"], ascending=[False, False, True])
        .iloc[0]
        .eps
    )
    fine = sel.sort_values(["min_topics_both_seeds", "eps"], ascending=[False, True]).iloc[0].eps
    sel["level"] = np.select([sel.eps == headline, sel.eps == fine], ["headline", "fine"], "")
    sel.to_csv(TABLES / "rq3_ensemble_selection.csv", index=False, float_format="%.6g")

    topics = pd.DataFrame(topic_rows)
    topics = topics[topics.eps.isin([headline, fine])].copy()
    topics.insert(0, "level", np.where(topics.eps == headline, "headline", "fine"))
    topics["reproduced"] = topics.jaccard_with_seed7_match >= J_REPRODUCED
    topics.to_csv(TABLES / "rq3_stable_topics.csv", index=False, float_format="%.6g")

    with pd.option_context("display.width", 250, "display.max_colwidth", 90):
        print(sel.round(3).to_string(index=False))
        print(f"\nheadline eps = {headline}, fine eps = {fine}")
        print(
            topics[["level", "topic", "shared_top10_words", "reproduced", "top_words"]].to_string(index=False)
        )


if __name__ == "__main__":
    main()
