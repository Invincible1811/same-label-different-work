"""Step 9: RQ3 triangulation with an embedding-based topic model (BERTopic; Grootendorst, 2022).

Purpose: test whether the RQ3 conclusions depend on the topic-modelling method. Recent
studies of AI-agent and issue-tracker data use BERTopic (arXiv 2512.24630, 2412.06757), so
agreement between LDA (word co-occurrence) and BERTopic (sentence embeddings) is strong
evidence; disagreement is itself a finding.

Fair-comparison design:
  * same 3,816 documents and the same cleaning as LDA (04_rq3_preprocess.py);
  * document embeddings from 08_rq3_embed.py (whole-document, chunked mean pooling);
  * topic words from c-TF-IDF over the SAME lemmatised tokens and vocabulary LDA uses,
    so top words are comparable;
  * clustering with k-means and k = number of fine-view LDA topics (same granularity, no outlier topic);
  * three seeds (UMAP + k-means) and the same stability measures as for LDA.
Compares with the fine-view Ensemble-LDA assignments from 07_rq3_topics.py.

Reads data/processed/rq3_corpus.csv, rq3_doc_topics.csv, rq3_embeddings_chunked.npy and
results/tables/rq3_T1_topics.csv (run 07 and 08 first).
Run:  python src/09_rq3_bertopic.py
"""
import os

# PyTorch (loaded by BERTopic) and scikit-learn each bundle an OpenMP runtime; on macOS their
# thread pools collide and the process ends in a segmentation fault. One OpenMP thread avoids
# this and also fixes the order of floating-point operations. Must be set before the imports.
os.environ["OMP_NUM_THREADS"] = "1"

import logging  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from bertopic import BERTopic  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer  # noqa: E402
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score  # noqa: E402
from umap import UMAP  # noqa: E402

import plotstyle as ps  # noqa: E402
from config import FIGURE_DATA, FIGURES, PROCESSED, SEED, TABLES  # noqa: E402
from rq3_common import build  # noqa: E402
from stats_utils import permutation_chi2, v_label  # noqa: E402

logging.disable(logging.WARNING)
ps.apply()
SEEDS = (SEED, SEED + 1, SEED + 2)

c = pd.read_csv(PROCESSED / "rq3_corpus.csv")
lda_docs = pd.read_csv(PROCESSED / "rq3_doc_topics.csv")
assert c.issue_id.equals(lda_docs.issue_id), "run 07_rq3_topics.py on the same corpus first"
K = sum(col.startswith("T") and col[1:].isdigit() for col in lda_docs.columns)  # fine-view topics
emb = np.load(PROCESSED / "rq3_embeddings_chunked.npy")
assert emb.shape[0] == len(c), "run 08_rq3_embed.py on the same corpus first"

docs = c.tokens.tolist()  # c-TF-IDF over the same tokens as LDA
# top words are ranked only among the LDA vocabulary (rq3_common.build: term in >= 10 documents and
# <= 50% of documents). c-TF-IDF weights are per word and each topic row is scaled by one constant, so
# this ranking equals that of a vocabulary-only model. (A fixed vocabulary inside BERTopic fails: its
# representative-document step refits c-TF-IDF on a sample in which some words are absent.)
LDA_VOCABULARY = set(build(c)[2].token2id)


def top_words(model, t, n=10):
    """Top-n c-TF-IDF words of BERTopic topic t, restricted to the LDA vocabulary."""
    terms = model.vectorizer_model.get_feature_names_out()
    row = model.c_tf_idf_[t + model._outliers].toarray().ravel()  # same row get_topic(t) reads
    ranked = [terms[j] for j in np.argsort(-row) if terms[j] in LDA_VOCABULARY]
    return ranked[:n]


def run(seed):
    model = BERTopic(
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=seed),
        hdbscan_model=KMeans(n_clusters=K, random_state=seed, n_init=10),
        vectorizer_model=CountVectorizer(token_pattern=r"\S+", lowercase=False),
        calculate_probabilities=False,
    )
    topics, _ = model.fit_transform(docs, embeddings=emb)
    words = [set(top_words(model, t)) for t in range(K)]
    return np.asarray(topics), words, model


runs = {s: run(s) for s in SEEDS}
ref_topics, ref_words, ref_model = runs[SEED]


def match(a, b):
    J = np.array([[len(x & y) / len(x | y) for y in b] for x in a])
    r, cc = linear_sum_assignment(-J)
    return J, r, cc


# -------------------- stability of BERTopic across seeds
stab = []
for s in SEEDS[1:]:
    J, r, cc = match(ref_words, runs[s][1])
    stab.append(
        {
            "comparison": f"seed {SEED} vs {s}",
            "mean_jaccard_top10": J[r, cc].mean(),
            "topics_reproduced_J_ge_0.5": int((J[r, cc] >= 0.5).sum()),
            "ARI_assignments": adjusted_rand_score(ref_topics, runs[s][0]),
        }
    )
stab = pd.DataFrame(stab)

# -------------------- topics of the reference BERTopic run
lda_words = pd.read_csv(TABLES / "rq3_T1_topics.csv").query("level == 'fine'").reset_index(drop=True)
lda_sets = [set(w.strip() for w in x.split(",")) for x in lda_words.top_words]
J, r, cc = match(lda_sets, ref_words)  # LDA topic i  <->  BERTopic topic cc
bt_rows = []
for t in range(K):
    lda_match = int(r[list(cc).index(t)]) + 1 if t in cc else np.nan
    bt_rows.append(
        {
            "bertopic_topic": t,
            "size": int((ref_topics == t).sum()),
            "top_words": ", ".join(top_words(ref_model, t)),
            "matched_lda_topic": lda_match,
            "jaccard_with_matched_lda_topic": J[lda_match - 1, t] if lda_match == lda_match else np.nan,
        }
    )
pd.DataFrame(bt_rows).to_csv(TABLES / "rq3_T7_bertopic_topics.csv", index=False, float_format="%.6g")

# -------------------- triangulation with Ensemble LDA
lda_dom = lda_docs.dominant.values - 1
mapping = {int(cc[i]): int(r[i]) for i in range(len(r))}  # BERTopic topic -> matched LDA topic
bt_mapped = np.array([mapping[t] for t in ref_topics])
agents = c.agent_type.values
r_lda = permutation_chi2(agents, lda_dom)
tri = {
    "documents": len(c),
    "topics": K,
    "word_overlap_mean_jaccard_matched": J[r, cc].mean(),
    "ARI_LDA_vs_BERTopic": adjusted_rand_score(lda_dom, ref_topics),
    "AMI_LDA_vs_BERTopic": adjusted_mutual_info_score(lda_dom, ref_topics),
    "share_docs_same_matched_topic": (lda_dom == bt_mapped).mean(),
    "cramers_v_agent_x_topic_LDA": r_lda["v"],
    "effect_LDA": v_label(r_lda["v"], r_lda["df_star"]),
}
v_bt = []
for s in SEEDS:
    rb = permutation_chi2(agents, runs[s][0], n_perm=2000)
    v_bt.append(rb["v"])
    if s == SEED:
        r_bt = permutation_chi2(agents, bt_mapped)
tri.update(
    {
        "cramers_v_agent_x_topic_BERTopic": r_bt["v"],
        "effect_BERTopic": v_label(r_bt["v"], r_bt["df_star"]),
        "p_perm_BERTopic": r_bt["p"],
        "cramers_v_BERTopic_min_over_seeds": min(v_bt),
        "cramers_v_BERTopic_max_over_seeds": max(v_bt),
    }
)
# agent profiles: do the same agents over/under-use the same (matched) topics?
res_lda = r_lda["residuals"].values.ravel()
res_bt = (
    r_bt["residuals"]
    .reindex(index=r_lda["residuals"].index, columns=r_lda["residuals"].columns)
    .values.ravel()
)
tri["spearman_agent_topic_residuals_LDA_vs_BERTopic"] = pd.Series(res_lda).corr(
    pd.Series(res_bt), method="spearman"
)
tri_tab = pd.DataFrame([tri])
tri_tab.to_csv(TABLES / "rq3_T8_triangulation.csv", index=False, float_format="%.6g")
stab.to_csv(TABLES / "rq3_T8b_bertopic_stability.csv", index=False, float_format="%.6g")
agree = pd.DataFrame(
    {"agent_type": agents, "lda_topic": lda_dom + 1, "bertopic_matched_lda_topic": bt_mapped + 1}
)
agree.groupby("agent_type").apply(lambda g: (g.lda_topic == g.bertopic_matched_lda_topic).mean()).rename(
    "share_same_topic"
).to_csv(TABLES / "rq3_T8c_agreement_by_agent.csv", float_format="%.6g")

# BERTopic's own topics (no forced matching to LDA): document assignments and agent residuals,
# to check whether the agent-level patterns found with LDA recur in BERTopic's own structure
pd.DataFrame({"issue_id": c.issue_id, "agent_type": agents, "bertopic_topic": ref_topics}).to_csv(
    PROCESSED / "rq3_bertopic_doc_topics.csv", index=False
)
r_own = permutation_chi2(agents, ref_topics)
r_own["residuals"].round(3).to_csv(TABLES / "rq3_T8d_bertopic_adjusted_residuals.csv")

# -------------------- figure: LDA × BERTopic contingency (row %)
ct = pd.crosstab(lda_dom + 1, bt_mapped + 1, normalize="index") * 100
fig, ax = ps.plt.subplots(figsize=(5.6, 4.4))
im = ax.imshow(ct.values, cmap=ps.SEQ, vmin=0, vmax=100)
for i in range(ct.shape[0]):
    for j in range(ct.shape[1]):
        v = ct.values[i, j]
        ax.text(
            j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5, color=ps.text_color_on(ps.SEQ(v / 100))
        )
ax.set_xticks(range(ct.shape[1]), [f"T{x}" for x in ct.columns])
ax.set_yticks(range(ct.shape[0]), [f"T{x}" for x in ct.index])
ax.set_xlabel("BERTopic topic (matched to LDA topic by top words)")
ax.set_ylabel("Ensemble-LDA topic")
for sp_ in ax.spines.values():
    sp_.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=7)
cb.set_label("% of the LDA topic's documents", fontsize=7, color=ps.INK_2)
ax.set_title("Triangulation: where LDA topics land in BERTopic", loc="left")
fig.text(
    0.01,
    -0.03,
    f"ARI = {tri['ARI_LDA_vs_BERTopic']:.2f}, AMI = {tri['AMI_LDA_vs_BERTopic']:.2f}; "
    f"Cramér's V (agent × topic): LDA {tri['cramers_v_agent_x_topic_LDA']:.2f}, BERTopic "
    f"{tri['cramers_v_agent_x_topic_BERTopic']:.2f}.",
    fontsize=7,
    color=ps.MUTED,
)
ps.save(fig, FIGURES / "rq3_R4_triangulation")
ct.round(3).to_csv(FIGURE_DATA / "rq3_R4_triangulation.csv")

with pd.option_context("display.width", 250, "display.max_colwidth", 100):
    print(pd.DataFrame(bt_rows).round(2).to_string(index=False))
    print("\nBERTopic stability across seeds:\n", stab.round(3).to_string(index=False))
    print("\nTriangulation:\n", tri_tab.T.round(3).to_string(header=False))
