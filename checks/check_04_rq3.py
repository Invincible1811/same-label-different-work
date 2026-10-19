"""Independent check of RQ3 (src/04_rq3_preprocess.py to src/11_report_assets.py).

Re-derives the RQ3 results from their definitions, without the analysis code, and compares
them with results/tables/rq3_*.csv and the report macros (report/generated/numbers.tex):
  * corpus invariants: English issues of the 11 tested agents, >= 10 tokens, no agent names left;
  * vocabulary size from raw document frequencies (>= 10 documents, <= 50% of documents);
  * the two ensemble-level selection rules, re-applied to the selection table;
  * shared top-10 words between the two ensembles, by brute-force optimal matching;
  * document-topic inference is reproducible (same dominant topic for every document);
  * agent x topic chi-square, Cramer's V and adjusted residuals from the contingency table;
  * mean topic shares per agent; CodeRabbit topic shares by authorship role;
  * LDA vs BERTopic adjusted Rand index by pair counting; Cramer's V on BERTopic's own topics.
Uses gensim only to load the saved ensembles and to run topic inference.

Run from the repository root:  python checks/check_04_rq3.py      (about 1 minute)
"""
import itertools
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import EnsembleLda

ROOT = Path(__file__).resolve().parents[1]
T, P = ROOT / "results/tables", ROOT / "data/processed"
SEED = 42
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")


def close(name, expected, actual, tol):
    ok = abs(float(expected) - float(actual)) <= max(tol, 1e-5 * abs(float(actual)))
    check(name, ok, f"expected {float(expected):.6g}, got {float(actual):.6g}")


def chi2_v_residuals(groups, cats):
    """Pearson chi-square, Cramer's V and adjusted standardised residuals from first principles."""
    obs = pd.crosstab(pd.Series(groups, name="g"), pd.Series(cats, name="c"))
    n = obs.values.sum()
    row, col = obs.sum(1).values[:, None], obs.sum(0).values[None, :]
    exp = row * col / n
    chi2 = ((obs.values - exp) ** 2 / exp).sum()
    v = math.sqrt(chi2 / (n * (min(obs.shape) - 1)))
    adj = (obs.values - exp) / np.sqrt(exp * (1 - row / n) * (1 - col / n))
    return chi2, v, pd.DataFrame(adj, index=obs.index, columns=obs.columns)


def ari(a, b):
    """Adjusted Rand index by pair counting (Hubert & Arabie, 1985)."""
    comb = lambda x: x * (x - 1) / 2  # noqa: E731
    ct = pd.crosstab(a, b).values
    s_ij = comb(ct).sum()
    s_a, s_b = comb(ct.sum(1)).sum(), comb(ct.sum(0)).sum()
    expected = s_a * s_b / comb(ct.sum())
    return (s_ij - expected) / ((s_a + s_b) / 2 - expected)


macros = dict(
    re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", (ROOT / "report/generated/numbers.tex").read_text())
)


def macro_num(name):
    return float(macros[name].replace("{,}", "").replace("\\ensuremath{-}", "-"))


# -------------------- corpus invariants
issues = pd.read_csv(P / "issues_clean.csv")
corpus = pd.read_csv(P / "rq3_corpus.csv")
meta = issues.set_index("issue_id")
closed_n = issues[issues.is_closed].groupby("agent_type").size()
tested = set(closed_n[closed_n >= 30].index)
check("corpus: 11 tested agents only", set(corpus.agent_type) == tested and len(tested) == 11)
check("corpus: English issues only", (meta.loc[corpus.issue_id, "lang_group"] == "english").all())
check("corpus: every document has >= 10 tokens", (corpus.tokens.str.split().str.len() >= 10).all())
close("corpus: documents", 3816, len(corpus), 0)
agent_words = {"coderabbit", "devin", "sweep", "aider", "kiro", "jules", "claude", "gemini", "swe_agent"}
leaks = Counter(t for doc in corpus.tokens.str.split() for t in doc if t in agent_words)
check("corpus: no agent-name tokens left", not leaks, str(dict(leaks)) if leaks else "")

texts = corpus.tokens.str.split().tolist()
df = Counter(t for doc in texts for t in set(doc))
vocab = {t for t, k in df.items() if k >= 10 and k / len(texts) <= 0.5}
close("vocabulary size (df >= 10, <= 50%)", vocab.__len__(), macro_num("NVocab"), 0)

# -------------------- ensemble level selection rules
sel = pd.read_csv(T / "rq3_ensemble_selection.csv")
best = max(sel.itertuples(), key=lambda r: (r[4], r.mean_jaccard, -r.eps))  # reproduced, mean J, smaller eps
fine = max(sel.itertuples(), key=lambda r: (min(r.stable_topics_seed42, r.stable_topics_seed7), -r.eps))
check(
    "headline eps = rule fixed in advance",
    sel.loc[sel.level == "headline", "eps"].iloc[0] == best.eps,
    f"eps {best.eps}",
)
check(
    "fine eps = most topics found by both seeds",
    sel.loc[sel.level == "fine", "eps"].iloc[0] == fine.eps,
    f"eps {fine.eps}",
)

# -------------------- shared top words between the two ensembles
dictionary = Dictionary(texts)
dictionary.filter_extremes(no_below=10, no_above=0.5, keep_n=None)
bows = [dictionary.doc2bow(t) for t in texts]
stable = pd.read_csv(T / "rq3_stable_topics.csv")
for level, eps in [("headline", best.eps), ("fine", fine.eps)]:
    tops = {}
    for s in (42, 7):
        e = EnsembleLda.load(str(P / f"rq3_ensemble_seed{s}.model"))
        e.recluster(eps=eps, min_samples=None, min_cores=None)
        tops[s] = [set(dictionary[j] for j in np.argsort(-row)[:10]) for row in e.get_topics()]
    a, b = tops[42], tops[7]
    # brute-force optimal one-to-one matching maximising total Jaccard
    jac = lambda x, y: len(x & y) / len(x | y)  # noqa: E731
    perm = max(
        itertools.permutations(range(len(b)), len(a)),
        key=lambda p: sum(jac(a[i], b[j]) for i, j in enumerate(p)),
    )
    shared = [len(a[i] & b[j]) for i, j in enumerate(perm)]
    stored = stable[stable.level == level].sort_values("topic").shared_top10_words.tolist()
    check(f"{level}: shared top-10 words per topic", shared == stored, f"{shared}")

# -------------------- inference is reproducible
docs = pd.read_csv(P / "rq3_doc_topics.csv")
e = EnsembleLda.load(str(P / "rq3_ensemble_seed42.model"))
e.recluster(eps=fine.eps, min_samples=None, min_cores=None)
lda = e.generate_gensim_representation()
lda.random_state = np.random.RandomState(SEED)
dom = [max(lda.get_document_topics(b, minimum_probability=0.0), key=lambda x: x[1])[0] + 1 for b in bows]
check(
    "fine view: dominant topic of every document reproduced",
    (np.array(dom) == docs.dominant.values).all(),
    f"{(np.array(dom) != docs.dominant.values).sum()} differ",
)

# -------------------- association, residuals, shares
assoc = pd.read_csv(T / "rq3_T3_association.csv")
flags = meta[["author_is_bot", "s3_precision", "own_bot_author"]]
d = docs.join(flags, on="issue_id")
for level, col in [("headline", "dominant_headline"), ("fine", "dominant")]:
    rows = assoc[assoc.level == level].set_index("analysis")
    for tag, label, sub in [
        ("main", "Main: agent × dominant topic", d),
        ("S2", "S2: human-authored only", d[~d.author_is_bot]),
        ("S3", "S3: CodAGE precision subset", d[d.s3_precision]),
    ]:
        chi2, v, adj = chi2_v_residuals(sub.agent_type.values, sub[col].values)
        close(f"{level} {tag}: chi2", chi2, rows.loc[label, "chi2"], 1e-3)
        close(f"{level} {tag}: Cramer's V", v, rows.loc[label, "cramers_v"], 1e-5)
_, _, adj = chi2_v_residuals(d.agent_type.values, d.dominant.values)
res = pd.read_csv(T / "rq3_T3b_adjusted_residuals_fine.csv", index_col=0)
close("fine residuals: max |difference| over all cells", 0, np.abs(adj.values - res.values).max(), 1e-3)

shares = pd.read_csv(T / "rq3_T2_agent_topic_shares.csv")
fs = shares[shares.level == "fine"].set_index(["agent", "topic"]).mean_share_pct
mine = d.groupby("agent_type")[[c for c in d.columns if re.fullmatch(r"T\d+", c)]].mean() * 100
diff = max(abs(mine.loc[a, f"T{t}"] - fs.loc[(a, t)]) for a, t in fs.index)
close("fine mean topic shares: max |difference|", 0, diff, 1e-3)

t1 = pd.read_csv(T / "rq3_T1_topics.csv")
code = int(t1[(t1.level == "fine") & t1.top_words.str.contains("function")].topic.iloc[0])
cr = d[d.agent_type == "CodeRabbit"]
close(
    "CodeRabbit own-bot share of code-level topic (%)",
    cr[cr.own_bot_author][f"T{code}"].mean() * 100,
    macro_num("CodeShareOwnCodeRabbit"),
    0.05,
)
close(
    "CodeRabbit other share of code-level topic (%)",
    cr[~cr.own_bot_author][f"T{code}"].mean() * 100,
    macro_num("CodeShareOtherCodeRabbit"),
    0.05,
)
close(
    "CodeRabbit share of code-level-dominated documents (%)",
    (d[d.dominant == code].agent_type == "CodeRabbit").mean() * 100,
    macro_num("CodeTopicShareCodeRabbit"),
    0.05,
)

# -------------------- BERTopic triangulation
bt = pd.read_csv(P / "rq3_bertopic_doc_topics.csv")
tri = pd.read_csv(T / "rq3_T8_triangulation.csv").iloc[0]
check("BERTopic assignments align with the corpus", bt.issue_id.equals(docs.issue_id))
close(
    "ARI LDA vs BERTopic (pair counting)",
    ari(docs.dominant.values, bt.bertopic_topic.values),
    tri.ARI_LDA_vs_BERTopic,
    1e-5,
)
_, v_bt, adj_bt = chi2_v_residuals(bt.agent_type.values, bt.bertopic_topic.values)
close("Cramer's V on BERTopic's own topics", v_bt, tri.cramers_v_agent_x_topic_BERTopic, 1e-5)
res_bt = pd.read_csv(T / "rq3_T8d_bertopic_adjusted_residuals.csv", index_col=0)
close("BERTopic residuals: max |difference|", 0, np.abs(adj_bt.values - res_bt.values).max(), 1e-3)

# -------------------- report macros agree with the recomputed values
close(
    "macro VTFMain",
    round(chi2_v_residuals(d.agent_type.values, d.dominant.values)[1], 2),
    macro_num("VTFMain"),
    1e-9,
)
close("macro BTV", round(v_bt, 2), macro_num("BTV"), 1e-9)
close("macro ResCodeRabbitCode", round(adj.loc["CodeRabbit", code], 1), macro_num("ResCodeRabbitCode"), 1e-9)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
