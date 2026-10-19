"""Step 12: write results/MANIFEST.csv: every generated file, the script that produced it,
what it contains, its size (rows x columns for CSVs) and a SHA-256 fingerprint.

A replicator who reruns the pipeline on the same raw CSVs should get identical
fingerprints for every table. The processed data (data/processed/) is not redistributed
(the assignment forbids re-uploading the data), but its fingerprints are listed so a
replicator can confirm they regenerated it exactly. Every file must have a producer and a
description; the script stops if a new file is not documented here.

Run last:  python src/12_manifest.py
"""
import csv
import hashlib
import sys
from pathlib import Path

from config import ROOT

csv.field_size_limit(sys.maxsize)

# (file-name prefix, producing script); the first matching prefix wins, so specific prefixes come first
PRODUCER = [
    ("validation_", "10_validation.py"),
    ("rq3_preprocessing_log", "04_rq3_preprocess.py"),
    ("rq3_template_lines", "04_rq3_preprocess.py"),
    ("rq3_corpus", "04_rq3_preprocess.py"),
    ("rq3_model_selection", "05_rq3_select.py"),
    ("rq3_selected_model", "05_rq3_select.py"),
    ("rq3_single_lda_stability", "05_rq3_select.py"),
    ("rq3_S1_coherence", "05_rq3_select.py"),
    ("rq3_ensemble", "06_rq3_ensemble.py"),
    ("rq3_stable_topics", "06_rq3_ensemble.py"),
    ("rq3_embeddings", "08_rq3_embed.py"),
    ("rq3_T7", "09_rq3_bertopic.py"),
    ("rq3_T8", "09_rq3_bertopic.py"),
    ("rq3_R4", "09_rq3_bertopic.py"),
    ("rq3_bertopic", "09_rq3_bertopic.py"),
    ("rq3_", "07_rq3_topics.py"),
    ("rq2_", "03_rq2.py"),
    ("rq1_", "02_rq1.py"),
    ("", "01_clean.py"),
]
DESCRIPTION = {
    # data/processed (not redistributed)
    "issues_clean.csv": "One row per retained issue with derived fields and flags",
    "labels_clean.csv": "One row per retained issue-label pair with its category (D9)",
    "rq3_corpus.csv": "RQ3 modelled documents: issue, agent, tokens and cleaned prose",
    "rq3_doc_topics.csv": "RQ3 document-topic proportions (fine view) and dominant topics at both levels",
    "rq3_ensemble_seed42.model": "Ensemble LDA, seed 42 (final model; gensim format)",
    "rq3_ensemble_seed7.model": "Ensemble LDA, seed 7 (independent replicate; gensim format)",
    "rq3_embeddings_chunked.npy": "Sentence embeddings of the RQ3 documents (chunked mean pooling)",
    "rq3_bertopic_doc_topics.csv": "BERTopic topic of every RQ3 document (reference seed)",
    # data processing
    "cleaning_log.csv": "Every cleaning step and flag with rows before/after (D1-D15)",
    "dropped_duplicates.csv": "issue_ids removed as re-submitted duplicates (D3)",
    "label_category_map.csv": "Every distinct normalised label, its category (D9) and issue count",
    "release_dates.csv": "Agent public release dates with sources and confidence (D8)",
    "dataset_summary_before_after.csv": "Dataset description before vs after cleaning",
    "dataset_summary_by_agent.csv": "Issues per agent before/after cleaning, closed issues, RQ2 eligibility",
    # RQ1
    "rq1_T1_agent_profile.csv": "RQ1 T1: agent profile (M1.1-M1.6, M1.9, M1.10), Wilson CIs, repo Gini",
    "rq1_T2_label_categories.csv": "RQ1 T2: % of issues per label category per agent (M1.7)",
    "rq1_T2_top_labels.csv": "RQ1 T2: top-5 raw labels per agent",
    "rq1_T3_association_tests.csv": "RQ1 T3: chi-square (permutation p), Cramer's V, Holm (M1.8-M1.10)",
    "rq1_X2_adjusted_residuals.csv": "RQ1 X2: adjusted residuals for every agent x category cell",
    "rq1_X3_monthly_volume.csv": "RQ1 X3: issues created per agent per month (long format)",
    "rq1_FA_own_bot_closure.csv": "RQ1 finding A: own-bot vs other closure, two-proportion test, age check",
    "rq1_FB_labelling_illusion.csv": "RQ1 finding B: any-label vs triage-label rates and ranks",
    # RQ2
    "rq2_T1_distribution_check.csv": "RQ2 M2.3: skewness, Shapiro-Wilk and Lilliefors per agent",
    "rq2_T2_ttc_summary.csv": "RQ2 M2.1-M2.2: TTC median (exact 95% CI), IQR, P90, mean, SD per agent",
    "rq2_T3_pairwise_dunn_cliff.csv": "RQ2 M2.5: all 55 agent pairs, Dunn p (Holm), Cliff's delta",
    "rq2_T4_kaplan_meier.csv": "RQ2 M2.6: Kaplan-Meier median TTC (95% CI), open issues censored",
    "rq2_T5_ttc_by_outcome.csv": "RQ2 M2.7: TTC per agent for completed / not planned / duplicate closures",
    "rq2_T6_tests.csv": "RQ2: Kruskal-Wallis (all, completed, not planned) with epsilon-squared; log-rank",
    "rq2_T7_sensitivity.csv": "RQ2: sensitivity checks S1-S4 (Kruskal-Wallis, epsilon2, rank stability)",
    "rq2_T8_ranking_closed_vs_km.csv": "RQ2: agent ranking by closed-only vs Kaplan-Meier median",
    "rq2_Y1_own_bot_ttc.csv": "RQ2 extra Y1: own-bot vs other TTC within CodeRabbit and Devin",
    # RQ3
    "rq3_preprocessing_log.csv": "RQ3 corpus construction: documents and lines kept/removed at each step",
    "rq3_template_lines.csv": "RQ3 template lines removed (masked line repeated in >= 10 issues)",
    "rq3_model_selection.csv": "RQ3 single LDA: C_v and UMass, k = 5-20 x 3 seeds, then priors",
    "rq3_selected_model.csv": "RQ3 single LDA: selected k and priors",
    "rq3_single_lda_stability.csv": "RQ3 single LDA: topic reproducibility across seeds per k",
    "rq3_ensemble_selection.csv": "RQ3 Ensemble LDA: stable/reproduced topics per eps; both levels",
    "rq3_stable_topics.csv": "RQ3 Ensemble LDA: topics at both levels with overlap with the second ensemble",
    "rq3_T1_topics.csv": "RQ3 M3.2: topics at both levels: label, stability grade, top words, prevalence",
    "rq3_T1b_representative_issues.csv": "RQ3 M3.2: three most representative issues per topic",
    "rq3_T2_agent_topic_shares.csv": "RQ3 M3.3: mean topic share per agent with bootstrap 95% CIs",
    "rq3_T3_association.csv": "RQ3 M3.4: agent x dominant topic chi-square, Cramer's V; S2, S3 and refits",
    "rq3_T3b_adjusted_residuals_headline.csv": "RQ3: adjusted residuals agent x topic, headline level",
    "rq3_T3b_adjusted_residuals_fine.csv": "RQ3: adjusted residuals agent x topic, fine view",
    "rq3_T3c_adjusted_residuals_S3_headline.csv": "RQ3: adjusted residuals on the S3 subset, headline",
    "rq3_T3c_adjusted_residuals_S3_fine.csv": "RQ3: adjusted residuals on the S3 subset, fine view",
    "rq3_T4_stability.csv": "RQ3 Z1: topic overlap with the 2nd ensemble and the >= 5-token refit",
    "rq3_T5_topic_ttc.csv": "RQ3 Z2: TTC per fine topic (closed median, Kaplan-Meier median)",
    "rq3_T5b_topic_ttc_tests.csv": "RQ3 Z2: Kruskal-Wallis and log-rank tests of TTC by topic",
    "rq3_T6_topic_roles.csv": "RQ3 Z3: topic shares by authorship role within CodeRabbit and Devin",
    "rq3_T7_bertopic_topics.csv": "RQ3 BERTopic: topics, sizes, top words, matched LDA topic",
    "rq3_T8_triangulation.csv": "RQ3 BERTopic vs LDA: word overlap, ARI/AMI, Cramer's V for both methods",
    "rq3_T8b_bertopic_stability.csv": "RQ3 BERTopic: stability across seeds",
    "rq3_T8c_agreement_by_agent.csv": "RQ3 BERTopic vs LDA: share in the matched topic per agent",
    "validation_V1_key.csv": "Manual validation V1: our attribution of the 99 sampled issues",
    "validation_V3_key.csv": "Manual validation V3: our language flag for the 40 sampled issues",
    "validation_summary.csv": "Manual validation: author's answers vs our classifications, Wilson 95% CIs",
    "rq3_T8d_bertopic_adjusted_residuals.csv": "RQ3 BERTopic: residuals agent x BERTopic's own topics",
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def shape(p):
    if p.suffix != ".csv":
        return "", ""
    with open(p, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        return sum(1 for _ in r), len(header)


rows, undocumented = [], []
for folder in ["data/processed", "results/tables", "results/figure_data", "results/figures"]:
    for p in sorted((ROOT / folder).glob("*")):
        if p.is_dir() or p.name.startswith("."):
            continue
        producer = next(v for k, v in PRODUCER if p.name.startswith(k))
        if folder == "results/figure_data":
            desc = f"Exact values plotted in figure {p.stem}"
        elif folder == "results/figures":
            desc = f"Figure {p.stem} ({p.suffix[1:]})"
        else:
            desc = DESCRIPTION.get(p.name)
            if desc is None:
                undocumented.append(f"{folder}/{p.name}")
        n_rows, n_cols = shape(p)
        rows.append(
            {
                "path": f"{folder}/{p.name}",
                "produced_by": str(Path("src", producer)).replace("src/../", ""),
                "description": desc,
                "rows": n_rows,
                "columns": n_cols,
                "redistributed": folder != "data/processed",
                "sha256": sha256(p),
            }
        )
if undocumented:
    sys.exit("undocumented files (add them to DESCRIPTION):\n  " + "\n  ".join(undocumented))

with open(ROOT / "results" / "MANIFEST.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"MANIFEST.csv: {len(rows)} files")
