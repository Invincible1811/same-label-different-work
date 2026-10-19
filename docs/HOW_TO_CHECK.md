# How to check the results

This guide describes four ways to verify the study, from a two-minute automatic check to a
careful reading of the reasoning. All commands run from the repository folder, with the
Python environment described in the README.

## 1. Automatic checks (about 2 minutes)

Each check script recomputes key results with code that is independent of the pipeline
(Python's `csv` module instead of pandas, textbook formulas, hand-written tests) and prints
PASS or FAIL for every item.

| Command | Expected last line | What it recomputes |
|---|---|---|
| `python checks/check_00_environment.py` | `24/24 checks passed` | Every library is at the version pinned in `requirements.txt` |
| `python checks/check_01_clean.py` | `14/14 checks passed` | The cleaning counts, read from the raw CSV with Python's `csv` module |
| `python checks/check_02_rq1.py` | `54/54 checks passed` | RQ1 rates, Wilson intervals, χ² and Cramér's V |
| `python checks/check_03_rq2.py` | `64/64 checks passed` | Each RQ2 test from its definition: Kruskal–Wallis with tie correction, Dunn with Holm, Cliff's δ over all pairs, a hand-written Kaplan–Meier estimator, Mann–Whitney U |
| `python checks/check_04_rq3.py` | `35/35 checks passed` | The RQ3 corpus rules, vocabulary size, both topic-level rules, shared topic words, reproducible topic assignment, χ², V and residuals, and the adjusted Rand index (about 1 minute) |
| `python checks/check_05_claims.py` | `248/248 claims verified` | Every number typed by hand in the documentation, so a written figure cannot drift from the data |
| `python checks/check_06_montecarlo.py` | `5/5 checks passed` | Permutation and bootstrap results do not depend on the random seed |
| `python checks/check_08_validation.py` | `32/32 checks passed` | The manual-validation scores, recomputed with a hand-written Wilson interval |
| `python checks/check_09_references.py` | `53/53 references verified` | Every cited work exists, checked against Crossref, arXiv or its official page (needs internet) |

The report contains no hand-typed numbers: `src/11_report_assets.py` writes every number and
table in it from the result files, to `report/generated/`.

## 2. Reproduce everything (about 1 to 1.5 hours)

```bash
bash run_all.sh
git diff --stat results/
```

`git diff` should show no changes. The pipeline is deterministic, so every table and
figure-data file comes out byte-identical on the same machine, and `results/MANIFEST.csv`
lists each file with its SHA-256 fingerprint for comparison. Running only `src/01_clean.py`,
`src/02_rq1.py` and `src/03_rq2.py` gives a quick partial check in under a minute.

## 3. Check by hand in a spreadsheet (20 to 30 minutes)

Open `data/raw/issues.csv`, the original file, and compare it with the results.

| What to check | How | Expected |
|---|---|---|
| Raw size | Count the rows | 5,000 |
| Pull requests removed | Filter `is_pull_request` = TRUE | 18 rows |
| CodeRabbit before cleaning | Filter `agent_type` = CodeRabbit | 1,960, and 1,942 after cleaning (`results/tables/dataset_summary_by_agent.csv`) |
| A removed duplicate is real | Take `snakemake/snakemake#3120` from `results/tables/dropped_duplicates.csv` and search the raw file for its title | Two issues with the same repository, title and body; the lower number (#3119) is kept |
| Closure rate of Amazon Q | Filter `agent_type` = Amazon_Q and `is_pull_request` = FALSE, then count `api_state` = closed | 39 of 108 rows. One of the 108 is a duplicate removed in cleaning, so the result table reports 39 of 107, or 36.4% (`rq1_T1_agent_profile.csv`, column `closed_pct`) |
| Closure of CodeRabbit's own issues | Filter `agent_type` = CodeRabbit and `author_login` = `coderabbitai[bot]`, then count closed | About 57% closed, against about 75% for CodeRabbit's other issues |
| Sweep's labels | In `data/raw/issue_labels.csv`, filter `label` = sweep | About 450 issues carry the `sweep` label, most of Sweep's labelled issues |
| A figure matches its data | Compare `results/figure_data/rq1_F1_volume_state.csv` with `results/figures/rq1_F1_volume_state.png` | Every bar and percentage matches a row of the CSV |

Counts made by hand can differ by one or two issues, for instance when a filter also catches
one of the 30 removed duplicates; the scripts are exact.

For RQ2, use `data/processed/issues_clean.csv`, which the pipeline creates:

| What to check | How | Expected |
|---|---|---|
| Median time-to-close of Claude Code | Filter `agent_type` = Claude_Code and `rq2_eligible` = True; take the median of `ttc_hours` | 11.1 hours (`rq2_T2_ttc_summary.csv`) |
| Median time-to-close of CodeRabbit | The same for CodeRabbit | 180.3 hours |
| Why closed issues alone mislead | For PR_Agent, count `is_closed` = False against all its issues | 25 of 55 are still open (45.5%), so its Kaplan–Meier median is long although its closed issues close quickly |
| A figure matches its data | Compare `results/figure_data/rq2_R2_cliffs_delta.csv` with `rq2_R2_cliffs_delta.png` | Every cell matches |

For RQ3:

| What to check | How | Expected |
|---|---|---|
| The topics | Open `results/tables/rq3_T1_topics.csv` | Two headline topics and five fine topics; columns `stability` and `interpreted` show which fine topics are used |
| Who writes the code-level issues | In `data/processed/rq3_doc_topics.csv`, filter `dominant` = 4 and count `agent_type` = CodeRabbit | 771 of 1,140 issues (67.6%) |
| Topic share of an agent | In `results/tables/rq3_T2_agent_topic_shares.csv`, level `fine`, agent CodeRabbit, topic 4 | `mean_share_pct` of 41.7 (rounded) |
| Whether the labels fit | Tab `V2_topic_labels` of `docs/validation/VALIDATION_FILLED_BY_AUTHOR.xlsx` | Topics 1 and 4 pass (4 and 5 of 5); topics 3 and 5 fail (2 and 1 of 5) and are not interpreted |
| A figure matches its data | Compare `results/figure_data/rq3_R2_agent_topic_heatmap.csv` with the PNG of the same name | Every cell matches |

## 4. Judge the reasoning (about 30 minutes)

1. Read the cleaning decisions D1 to D15 in `docs/DECISION_LOG.md` (section 4). Is each
   removal justified?
2. Check that the row counts in `results/tables/cleaning_log.csv` add up:
   5,000 − 18 − 30 = 4,952.
3. In `rq1_FA_own_bot_closure.csv`, the age check addresses the obvious objection that
   bot-written issues are simply newer. They are in fact older and are still closed less often.
4. In `rq1_FB_labelling_illusion.csv`, the `sweep` label acts as a trigger rather than a
   triage label. Opening three Sweep issues on GitHub (column `html_url`) shows this directly.
5. Open three CodeRabbit issues written by `coderabbitai[bot]`: they are follow-up issues
   created from a pull-request review.
6. Single LDA runs were not reproducible, so the topics come from Ensemble LDA at two levels.
   The headline rule was fixed in advance; the fine-view rule and the "partly reproduced"
   grade were added afterwards, which the report and the decision log state openly.
7. Compare any remaining concern with the Threats to Validity section of the report.

The README lists every pipeline step and what it produces, and `results/MANIFEST.csv` names
the script behind every output file.
