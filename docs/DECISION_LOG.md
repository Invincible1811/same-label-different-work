# Decision log

This file records how the study was designed and every decision taken while carrying it out,
in the order it was taken, with the reason for it. It is the audit trail behind the report.

Note on names: the pipeline scripts and the checks were renumbered during the study, so the
early entries below use the names of the time. The current names are listed in the README;
the main changes were that the manifest became `12_manifest.py`, the report numbers
`11_report_assets.py`, the validation step `10_validation.py`, the BERTopic step was split into
`08_rq3_embed.py` and `09_rq3_bertopic.py`, and the checks were renumbered in pipeline order.
This file was previously `docs/PLAN.md`. Numbers quoted in early entries are those of the time;
the current values are in `results/` and in the report.

## 1. First profile of the raw data (2026-10-17, before cleaning)

- 5,000 issues, 39 columns, no duplicate `issue_id`s, 3,477 repos.
- `agent_type` has 13 values: CodeRabbit 1,960, Claude_Code 880, Sweep 696, Devin 547, SWE_agent 310, Gemini_Code_Assist 257, Amazon_Q 108, Aider 68, Kiro 64, PR_Agent 55, Jules 49, Copilot 3 and Cursor 3.
- `api_state`: 3,309 closed / 1,691 open (34% still open, so time-to-close is right-censored).
- `time_to_close_hours` and `closed_at` (GH Archive) are unreliable; recompute from `api_closed_at − api_created_at`. No negative durations.
- `state_final` disagrees with `api_state` for 1,959 issues; the schema says `api_state` is current, so use it.
- 18 rows are pull requests (`is_pull_request = True`).
- 129 issues have an empty body.
- Author type is confounded with agent: CodeRabbit 85% Bot, Devin 86% Bot; Claude_Code, Sweep, Gemini mostly User accounts.
- `api_created_at` spans 2019-03 to 2026-07 (check for agent labels that predate the agent's release; possible misattribution).
- `state_reason`: completed 2,888, not_planned 372, duplicate 34, reopened 34, empty 1,672.

---

## 2. Goal, questions and metrics (GQM, approved 2026-10-17)

Follows the strict GQM goal template (Basili et al., 1994).

**Motivation.** AI coding agents now open issues in thousands of repositories, but maintainers have no evidence on whether the issues from different agents look alike, get resolved equally fast, or concern the same kind of work. Without that evidence, choosing or configuring an agent is guesswork.

**Goal.**
> **Analyze** GitHub issues attributed to AI coding agents,
> **for the purpose of** understanding how these issues differ between agents,
> **with respect to** their characteristics (volume, state, labelling), their resolution time, and their topics,
> **from the viewpoint of** open-source maintainers and software engineering researchers,
> **in the context of** 4,952 public GitHub issues (after cleaning, section 4) from 3,466 repositories, attributed to 13 AI coding agents (2019–2026).

Three RQs. 21 metrics (M1.9, M1.10, M2.7 added 2026-10-17). Each metric is tagged A (atomic) or C (composite).

### RQ1 (descriptive): How do issues attributed to different AI coding agents differ in volume, current state, and labelling practice?
*Why:* maintainers deciding whether to adopt an agent need a baseline of what its issues look like in practice.

| ID | Metric | Type | Definition and source |
|----|--------|------|-----------------------|
| M1.1 | Issue volume | A | Issues per agent (n, % of dataset) |
| M1.2 | Repository spread | C | Distinct repos per agent; issues per repo (median, SD); share held by the top repo |
| M1.3 | Closure rate | C | closed ÷ n per agent (`api_state`) |
| M1.4 | Closure-reason mix | C | % completed / not_planned / duplicate among closed issues (`state_reason`) |
| M1.5 | Labelling rate | C | % issues with `n_labels` ≥ 1 |
| M1.6 | Labels per issue | A | `n_labels` (median, mean, SD) |
| M1.7 | Label category mix | C | % bug / feature / docs / agent-workflow (e.g. `sweep`) / other; top-5 raw labels per agent |
| M1.8 | Association strength | C | χ² test of independence + Cramér's V for agent × state and agent × labelled |
| M1.9 | Language mix | C | % of each agent's issues by language group (English / non-Latin script / Latin non-English), from D13; χ² + Cramér's V for agent × language |
| M1.10 | Attribution-signal mix | C | % of each agent's issues by how the agent is linked: agent's own bot login / `github-actions[bot]` echo / self-referential repo / other signal (D15, CodAGE recipe); χ² + Cramér's V |

### RQ2 (quantitative comparison): Does time-to-close differ across AI coding agents, and how large are the differences?
*Why:* time-to-close is a standard proxy for how quickly maintainers act on an issue, and it tells adopters whether one agent's issues sit idle longer than another's.
*Hypotheses:* H₀: the time-to-close distribution is the same for all agents. H₁: at least one agent's distribution differs.

| ID | Metric | Type | Definition and source |
|----|--------|------|-----------------------|
| M2.1 | Time-to-close (TTC) | A | `api_closed_at − api_created_at`, in hours; closed issues only; agents with ≥ 30 closed issues (11 agents) |
| M2.2 | TTC summary | C | Median (primary, robust to outliers), IQR, mean, SD per agent |
| M2.3 | Distribution check | C | Histograms + Q–Q plots; Shapiro–Wilk per agent (Kolmogorov–Smirnov as a second check); skewness |
| M2.4 | Overall test + effect size | C | Kruskal–Wallis H and p; ε² = H(n+1)/(n²−1) (Tomczak & Tomczak, 2014), interpreted with Cohen's (1988) η² benchmarks: 0.01 small, 0.06 medium, 0.14 large |
| M2.5 | Pairwise differences | C | Dunn's test with Holm-adjusted p; Cliff's δ with Romano et al. (2006) thresholds (0.147 / 0.33 / 0.474) |
| M2.6 | Robustness to open issues | C | Kaplan–Meier median TTC with open issues treated as censored; log-rank test |
| M2.7 | TTC by closure outcome | C | Median/IQR of TTC per agent separately for *completed* and *not planned* closures (as Li et al., 2025 split accepted vs rejected PRs); Kruskal–Wallis within each outcome |

### RQ3 (qualitative, topic modelling): What topics do agent-attributed issues raise, and how does topic prevalence differ across agents?
*Why:* this shows what kind of work each agent is actually used for (bug reports, refactoring, review follow-ups, and so on), which explains RQ1 and RQ2 rather than just restating them.

| ID | Metric | Type | Definition and source |
|----|--------|------|-----------------------|
| M3.1 | Model quality | C | C_v coherence for k = 5 to 20; choose k from coherence plus whether the topics read sensibly |
| M3.2 | Topic description | C | Top-10 terms per topic + a label assigned by hand after reading 3 representative issues (the mixed-methods step) |
| M3.3 | Topic prevalence per agent | C | Mean document–topic proportion per agent (primary); share of issues where the topic is dominant (secondary) |
| M3.4 | Association strength | C | χ² test + Cramér's V for agent × dominant topic |

*Algorithm choice (final):* Ensemble LDA with topics graded by reproducibility, confirmed with BERTopic; single LDA runs proved not reproducible (see the decision table, 2026-10-18).

**Sensitivity checks.** (S1) rerun the key RQ1/RQ2 numbers on issues created after each agent's public release; (S2) rerun them on User-authored issues only, to separate the agent from the bot-versus-human confound; (S3) rerun them on the CodAGE precision subset (`s3_precision`: drop `github-actions[bot]` authors and self-referential repos, 4,593 issues, 3,049 for RQ2). The main analysis keeps `agent_type` as given, as the brief instructs.

---

## 3. Data source
The course dataset appears to be derived from CodAGE (Ghaleb, 2026). This is the author's inference; the brief does not state it. Evidence: the same author, the same `agent_type` scheme and agent names, all 13 agents of this dataset in CodAGE's list of 16, and the latest GH Archive event in the data (2026-04-15 23:57:59) is exactly CodAGE's documented end date. One mismatch: 54 issues have GH Archive events before CodAGE's documented start (2024-01-01). CodAGE (Hugging Face `taher-ghaleb/CodAGE`, CC BY 4.0) was built from GH Archive (2024-01-01 to 2026-04-15); the course data adds live GitHub API fields. CodAGE's own documentation says `agent_type` is a recall-oriented "candidate attribution, not a verified label"; bot-login signals are high-precision, while branch and free-text signals are the main source of false positives. The licence requires crediting both CodAGE and GH Archive.

## 4. Data processing (`src/01_clean.py`)

Every step is logged with row counts in `results/tables/cleaning_log.csv`.
Result: 5,000 issues reduced to 4,952 (3,466 repos; 3,275 closed; 3,269 eligible for RQ2 across 11 agents).

| # | Decision | Outcome |
|---|----------|---------|
| D1 | Raw CSVs never modified; logged pipeline | `issues_clean.csv`, `labels_clean.csv`, `cleaning_log.csv` |
| D2 | Drop pull requests | −18 |
| D3 | Drop re-submitted duplicates: same repo, same title and body after normalising case/whitespace/punctuation (numbers kept), re-posted within 24 h; first copy kept | −30 (IDs in `dropped_duplicates.csv`): 27 exact copies + 3 Sweep re-posts differing only in whitespace. Deep dive (2026-10-18): 24 of 27 exact pairs posted within 44 min, mostly same account; in 4 pairs maintainers closed one copy as duplicate. Number-masking and body-only matching were rejected (they merge dated recurring reports and template-based distinct issues). 34 issues *closed as* duplicate are kept |
| D4 | API fields only; TTC = `api_closed_at − api_created_at` (hours) | archive state/closed_at/time_to_close ignored |
| D5 | No outlier trimming | 47 instant closes (< 1 min) and 119 with TTC > 1 year kept and reported |
| D6 | Open issues kept, censored | 1,677 open (33.9%); 0 closed issues lacked a reason after D2/D3 |
| D7 | Inference only for agents with ≥ 30 closed issues | Copilot and Cursor (3 each) appear in RQ1 only |
| D8 | `agent_type` kept as given; flags `author_is_bot`, `pre_release` | 2,278 bot-authored (46.0%); 10 created before release (`release_dates.csv`, with sources) |
| D9 | Labels lower-cased and mapped to 5 categories (`config.LABEL_RULES`) | label–issue pairs: feature 767, agent_workflow 664, bug 406, docs 157, other 2,734 (`label_category_map.csv`) |
| D10 | `text = title + body`; text cleaning deferred to RQ3 | 128 empty bodies (title kept) |
| D11 | `issue_comments.csv` not used | no Set-5 metric needs it |
| D12 | Censoring date = latest `api_updated_at` | 2026-10-09T17:48:25Z |
| D13 | Language flag (alphabet + function-word heuristic) | 695 non-Latin script (14.0%), 48 Latin non-English (1.0%); kept for RQ1 and RQ2, excluded from the topic models. Spot-check: about 1 in 5 Latin-script flags are English false positives (≈ 0.2% of data) |
| D14 | Renamed repos / transferred issues flagged | 533 renamed, 13 transferred; timestamps verified |
| D15 | Attribution signal (CodAGE precision recipe, steps 1–3) | agent's own bot 2,091 (42.2%), `github-actions[bot]` echo 134 (2.7%), self-referential repo 189 (3.8%), other 2,538 (51.3%); S3 subset = 4,591 |

Not done, on purpose: repo-quality filtering (stars/forks/activity). This needs extra API data the brief does not provide, so it is treated as a threat to external validity.

## 5. Dated decision log
| Date | Decision | Why |
|------|----------|-----|
| 2026-10-17 | Chose RQ Set 5 | Uses the core `agent_type` field; enough data for all three RQs |
| 2026-10-17 | GQM approved: one goal in the slide-12 template, 3 RQs, 18 metrics tagged atomic/composite | Matches Lecture 4 exactly |
| 2026-10-17 | Kruskal–Wallis + ε², Dunn/Holm + Cliff's δ for RQ2 | Skewed TTC; lecture names Kruskal–Wallis and insists on effect size |
| 2026-10-17 | LDA for RQ3 | Lecture: LDA for long text; bodies are long (median ~1,040 chars) |
| 2026-10-17 | Cleaning D1–D14 approved and run (5,000 to 4,955) | See section 2 |
| 2026-10-17 | ε² thresholds changed to Cohen (1988) η² benchmarks | The threshold values could not be verified in Tomczak & Tomczak's text (the formula could); cite only what is verified |
| 2026-10-17 | Added M1.9 language mix to RQ1 | Strong difference found during cleaning (final rule: Devin 33.8% vs Sweep 2.2% non-English) |
| 2026-10-17 | Six additions from the in-depth related-work review approved: M1.10, S3, M2.7, the CodAGE/GH Archive citation, the report structure, the group-size threshold note; cleaning step D15 added | Align with the instructor's own data documentation and papers |
| 2026-10-17 | RQ1 approved with extras X1 (Wilson CIs), X2 (adjusted residuals), X3 (monthly volume), X4 (repo Gini); permutation χ² for sparse tables; df-adjusted Cramér's V benchmarks | Rigor beyond the closest related work, which reports no CIs and does not adjust V interpretation for df |
| 2026-10-17 | RQ1 findings A and B chosen; within-agent two-proportion test (Newcombe CI, Cohen's h, Holm) and age check added for A | Author's choice; age check guards the main confounder |
| 2026-10-17 | D3 tie-break: duplicates with identical timestamps keep the lower issue number (stable sort); found by the independent check (snakemake #3119/#3120); no reported number changed | Determinism |
| 2026-10-17 | Replication: per-figure CSVs, MANIFEST with SHA-256, independent checks (13/13, 54/54), HOW_TO_CHECK guide, before/after dataset summary; PDFs made byte-reproducible | Author's request; replication package rubric (10%) |
| 2026-10-17 | Standing rule: notable facts found during analysis are added as extra statistics, each logged here | Author's request; keeps exploratory additions transparent |
| 2026-10-17 | `agent_type` kept as given; release-date and author-type checks only as sensitivity analyses | Brief says use it as is |
| 2026-10-18 | D3 refined after a deep dive on duplicates (author-approved): 27 to 30 removed, 4,955 to 4,952; all written numbers re-verified by the claims audit | 3 Sweep re-posts differed only in whitespace; looser rules rejected with evidence |
| 2026-10-18 | Claims audit (checks/check_03_claims.py, 147 claims): wording corrected — CodAGE origin stated as inference with evidence; Sweep rank and repo-share statements made precise; AIDev size stated per version | No overstated claims |
| 2026-10-18 | RQ2 built: KW + Dunn/Holm + Cliff's δ + KM/log-rank, by-outcome tests, S1–S4, Y1; S4 (repository clustering) and closed-only vs KM ranking added during the validity review | Independence and selection-bias checks an examiner would ask for |
| 2026-10-18 | RQ2 findings A and B chosen by the author | — |
| 2026-10-18 | LDA η grid changed from {symmetric, auto} to {symmetric, 0.01, 0.1} | In gensim 4.3.0, η = "auto" updates LdaModel.eta but topics are computed from LdaState.eta, which keeps its initial value: topic-word matrices were identical (max difference 0.0). Fixed η values take effect. Reported as a tool limitation |
| 2026-10-18 | Statement audit of RQ2: 5 wordings corrected (smallest groups, median range, 'fastest under both views', 'slowest', causal 'comes from'); statement checks added to the claims audit | Numbers were right; wording overstated |
| 2026-10-18 | Median CIs: percentile bootstrap replaced by the exact order-statistic (binomial) CI | Seed-robustness check showed bootstrap median CIs move with the seed (Aider upper bound 181.7 vs 279.2 h: adjacent order statistics); the exact CI is deterministic with guaranteed ≥ 95% coverage (validated: textbook n = 20 case; 96.4% simulated coverage) |
| 2026-10-18 | Library versions pinned exactly (requirements.txt); seaborn removed (unused); scikit-learn pin corrected to 1.2.2 (the version actually used) | Replicability |
| 2026-10-18 | gensim pinned to 4.3.3 (was 4.3.0) | Clean-room replication failed: gensim 4.3.0 has no pre-built wheel for macOS arm64 / Python 3.11 and does not compile; 4.3.3 has wheels. RQ3 is refitted under 4.3.3 |
| 2026-10-18 | RQ3 preprocessing fixes: (1) multi-word product names ("Q Developer", "Code Assist", …) removed at text level (leakage into LDA tokens and embeddings); (2) BUG FIX: the old `^sweep:` signature rule deleted whole Sweep issue titles, so many Sweep issues fell below 10 tokens — now only the trigger word is removed and titles are kept; (3) signature lines are dropped before product names are stripped (else "Generated with Claude Code" would leave "generate"). Corpus 3,736 to 3,816 documents; Sweep 384 to 468 | Found by checking the BERTopic prose corpus for leakage |
| 2026-10-18 | RQ3 single-model selection on the corrected corpus (3,816 documents): k = 8, symmetric α, η = 0.1, C_v = 0.524 | C_v is flat across k (0.47–0.51 in stage 1). The k = 8 model reproduces only 12% of its topics with other seeds, and k = 17 (highest stage-1 C_v) only 3%, so single LDA runs are not used for findings |
| 2026-10-18 | Ensemble LDA under the pre-set rule (max topics reproduced across seeds 42/7, J ≥ 0.5 = ≥ 7 of 10 shared words): eps = 0.15, 2 reproduced topics | Rule fixed before looking at topics; result kept as the headline |
| 2026-10-18 | Diagnostic (scratch, not in pipeline): at eps 0.20 both seeds find 5 topics; shared top-10 words with the seed-7 match 8, 8, 6, 6, 1; at eps 0.25 a UI topic appears in both seeds (9/10). Document-level ARI between seeds 0.70 (2 topics), 0.37 (eps 0.20), 0.45 (eps 0.25) | The strict word rule undercounts near-identical themes; documented, rule not changed |
| 2026-10-18 | Author's decision (option A): two levels. Headline = 2 reproduced topics (pre-set rule). Fine view = eps with the most topics found by both seeds (the minimum over the two seeds; ties go to the smaller eps), which is eps 0.20 with 5 topics; every fine topic reported with its cross-seed overlap; non-reproduced topics flagged and not interpreted; extra association row without their documents; BERTopic triangulates the fine view | The fine-view rule was defined AFTER the headline result; this is stated in the report. Rejected: loosening J to ≥ 6 words post hoc (tuning to the data); larger ensembles (option B, ~1 h, outcome uncertain) |
| 2026-10-18 | Ensemble rerun with two-level 06 reproduced the diagnostic exactly (headline eps 0.15: 2 topics, 8 and 9 shared words; fine eps 0.20: 5 topics, 8, 1, 6, 8, 6 shared words) | Determinism confirmed |
| 2026-10-18 | Fine-view topics graded by cross-seed overlap: reproduced (≥ 7 of 10 shared words, pre-set rule), partly reproduced (5–6, at least half), not reproduced (< 5; flagged, not interpreted). Result: T1 and T4 reproduced, T3 and T5 partly, T2 not | Under the strict rule alone 3 of 5 fine topics would be flagged, although T3/T5 recur with 6 shared words; the "at least half" cut is new and stated as such |
| 2026-10-18 | BERTopic step split into 08_rq3_embed.py (embeddings) and 09_rq3_bertopic.py; 09 runs with OMP_NUM_THREADS=1; report assets became step 10 and the manifest step 11 | Segmentation faults traced by isolated tests: PyTorch (loaded by BERTopic itself) and scikit-learn/numba OpenMP runtimes collide on macOS; with one OpenMP thread, 3/3 runs succeed with identical clusters. Removed the arXiv 2607.21832 citation from the script header (its abstract does not confirm BERTopic) |
| 2026-10-18 | RQ3 extra: adjusted residuals on the S3 precision subset saved (rq3_T3c) | Checks that the agent-as-subject pattern (Aider, SWE-agent, Kiro, Amazon Q, Gemini over-represented in the LLM/agent topic) is not driven by self-referential repositories: it holds (z = +5.4 to +11.7) |
| 2026-10-18 | BERTopic topic words ranked within the LDA vocabulary (c-TF-IDF over all tokens; listing restricted to LDA terms); helper verified against BERTopic's get_topic() (a mapper-based version read the wrong topic and was rejected) | A fixed vocabulary inside BERTopic fails (its representative-document step refits c-TF-IDF on a sample with absent words); restricting the ranking gives the same order as a vocabulary-only model |
| 2026-10-18 | BERTopic own-topic residuals saved (rq3_T8d) and document assignments (rq3_bertopic_doc_topics.csv) | Forced LDA↔BERTopic matching pairs unrelated topics (one match has J = 0); patterns must be checked on BERTopic's own topics |
| 2026-10-18 | Ensemble inference seeded: gensim's generate_gensim_representation() builds the LdaModel without random_state, so document–topic inference drew from numpy's unseeded global generator; one sensitivity row changed between runs (χ² 144.19 vs 143.49). Fixed by a fixed RandomState before inference; RQ3 rerun; second run compared by SHA-256 | Every number must be reproducible |
| 2026-10-18 | Replication-package cleanup (author-approved): one entry point `run_all.sh`; checks renumbered in pipeline order (00 environment, 01 clean, 02 RQ1, 03 RQ2, 04 RQ3, 05 claims, 06 Monte Carlo, 07 replication; earlier entries in this log use the old numbers); working notes moved to a separate folder (kept private since 2026-10-18); course handouts moved out of the shared tree (`docs/course_material/`, git-ignored); code formatted with black (line length 110) and checked with pyflakes and pycodestyle (no logic change: black verifies the syntax tree); figure labels use readable agent names (`plotstyle.name`); manifest rewritten (all 98 files documented; stale "4,955" removed); README and HOW_TO_CHECK rewritten | Clear package for the replicator. Proof of no change: SHA-256 of all 66 result/processed CSVs before vs after a full `run_all.sh` rerun |
| 2026-10-18 | Two numbers in HOW_TO_CHECK caught before release: CodeRabbit topic-4 share written as 41.6 then 41.8; true 41.7465 to 41.7 (added to the claims audit, now 248 claims) | Every hand-written number must be audited |
| 2026-10-18 | Manual validation protocol (author's choice: 9 issues per agent): V1 attribution, 99 issues stratified by tested agent (≈ ±10 points at 95% confidence overall); V2 topic labels, 25 issues; V3 language flag, 40 issues (20 flagged non-English, 20 English). Samples drawn by the validation script (seed 42); blind sheets (our classification kept in `results/tables/validation_*_key.csv`); coding rules in `docs/validation/README.md`; scored by `checks/check_08_validation.py` (Wilson CIs) | Heuristic labels must be validated by hand (CodAGE: `agent_type` is a candidate attribution). One rater (the author); the professor can re-rate the saved sheets (`check_08_validation.py <folder>` scores another rater's copy). Scorer tested on simulated answers in a scratch copy |
| 2026-10-18 | Manual-validation answers recorded in one Excel workbook (`docs/validation/VALIDATION_FILLED_BY_AUTHOR.xlsx`: instructions tab, one tab per check, drop-down answer codes, clickable links), created once and never overwritten; blank samples still published as CSV; scorer reads the workbook (tested on a scratch copy) | Plain CSV opened in Excel can garble non-Latin titles and change format on save; drop-downs prevent invalid codes |
| 2026-10-18 | Cited literature: 23 open-access PDFs (arXiv, ACL Anthology, JMLR, authors' pages; 47 MB) saved locally for reference (not redistributed: third-party copyright); 32 works without a legitimate free copy listed with their identifiers | Author's request; copies are for reference and are not redistributed |
| 2026-10-18 | Full `run_all.sh` rerun after the cleanup: every step and check passed (14/14, 54/54, 64/64, 35/35, 248/248, 5/5); all 66 result and processed CSVs byte-identical to the pre-cleanup baseline | Proof that the cleanup changed no result |
| 2026-10-18 | Validation sampling wired into the pipeline as step 11; manifest renumbered to step 12; `check_08_validation.py` added to `run_all.sh` | Run only after the rerun finished (bash reads a running script incrementally) |
| 2026-10-18 | Observation while preparing a worked example for the V1 coding rules: V1 item 1 (Jules) was opened by the account `jules-fleet` (a bot), which is not in our own-bot login list (`config.OWN_BOT_LOGINS`), so D15 classifies it as "other signal". Not changed now: the blind validation will measure how often our own-bot flag misses agent-written issues, and any correction is decided afterwards and reported | Changing the classification during the blind rating would bias it |
| 2026-10-18 | Validation workbook rebuilt with a clearer Instructions tab (purpose, 6 steps, where to look on GitHub, code tables with meaning and how to recognise, worked example = V1 row 1, illustrative examples, tie rule W > T > M) and a help box on every answer cell; `--rebuild-empty-workbook` refuses if any answer is saved. Samples and keys byte-identical | Author's request; the first tab must explain the task on its own |
| 2026-10-18 | Manual validation completed by the author (164 judgements, blind). V1: 85.9% of attributions correct (85/99, CI 77.7–91.4); Aider 5/9 lowest; 11 of 14 errors from text signals ("Raider" read as Aider, "sweep" as an ordinary word, paper digests); own-bot flag precision 10/10, recall 10/25 (misses "Generated with Claude Code" footers, linear[bot] for Devin, jules-fleet); roles among correct: W 29.4%, T 20.0%, M 50.6%. V2: labels of T1 4/5 and T4 5/5 fit; T3 2/5 and T5 1/5 fail. V3: 19/20 non-English and 20/20 English flags correct | Scored by `src/10_validation.py`, verified by `check_08_validation.py` (32/32) |
| 2026-10-18 | Author's decisions after validation: (1) attribution analysis unchanged; precision and recall of the own-bot flag reported, and RQ1-A described as "issues opened by the agent's own bot account vs all other issues of the agent"; (2) RQ3 interprets only fine topics that are reproduced AND whose label passed validation (pre-set rule: a label fails at 2 or more of 5 "does not fit"), that is T1 and T4 (`config.RQ3_INTERPRETED`); T3, T5 not interpreted; the "smaller agents lean to LLM tooling" claim dropped (it rested on T3) | Tuning the attribution rules after seeing the answers would bias the validation; the label rule was fixed before the validation |
| 2026-10-18 | Pipeline order: validation scoring moved into the pipeline (`src/10_validation.py`), report numbers to step 11 (`11_report_assets.py`, now with validation macros), manifest step 12; `check_08` re-computes the scores independently | The report needs the validation results before its numbers are generated |
| 2026-10-18 | Reference verification: all 53 cited references confirmed real (44 against Crossref/arXiv records, 4 via official pages, 5 by hand where sites block scripts), title, first author and year compared; re-runnable `checks/check_09_references.py`. Fixes: SWE-bench arXiv note added (ICLR has no DOI); earlier Wessel 2018 title fix confirmed by its DOI | Author's request: no reference may be fabricated |
| 2026-10-18 | References folder for the instructor (outside the repository, shared privately): 27 open-access PDFs plus the CodAGE dataset card, the verification table, and an index with the DOI or link of every reference; 3 open-access papers must be saved from the publisher (scripted downloads blocked), 22 books/paywalled papers via library | Third-party copyright: copies for checking, sent privately, not in the public package |
| 2026-10-18 | Final full run of `run_all.sh`: every step and check passed (cleaning 14/14, RQ1 54/54, RQ2 64/64, RQ3 35/35, claims 248/248, Monte Carlo 5/5, validation 32/32). Against the pre-cleanup baseline, 62 result and processed files are byte-identical; the other 4 (RQ3 topic tables) differ only by the added `interpreted` column, with every number identical | Proof that the late changes altered no result |
| 2026-10-18 | Package tidied for sharing: working notes (study plan, lecture notes, related-work notes, study guide, to-do list) moved to a private, git-ignored folder; this decision log kept as the record; an Excel lock file removed; the unused LaTeX copy of RQ1 table T1 no longer written; two uncited references removed; stale comments fixed; reference check added | The shared package should hold only what a reader needs |
| 2026-10-18 | Report front matter: title extended to "Same Label, Different Work: Roles, Resolution Time and Topics of GitHub Issues Attributed to Thirteen AI Coding Agents"; student ID removed (the paper will be public); course named in a footnote; Times-matched math font; generative-AI disclosure worded by the author and checked for accuracy, and repeated in the README | Author's decisions |
