#!/usr/bin/env bash
# Reproduce every result of the study, then verify it.
#
#   1. put the course dataset (issues.csv, issue_labels.csv, issue_comments.csv,
#      issue_assignees.csv) in data/raw/
#   2. pip install -r requirements.txt  and  python -m nltk.downloader wordnet omw-1.4
#   3. bash run_all.sh            (about 1-1.5 hours on a laptop; step 8 downloads a
#                                  90 MB embedding model from Hugging Face once)
#
# Use another interpreter with:  PYTHON=/path/to/python bash run_all.sh
# The script stops at the first failing step or check.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"

for f in issues.csv issue_labels.csv; do
    [[ -f "data/raw/$f" ]] || { echo "missing data/raw/$f (the dataset is not redistributed)"; exit 1; }
done

step() { echo; echo "=== $1"; shift; "$@"; }

step "environment: pinned library versions" "$PY" checks/check_00_environment.py

step "1  data cleaning"                  "$PY" src/01_clean.py
step "2  RQ1 issue traits"               "$PY" src/02_rq1.py
step "3  RQ2 time-to-close"              "$PY" src/03_rq2.py
step "4  RQ3 text preprocessing"         "$PY" src/04_rq3_preprocess.py
step "5  RQ3 single-LDA model selection" "$PY" src/05_rq3_select.py
step "6  RQ3 Ensemble LDA"               "$PY" src/06_rq3_ensemble.py
step "7  RQ3 topics and associations"    "$PY" src/07_rq3_topics.py
step "8  RQ3 sentence embeddings"        "$PY" src/08_rq3_embed.py
step "9  RQ3 BERTopic triangulation"     "$PY" src/09_rq3_bertopic.py
step "10 manual validation"              "$PY" src/10_validation.py
step "11 report numbers and tables"      "$PY" src/11_report_assets.py
step "12 manifest with SHA-256"          "$PY" src/12_manifest.py

step "check: cleaning"          "$PY" checks/check_01_clean.py
step "check: RQ1"               "$PY" checks/check_02_rq1.py
step "check: RQ2"               "$PY" checks/check_03_rq2.py
step "check: RQ3"               "$PY" checks/check_04_rq3.py
step "check: claims audit"      "$PY" checks/check_05_claims.py
step "check: Monte Carlo seeds" "$PY" checks/check_06_montecarlo.py
step "check: manual validation" "$PY" checks/check_08_validation.py

echo
echo "All steps and checks passed."
echo "To compare with the published results:  $PY checks/check_07_replication.py <reference results dir>"
echo "To rebuild the report PDF:  cd report && tectonic main.tex"
