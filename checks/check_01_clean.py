"""Independent check of the cleaning step (src/01_clean.py).

Re-derives the key numbers straight from data/raw/issues.csv with Python's csv
module (no pandas, none of the pipeline's code) and compares them with
data/processed/issues_clean.csv and results/tables/cleaning_log.csv.

Run from the repository root:  python checks/check_01_clean.py
Exit code 0 = all checks pass.
"""
import csv
import re
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
csv.field_size_limit(sys.maxsize)
results = []


def check(name, expected, actual):
    ok = expected == actual
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: expected {expected}, got {actual}")


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


raw = list(csv.DictReader(open(ROOT / "data/raw/issues.csv", newline="", encoding="utf-8")))
clean = list(csv.DictReader(open(ROOT / "data/processed/issues_clean.csv", newline="", encoding="utf-8")))
log = {r["step"]: r for r in csv.DictReader(open(ROOT / "results/tables/cleaning_log.csv", newline=""))}

# 1. pull requests removed
no_pr = [r for r in raw if r["is_pull_request"].strip().lower() != "true"]
check("rows after dropping pull requests", len(no_pr), int(log["1 drop pull requests"]["rows_after"]))


# 2. re-submitted duplicates: same repo + title + body after normalising case, whitespace and
#    punctuation, re-posted within 24 hours of the first copy (first copy kept)
def norm(x):
    return " ".join(re.sub(r"[^\w\s]", " ", x.lower()).split())


first, kept = {}, []
for r in sorted(no_pr, key=lambda r: (r["api_created_at"], int(r["issue_number"]))):
    key = (r["repo_name"], norm(r["title"]), norm(r["body"]))
    t = ts(r["api_created_at"])
    if key in first and (t - first[key]).total_seconds() <= 24 * 3600:
        continue
    first[key] = t
    kept.append(r)
check("rows after dropping re-submitted duplicates", len(kept), len(clean))
check(
    "duplicates removed",
    len(no_pr) - len(kept),
    len(open(ROOT / "results/tables/dropped_duplicates.csv").read().split()) - 1,
)
ids_check, ids_pipe = {r["issue_id"] for r in kept}, {r["issue_id"] for r in clean}
check("issue_ids kept that differ from the pipeline (should be none)", set(), (ids_check ^ ids_pipe))

# 3. state and time-to-close from API timestamps
closed = [r for r in kept if r["api_state"] == "closed"]
check("closed issues", len(closed), sum(r["is_closed"] == "True" for r in clean))
ttc = [(ts(r["api_closed_at"]) - ts(r["api_created_at"])).total_seconds() / 3600 for r in closed]
med_clean = statistics.median(float(r["ttc_hours"]) for r in clean if r["ttc_hours"])
check("median time-to-close (hours, 2 dp)", round(statistics.median(ttc), 2), round(med_clean, 2))
check("no negative time-to-close", True, min(ttc) >= 0)

# 4. RQ2-eligible: agents with >= 30 closed issues
per_agent = Counter(r["agent_type"] for r in closed)
eligible = {a for a, n in per_agent.items() if n >= 30}
check("agents with >= 30 closed issues", 11, len(eligible))
check(
    "RQ2-eligible closed issues",
    sum(per_agent[a] for a in eligible),
    sum(r["rq2_eligible"] == "True" for r in clean),
)

# 5. repositories, bot authors, own-bot authors
check("distinct repositories", len({r["repo_name"] for r in kept}), len({r["repo_name"] for r in clean}))
check(
    "bot-authored issues",
    sum(r["author_type"] == "Bot" for r in kept),
    sum(r["author_is_bot"] == "True" for r in clean),
)
own = Counter(
    r["agent_type"]
    for r in kept
    if (r["agent_type"], r["author_login"].lower())
    in {
        ("CodeRabbit", "coderabbitai[bot]"),
        ("CodeRabbit", "coderabbit[bot]"),
        ("Devin", "devin-ai-integration[bot]"),
        ("Sweep", "sweep-ai[bot]"),
        ("Claude_Code", "claude[bot]"),
    }
)
check(
    "own-bot issues: CodeRabbit",
    own["CodeRabbit"],
    sum(r["own_bot_author"] == "True" and r["agent_type"] == "CodeRabbit" for r in clean),
)
check(
    "own-bot issues: Devin",
    own["Devin"],
    sum(r["own_bot_author"] == "True" and r["agent_type"] == "Devin" for r in clean),
)
check(
    "github-actions[bot] issues",
    sum(r["author_login"] == "github-actions[bot]" for r in kept),
    sum(r["automation_author"] == "True" for r in clean),
)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
