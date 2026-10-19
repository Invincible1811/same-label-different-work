"""Step 1: data cleaning (decisions D1-D15 in docs/DECISION_LOG.md).

Reads data/raw/*.csv (never modified) and writes
  data/processed/issues_clean.csv    one row per retained issue + derived fields/flags
  data/processed/labels_clean.csv    one row per (retained issue, label) with category
  results/tables/cleaning_log.csv    rows before/after every step, plus flag counts
  results/tables/label_category_map.csv
  results/tables/release_dates.csv

Run:  python src/01_clean.py
"""
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from config import (
    AUTOMATION_LOGINS,
    LABEL_RULES,
    MIN_CLOSED_RQ2,
    OTHER_LABEL,
    OWN_BOT_LOGINS,
    PROCESSED,
    RAW,
    RELEASE_DATES,
    SELF_REF_REPO_PATTERNS,
    SNAPSHOT_DATE,
    TABLES,
)

log = []


def record(step, decision, n_before, n_after, note):
    log.append(
        {
            "step": step,
            "decision": decision,
            "rows_before": n_before,
            "rows_after": n_after,
            "removed": n_before - n_after,
            "note": note,
        }
    )


def flag(name, decision, count, total, note):
    log.append(
        {
            "step": f"flag: {name}",
            "decision": decision,
            "rows_before": total,
            "rows_after": total,
            "removed": 0,
            "note": f"{count} flagged ({count / total:.1%}). {note}",
        }
    )


# ---------- language heuristic (D13) ----------
NON_LATIN = re.compile(r"[Ѐ-ӿ֐-׿؀-ۿऀ-ॿ฀-๿" r"぀-ヿ㐀-䶿一-鿿가-힯]")
LATIN = re.compile(r"[A-Za-zÀ-ɏ]")
CODE_OR_URL = re.compile(r"```.*?```|`[^`]*`|https?://\S+", re.S)
WORD = re.compile(r"[a-zà-öø-ÿ]+")
# Function words of the main Latin-script languages other than English
# (Spanish, Portuguese, French, German, Italian); words that are also English
# stop words are removed so they cannot count for both sides.
FOREIGN_FUNCTION_WORDS = (
    set(
        """
de la que el los se del las por para con una es al lo como más pero sus ya este porque esta entre
cuando muy sin sobre también hasta hay donde desde todo durante todos les contra otros ese eso ante
esto antes algunos qué unos otro otras otra tanto esa estos mucho nada muchos cual poco ella estar
estas algo actualmente não uma os na com mais foi ao ele das tem seu sua ou ser quando muito há já
está também só pelo pela até isso entre era depois sem mesmo aos ter seus quem nas esse eles estão
você essa num nem suas meu às minha numa pelos elas seja será deles essas esses pelas fosse dele
desta deste le les des et est un une du en qui dans pas sur au avec ce il ne plus sont mais nous vous
leur cette être aux été der die und den das ist nicht mit sich auf für ein eine als auch werden aus
er hat dass sie nach wird bei einer um am sind noch wie einem über einen zum war haben nur oder aber
vor zur bis mehr durch man sein wurde il di che è per gli della sono questo anche come più
""".split()
    )
    - set(ENGLISH_STOP_WORDS)
)


def language_group(text):
    """Script-based language grouping, deterministic and dependency-free.
    'non_latin'          : >20% of letters in a non-Latin script (CJK, Cyrillic, Arabic, ...)
    'latin_non_english'  : Latin script, >= 30 words, and more ES/PT/FR/DE/IT function
                           words than English stop words
    'english'            : everything else (incl. short texts that cannot be judged)"""
    t = CODE_OR_URL.sub(" ", text)
    nl, la = len(NON_LATIN.findall(t)), len(LATIN.findall(t))
    if nl / max(nl + la, 1) > 0.20:
        return "non_latin"
    words = WORD.findall(t.lower())
    if len(words) >= 30:
        foreign = sum(w in FOREIGN_FUNCTION_WORDS for w in words)
        english = sum(w in ENGLISH_STOP_WORDS for w in words)
        if foreign > english:
            return "latin_non_english"
    return "english"


def label_category(label):
    for cat, pattern in LABEL_RULES:
        if re.search(pattern, label):
            return cat
    return OTHER_LABEL


DUP_WINDOW_HOURS = 24


def norm_text(x):
    """Lower-case, punctuation -> space, collapse whitespace (numbers kept)."""
    s = str(x).lower() if pd.notna(x) else ""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


def n_resubmitted(df):
    df = df.assign(
        _t=df.title.map(norm_text), _b=df.body.map(norm_text), _ts=pd.to_datetime(df.api_created_at, utc=True)
    ).sort_values(["api_created_at", "issue_number"], kind="mergesort")
    first, n = {}, 0
    for k, ts in zip(zip(df.repo_name, df._t, df._b), df._ts):
        if k in first and (ts - first[k]).total_seconds() <= DUP_WINDOW_HOURS * 3600:
            n += 1
        else:
            first[k] = ts
    return n


def summarise(df, n_label_rows):
    """Dataset-level description used for the before/after cleaning summary."""
    created = pd.to_datetime(df.api_created_at, utc=True)
    closed_at = pd.to_datetime(df.api_closed_at, utc=True)
    ttc = (closed_at - created).dt.total_seconds() / 3600
    closed = df.api_state.eq("closed")
    return {
        "issues (rows)": len(df),
        "repositories": df.repo_name.nunique(),
        "agent types": df.agent_type.nunique(),
        "pull requests included": int(df.is_pull_request.astype(bool).sum()),
        "re-submitted duplicates (D3 rule)": n_resubmitted(df),
        "closed (API state)": int(closed.sum()),
        "open (API state)": int((~closed).sum()),
        "closed %": round(100 * closed.mean(), 1),
        "median time-to-close, hours (closed only)": round(ttc[closed].median(), 2),
        "issues with >= 1 label": int((df.n_labels > 0).sum()),
        "issue-label pairs": n_label_rows,
        "issues with >= 1 assignee": int((df.n_assignees > 0).sum()),
        "issues with >= 1 comment (API count)": int((df.comments_api > 0).sum()),
        "bot-authored issues": int(df.author_type.eq("Bot").sum()),
        "empty bodies": int(df.body.fillna("").str.strip().eq("").sum()),
        "earliest issue (API created)": str(created.min().date()),
        "latest issue (API created)": str(created.max().date()),
    }


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    issues = pd.read_csv(RAW / "issues.csv")
    labels = pd.read_csv(RAW / "issue_labels.csv")
    raw_issues, raw_label_rows = issues.copy(), len(labels)
    n0 = len(issues)
    record("0 load", "D1", n0, n0, "Raw issues.csv loaded; raw files are never modified.")

    # integrity checks (fail loudly rather than silently analyse broken data)
    assert issues.issue_id.is_unique
    assert (issues.issue_id == issues.repo_name + "#" + issues.issue_number.astype(str)).all()
    assert labels.issue_id.isin(issues.issue_id).all()

    # D2: pull requests are not issues
    n = len(issues)
    issues = issues[~issues.is_pull_request.astype(bool)].copy()
    record(
        "1 drop pull requests",
        "D2",
        n,
        len(issues),
        "is_pull_request == True; PRs follow a review/merge lifecycle, not an issue lifecycle.",
    )

    # D4: API timestamps are authoritative
    for c in ["api_created_at", "api_closed_at", "api_updated_at"]:
        issues[c] = pd.to_datetime(issues[c], utc=True)

    # D3: re-submitted duplicates = same repo, same title and body after normalising case,
    # whitespace and punctuation (numbers kept, so dated recurring reports stay distinct),
    # posted within DUP_WINDOW_HOURS of the first (kept) copy.
    n = len(issues)
    issues["_t"] = issues.title.map(norm_text)
    issues["_b"] = issues.body.map(norm_text)
    # stable sort; ties on the timestamp are broken by the lower issue number (filed first)
    issues = issues.sort_values(["api_created_at", "issue_number"], kind="mergesort")
    first_seen, dup_ids = {}, []
    for iid, repo, t, b, ts in zip(
        issues.issue_id, issues.repo_name, issues._t, issues._b, issues.api_created_at
    ):
        key = (repo, t, b)
        if key in first_seen and (ts - first_seen[key]).total_seconds() <= DUP_WINDOW_HOURS * 3600:
            dup_ids.append(iid)
        else:
            first_seen[key] = ts
    dropped_dups = pd.Series(dup_ids, dtype=str)
    issues = issues[~issues.issue_id.isin(dup_ids)].drop(columns=["_t", "_b"])
    record(
        "2 drop re-submitted duplicates",
        "D3",
        n,
        len(issues),
        "Same repo, title and body (case/whitespace/punctuation normalised) re-posted within "
        f"{DUP_WINDOW_HOURS} h; "
        "first copy kept. Recurring reports further apart are distinct events. Issues *closed as* "
        "'duplicate' by maintainers are a real outcome and are kept.",
    )

    # D4/D6/D12: state, time-to-close, censoring
    snapshot = pd.Timestamp(SNAPSHOT_DATE)
    issues["is_closed"] = issues.api_state.eq("closed")
    issues["ttc_hours"] = (issues.api_closed_at - issues.api_created_at).dt.total_seconds() / 3600
    issues["duration_hours"] = np.where(
        issues.is_closed, issues.ttc_hours, (snapshot - issues.api_created_at).dt.total_seconds() / 3600
    )
    issues["event_closed"] = issues.is_closed.astype(int)  # 0 = right-censored (still open)
    issues["closure_reason"] = np.where(issues.is_closed, issues.state_reason.fillna("unknown"), "open")
    assert (issues.ttc_hours.dropna() >= 0).all()
    assert (issues.duration_hours >= 0).all()
    unknown = int((issues.closure_reason == "unknown").sum())
    flag(
        "closure_reason unknown",
        "D6",
        unknown,
        len(issues),
        "Closed issues with an empty state_reason, recoded to 'unknown'.",
    )
    flag(
        "open (censored)",
        "D6/D12",
        int((~issues.is_closed).sum()),
        len(issues),
        f"Kept; TTC undefined; censored at snapshot {SNAPSHOT_DATE} for Kaplan-Meier.",
    )

    # D5: no trimming, but make extreme values visible
    flag(
        "instant close (< 1 min)",
        "D5",
        int((issues.ttc_hours < 1 / 60).sum()),
        len(issues),
        "Kept (genuine open-then-close workflows); no outlier trimming.",
    )
    flag(
        "TTC > 1 year",
        "D5",
        int((issues.ttc_hours > 24 * 365).sum()),
        len(issues),
        "Kept; handled by median-based, non-parametric statistics.",
    )

    # D8: keep agent_type as given; add sensitivity flags
    issues["author_is_bot"] = issues.author_type.eq("Bot")
    rel = {a: pd.Timestamp(d, tz="UTC") for a, (d, _, _) in RELEASE_DATES.items()}
    issues["release_date"] = issues.agent_type.map(rel)
    issues["pre_release"] = issues.api_created_at < issues.release_date
    flag(
        "bot-authored",
        "D8",
        int(issues.author_is_bot.sum()),
        len(issues),
        "agent_type kept as given; used for the User-only sensitivity check (S2).",
    )
    flag(
        "created before agent release",
        "D8",
        int(issues.pre_release.sum()),
        len(issues),
        "agent_type kept as given; excluded only in sensitivity check S1.",
    )

    # D15: attribution signal (CodAGE precision recipe, steps 1-3) -> M1.10 and S3
    login = issues.author_login.fillna("").str.lower()
    issues["own_bot_author"] = [lg in OWN_BOT_LOGINS.get(a, set()) for lg, a in zip(login, issues.agent_type)]
    issues["automation_author"] = login.isin(AUTOMATION_LOGINS)
    issues["self_ref_repo"] = [
        re.search(SELF_REF_REPO_PATTERNS.get(a, r"$^"), r.lower()) is not None
        for r, a in zip(issues.repo_name, issues.agent_type)
    ]
    issues["attribution_signal"] = np.select(
        [issues.own_bot_author, issues.automation_author, issues.self_ref_repo],
        ["agent_own_bot", "automation_echo", "self_referential_repo"],
        default="other_signal",
    )
    issues["s3_precision"] = ~(issues.automation_author | issues.self_ref_repo)
    for sig in ["agent_own_bot", "automation_echo", "self_referential_repo", "other_signal"]:
        flag(
            f"attribution: {sig}",
            "D15",
            int((issues.attribution_signal == sig).sum()),
            len(issues),
            "Mutually exclusive, in this priority order (M1.10).",
        )
    flag(
        "excluded from S3 precision subset",
        "D15",
        int((~issues.s3_precision).sum()),
        len(issues),
        "github-actions[bot] authors or self-referential repos (CodAGE recipe steps 1 and 3).",
    )

    # D7: small groups
    closed_per_agent = issues[issues.is_closed].agent_type.value_counts()
    rq2_agents = sorted(closed_per_agent[closed_per_agent >= MIN_CLOSED_RQ2].index)
    issues["rq2_eligible"] = issues.is_closed & issues.agent_type.isin(rq2_agents)
    small = sorted(set(issues.agent_type) - set(rq2_agents))
    flag(
        "agent below inference threshold",
        "D7",
        int(issues.agent_type.isin(small).sum()),
        len(issues),
        f"{', '.join(small)} (< {MIN_CLOSED_RQ2} closed issues): RQ1 descriptives only.",
    )

    # D14: renamed repos / transferred issues (informational)
    url = issues.html_url.str.extract(r"github\.com/([^/]+/[^/]+)/issues/(\d+)")
    issues["issue_transferred"] = url[1].astype(int) != issues.issue_number
    issues["repo_renamed"] = (url[0].str.lower() != issues.repo_name.str.lower()) & ~issues.issue_transferred
    flag(
        "repo renamed since archive",
        "D14",
        int(issues.repo_renamed.sum()),
        len(issues),
        "Kept; repo_name used as the repository key.",
    )
    flag(
        "issue transferred to another repo",
        "D14",
        int(issues.issue_transferred.sum()),
        len(issues),
        "Kept; API timestamps verified consistent with the archive.",
    )

    # D10: text field for RQ3 (cleaning of the text itself happens in the RQ3 script)
    issues["body_empty"] = issues.body.fillna("").str.strip().eq("")
    issues["text"] = issues.title.fillna("") + "\n\n" + issues.body.fillna("")
    flag("empty body", "D10", int(issues.body_empty.sum()), len(issues), "Kept; title still provides text.")

    # D13: language group
    issues["lang_group"] = issues.text.map(language_group)
    for g in ["non_latin", "latin_non_english"]:
        flag(
            f"language: {g}",
            "D13",
            int((issues.lang_group == g).sum()),
            len(issues),
            "Kept for RQ1/RQ2; excluded from the RQ3 topic model (English LDA).",
        )

    # D9: labels
    labels = labels[labels.issue_id.isin(issues.issue_id)].copy()
    labels["label_norm"] = labels.label.astype(str).str.strip().str.lower()
    labels["category"] = labels.label_norm.map(label_category)
    for cat in [c for c, _ in LABEL_RULES] + [OTHER_LABEL]:
        ids = set(labels.loc[labels.category == cat, "issue_id"])
        issues[f"has_{cat}_label"] = issues.issue_id.isin(ids)
    issues["n_labels_check"] = issues.issue_id.map(labels.groupby("issue_id").size()).fillna(0).astype(int)
    assert (issues.n_labels_check == issues.n_labels).all()
    issues = issues.drop(columns="n_labels_check")

    # outputs
    issues = issues.sort_values("issue_id").reset_index(drop=True)
    record(
        "3 final analysis table",
        "D1",
        n0,
        len(issues),
        f"{len(issues)} issues, {issues.repo_name.nunique()} repos, "
        f"{int(issues.is_closed.sum())} closed, {int(issues.rq2_eligible.sum())} RQ2-eligible "
        f"across {len(rq2_agents)} agents. issue_comments.csv not used (D11).",
    )

    issues.to_csv(PROCESSED / "issues_clean.csv", index=False)
    labels.to_csv(PROCESSED / "labels_clean.csv", index=False)
    pd.DataFrame(log).to_csv(TABLES / "cleaning_log.csv", index=False)

    # before / after summary (requested by the author for transparency)
    before, after = summarise(raw_issues, raw_label_rows), summarise(issues, len(labels))
    summary = pd.DataFrame(
        {
            "measure": list(before),
            "before_cleaning": list(before.values()),
            "after_cleaning": [after[k] for k in before],
        }
    )
    summary.to_csv(TABLES / "dataset_summary_before_after.csv", index=False)
    b = raw_issues.agent_type.value_counts().rename("before_cleaning")
    a = issues.agent_type.value_counts().rename("after_cleaning")
    by_agent = pd.concat([b, a], axis=1).fillna(0).astype(int)
    by_agent["removed"] = by_agent.before_cleaning - by_agent.after_cleaning
    by_agent["closed_after"] = (
        issues[issues.is_closed].agent_type.value_counts().reindex(by_agent.index).fillna(0).astype(int)
    )
    by_agent["rq2_eligible"] = by_agent.closed_after >= MIN_CLOSED_RQ2
    by_agent.rename_axis("agent_type").to_csv(TABLES / "dataset_summary_by_agent.csv")
    (
        labels.groupby(["label_norm", "category"])
        .size()
        .rename("n_issues")
        .reset_index()
        .sort_values("n_issues", ascending=False)
        .to_csv(TABLES / "label_category_map.csv", index=False)
    )
    pd.DataFrame(
        [
            {"agent_type": a, "release_date": d, "source": s, "confidence": c}
            for a, (d, s, c) in RELEASE_DATES.items()
        ]
    ).to_csv(TABLES / "release_dates.csv", index=False)
    pd.Series(sorted(dropped_dups), name="issue_id").to_csv(TABLES / "dropped_duplicates.csv", index=False)

    with pd.option_context("display.max_colwidth", 120, "display.width", 250):
        print(
            pd.DataFrame(log)[["step", "decision", "rows_before", "rows_after", "removed", "note"]].to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
