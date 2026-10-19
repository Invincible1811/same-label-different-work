"""Step 4: RQ3 text preprocessing for LDA (English issues of the 11 tested agents).

Each step and its reason (lecture slide 16: preprocessing must match the algorithm):
  1. Strip non-prose: fenced/inline code, HTML, URLs, image links, @mentions, issue refs, hashes, numbers.
  2. Remove agent signatures and trigger lines (the same text CodAGE uses to label agents), so topics
     cannot rediscover the label — this avoids circularity.
  3. Remove template text: a line whose masked form (users, links, numbers masked) occurs in
     >= RQ3_TEMPLATE_MIN_DOCS different issues (headings, bot report templates, form fields).
  4. Tokenise (letters only, >= 3 chars), remove stop words (scikit-learn ∪ gensim ∪ issue-tracker words)
     and agent/vendor names, lemmatise with WordNet (verb, then noun).
  5. Detect frequent two-word phrases (gensim Phrases) and drop documents with < RQ3_MIN_TOKENS tokens.

Writes data/processed/rq3_corpus.csv (NOT redistributed) and
results/tables/rq3_preprocessing_log.csv, rq3_template_lines.csv.
Run:  python src/04_rq3_preprocess.py   (needs: python -m nltk.downloader wordnet omw-1.4)
"""
import re
from collections import Counter

import pandas as pd
from gensim.models.phrases import Phrases
from gensim.parsing.preprocessing import STOPWORDS as GENSIM_STOP
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from config import (
    MIN_CLOSED_RQ2,
    PROCESSED,
    RQ3_AGENT_NAMES,
    RQ3_AGENT_PHRASES,
    RQ3_AGENT_SUBSTRINGS,
    RQ3_EXTRA_STOPWORDS,
    RQ3_MIN_TOKENS,
    RQ3_SIGNATURE_LINES,
    RQ3_TEMPLATE_MIN_DOCS,
    TABLES,
)

CODE_BLOCK = re.compile(r"```.*?(```|\Z)", re.S)
INLINE_CODE = re.compile(r"`[^`\n]*`")
HTML = re.compile(r"<[^>\n]+>")
MD_IMAGE_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")  # keep link text, drop target
URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"@[\w-]+")
ISSUE_REF = re.compile(r"(?<!\w)#\d+")
HASH = re.compile(r"\b[0-9a-f]{7,40}\b")
SIGNATURE = re.compile("|".join(RQ3_SIGNATURE_LINES), re.I)
AGENT_PHRASE = re.compile("|".join(RQ3_AGENT_PHRASES), re.I)
STOP = set(ENGLISH_STOP_WORDS) | set(GENSIM_STOP) | RQ3_EXTRA_STOPWORDS
WORD = re.compile(r"[a-z]+")
LEM = WordNetLemmatizer()


def strip_non_prose(text):
    t = CODE_BLOCK.sub("\n", text)
    t = INLINE_CODE.sub(" ", t)
    t = MD_IMAGE_LINK.sub(r"\1", t)
    t = HTML.sub(" ", t)
    t = URL.sub(" ", t)
    t = MENTION.sub(" ", t)
    t = ISSUE_REF.sub(" ", t)
    return HASH.sub(" ", t)


def mask(line):
    """Masked form of a line for template detection (users, links, numbers masked)."""
    m = re.sub(r"\s+", " ", line.strip().lower())
    m = re.sub(r"@[\w-]+", "@USER", m)
    m = re.sub(r"https?://\S+", "URL", m)
    return re.sub(r"\d+", "N", m)


def is_agent_token(w):
    return w in RQ3_AGENT_NAMES or any(s in w for s in RQ3_AGENT_SUBSTRINGS)


AGENT_WORD = re.compile(
    r"\b("
    + "|".join(sorted(RQ3_AGENT_NAMES, key=len, reverse=True))
    + r")\b|\S*("
    + "|".join(RQ3_AGENT_SUBSTRINGS)
    + r")\S*",
    re.I,
)


def prose(text):
    """Cleaned natural text for embedding models: agent/vendor names removed, markdown
    symbols and whitespace collapsed; sentences and word order kept."""
    t = AGENT_WORD.sub(" ", text)
    t = re.sub(r"[#*_>|`~]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def lemma(w):
    return LEM.lemmatize(LEM.lemmatize(w, "v"), "n")


def tokens(text):
    out = []
    for w in WORD.findall(text.lower()):
        if len(w) < 3 or w in STOP or is_agent_token(w):
            continue
        w = lemma(w)
        if len(w) >= 3 and w not in STOP and not is_agent_token(w):
            out.append(w)
    return out


def run(min_tokens=RQ3_MIN_TOKENS, write=True):
    """Preprocess; returns the modelled documents. write=False is used by sensitivity checks."""
    d = pd.read_csv(PROCESSED / "issues_clean.csv")
    closed_n = d[d.is_closed].agent_type.value_counts()
    agents = closed_n[closed_n >= MIN_CLOSED_RQ2].index  # the 11 tested agents (D7)
    log = [("all cleaned issues", len(d))]
    e = d[d.agent_type.isin(agents)]
    log.append(("issues of the 11 tested agents", len(e)))
    e = e[e.lang_group == "english"].copy()
    log.append(("English issues (D13)", len(e)))

    # step 1 + 2: non-prose and signature lines
    lines = []
    for t in e.text.fillna(""):
        # signature lines are dropped first; product names are then stripped from the remaining lines
        # (e.g. the trigger "Sweep:" is removed but the rest of the title is kept as content)
        kept = [
            AGENT_PHRASE.sub(" ", ln)
            for ln in strip_non_prose(t).splitlines()
            if ln.strip() and not SIGNATURE.search(ln)
        ]
        kept = [ln for ln in kept if ln.strip()]
        lines.append(kept)
    sig_removed = sum(
        len([ln for ln in strip_non_prose(t).splitlines() if ln.strip() and SIGNATURE.search(ln)])
        for t in e.text.fillna("")
    )

    # step 3: template lines (masked line repeated in >= N issues)
    df = Counter()
    for kept in lines:
        df.update({mask(ln) for ln in kept})
    template = {m for m, c in df.items() if c >= RQ3_TEMPLATE_MIN_DOCS}
    if write:
        pd.DataFrame(
            sorted(((m, c) for m, c in df.items() if m in template), key=lambda x: (-x[1], x[0])),
            columns=["masked_line", "n_issues"],
        ).to_csv(TABLES / "rq3_template_lines.csv", index=False)
    texts = ["\n".join(ln for ln in kept if mask(ln) not in template) for kept in lines]
    tpl_removed = sum(1 for kept in lines for ln in kept if mask(ln) in template)

    # step 4: tokens; step 5: phrases + minimum length
    toks = [tokens(t) for t in texts]
    phrases = Phrases(toks, min_count=20, threshold=10)
    toks = [[w for w in phrases[t] if not is_agent_token(w)] for t in toks]
    e["tokens"] = [" ".join(t) for t in toks]
    e["prose"] = [prose(t) for t in texts]  # same cleaning, sentences kept (for BERTopic, 08)
    e["n_tokens"] = [len(t) for t in toks]
    kept = e[e.n_tokens >= min_tokens]
    log.append((f"documents with >= {min_tokens} tokens after cleaning (modelled)", len(kept)))
    if not write:
        return kept

    kept[["issue_id", "agent_type", "n_tokens", "tokens", "prose"]].to_csv(
        PROCESSED / "rq3_corpus.csv", index=False
    )
    stats = pd.DataFrame(log, columns=["stage", "documents"])
    extra = pd.DataFrame(
        [
            ("signature/trigger lines removed", sig_removed),
            (f"template lines removed (masked line in >= {RQ3_TEMPLATE_MIN_DOCS} issues)", tpl_removed),
            ("distinct template line patterns", len(template)),
            ("median tokens per modelled document", int(kept.n_tokens.median())),
            ("distinct terms before vocabulary filtering", len({w for t in toks for w in t})),
            ("phrases (bigrams) in vocabulary", len({w for t in toks for w in t if "_" in w})),
        ],
        columns=["stage", "documents"],
    )
    pd.concat([stats, extra]).rename(columns={"documents": "value"}).to_csv(
        TABLES / "rq3_preprocessing_log.csv", index=False
    )
    print(pd.concat([stats, extra]).to_string(index=False))
    print("\nDocuments per agent (modelled):", kept.agent_type.value_counts().to_dict())
    return kept


def main():
    run()


if __name__ == "__main__":
    main()
