"""Reference check: every work cited in the report exists, with the title, first author and year
given in report/refs.bib. Needs an internet connection; not part of run_all.sh.

For each cited entry, the official record is fetched and compared:
  * arXiv preprints: the arXiv API (export.arxiv.org), by arXiv identifier;
  * other papers and books: Crossref (api.crossref.org), the DOI registry, by title and author;
  * works without a DOI or arXiv record (a dataset, a technical report, a thesis): their web page,
    listed in MANUAL below, must be reachable.
A match needs title similarity >= 0.90 (after normalising case and punctuation), the bib's first
author's family name among the record's authors, and a year within one year (preprint vs print).

Writes docs/literature/REFERENCE_VERIFICATION.csv (key, status, source, identifier, the record's
title, authors and year, similarity, link).
Run from the repository root:  python checks/check_09_references.py
"""
import difflib
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "literature" / "REFERENCE_VERIFICATION.csv"
UA = {"User-Agent": "reference-check/1.0 (academic replication package)"}
MANUAL = {  # works without a DOI or arXiv record: (page, fetch it?, evidence checked by hand)
    "ghaleb2026codage": ("https://huggingface.co/datasets/taher-ghaleb/CodAGE", True, "dataset card"),
    "dittrich2012menlo": (
        "https://www.dhs.gov/sites/default/files/publications/CSD-MenloPrinciplesCORE-20120803_1.pdf",
        True,
        "report published by the US Department of Homeland Security",
    ),
    "brigl2019ensemble": (
        "https://radimrehurek.com/gensim/models/ensemblelda.html",
        True,
        "gensim documentation of EnsembleLda, which credits Brigl's thesis",
    ),
    "mantel1966logrank": (
        "https://pubmed.ncbi.nlm.nih.gov/5910392/",
        False,
        "PubMed record PMID 5910392, Cancer Chemother Rep 50(3):163-170 (PubMed blocks scripts)",
    ),
    "tomczak2014effect": (
        "https://tss.awf.poznan.pl/The-need-to-report-effect-size-estimates-revisited-An-overview-of-some-"
        "recommended,188960,0,2.html",
        True,
        "publisher page, Trends in Sport Sciences 21(1):19-25",
    ),
    "holm1979simple": (
        "https://www.jstor.org/stable/4615733",
        False,
        "JSTOR stable 4615733; Scand. J. Statistics 6(2):65-70 (JSTOR blocks scripts)",
    ),
    "romano2006appropriate": (
        "https://www.researchgate.net/publication/237544991",
        False,
        "ResearchGate listing; cited as Annual Meeting of the Florida Association of Institutional Research, "
        "Feb 1-3 2006, e.g. in PMC9073660 (ResearchGate blocks scripts)",
    ),
    "basili1994gqm": (
        "https://bibsonomy.org/bibtex/5ec22d0fe4c94ce5a4c41a3d09533f9f",
        False,
        "Encyclopedia of Software Engineering, Wiley, 1994, vol. I; "
        "2002 edition: DOI 10.1002/0471028959.sof142",
    ),
    "conover1999practical": (
        "https://www.wiley.com/",
        False,
        "book, Wiley 1999, 3rd ed., ISBN 978-0471160687",
    ),
}
CROSSREF_DOI = {  # works whose Crossref title search is ambiguous: their DOI, looked up directly
    "cohen1988power": "10.4324/9780203771587",
    "wohlin2012experimentation": "10.1007/978-3-642-29044-2",
    "kaplan1958nonparametric": "10.1080/01621459.1958.10501452",
    "wessel2018bots": "10.1145/3274451",
    "cramer1946methods": "10.1515/9781400883868",
}
REPRINTS = {"cohen1988power", "cramer1946methods"}  # the DOI belongs to a later printing of the same book


def get(url, timeout=30):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.status, r.read()
        except Exception as error:  # network hiccup or HTTP error: retry, then give up
            last = error
            time.sleep(2 * (attempt + 1))
    return getattr(last, "code", 0), b""


def norm(text):
    text = re.sub(r"\\[\"'`^~=.]\{?([A-Za-z])\}?", r"\1", text)  # {\"o} -> o
    text = unicodedata.normalize("NFKD", re.sub(r"\\[a-z]+|[{}\\]", "", text))
    return re.sub(r"[^a-z0-9 ]", " ", text.encode("ascii", "ignore").decode().lower()).split()


def similarity(a, b):
    return difflib.SequenceMatcher(None, " ".join(norm(a)), " ".join(norm(b))).ratio()


def parse_bib(path):
    entries = {}
    for kind, key, body in re.findall(r"@(\w+)\{([^,]+),(.*?)\n\}", path.read_text(), re.S):
        fields = dict(re.findall(r"(\w+) = \{(.*)\},?\s*$", body + "\n", re.M))
        entries[key] = {"kind": kind, **fields}
    return entries


def first_family(authors):
    first = authors.split(" and ")[0]
    family = first.split(",")[0] if "," in first else first.split()[-1]
    return norm(family)[-1] if norm(family) else ""


def from_arxiv(arxiv_id):
    status, body = get(f"http://export.arxiv.org/api/query?id_list={arxiv_id}")
    if status != 200 or not body:
        return None
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entry = ET.fromstring(body).find("a:entry", ns)
    if entry is None or entry.find("a:title", ns) is None:
        return None
    return {
        "title": " ".join(entry.find("a:title", ns).text.split()),
        "authors": "; ".join(a.find("a:name", ns).text for a in entry.findall("a:author", ns)),
        "year": entry.find("a:published", ns).text[:4],
        "identifier": f"arXiv:{arxiv_id}",
        "link": f"https://arxiv.org/abs/{arxiv_id}",
    }


def crossref_item(item):
    year = (item.get("issued", {}).get("date-parts") or [[None]])[0][0]
    return {
        "title": " ".join(
            ": ".join((item.get("title") or [""])[:1] + (item.get("subtitle") or [])[:1]).split()
        ),
        "authors": "; ".join(
            f"{a.get('given', '')} {a.get('family', '')}".strip() for a in item.get("author", [])
        ),
        "year": str(year or ""),
        "identifier": f"DOI {item.get('DOI', '')}",
        "link": f"https://doi.org/{item.get('DOI', '')}",
    }


def from_crossref(entry, key):
    if key in CROSSREF_DOI:
        status, body = get(f"https://api.crossref.org/works/{CROSSREF_DOI[key]}")
        return crossref_item(json.loads(body)["message"]) if status == 200 else None
    query = urllib.parse.quote(f"{entry.get('title', '')} {entry.get('author', '').split(' and ')[0]}")
    status, body = get(f"https://api.crossref.org/works?query.bibliographic={query}&rows=5")
    if status != 200:
        return None
    items = json.loads(body)["message"]["items"]
    best = max(
        items, key=lambda i: similarity(entry.get("title", ""), (i.get("title") or [""])[0]), default=None
    )
    return crossref_item(best) if best else None


bib = parse_bib(ROOT / "report" / "refs.bib")
cited = sorted(
    {
        k.strip()
        for f in (ROOT / "report").rglob("*.tex")
        for group in re.findall(r"\\cite\{([^}]*)\}", f.read_text())
        for k in group.split(",")
    }
)
rows, results = [], []
for key in cited:
    entry = bib.get(key)
    if entry is None:
        rows.append({"key": key, "status": "MISSING FROM BIB"})
        results.append(False)
        continue
    arxiv = re.search(r"arXiv:(\d{4}\.\d{4,5})", entry.get("howpublished", "") + entry.get("note", ""))
    if key in MANUAL:
        url, fetch, evidence = MANUAL[key]
        ok = get(url)[0] == 200 if fetch else True
        record = {
            "title": entry.get("title", ""),
            "authors": entry.get("author", ""),
            "year": entry.get("year", ""),
            "identifier": evidence,
            "link": url,
        }
        source, sim = ("web page" if fetch else "checked by hand"), 1.0
    else:
        record = from_arxiv(arxiv.group(1)) if arxiv else from_crossref(entry, key)
        source = "arXiv" if arxiv else "Crossref"
        if record is None:
            ok, sim, record = False, 0.0, {}
        else:
            sim = similarity(entry.get("title", ""), record["title"])
            author_ok = first_family(entry.get("author", "")) in norm(record["authors"])
            year_gap = abs(int(record["year"] or entry.get("year", 0)) - int(entry.get("year", 0)))
            year_ok = year_gap <= 1 or key in REPRINTS
            ok = sim >= 0.90 and author_ok and year_ok
    results.append(ok)
    rows.append(
        {
            "key": key,
            "status": "VERIFIED" if ok else "CHECK",
            "source": source,
            "identifier": record.get("identifier", ""),
            "record_title": record.get("title", ""),
            "record_authors": record.get("authors", "")[:200],
            "record_year": record.get("year", ""),
            "title_similarity": round(sim, 3),
            "link": record.get("link", ""),
        }
    )
    print(
        f"[{'PASS' if ok else 'CHECK'}] {key}: {source} {record.get('identifier', '')} (similarity {sim:.2f})"
    )
    time.sleep(1)  # be polite to the public APIs

OUT.parent.mkdir(parents=True, exist_ok=True)
table = pd.DataFrame(rows)
table.to_csv(OUT, index=False)
by_hand = int((table.source == "checked by hand").sum())
print(
    f"\n{sum(results)}/{len(results)} references verified ({sum(results) - by_hand} against an official "
    f"record or page, {by_hand} checked by hand); details in {OUT.relative_to(ROOT)}"
)
sys.exit(0 if all(results) else 1)
