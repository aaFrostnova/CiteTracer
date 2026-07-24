#!/usr/bin/env python3
"""Harvest real biomedical citation seeds from Europe PMC (MEDLINE source).

Queries a spread of MeSH-ish topics x years, keeps only records with complete,
verifiable fields (title, >=2 named authors, journal + ISO abbreviation, year,
volume, pages, DOI), dedups by DOI. Output feeds generate_biomed_synth.py.
"""
from __future__ import annotations
import json, re, sys, time, urllib.parse, urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "synthetic_data" / "biomed_v1"
OUT.mkdir(parents=True, exist_ok=True)
BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

TOPICS = [
    "cancer immunotherapy", "CRISPR gene editing", "Alzheimer disease",
    "SARS-CoV-2 vaccine", "type 2 diabetes mellitus", "cardiovascular disease",
    "gut microbiome", "stem cell therapy", "antibiotic resistance",
    "tumor microenvironment", "single cell RNA sequencing", "Parkinson disease",
    "obesity metabolism", "breast cancer prognosis", "sepsis mortality",
    "hepatocellular carcinoma", "multiple sclerosis", "chronic kidney disease",
    "asthma inflammation", "major depressive disorder", "rheumatoid arthritis",
    "ischemic stroke", "HIV latency", "influenza epidemiology",
    "colorectal cancer screening", "neurodegeneration tau", "insulin resistance",
    "melanoma metastasis", "atrial fibrillation", "pulmonary fibrosis",
    "glioblastoma treatment", "inflammatory bowel disease", "osteoporosis bone",
    "psoriasis immunology", "diabetic retinopathy", "prostate cancer biomarker",
]
YEARS = [2019, 2020, 2021, 2022, 2023, 2024]


def fetch(query: str, page_size: int = 25) -> list[dict]:
    q = (f"{query} AND SRC:MED AND (HAS_DOI:Y) AND (LANG:eng)")
    url = (f"{BASE}?query={urllib.parse.quote(q)}&format=json"
           f"&pageSize={page_size}&resultType=core&sort=CITED%20desc")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=45) as r:
                return json.loads(r.read())["resultList"]["result"]
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                print(f"  [warn] {query}: {e}", file=sys.stderr)
                return []
            time.sleep(2 ** attempt)
    return []


def clean(rec: dict, topic: str, year: int) -> dict | None:
    al = rec.get("authorList", {}).get("author", [])
    authors = []
    for a in al:
        fn, ln = a.get("firstName"), a.get("lastName")
        if fn and ln:
            authors.append(f"{fn} {ln}")
        elif a.get("collectiveName"):
            # Consortium/collective authors carry ONLY collectiveName. Dropping them
            # made the rendered citation one author short of every index, which reads
            # downstream as a fabricated author deletion.
            authors.append(str(a["collectiveName"]).strip())
        elif a.get("fullName"):
            authors.append(a["fullName"])
    if len(authors) < 2:
        return None
    ji = rec.get("journalInfo", {}) or {}
    j = ji.get("journal", {}) or {}
    journal = j.get("title") or ""
    iso = j.get("isoabbreviation") or ""
    vol = str(ji.get("volume") or "")
    pages = rec.get("pageInfo") or ""
    doi = rec.get("doi") or ""
    title = re.sub(r"<[^>]+>", "", rec.get("title") or "").rstrip(".")
    title = re.sub(r"\s+", " ", title).strip()
    py = rec.get("pubYear")
    if not (title and journal and vol and pages and doi and py):
        return None
    if not any(c.isdigit() for c in pages):
        return None
    return {
        "title": title,
        "authors": authors,
        "venue": journal,
        "venue_iso": iso,
        "year": int(py),
        "doi": doi.lower(),
        "arxiv_id": "",
        "url": f"https://doi.org/{doi.lower()}",
        "volume": vol,
        "pages": pages,
        "publisher": "",
        "location": "",
        "_source": "europepmc",
        "_query_topic": topic,
        "_query_venue": iso or journal,
        "_query_year": int(py),
    }


def main() -> None:
    seen_doi: set[str] = set()
    seen_title: set[str] = set()
    seeds: list[dict] = []
    for topic in TOPICS:
        for year in YEARS:
            q = f"{topic} AND PUB_YEAR:{year}"
            for rec in fetch(q):
                s = clean(rec, topic, year)
                if not s:
                    continue
                tkey = "".join(ch for ch in s["title"].lower() if ch.isalnum())
                if s["doi"] in seen_doi or tkey in seen_title:
                    continue
                seen_doi.add(s["doi"]); seen_title.add(tkey)
                seeds.append(s)
            time.sleep(0.34)  # be polite to Europe PMC
        print(f"[{topic}] running total: {len(seeds)}", flush=True)
    (OUT / "seeds_raw.json").write_text(json.dumps(seeds, ensure_ascii=False, indent=1))
    print(f"\n[done] harvested {len(seeds)} unique complete biomedical seeds "
          f"-> {OUT/'seeds_raw.json'}")
    yrs = {}
    for s in seeds:
        yrs[s["year"]] = yrs.get(s["year"], 0) + 1
    print("by year:", dict(sorted(yrs.items())))


if __name__ == "__main__":
    main()
