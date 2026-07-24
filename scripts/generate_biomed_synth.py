#!/usr/bin/env python3
"""Generate a 1000-citation biomedical synthetic benchmark mirroring
data/synthetic_data/v2 (same 11-code taxonomy, mutation operators, and output
layout). Seeds come from scripts/harvest_biomed_seeds.py (Europe PMC).

Determinism: a fixed RNG seed drives every choice. The LLM (Bedrock) is used
ONLY for H1 title_paraphrase / title_fabrication for naturalness, with a
deterministic fallback if no key is available, so the script always completes.

Output (data/synthetic_data/biomed_v1/):
  <CLASS>.json         per-class citation lists (same item schema as v2)
  meta.json            {cid: {subtype,label,category,mutation_type,changed_fields,explanation,seed}}
  cleaned_bib.bib      BibTeX with H*/P*/R* keys (H=hallucinated, R=real, P=potential)
  manifest.json        {cid: {label, subtype}}
  _all.json            flat list of 1000 citations
"""
from __future__ import annotations
import argparse, copy, json, os, random, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "synthetic_data" / "biomed_v1"

# ---- 1000-scale target, proportional to v2 (sums to 1000) ----
TARGET = {"H1": 82, "H2": 81, "H3": 80, "H4": 80, "H5": 82, "H6": 68,
          "P1": 37, "P3": 73,
          "R1": 70, "R1_plus": 68, "R2": 71, "R2_plus": 68, "R3": 72, "R3_plus": 68}
LABEL = {"R": "REAL", "P": "POTENTIAL_HALLUCINATED", "H": "HALLUCINATED"}
CATEGORY = {"H1": "Title", "H2": "Authors", "H3": "Venue", "H4": "Year",
            "H5": "Identifier", "H6": "Peripheral", "P1": "Author variant",
            "P3": "Insufficient evidence", "R1": "Real", "R2": "Real", "R3": "Real"}

# deterministic biomedical synonym map for H1 word_substitution
SYN = {
    "treatment": "therapy", "therapy": "treatment", "analysis": "assessment",
    "role": "function", "effect": "impact", "effects": "impacts",
    "expression": "abundance", "response": "reaction", "outcome": "result",
    "outcomes": "results", "risk": "hazard", "patients": "subjects",
    "children": "pediatric patients", "regulation": "modulation",
    "activation": "stimulation", "inhibition": "suppression",
    "association": "correlation", "mechanism": "pathway", "novel": "new",
    "study": "investigation", "review": "survey", "evaluation": "appraisal",
    "management": "handling", "prevention": "prophylaxis", "diagnosis": "detection",
    "progression": "advancement", "survival": "persistence", "clinical": "medical",
    "disease": "disorder", "cancer": "carcinoma", "tumor": "neoplasm",
    "cells": "cellular units", "brain": "cerebral tissue", "blood": "hematologic",
    "infection": "colonization", "vaccine": "immunogen", "immune": "immunologic",
    "gene": "genetic", "protein": "polypeptide", "signaling": "signal transduction",
    "pathway": "cascade", "model": "framework", "using": "via", "based": "grounded",
    "targeting": "against", "induced": "triggered", "mediated": "driven",
    "dependent": "reliant", "associated": "linked", "improves": "enhances",
    "reduces": "lowers", "increases": "elevates", "predicts": "forecasts",
    "identification": "discovery", "characterization": "profiling", "profile": "signature",
    "development": "emergence", "function": "role", "impact": "effect",
    "therapeutic": "curative", "molecular": "biochemical", "cellular": "cell-level",
    "population": "cohort", "human": "patient-derived", "mouse": "murine",
    "acute": "sudden-onset", "chronic": "long-term", "severe": "advanced",
}
# strong deterministic swap for titles with no synonym hit: pick a wrong head noun
FALLBACK_NOUNS = ["biomarkers", "mechanisms", "outcomes", "therapeutics", "pathways",
                  "correlates", "predictors", "modulators", "phenotypes", "signatures"]
FAKE_FIRST = ["Elena", "Marcus", "Priya", "Chen", "Sofia", "Omar", "Lena",
              "Hiroshi", "Amara", "Viktor", "Ingrid", "Rafael", "Nadia", "Tobias"]
FAKE_LAST = ["Voss", "Marlowe", "Okafor", "Halvorsen", "Reyes", "Battaglia",
             "Nakamura", "Sorensen", "Delacroix", "Petrov", "Alvarez", "Kwon"]
REAL_JOURNAL_POOL = [
    "Nature Medicine", "The Lancet", "New England Journal of Medicine", "Cell",
    "Science Translational Medicine", "JAMA", "Nature Communications",
    "Journal of Clinical Oncology", "Blood", "Circulation", "Gut",
    "Journal of Experimental Medicine", "PNAS", "Nature Immunology",
]


def _int_hash(s: str) -> int:
    h = 0
    for ch in s:
        h = (h * 131 + ord(ch)) % (2 ** 31)
    return h


# ---------------- deterministic mutations ----------------
def m_word_sub(seed, rng):
    words = seed["title"].split()
    idxs = [i for i, w in enumerate(words) if re.sub(r"[^a-z]", "", w.lower()) in SYN]
    changes = []
    for i in rng.sample(idxs, min(2, len(idxs))) if idxs else []:
        w = words[i]; core = re.sub(r"[^A-Za-z]", "", w)
        rep = SYN[core.lower()]
        rep = rep.capitalize() if core[:1].isupper() else rep
        words[i] = w.replace(core, rep)
        changes.append(f"{core}→{rep}")
    # v2 convention: word_substitution changes >=2 content words so the title
    # no longer matches the source. Top up from FALLBACK_NOUNS if synonyms gave <2.
    if len(changes) < 2:
        used = {i for i in idxs}
        cand = sorted([i for i, w in enumerate(words)
                       if len(re.sub(r"[^A-Za-z]", "", w)) >= 5 and i not in used],
                      key=lambda i: -len(words[i]))
        for j, i in enumerate(cand):
            if len(changes) >= 2:
                break
            core = re.sub(r"[^A-Za-z]", "", words[i])
            rep = FALLBACK_NOUNS[(_int_hash(seed["title"]) + j) % len(FALLBACK_NOUNS)]
            rep = rep.capitalize() if core[:1].isupper() else rep
            words[i] = words[i].replace(core, rep)
            changes.append(f"{core}→{rep}")
    new = " ".join(words)
    return new, f"Word substitution: {', '.join(changes)}. Original: \"{seed['title']}\"."


def m_reorder(seed, rng):
    a = list(seed["authors"])
    if len(a) < 2:
        return a, "single author; no reorder"
    b = a[:]
    while b == a:
        rng.shuffle(b)
    return b, f"Author order changed. Original: {a}."


def m_add_del(seed, rng):
    a = list(seed["authors"])
    if rng.random() < 0.5 and len(a) > 2:
        pos = rng.randrange(len(a)); removed = a.pop(pos)
        return a, f"Removed real author '{removed}' from position {pos}. Original had {len(a)+1} authors."
    name = f"{rng.choice(FAKE_FIRST)} {rng.choice(FAKE_LAST)}"
    pos = rng.randrange(len(a) + 1); a.insert(pos, name)
    return a, f"Added fake author '{name}' at position {pos}. Original had {len(a)-1} authors."


def m_fabricate_authors(seed, rng):
    a = list(seed["authors"]); n = len(a)
    k = max(1, min(n, rng.randint(1, 2)))
    idxs = rng.sample(range(n), k)
    for i in idxs:
        a[i] = f"{rng.choice(FAKE_FIRST)} {rng.choice(FAKE_LAST)}"
    return a, f"Fabricated {k} author name(s) at positions {sorted(idxs)}. Original: {seed['authors']}."


def m_venue(seed, rng, also_year):
    old = seed["venue"]
    new = rng.choice([j for j in REAL_JOURNAL_POOL if j != old])
    if also_year:
        oy = seed["year"]; ny = oy + rng.choice([-3, -2, 2, 3])
        return new, ny, (f"Venue: \"{old}\" → \"{new}\"; year {oy} → {ny}. "
                         f"Paper was never published there.")
    return new, seed["year"], f"Venue: \"{old}\" → \"{new}\". Paper was never published at {new}."


def m_year(seed, rng):
    oy = seed["year"]; shift = rng.choice([-5, -4, -3, -2, -1, 1, 2, 3])
    return oy + shift, f"Year: {oy} → {oy+shift}."


def m_doi(seed, rng, nonexistent):
    old = seed["doi"]
    new = f"10.{rng.randint(1000,9999)}/{rng.randint(100000,999999)}"
    kind = "not resolve" if nonexistent else "resolve to a different paper"
    return new, f"DOI fabricated: \"{old}\" → \"{new}\". Will {kind}."


def m_peripheral(seed, rng, kind):
    s = copy.deepcopy(seed); changed = []; parts = []
    if kind == "pages_volume":
        op = seed["pages"]; np_ = f"{rng.randint(1,300)}-{rng.randint(301,600)}"
        ov = seed["volume"]; nv = str(rng.randint(1, 120))
        s["pages"], s["volume"] = np_, nv
        changed = ["pages", "volume"]
        parts = [f"Pages: \"{op}\" → \"{np_}\"", f"Volume: \"{ov}\" → \"{nv}\""]
    elif kind == "publisher":
        s["publisher"] = rng.choice(["Elsevier", "Springer Nature", "Wiley",
                                     "Oxford University Press", "Taylor & Francis"])
        changed = ["publisher"]; parts = [f"Publisher fabricated: \"\" → \"{s['publisher']}\""]
    else:  # location
        s["location"] = rng.choice(["London, UK", "Basel, Switzerland",
                                    "Boston, MA", "Amsterdam, Netherlands"])
        changed = ["location"]; parts = [f"Location fabricated: \"\" → \"{s['location']}\""]
    return s, changed, ("; ".join(parts) +
                        f" for a {seed['year']} {seed['venue']} paper (verifiable against source).")


def m_name_variant(seed, rng):
    a = list(seed["authors"]); i = rng.randrange(len(a))
    parts = a[i].split()
    old = a[i]
    if len(parts) >= 2 and len(parts[0]) > 3:
        short = parts[0][:3]
        a[i] = f"{short} {parts[-1]}"
        note = f"\"{old}\" → \"{a[i]}\". '{short}' is a plausible short form of '{parts[0]}'."
    else:
        a[i] = f"{parts[0][0]}. {parts[-1]}"
        note = f"\"{old}\" → \"{a[i]}\". Initial-only variant of a real author."
    return a, f"Author name variant: {note}"


def m_insufficient(seed, rng):
    s = copy.deepcopy(seed); changed = []; parts = []
    combo = rng.choice([["volume"], ["location"], ["volume", "location"],
                        ["pages", "volume", "location"]])
    for f in combo:
        if f == "volume":
            s["volume"] = str(rng.randint(1, 99)); parts.append(f"volume={s['volume']}")
        elif f == "location":
            s["location"] = rng.choice(["Geneva", "Kyoto", "Toronto", "Melbourne"])
            parts.append(f"location={s['location']}")
        elif f == "pages":
            s["pages"] = f"{rng.randint(1,50)}-{rng.randint(51,120)}"; parts.append(f"pages={s['pages']}")
        changed.append(f)
    return s, changed, (f"Fabricated {', '.join(parts)} for a {seed['year']} {seed['venue']} "
                        f"paper; no connector indexes these peripheral fields.")


def m_format_variant(seed, rng, plus):
    s = copy.deepcopy(seed)
    s["title"] = seed["title"].lower()
    s["authors"] = [f"{p.split()[0][0]}. {p.split()[-1]}" if len(p.split()) >= 2 else p
                    for p in seed["authors"]]
    if not plus and seed.get("venue_iso"):
        s["venue"] = seed["venue_iso"]
    note = ("Format-only changes: title lowercased; authors abbreviated to initials"
            + ("" if plus else "; venue → ISO abbreviation") + ". Still the same valid citation.")
    return s, note


def m_etal(seed, rng, plus):
    s = copy.deepcopy(seed); a = seed["authors"]
    keep = 2 if len(a) > 2 else 1
    marker = "et al." if plus else "Others"
    s["authors"] = a[:keep] + [marker]
    return s, (f"Author list truncated from {len(a)} to {keep} + '{marker}'. "
               f"All listed authors are correct.")


# ---------------- optional LLM for H1 paraphrase / fabrication ----------------
def _llm_titles(prompts: list[str], model: str) -> list[str | None]:
    try:
        import boto3
    except Exception:
        return [None] * len(prompts)
    if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        return [None] * len(prompts)
    os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    out = []
    for pr in prompts:
        try:
            r = client.converse(modelId=model,
                                 messages=[{"role": "user", "content": [{"text": pr}]}],
                                 inferenceConfig={"temperature": 0.7, "maxTokens": 60})
            out.append(r["output"]["message"]["content"][0]["text"].strip().strip('"'))
        except Exception:
            out.append(None)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=str(OUT / "seeds_raw.json"))
    ap.add_argument("--rng-seed", type=int, default=20260723)
    ap.add_argument("--use-llm", action="store_true",
                    help="Use Bedrock for H1 paraphrase/fabrication (needs AWS_BEARER_TOKEN_BEDROCK).")
    ap.add_argument("--model", default="qwen.qwen3-vl-235b-a22b")
    args = ap.parse_args()

    rng = random.Random(args.rng_seed)
    seeds = json.loads(Path(args.seeds).read_text())
    rng.shuffle(seeds)
    need = sum(TARGET.values())
    if len(seeds) < need:
        print(f"[abort] need {need} seeds, have {len(seeds)}", file=sys.stderr); sys.exit(1)
    seeds = seeds[:need]
    print(f"[gen] {need} citations from {len(seeds)} unique seeds", flush=True)

    # assign contiguous seed slices per class
    order = ["H1", "H2", "H3", "H4", "H5", "H6", "P1", "P3",
             "R1", "R1_plus", "R2", "R2_plus", "R3", "R3_plus"]
    idx = 0; assign = {}
    for cls in order:
        assign[cls] = seeds[idx: idx + TARGET[cls]]; idx += TARGET[cls]

    meta = {}; per_class = {}; counters = {}

    def cid(cls):
        counters[cls] = counters.get(cls, 0) + 1
        base = cls.replace("_plus", "")
        return f"{base}-{counters[cls]:04d}" if not cls.endswith("_plus") else f"{base}_plus-{counters[cls]:04d}"

    # pre-generate H1 paraphrase/fabrication titles (LLM optional)
    h1_seeds = assign["H1"]
    modes = (["word_substitution"] * 27 + ["title_paraphrase"] * 27 +
             ["title_fabrication"] * (len(h1_seeds) - 54))
    rng.shuffle(modes)
    para_prompts, fab_prompts, para_i, fab_i = [], [], [], []
    for i, (s, mode) in enumerate(zip(h1_seeds, modes)):
        if mode == "title_paraphrase":
            para_i.append(i)
            para_prompts.append(f"Paraphrase this biomedical paper title so it describes a "
                                f"different-but-plausible paper (change the meaning), 6-14 words, "
                                f"no quotes:\n{s['title']}")
        elif mode == "title_fabrication":
            fab_i.append(i)
            fab_prompts.append(f"Invent a plausible but fake biomedical paper title in the same "
                               f"subfield as: '{s['title']}'. 6-14 words, no quotes.")
    if args.use_llm:
        print(f"[gen] LLM titles: {len(para_prompts)} paraphrase + {len(fab_prompts)} fabricate",
              flush=True)
        para_out = _llm_titles(para_prompts, args.model)
        fab_out = _llm_titles(fab_prompts, args.model)
    else:
        para_out = [None] * len(para_prompts); fab_out = [None] * len(fab_prompts)
    para_map = dict(zip(para_i, para_out)); fab_map = dict(zip(fab_i, fab_out))

    def build_h1(i, s):
        mode = modes[i]
        if mode == "word_substitution":
            nt, note = m_word_sub(s, rng)
        elif mode == "title_paraphrase":
            nt = para_map.get(i) or (m_word_sub(s, rng)[0])
            note = f"Title paraphrased to a different paper. Original: \"{s['title']}\"."
        else:
            nt = fab_map.get(i) or f"Molecular determinants of {s['_query_topic']} in clinical cohorts"
            note = f"Title fabricated. Original: \"{s['title']}\"."
        c = copy.deepcopy(s); c["title"] = nt
        return c, mode, ["title"], note

    def finalize(cls, c, mut, changed, note, subtype=None):
        st = subtype or cls.replace("_plus", "")
        k = cid(cls)
        c = copy.deepcopy(c)
        c["citation_id"] = k
        per_class.setdefault(cls, []).append(c)
        meta[k] = {"subtype": st, "label": LABEL[st[0]], "category": CATEGORY.get(st, st),
                   "mutation_type": mut, "changed_fields": changed, "explanation": note,
                   "seed": {kk: vv for kk, vv in c.items()
                            if kk not in ("citation_id",) and not kk.startswith("_mut")}}
        return k

    # H1
    for i, s in enumerate(assign["H1"]):
        c, mut, cf, note = build_h1(i, s); finalize("H1", c, mut, cf, note)
    # H2
    h2modes = (["author_addition_deletion"] * 27 + ["author_reordering"] * 27 +
               ["author_fabrication"] * (len(assign["H2"]) - 54))
    rng.shuffle(h2modes)
    for s, mode in zip(assign["H2"], h2modes):
        c = copy.deepcopy(s)
        if mode == "author_reordering":
            c["authors"], note = m_reorder(s, rng)
        elif mode == "author_addition_deletion":
            c["authors"], note = m_add_del(s, rng)
        else:
            c["authors"], note = m_fabricate_authors(s, rng)
        finalize("H2", c, mode, ["authors"], note)
    # H3
    for j, s in enumerate(assign["H3"]):
        also = j % 2 == 1
        nv, ny, note = m_venue(s, rng, also)
        c = copy.deepcopy(s); c["venue"] = nv; c["year"] = ny
        finalize("H3", c, "venue_year_fabrication" if also else "venue_fabrication",
                 ["venue", "year"] if also else ["venue"], note)
    # H4
    for s in assign["H4"]:
        ny, note = m_year(s, rng); c = copy.deepcopy(s); c["year"] = ny
        finalize("H4", c, "date_error", ["year"], note)
    # H5
    for j, s in enumerate(assign["H5"]):
        non = j % 2 == 1
        nd, note = m_doi(s, rng, non); c = copy.deepcopy(s); c["doi"] = nd
        c["url"] = f"https://doi.org/{nd}"
        finalize("H5", c, "doi_nonexistent" if non else "doi_fabrication", ["doi"], note)
    # H6
    h6kinds = (["pages_volume"] * 27 + ["publisher"] * 27 +
               ["location"] * (len(assign["H6"]) - 54))
    rng.shuffle(h6kinds)
    for s, kind in zip(assign["H6"], h6kinds):
        c, changed, note = m_peripheral(s, rng, kind)
        mt = {"pages_volume": "pages_volume_fabrication", "publisher": "publisher_fabrication",
              "location": "location_fabrication"}[kind]
        finalize("H6", c, mt, changed, note)
    # P1
    for s in assign["P1"]:
        c = copy.deepcopy(s); c["authors"], note = m_name_variant(s, rng)
        finalize("P1", c, "author_name_variant", ["authors"], note)
    # P3
    for s in assign["P3"]:
        c, changed, note = m_insufficient(s, rng)
        finalize("P3", c, "insufficient_field_evidence", changed, note)
    # R1 / R1_plus (no mutation)
    for cls in ("R1", "R1_plus"):
        for s in assign[cls]:
            finalize(cls, copy.deepcopy(s), "none", [],
                     "No mutation. All fields match the original valid citation.")
    # R2 / R2_plus
    for cls in ("R2", "R2_plus"):
        for s in assign[cls]:
            c, note = m_format_variant(s, rng, plus=cls.endswith("_plus"))
            finalize(cls, c, "format_variant", ["title", "authors", "venue"], note)
    # R3 / R3_plus
    for cls in ("R3", "R3_plus"):
        for s in assign[cls]:
            c, note = m_etal(s, rng, plus=cls.endswith("_plus"))
            finalize(cls, c, "et_al_abbreviation", ["authors"], note)

    # ---- write outputs ----
    OUT.mkdir(parents=True, exist_ok=True)
    allc = []
    for cls, items in per_class.items():
        (OUT / f"{cls}.json").write_text(json.dumps(items, ensure_ascii=False, indent=1))
        allc.extend(items)
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    (OUT / "manifest.json").write_text(json.dumps(
        {k: {"label": v["label"], "subtype": v["subtype"]} for k, v in meta.items()},
        ensure_ascii=False, indent=1))
    (OUT / "_all.json").write_text(json.dumps(allc, ensure_ascii=False, indent=1))

    # bib
    def esc(x): return str(x).replace("&", "\\&").replace("_", "\\_")
    lines = ["% Biomedical synthetic benchmark (biomed_v1); 1000 citations.",
             "% Key prefix: H=hallucinated, R=real, P=potential.", ""]
    for c in allc:
        k = c["citation_id"]
        etype = "article"
        lines.append(f"@{etype}{{{k},")
        lines.append(f"  title = {{{esc(c['title'])}}},")
        lines.append(f"  author = {{{' and '.join(esc(a) for a in c['authors'])}}},")
        lines.append(f"  journal = {{{esc(c['venue'])}}},")
        lines.append(f"  year = {{{c['year']}}},")
        if c.get("volume"): lines.append(f"  volume = {{{esc(c['volume'])}}},")
        if c.get("pages"): lines.append(f"  pages = {{{esc(c['pages'])}}},")
        if c.get("doi"): lines.append(f"  doi = {{{esc(c['doi'])}}},")
        if c.get("publisher"): lines.append(f"  publisher = {{{esc(c['publisher'])}}},")
        if c.get("location"): lines.append(f"  address = {{{esc(c['location'])}}},")
        lines.append("}\n")
    (OUT / "cleaned_bib.bib").write_text("\n".join(lines))

    from collections import Counter
    print("[done] wrote", OUT)
    print("  counts:", dict(Counter(v["subtype"] for v in meta.values())))
    print("  total:", len(meta))


if __name__ == "__main__":
    main()
