#!/usr/bin/env python3
"""Repair the three label defects found in data/synthetic_data/biomed_v1.

1. H6/location (14): fabricating a conference location on a journal article is
   the SAME construction as the P3 class, and no connector indexes the field
   (0/3124 candidate records carried a location). The H6 label ("verifiable
   against a source") is indefensible, so these are regenerated as genuine
   verifiable peripheral errors (pages+volume, which Crossref/PubMed do index).
2. H6/publisher null mutations: the fabricated publisher is drawn from a fixed
   pool and sometimes equals the journal's real publisher (Springer Nature for
   Nature, Elsevier for EBioMedicine). Those are re-drawn against the true value.
3. P1 (37): the mutation truncates the citation's given name ("Yuanyuan"->"Yua").
   That is only detectable if some source returns the FULL given name; when every
   source answers in NLM style ("An Y") the class is undecidable in principle.
   Seeds are probed live and only those whose authors come back with full given
   names are used.

Collective-author restoration is handled in harvest_biomed_seeds.py; existing
seeds are re-probed here so their author lists match what indexes report.
"""
from __future__ import annotations
import json, random, re, sys, time, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data" / "synthetic_data" / "biomed_v1"
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
UA = {"User-Agent": "citecheck/1.0 (mailto:eval@example.org)"}


def epmc_by_doi(doi: str) -> dict | None:
    url = (f"{EPMC}?query={urllib.parse.quote('DOI:' + doi)}"
           f"&format=json&resultType=core&pageSize=1")
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=25) as r:
            res = json.loads(r.read())["resultList"]["result"]
            return res[0] if res else None
    except Exception:
        return None


def crossref_by_doi(doi: str) -> dict | None:
    try:
        req = urllib.request.Request(
            f"https://api.crossref.org/works/{urllib.parse.quote(doi)}", headers=UA)
        with urllib.request.urlopen(req, timeout=25) as r:
            return json.loads(r.read())["message"]
    except Exception:
        return None


def authors_full_from_epmc(item: dict) -> tuple[list[str], bool]:
    """(authors, has_full_given_names)"""
    out, full = [], False
    for a in ((item.get("authorList") or {}).get("author") or []):
        fn, ln = (a.get("firstName") or "").strip(), (a.get("lastName") or "").strip()
        if fn and ln:
            out.append(f"{fn} {ln}")
            if len(fn) > 1:
                full = True
        elif a.get("collectiveName"):
            out.append(str(a["collectiveName"]).strip())
        elif a.get("fullName"):
            out.append(str(a["fullName"]).strip())
    return out, full


def crossref_has_full_names(m: dict) -> bool:
    for a in (m.get("author") or []):
        given = (a.get("given") or "").replace(".", "").strip()
        if len(given.split(" ")[0]) > 1:
            return True
    return False


def main() -> None:
    rng = random.Random(20260724 + 7)
    meta = json.loads((D / "meta.json").read_text())
    allc = json.loads((D / "_all.json").read_text())
    byid = {c["citation_id"]: c for c in allc}
    seeds = json.loads((D / "seeds_raw.json").read_text())
    used = {c.get("doi", "") for c in allc}
    free = [s for s in seeds if s["doi"] not in used]
    rng.shuffle(free)
    free_i = 0
    changed: dict[str, str] = {}

    # ---- defect 1: H6/location -> verifiable pages+volume -------------------
    h6loc = sorted(k for k, m in meta.items()
                   if m["subtype"] == "H6" and m["changed_fields"] == ["location"])
    print(f"[1] H6/location -> pages+volume: {len(h6loc)} 条", flush=True)
    for cid in h6loc:
        c = byid[cid]
        seed = meta[cid]["seed"]
        true_pages, true_vol = seed.get("pages", ""), seed.get("volume", "")
        c.pop("location", None)
        c["location"] = ""
        np_ = f"{rng.randint(1, 300)}-{rng.randint(301, 600)}"
        while np_ == true_pages:
            np_ = f"{rng.randint(1, 300)}-{rng.randint(301, 600)}"
        nv = str(rng.randint(1, 120))
        while nv == str(true_vol):
            nv = str(rng.randint(1, 120))
        c["pages"], c["volume"] = np_, nv
        meta[cid]["changed_fields"] = ["pages", "volume"]
        meta[cid]["mutation_type"] = "pages_volume_fabrication"
        meta[cid]["explanation"] = (
            f'Pages: "{true_pages}" -> "{np_}"; Volume: "{true_vol}" -> "{nv}" for a '
            f'{seed.get("year")} {seed.get("venue")} paper. Both fields are indexed by '
            f"CrossRef/PubMed/Europe PMC, so the error is verifiable against a source.")
        meta[cid]["seed"]["location"] = ""
        changed[cid] = "H6-location->pages_volume"

    # ---- defect 2: H6/publisher null mutations ------------------------------
    h6pub = sorted(k for k, m in meta.items()
                   if m["subtype"] == "H6" and m["changed_fields"] == ["publisher"])
    POOL = ["Elsevier", "Springer Nature", "Wiley", "Oxford University Press",
            "Taylor & Francis", "SAGE Publications", "Karger", "IOP Publishing"]
    print(f"[2] 核验 H6/publisher {len(h6pub)} 条是否撞上真实出版商 ...", flush=True)
    nulls = 0
    for cid in h6pub:
        m = crossref_by_doi(meta[cid]["seed"].get("doi", ""))
        time.sleep(0.1)
        true_pub = (m or {}).get("publisher") or ""
        inj = byid[cid].get("publisher", "")
        def same(a: str, b: str) -> bool:
            na = re.sub(r"[^a-z]", "", a.lower())
            nb = re.sub(r"[^a-z]", "", b.lower())
            return bool(na and nb) and (na in nb or nb in na)
        if same(inj, true_pub):
            alt = [p for p in POOL if not same(p, true_pub)]
            newp = rng.choice(alt)
            byid[cid]["publisher"] = newp
            meta[cid]["explanation"] = (
                f'Publisher fabricated: "" -> "{newp}" (true publisher is '
                f'"{true_pub}") for a {meta[cid]["seed"].get("year")} '
                f'{meta[cid]["seed"].get("venue")} paper (verifiable against source).')
            changed[cid] = f"H6-publisher null-mutation redrawn ({inj}->{newp})"
            nulls += 1
    print(f"    空变异修正: {nulls} 条", flush=True)

    # ---- defect 3: P1 on seeds whose sources return FULL given names --------
    p1 = sorted(k for k, m in meta.items() if m["subtype"] == "P1")
    print(f"[3] P1 {len(p1)} 条: 只保留所有源都返回全名的种子", flush=True)
    kept, replaced = 0, 0
    for cid in p1:
        doi = meta[cid]["seed"].get("doi", "")
        it = epmc_by_doi(doi); time.sleep(0.12)
        cr = crossref_by_doi(doi); time.sleep(0.1)
        ok_e = authors_full_from_epmc(it)[1] if it else False
        ok_c = crossref_has_full_names(cr) if cr else False
        if ok_e and ok_c:
            kept += 1
            continue
        # replace with a probed seed that satisfies the requirement
        newseed = None
        global_i = free_i
        while global_i < len(free):
            s = free[global_i]; global_i += 1
            it2 = epmc_by_doi(s["doi"]); time.sleep(0.12)
            cr2 = crossref_by_doi(s["doi"]); time.sleep(0.1)
            if not it2 or not cr2:
                continue
            a2, full2 = authors_full_from_epmc(it2)
            if full2 and crossref_has_full_names(cr2) and len(a2) >= 2:
                newseed = dict(s); newseed["authors"] = a2
                break
        free_i = global_i
        if not newseed:
            print(f"    [warn] {cid}: 找不到合格替换种子, 保留原样", flush=True)
            continue
        # apply the P1 mutation: truncate one given name to a 3-char prefix
        auth = list(newseed["authors"])
        idxs = [i for i, a in enumerate(auth)
                if len(a.split()) >= 2 and len(a.split()[0]) > 4]
        if not idxs:
            print(f"    [warn] {cid}: 替换种子无可截断名, 保留原样", flush=True)
            continue
        i = rng.choice(idxs)
        old = auth[i]; parts = old.split()
        auth[i] = f"{parts[0][:3]} {parts[-1]}"
        c = dict(newseed); c["citation_id"] = cid; c["authors"] = auth
        byid[cid] = c
        meta[cid] = {
            "subtype": "P1", "label": "POTENTIAL_HALLUCINATED", "category": "Author variant",
            "mutation_type": "author_name_variant", "changed_fields": ["authors"],
            "explanation": (
                f'Author name variant: "{old}" -> "{auth[i]}". '
                f"'{parts[0][:3]}' is a plausible short form of '{parts[0]}'. Seed verified: "
                f"both Europe PMC and CrossRef return full given names for this paper, so the "
                f"truncation is detectable rather than indistinguishable from NLM initials."),
            "seed": {k: v for k, v in newseed.items() if k != "citation_id"},
        }
        replaced += 1
        changed[cid] = "P1 seed replaced (sources verified non-NLM)"
    print(f"    保留 {kept} 条, 替换 {replaced} 条", flush=True)

    # ---- write everything back ---------------------------------------------
    allc = [byid[c["citation_id"]] for c in allc]
    (D / "_all.json").write_text(json.dumps(allc, ensure_ascii=False, indent=1))
    (D / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    for cls in {m["subtype"] for m in meta.values()} | {"R1_plus", "R2_plus", "R3_plus"}:
        ids = [k for k in meta if k.split("-")[0] == cls]
        if ids:
            (D / f"{cls}.json").write_text(json.dumps(
                [byid[i] for i in sorted(ids)], ensure_ascii=False, indent=1))

    def esc(x): return str(x).replace("&", "\\&").replace("_", "\\_")
    def entry(c):
        L = [f"@article{{{c['citation_id']},", f"  title = {{{esc(c['title'])}}},",
             f"  author = {{{' and '.join(esc(a) for a in c['authors'])}}},",
             f"  journal = {{{esc(c['venue'])}}},", f"  year = {{{c['year']}}},"]
        for f_, k_ in [("volume", "volume"), ("pages", "pages"), ("doi", "doi"),
                       ("publisher", "publisher"), ("location", "address")]:
            if c.get(f_):
                L.append(f"  {k_} = {{{esc(c[f_])}}},")
        L.append("}\n")
        return "\n".join(L)
    hdr = ("% Biomedical synthetic benchmark (biomed_v1); 1000 citations.\n"
           "% Key prefix: H=hallucinated, R=real, P=potential.\n\n")
    (D / "cleaned_bib.bib").write_text(hdr + "\n".join(entry(c) for c in allc))
    bib = (D / "cleaned_bib.bib").read_text()
    ents = [b for b in re.split(r"(?=@article\{)", bib)[1:] if b.strip().startswith("@")]
    sd = D / "shards"
    for f in sd.glob("*.bib"):
        f.unlink()
    for i in range(0, len(ents), 20):
        sd.joinpath(f"shard_{i//20+1:03d}.bib").write_text(hdr + "\n".join(ents[i:i+20]))
    (D / "repair_log.json").write_text(json.dumps(changed, ensure_ascii=False, indent=1))
    print(f"\n[done] 修改 {len(changed)} 条 -> repair_log.json; 已重建 bib 与 {len(list(sd.glob('*.bib')))} 分片")


if __name__ == "__main__":
    main()
