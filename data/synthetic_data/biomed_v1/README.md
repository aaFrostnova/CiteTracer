# Biomedical synthetic citation benchmark (biomed_v1)

1000 synthetic citations built to mirror `data/synthetic_data/v2` (the CS/ML
benchmark) so the detector can be evaluated on a second domain, addressing the
reviewer note that only CS papers were tested.

## Provenance
- **Seeds**: 2095 real biomedical papers harvested from Europe PMC (MEDLINE
  source, English, complete metadata) across 36 clinical/biology topics x
  2019-2024. Script: `scripts/harvest_biomed_seeds.py` -> `seeds_raw.json`.
  1000 unique seeds are used; each citation derives from a distinct real paper.
- **Generation**: `scripts/generate_biomed_synth.py` (deterministic, RNG seed
  20260723). Same 11-code taxonomy and mutation operators as v2. The LLM
  (Bedrock qwen3-vl-235b) is used only for H1 title paraphrase/fabrication;
  every other mutation is deterministic.

## Taxonomy and counts (proportional to v2, sums to 1000)
| code | n | label | mutation |
|---|---|---|---|
| H1 | 82 | HALLUCINATED | title word-substitution / paraphrase / fabrication |
| H2 | 81 | HALLUCINATED | author add/delete / reorder / fabricate |
| H3 | 80 | HALLUCINATED | venue (and sometimes year) swapped to a different real journal |
| H4 | 80 | HALLUCINATED | year shifted +/-1..5 |
| H5 | 82 | HALLUCINATED | DOI fabricated (non-existent or wrong paper) |
| H6 | 68 | HALLUCINATED | pages+volume / publisher / location fabricated (verifiable) |
| P1 | 37 | POTENTIAL | author name variant (nickname / initials) |
| P3 | 73 | POTENTIAL | volume/pages/publisher/location that no source indexes |
| R1 / R1_plus | 70 / 68 | REAL | no mutation |
| R2 / R2_plus | 71 / 68 | REAL | format variant (lowercase title, initials, ISO venue) |
| R3 / R3_plus | 72 / 68 | REAL | et al. / Others author truncation |

Real 417, Potential 110, Hallucinated 473.

## Files
- `<CODE>.json` per-class citation lists (same item schema as v2).
- `meta.json` `{cid: {subtype,label,category,mutation_type,changed_fields,explanation,seed}}`.
- `manifest.json` `{cid: {label, subtype}}`.
- `cleaned_bib.bib` BibTeX; key prefix encodes truth (H=hallucinated, R=real,
  P=potential), e.g. `@article{H3-0001, ...}`.
- `_all.json` flat list of 1000 citations.
- `seeds_raw.json` the harvested real-paper pool.

## Validation performed
- 0 label/prefix mismatches, 0 missing core fields, 0 duplicate titles, 0 truth
  leakage into bib fields, bib parses (1000/1000 entries).
- Every hallucinated field verified to differ from the true value; H1 word
  substitutions all change >=2 content words (v2 convention).
- 14-citation stratified smoke run through the real pipeline (`apps.bib_checker`)
  with live connectors + LLM: 11/14 matched the ground truth. The 3 non-matches
  were pipeline behavior, not data defects:
  - R1-0001 flagged H5 by a transient Phase-0 DOI-resolution failure under
    connector rate-limiting; the DOI resolves perfectly on Crossref (fields
    identical). See note below.
  - R2_plus / P3: known borderline classes (initials read as P1; P3 peripheral
    fabrication), same behavior as on the CS set.

## Note for running the full eval
Unlike the CS set, **every biomedical citation carries a real DOI**, so Phase-0
DOI resolution runs on all 1000. Under connector rate-limiting this can
transiently mislabel real papers as H5 (see R1-0001). Before the full run, seed
/ reuse the connector cache and keep connector concurrency modest so REAL recall
is not depressed by transient DOI-fetch failures.

Suggested run (BibTeX input, verifier-only style like the CS eval):
```
AWS_BEARER_TOKEN_BEDROCK=<key> python -m apps.bib_checker.run \
  --input data/synthetic_data/biomed_v1/cleaned_bib.bib \
  --out artifacts/biomed_v1_report --cache-path data/cache/biomed_v1_cache.sqlite \
  --citation-workers 16 --connector-workers 8
```
