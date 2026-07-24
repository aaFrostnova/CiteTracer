"""Bedrock LLM implementations of the 3 cascading agents with in-context learning."""
from __future__ import annotations

import json
import re
from typing import Any

from .models import (
    AuthorVariantResult,
    CandidateMatch,
    CitationRecord,
    DiscrepancyEvidence,
    DownstreamAgentResult,
    FieldComparisonResult,
    ValidAgentResult,
    VenueEquivalenceResult,
    VerdictLabel,
)


def _call_bedrock(
    client, model_id: str, prompt: str, max_tokens: int = 1024,
    max_retries: int = 2,
) -> dict:
    """Call Bedrock and parse JSON from response. Retries on transient errors
    (throttling, timeouts, JSON parse failures) with exponential backoff.
    """
    import sys as _sys
    import time as _time

    for attempt in range(max_retries + 1):
        try:
            response = client.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"temperature": 0, "maxTokens": max_tokens},
            )
            text = response["output"]["message"]["content"][0]["text"]
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                return json.loads(m.group(0))
            # LLM returned no JSON object at all — not a retryable condition.
            return {}
        except Exception as exc:
            msg = str(exc)
            is_throttle = (
                "Throttling" in msg or "TooManyRequests" in msg
                or "rate" in msg.lower() or "quota" in msg.lower()
            )
            if is_throttle:
                print(
                    f"[rate-limit] bedrock 429/Throttling on model={model_id}: {msg[:200]}",
                    file=_sys.stderr, flush=True,
                )
            if attempt >= max_retries:
                raise
            # Exponential backoff: 0.5s, 1s (then raise on the 3rd failure)
            _time.sleep(0.5 * (2 ** attempt))
    return {}  # unreachable


def _field_summary(fs: dict) -> str:
    lines = []
    for fname, info in fs.items():
        if not isinstance(info, dict):
            continue
        status = info.get("status", "")
        if status in ("both_missing",):
            continue
        ref = str(info.get("reference", ""))[:60]
        cand = str(info.get("candidate", ""))[:60]
        lines.append(f"  {fname}: {status} (ref='{ref}' cand='{cand}')")
    return "\n".join(lines) or "  (no fields to compare)"


def _parse_taxonomy(raw) -> list[str]:
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if str(t).strip()]
    if isinstance(raw, str):
        return [t.strip() for t in raw.split(",") if t.strip()]
    return []


# Map LLM-authored short issue names (e.g. "volume_missing") to the canonical
# forms produced by packages.core.cascading_agents.compute_hint. Downstream
# logic (P3 fallback, R4 coverage, HallucinatedAgent fallback taxonomy) relies
# on the canonical `*_candidate_missing` suffix; LLM output tends to drop
# `_candidate_` and say `volume_missing` instead.
_ISSUE_NAME_ALIASES = {
    "volume_missing":    "volume_candidate_missing",
    "pages_missing":     "pages_candidate_missing",
    "publisher_missing": "publisher_candidate_missing",
    "location_missing":  "location_candidate_missing",
    "venue_missing":     "venue_candidate_missing",
    "year_missing":      "year_candidate_missing",
    "doi_missing":       "doi_candidate_missing",
    "title_missing":     "title_candidate_missing",
    "author_missing":    "authors_candidate_missing",
    "authors_missing":   "authors_candidate_missing",
    "arxiv_id_missing":  "arxiv_id_candidate_missing",
}


def _normalize_issue_names(issues) -> list[str]:
    """Remap LLM-authored issue names to the canonical `compute_hint` forms."""
    out: list[str] = []
    for item in issues or []:
        if not isinstance(item, str):
            continue
        name = item.strip()
        if not name:
            continue
        out.append(_ISSUE_NAME_ALIASES.get(name, name))
    return out


def _filter_llm_issues_by_field_status(
    issues: list[str], field_status: dict
) -> list[str]:
    """Drop LLM-emitted *_candidate_missing issues that contradict field_status.

    The rule-based comparator is authoritative. LLM sometimes labels
    "reference has no volume, candidate has one" as volume_missing, which the
    alias map converts to volume_candidate_missing — triggering a false P3
    downgrade. Here we reject any *_candidate_missing issue when the actual
    field status is not candidate_missing.
    """
    out: list[str] = []
    suffix = "_candidate_missing"
    for issue in issues or []:
        if not issue.endswith(suffix):
            out.append(issue)
            continue
        field = issue[: -len(suffix)]
        actual = (field_status.get(field) or {}).get("status", "")
        if actual == "candidate_missing":
            out.append(issue)
    return out


# ===========================================================================
# FieldClassifier — unified authors + venue + publisher classification
# ===========================================================================
#
# Single LLM call per candidate. Runs BEFORE ValidAgent. Produces authoritative
# verdicts on three fields:
#   - authors:   exact / r2_initial / p1_variant / h2_error
#   - venue:     exact / alias / different
#   - publisher: exact / alias / different
#
# ValidAgent consumes these verdicts (without re-deriving), and R4 coverage
# respects them (p1_variant does NOT count as match; venue/publisher "alias"
# DOES count as match).

FIELD_CLASSIFIER_PROMPT = """You are a citation field equivalence classifier. For a single (citation, candidate) pair, produce authoritative verdicts for three fields at once: authors, venue, publisher.

====================================================================
AUTHORS — 4 categories
====================================================================

### Name decomposition (apply FIRST to every author pair)

Before comparing, decompose each author into (given_names, surname):
  - Default order is "Given ... Surname" — the SURNAME is the LAST token
    (after trimming DBLP numeric suffixes like "0001"/"0007" and honorifics).
    Example: "Xiangyang Liu" → given="Xiangyang", surname="Liu".
             "Xiang Liu"     → given="Xiang",     surname="Liu".
  - "Surname, Given" form — recognize when the first token ends with a comma
    OR the last token is an initial and first is a full word capitalized.
    Examples: "Guedj, B"         → given="B",    surname="Guedj".
              "Afouras T"        → given="T",    surname="Afouras"  (last-first + initial)
              "Mittal, T."       → given="T.",   surname="Mittal".
  - Middle names / initials belong to the given-name cluster, not the surname.
    Example: "Daniel J. Michael" → given="Daniel J.", surname="Michael".

### Hard rule — SURNAME EQUALITY

After decomposition, the SURNAMES must be byte-identical (case-insensitive,
diacritics-preserved but equal if same Unicode codepoints). If surnames
differ → h2_error. Do NOT treat first-name-like shorter strings as surname.

Examples that are SAME surname (→ continue to first-name rules):
  "Xiang Liu" vs "Xiangyang Liu" → surnames both "Liu" ✓ (first names differ)
  "Den Cai"   vs "Deng Cai"      → surnames both "Cai" ✓ (first names differ)
  "Mike Chen" vs "Michael Chen"  → surnames both "Chen" ✓
  "Afouras T" vs "Triantafyllos Afouras" → surnames both "Afouras" ✓

Examples where SURNAMES DIFFER (→ h2_error):
  "Xiang Liu"  vs "Xiang Wang"    → "Liu" ≠ "Wang"
  "Jian Zhou"  vs "Jian Chen"     → "Zhou" ≠ "Chen"

### 4 categories (evaluated on FIRST NAMES only, after surname equality is confirmed)

- **exact**: Every author pair is byte-identical after case normalization.

- **r2_initial**: Same people; the ONLY first-name difference is single-letter
  initial expansion (ONE alphabetic char, optionally period-terminated).

  ### HOW TO CHECK (apply to EACH differing first-name pair):
    STEP 1: Identify the shorter side (by alphabetic-char count).
    STEP 2: Count letters in the shorter side's first-name token(s).
    STEP 3:
       - If the shorter side is EXACTLY 1 alphabetic letter (with or
         without trailing period) AND the longer side is a full word
         starting with that same letter → this pair IS r2_initial.
       - If the shorter side has 2 OR MORE letters → this pair is NOT
         r2_initial (it is p1_variant — multi-letter truncation).
    Direction does NOT matter: "A. Bulatov" vs "Aydar Bulatov" and
    "Aydar Bulatov" vs "A. Bulatov" are BOTH r2_initial.

  Valid r2_initial examples (all 1-char initial expansion):
    "G. Hao"        ↔ "Gao Hao"               (G = 1 char)
    "S. Bubeck"     ↔ "Sébastien Bubeck"      (S = 1 char)
    "A. Bulatov"    ↔ "Aydar Bulatov"         (A = 1 char)
    "J. Liu"        ↔ "Jiduan Liu"            (J = 1 char)
    "K. Alshammari" ↔ "Khaznah Alshammari"    (K = 1 char)
    "D. D. Johnson" ↔ "Daniel D. Johnson"     (D = 1 char, middle D. unchanged)
    "J. E. Santos"  ↔ "Javier E. Santos"      (J = 1 char, E. unchanged)
    "V. Manohar"    ↔ "Vimal Manohar"         (V = 1 char)
    "Erik L Bolager"  ↔ "Erik Lien Bolager"   (middle initial L → Lien; first token "Erik" unchanged)
    "Daniel D Johnson" ↔ "Daniel David Johnson" (middle initial D → David; "Daniel" unchanged)
    "Mary J Smith"    ↔ "Mary Jane Smith"     (middle initial J → Jane; "Mary" unchanged)
    "Sun, Y"        ↔ "Yu Sun"                (Y = 1 char; last-first notation, decompose to given=Y, surname=Sun → r2)
    "Wohlberg, B"   ↔ "Brendt Wohlberg"       (B = 1 char; last-first notation → r2)
    "Kamilov, U. S" ↔ "Ulugbek S. Kamilov"    (U = 1 char; middle "S" matches → r2)
    "Afouras T"     ↔ "Triantafyllos Afouras" (T = 1 char; space-separated last-first → r2)

  ### TOKEN-BY-TOKEN RULE for multi-token first names:
    Split each side's first-name into tokens. Pair them positionally.
    For each token-pair, apply the 1-char check independently. If every
    differing pair is a 1-char initial expansion (regardless of position
    within the first name), the overall verdict is r2_initial.
    DO NOT count letters across the entire first-name string; count only
    within each individual token. A 1-char middle-initial token is r2_initial
    even when other tokens are full words.

  NOT r2_initial (shorter side is 2+ letters → use p1_variant):
    "Chao Xiao"     ↔ "Chaowei Xiao"   (Chao = 4 chars → p1 multi-letter trunc)
    "Stef Jegelka"  ↔ "Stefanie Jegelka" (Stef = 4 chars → p1)
    "Erk Zhu"       ↔ "Erkang Zhu"     (Erk = 3 chars → p1)
    "Mike Chen"     ↔ "Michael Chen"   (Mike = 4 chars → p1 nickname)

  ### COMMON MISCLASSIFICATIONS TO AVOID:
    - Do NOT mark "A. Bulatov" ↔ "Aydar Bulatov" as p1_variant.
      "A." is exactly 1 character → this is r2_initial by rule.
    - Do NOT mark "J. Liu" ↔ "Jiduan Liu" as p1_variant.
      "J." is exactly 1 character → r2_initial.
    - Do NOT mark "K. Alshammari" ↔ "Khaznah Alshammari" as p1_variant.
    - Do NOT mark "Sun, Y" ↔ "Yu Sun" as p1_variant. The comma is just
      "Surname, Given" notation. Decompose first → given="Y", surname="Sun".
      Then 1-char rule applies → r2_initial.
    - Do NOT mark "Afouras T" ↔ "Triantafyllos Afouras" as p1_variant.
      Last-first space-separated notation → decompose to given="T",
      surname="Afouras". 1-char initial → r2_initial.
    - When EVERY differing pair is 1-char-initial-expansion, the
      overall verdict MUST be r2_initial, not p1_variant.

- **p1_variant**: Same people (SURNAMES equal), but first-name form differs by:
    * Nickname / diminutive: "Mike" ← "Michael", "Nando" ← "Fernando",
      "Dima" ← "Dmitrii", "Masa" ← "Masashi"
    * Multi-letter truncation (≥2 preserved chars): "Chao" ← "Chaowei",
      "Stef" ← "Stefanie", "Edw." ← "Edward", "Xiang" ← "Xiangyang"
    * Single-char typo / transliteration: "Ulle" ↔ "Ullie", "Den" ↔ "Deng",
      "Wei" ↔ "William"
    * NOTE on last-first / "Surname, Given" notation: this is just citation
      formatting, not a name variant. Apply the decomposition rule above
      first; the surface order does NOT influence p1 vs r2. After
      decomposition, classify by the first-name comparison rules:
        - "Sun, Y"     ↔ "Yu Sun"            → r2_initial (Y is 1 char)
        - "Wohlberg, B" ↔ "Brendt Wohlberg"  → r2_initial (B is 1 char)
        - "Afouras T"  ↔ "Triantafyllos Afouras" → r2_initial (T is 1 char)
        - "Liu, Mike"  ↔ "Michael Liu"       → p1_variant (Mike is nickname)
    * MIDDLE-NAME VARIATIONS — any of these five subtypes (all → p1_variant):
        a. Middle initial dropped:    "Dmitry P. Vetrov"   ↔ "Dmitry Vetrov"
        b. Middle initial added:      "Dmitry Vetrov"      ↔ "Dmitry P. Vetrov"
        c. Middle full name dropped:  "Aditya Manish Kane" ↔ "Aditya Kane"
        d. Middle full name added:    "Aditya Kane"        ↔ "Aditya Manish Kane"
        e. Middle initial ↔ full:     "Dmitry P. Vetrov"   ↔ "Dmitry Petrov Vetrov"
                                      "Richard S. Zemel"   ↔ "Richard Samuel Zemel"
      The SURNAME (last token) must still match byte-identically. Middle-name
      add/drop NEVER escalates to h2_error — it is always same-person P1.
    * Dropped middle word(s) in longer name: "Sai Zhang" ↔ "Sai Qian Zhang"
      (same as middle-full-name-dropped above).
  DBLP disambiguation suffixes ("Ting Chen 0007") = SAME person, strip suffix
  before decomposition.

- **h2_error**: Genuinely different people OR authors reordered. Applies when
  ANY of the following — ALL are h2_error, NOT p1_variant, NOT r2_initial:
    * Different SURNAME (after decomposition).
    * **COUNT MISMATCH — different number of authors, with NO truncation
      marker (et al. / and others / Others / …) on the citation side**
      (see dedicated section below — this is the #2 missed rule).
    * Added author with no truncation marker to justify.
    * Deleted author with no truncation marker to justify.
    * **REORDERED authors — same set of people, different position order**
      (see dedicated section below — this is the #1 missed rule).
    * Fabricated name: surname not present in the candidate at all.

### ★★★ COUNT MISMATCH = h2_error — DEDICATED RULE ★★★

Apply this check FIRST, before looking at r2_initial / p1_variant.

COUNT CHECK PROCEDURE:
  1. Strip truncation markers from the citation list ("et al.", "and others",
     "Others", "…"). If any marker is present, the citation is allowed to
     have fewer listed authors than the candidate — skip the count check.
  2. Drop COLLECTIVE / CORPORATE author rows from BOTH sides before counting.
     These are consortium or working-group names, not people, and indexes
     disagree wildly on whether to list them (and on how many members to
     enumerate). Recognise them by the absence of a personal-name shape:
     they contain words like "Group", "Consortium", "Investigators",
     "Collaboration", "Collaborators", "Network", "Team", "Study",
     "Committee", "Trial", "Authors", or are an ALL-CAPS acronym.
     Examples: "IMbrave150 Investigators", "ESC Scientific Document Group",
     "Darwin Tree of Life Consortium", "NGS-SA", "COMMIT-KZN Team".
     A citation omitting such a row is NOT an author deletion.
  3. Otherwise, if the remaining len(citation) != len(candidate) →
     **h2_error, count mismatch.** Return immediately; do NOT try to explain
     the extra/missing author as a name variant or initial expansion.

COUNT EXAMPLES — ALL h2_error:

  Example 1 — citation has ONE EXTRA author:
    citation : ["A. Raffin", "Ashley Hill", "A. Gleave", "Elena Voss",
                "A. Kanervisto", "M. Ernestus", "Noah Dormann"]   (7)
    candidate: ["Antonin Raffin", "Ashley Hill", "Adam Gleave",
                "Anssi Kanervisto", "Maximilian Ernestus", "Noah Dormann"] (6)
    No truncation marker; counts differ → h2_error (fabricated author
    "Elena Voss" injected into citation).
    Correct reason: "Citation lists 7 authors, candidate lists 6, no et al.
    marker → fabricated author. h2_error."
    WRONG output: "r2_initial" (ignoring the extra author is forbidden).

  Example 2 — citation is MISSING an author:
    citation : ["Jiayu Qin", "Jian Chen", "Rohan Sharma", "Jingchen Sun"]  (4)
    candidate: ["Jiayu Qin", "Jian Chen", "Rohan Sharma", "Jingchen Sun",
                "Changyou Chen"]                                             (5)
    No truncation marker; counts differ → h2_error (deletion of one author
    by the citation).
    Correct reason: "Citation has 4 authors, candidate has 5, no et al.
    → deleted author. h2_error."
    WRONG output: "exact" (ignoring the missing author is forbidden).

  Example 3 — count differs BUT citation has truncation marker → NOT h2_error:
    citation : ["Canwen Xu", "Zexue He", "Zhankui He", "and others"]   (3+marker)
    candidate: ["Canwen Xu", "Zexue He", "Zhankui He", "Julian McAuley"] (4)
    Truncation marker present → count difference is allowed.
    Proceed to per-pair surname/first-name analysis on the explicit authors.

### ★★★ REORDER = h2_error — DEDICATED RULE (do NOT merge into p1_variant) ★★★

Reorder means: the citation's author list and the candidate's author list
contain the SAME PEOPLE (surnames identical as a multiset), but their
POSITIONS do not align. Even though no new person was added and none was
deleted, different order is treated as a fabrication/error by this pipeline.

REORDER CHECK PROCEDURE (apply after confirming count + surname-set match):
  For i in 0..len-1:
    if normalized_surname(citation[i]) != normalized_surname(candidate[i]):
        → REORDER detected → return h2_error
  (A one-off swap at ANY position is enough; you do NOT need most or all
  positions to be misaligned.)

REORDER EXAMPLES — ALL h2_error:

  Example A — plain swap:
    citation : ["Alice Zhang", "Bob Li", "Carol Wu"]
    candidate: ["Bob Li", "Alice Zhang", "Carol Wu"]
    Position 0: surname "Zhang" vs "Li" → different → REORDER → h2_error.
    Correct reason: "Authors are the same three people but positions 0 and 1
    are swapped. Reorder → h2_error."
    Do NOT say "same set of people → p1_variant".

  Example B — reorder with DBLP suffix that must be stripped first:
    citation : ["Zhengdao Chen", "Lei Chen", "Joan Bruna"]
    candidate: ["Joan Bruna", "Zhengdao Chen", "Lei Chen 0062"]
    Strip DBLP suffix: "Lei Chen 0062" → "Lei Chen".
    Position 0: "Chen" vs "Bruna" → different → h2_error.

  Example C — reorder with initial expansion that would otherwise look like r2_initial:
    citation : ["Wotao Yin", "Lin F. Yang", "Simon S. Du"]
    candidate: ["W. Yin", "S. Du", "Lin F. Yang"]
    Even though each pair COULD individually be explained as initial
    expansion, the POSITIONS are wrong (position 1 is Yang vs Du).
    REORDER detected → h2_error. **REORDER OVERRIDES r2_initial.**

  Example D — reorder with last-first format:
    citation : ["Yuhang Zhou", "Jing Zhu"]
    candidate: ["Zhu, Jing", "Zhou, Yuhang"]
    Even though "Zhou, Yuhang" ≡ "Yuhang Zhou" by last-first format, the
    positions are swapped → h2_error.

COUNTER-EXAMPLE (NOT reorder, still p1_variant):
    citation : ["Sai Zhang", "Bob Li"]
    candidate: ["Sai Qian Zhang", "Bob Li"]
    Position 0: "Zhang" vs "Zhang" — same surname, only middle name
    difference → NOT reorder → p1_variant.

### COMMON MISCLASSIFICATIONS TO AVOID (REORDER EDITION):

  - "Same set of authors, just different order → p1_variant"
    WRONG. Same set + different order = REORDER = h2_error.

  - "All surnames present in both lists → no author added or deleted → p1_variant"
    WRONG. Positional mismatch still triggers h2_error even if set-equal.

  - "DBLP suffix + reordering → p1_variant"
    WRONG. Strip suffix first, then check positions. Reorder → h2_error.

### Authors decision order (first match wins):
0. **Truncation marker**: if citation's LAST author is "et al.", "et al",
   "Others", "and others", "et.al.", "etal", "...", "etc." (case-insensitive),
   the citation is truncated. Check only the N = len(listed non-marker)
   citation authors vs the candidate's first N authors. The candidate having
   MORE authors is NOT an error. Apply rules 3–5 to the N listed pairs to
   pick the strongest variant. If any listed author is missing from the
   candidate's first N positions → h2_error. NEVER h2_error just because
   count differs with a truncation marker present.
1. Count differs AND no truncation marker → h2_error.
2. **POSITIONAL SURNAME CHECK**: strip DBLP suffixes, decompose each author
   into (given, surname). Walk positions i = 0..N-1. If at ANY position
   i the citation's surname ≠ candidate's surname → h2_error.
   This catches two sub-cases:
     - Different surname sets (person X fabricated, person Y missing).
     - Same set but reordered positions (REORDER).
   **Rule 2 overrides rules 3/4/5.** If rule 2 fires, you MUST return
   h2_error even if every pair would individually look like exact / r2_initial
   / p1_variant.
3. All pairs byte-identical after case/whitespace normalization → exact.
4. **All first-name differences are single-letter initial only → r2_initial.**
   (Rule 4 has PRIORITY over rule 5. If every differing pair passes the
   3-step check above with shorter-side = 1 char, you MUST return
   r2_initial, not p1_variant.)
5. Otherwise (surnames all match AT SAME POSITIONS; at least ONE differing
   pair has shorter-side ≥ 2 letters, or involves nickname / typo /
   transliteration / dropped middle) → p1_variant.

====================================================================
VENUE — 3 categories
====================================================================

- **exact**: Byte-identical after case/punctuation normalization.

- **alias**: Same venue, expressed differently. Acceptable patterns:
    * Acronym ↔ full name: "NeurIPS" ↔ "Advances in Neural Information
      Processing Systems"; "ICML" ↔ "International Conference on Machine
      Learning"; "EMNLP" ↔ "Conference on Empirical Methods in NLP".
    * Proceedings prefix: "Proceedings of the 60th Annual Meeting of the
      Association for Computational Linguistics" ↔ "ACL".
    * Dot-abbreviated: "J. Mach. Learn. Res." ↔ "Journal of Machine
      Learning Research".
    * Year-tagged form: "EMNLP 2024" ↔ "EMNLP".
    * Preprint server synonyms: "Preprint" ↔ "arXiv" ↔ "CoRR".
    * Sub-track of the same conference, expressed in shorthand. Citations
      routinely abbreviate "Findings of the Association for Computational
      Linguistics: EMNLP 2024" or "Proceedings of the 60th Annual Meeting
      of the ACL: Student Research Workshop" as just the parent acronym
      ("ACL" or "EMNLP"). Treat these as alias when the candidate side
      clearly names the parent conference (ACL / EMNLP / NAACL / EACL /
      AACL) and only adds a track / workshop / Findings qualifier.
      Examples that are alias:
        cit="ACL"   cand="Findings of the Association for Computational Linguistics: EMNLP 2024" → alias
        cit="ACL"   cand="Findings of the Association for Computational Linguistics: ACL 2024" → alias
        cit="ACL"   cand="Proceedings of the 60th Annual Meeting of the ACL: Student Research Workshop" → alias
        cit="EMNLP" cand="Findings of the Association for Computational Linguistics: EMNLP 2023" → alias
        cit="ACL"   cand="Proceedings of the 58th Annual Meeting of the ACL" → alias

- **different**: Genuinely different conferences / journals. Do NOT invent
  mappings beyond the above patterns. In particular:
    * "ICLR" ≠ "ACL Proceedings"
    * "ACL" ≠ "NAACL" ≠ "EACL" ≠ "AACL" (all distinct conferences with
      their own program committees)
    * "EMNLP" ≠ "Annual Meeting of ACL" (when the candidate clearly names
      ACL as the host conference, not EMNLP)
    * "KDD" ≠ "ACL"
    * "NeurIPS" ≠ "ICML"
    * "CVPR" ≠ "ECCV"
    * "Nature" ≠ "EMNLP"
    * "IEEE TPAMI" ≠ "ICLR"
    * Published venue ≠ arXiv preprint (treat "TPAMI" vs "arXiv" as different,
      unless the citation explicitly says Preprint/arXiv/CoRR on the reference
      side).

### ★★★ VENUE ALIAS GUARD — DO NOT INVENT MAPPINGS ★★★
A venue is alias ONLY if one side is a well-known acronym / abbreviation of
the other side's FULL literal name, OR if the candidate side names the same
parent conference as the citation acronym (with an optional track / workshop
/ Findings qualifier). Two different conferences that merely share a parent
society (ACL vs NAACL, ICML vs NeurIPS) are NOT aliases.

ACL-family disambiguation — quick reference:

  A) cit="ACL"   cand="Proceedings of the 60th Annual Meeting of the ACL" → alias
  B) cit="ACL"   cand="Findings of the Association for Computational Linguistics: ACL 2024" → alias
  C) cit="ACL"   cand="Findings of the Association for Computational Linguistics: EMNLP 2024" → alias (Findings is a single cross-conference track that ACL-family citations routinely tag with whichever parent conference happened to host it; the citation acronym still resolves to the same publication track)
  D) cit="ACL"   cand="Proceedings of the 60th Annual Meeting of the ACL: Student Research Workshop" → alias (workshop track of ACL is still ACL-family)
  E) cit="EMNLP" cand="Findings of the Association for Computational Linguistics: EMNLP 2023" → alias
  F) cit="ACL"   cand="NAACL 2024" → different (NAACL is a separate conference, not an ACL track)
  G) cit="EMNLP" cand="Proceedings of the 58th Annual Meeting of the ACL" → different (host conference is ACL, not EMNLP)

Decision rule: if the citation is a bare acronym (ACL / EMNLP / NAACL / EACL /
AACL) and the candidate side names the SAME acronym anywhere in its title
(possibly with "Findings of", "Workshop", "Student Research", "Proceedings of
the N-th Annual Meeting of the" qualifiers), say **alias**. If the candidate
side names a DIFFERENT acronym from that family, say **different**.

====================================================================
PUBLISHER — 3 categories (same structure as venue)
====================================================================

- **exact**: Byte-identical after normalization.

- **alias**: Same publisher, different form. Acceptable patterns:
    * Acronym ↔ full name: "ACM" ↔ "Association for Computing Machinery";
      "IEEE" ↔ "Institute of Electrical and Electronics Engineers"; "AAAI"
      ↔ "Association for the Advancement of Artificial Intelligence"; "ACL"
      (as publisher) ↔ "Association for Computational Linguistics".
    * "PMLR" ↔ "Proceedings of Machine Learning Research".
    * Minor variants: "Springer" ↔ "Springer Nature" ↔ "Springer-Verlag";
      "Elsevier" ↔ "Elsevier B.V."; "MIT Press" ↔ "The MIT Press".

- **different**: Unrelated publishers. E.g. "Elsevier" vs "Springer",
  "Morgan Kaufmann" vs "PMLR". Critical negative examples:
    * "ACM" ≠ "ACL" / "Association for Computational Linguistics"
      (ACM = Association for Computing **Machinery**, a completely different
      organization from ACL. Do NOT alias these just because both acronyms
      start with "A-C".)
    * "IEEE" ≠ "ACM" (two separate societies)
    * "Springer" ≠ "Elsevier" (both big, still distinct)
    * "Nature Publishing Group" ≠ "AAAI"

### ★★★ PUBLISHER ALIAS GUARD — ACRONYM LOOK-ALIKES ARE NOT ALIASES ★★★
Publisher alias is allowed ONLY when one side literally expands the other:
  - "ACM" ↔ "Association for Computing Machinery" — OK
  - "ACL" ↔ "Association for Computational Linguistics" — OK
  - "ACM" ↔ "Association for Computational Linguistics" — **NEVER** (different
    organizations; same-ish 3-letter acronym is coincidence, not equivalence).
If you find yourself reasoning "ACM is an alias for X because Y publishes
at venue Z", STOP — that is invented. Default to **different**.

====================================================================
INPUT
====================================================================
Citation authors:   {citation_authors}
Candidate authors:  {candidate_authors}

Citation venue:     {citation_venue!r}
Candidate venue:    {candidate_venue!r}

Citation publisher: {citation_publisher!r}
Candidate publisher: {candidate_publisher!r}

====================================================================
OUTPUT (JSON only, no other text)
====================================================================

★★★ COMMIT RULE — ORDER: reason FIRST, then overall ★★★

For each field, write the "reason" key FIRST. Reason is where you reason
through the evidence step by step. The "overall" key comes AFTER the reason
and MUST match the final conclusion of that reason.

  NEVER output a JSON where reason ends with "Final verdict: different"
  but overall="alias". NEVER say "ACM ≠ ACL, this should be different"
  in reason and then write "alias". If your analysis concludes X, overall
  is X — full stop, no self-correction tail, no flip-flop.

If you catch yourself writing a self-correcting reason ("Correction: ...",
"But actually ...", "Final verdict: ..."), re-read your own conclusion and
set overall to THAT conclusion before closing the JSON.

Required shape (note the key order: reason first, overall second):

{{
  "authors":   {{"reason": "...", "overall": "exact"|"r2_initial"|"p1_variant"|"h2_error"}},
  "venue":     {{"reason": "...", "overall": "exact"|"alias"|"different"}},
  "publisher": {{"reason": "...", "overall": "exact"|"alias"|"different"}}
}}

If a side is empty for venue or publisher (e.g. citation provides venue but
candidate does not), return "different" with a reason like "candidate missing"
— the caller will treat missing-side the same as rule-based candidate_missing."""


class BedrockFieldClassifier:
    """Unified LLM classifier for authors + venue + publisher in one call."""

    def __init__(self, client, model_id: str) -> None:
        self.client = client
        self.model_id = model_id

    _TRUNCATION_MARKERS = {
        "et al.", "et al", "et.al.", "et.al", "etal",
        "others", "and others", "... et al.", "...", "etc.",
    }

    def _is_truncation_marker(self, s: str) -> bool:
        return str(s or "").strip().lower() in self._TRUNCATION_MARKERS

    @staticmethod
    def _canon(s: str) -> str:
        return " ".join(str(s or "").strip().lower().split())

    @staticmethod
    def _name_parts(name: str) -> tuple[list[str], str]:
        """(given tokens, surname) with accents and periods stripped."""
        import unicodedata
        s = "".join(c for c in unicodedata.normalize("NFD", str(name or ""))
                    if unicodedata.category(c) != "Mn")
        s = " ".join(s.replace(".", " ").split())
        t = [x for x in s.split(" ") if x]
        if not t:
            return [], ""
        # NLM style puts the surname first and a concatenated initials block last
        # ("ElSayed NA", "Kombe Kombe AJ"): the trailing all-caps block is the
        # given-name initials, everything before it is the surname.
        if len(t) >= 2 and 1 <= len(t[-1]) <= 3 and t[-1].isupper() and t[-1].isalpha():
            return list(t[-1]), " ".join(t[:-1]).lower()
        return t[:-1], t[-1].lower()

    @staticmethod
    def _expand_initial_blocks(given: list[str]) -> list[str]:
        """'NA' -> ['N','A'] for NLM-style concatenated initials; else unchanged."""
        out: list[str] = []
        for g in given:
            if 1 < len(g) <= 3 and g.isupper() and g.isalpha():
                out += list(g)
            else:
                out.append(g)
        return out

    @classmethod
    def _initials_equivalent(
        cls, citation_authors: list[str], candidate_authors: list[str],
    ) -> str | None:
        """'exact' / 'r2_initial' when every aligned pair differs only by initials
        formatting; None when any pair is ambiguous (so the LLM still decides)."""
        def pair_ok(a: str, b: str) -> bool:
            if a.lower() == b.lower():
                return True
            if len(a) == 1 and len(b) > 1 and a.upper() == b[0].upper():
                return True
            if len(b) == 1 and len(a) > 1 and b.upper() == a[0].upper():
                return True
            return False  # multi-letter truncation stays ambiguous

        # Lists must be the same length. Comparing only the aligned prefix would
        # silently mask an author added or removed at the tail (H2), so any length
        # difference is left to the LLM / existing truncation handling.
        if not citation_authors or len(citation_authors) != len(candidate_authors):
            return None
        n = len(citation_authors)
        saw_expansion = False
        for i in range(n):
            ga, sa = cls._name_parts(citation_authors[i])
            gb, sb = cls._name_parts(candidate_authors[i])
            if not sa or sa != sb:
                return None
            ga, gb = cls._expand_initial_blocks(ga), cls._expand_initial_blocks(gb)
            if len(ga) != len(gb):
                return None  # middle-name add/drop: leave to existing semantics
            for x, y in zip(ga, gb):
                if not pair_ok(x, y):
                    return None
                if x.lower() != y.lower():
                    saw_expansion = True
        return "r2_initial" if saw_expansion else "exact"

    def _fast_authors(
        self, citation_authors: list[str], candidate_authors: list[str],
    ) -> AuthorVariantResult | None:
        """Deterministic authors fast-path; None if LLM needed."""
        if not citation_authors and not candidate_authors:
            return AuthorVariantResult(overall="exact", reason="both lists empty")
        ca = [self._canon(x) for x in (citation_authors or [])]
        da = [self._canon(x) for x in (candidate_authors or [])]
        if ca and ca == da:
            return AuthorVariantResult(overall="exact", reason="byte-identical after normalization")
        # Initials-convention fast path. Covers ONLY unambiguous formatting deltas:
        #   1. initials punctuation ("Anne F Peery" vs "Anne F. Peery")
        #   2. single-letter <-> full-word expansion ("M. Biton" vs "Moshe Biton")
        #   3. NLM concatenated initials block ("ElSayed NA" vs "Nuha A ElSayed")
        # Multi-letter truncation ("Ty" vs "Tyler") and middle-name add/drop are
        # deliberately NOT accepted here: those stay p1_variant, preserving the
        # existing taxonomy semantics.
        if citation_authors and candidate_authors:
            _v = self._initials_equivalent(citation_authors, candidate_authors)
            if _v:
                return AuthorVariantResult(
                    overall=_v,
                    reason=("Same people; difference is initials formatting only "
                            "(punctuation, initial expansion, or NLM initials block)."),
                )
        # Truncation marker: strip last + match prefix
        if citation_authors and self._is_truncation_marker(citation_authors[-1]):
            listed_canon = [self._canon(a) for a in citation_authors[:-1]]
            if (
                len(listed_canon) <= len(da)
                and listed_canon == da[: len(listed_canon)]
            ):
                return AuthorVariantResult(
                    overall="exact",
                    reason=(
                        f"Citation truncated with '{citation_authors[-1]}'; "
                        f"{len(listed_canon)} listed authors match candidate's "
                        f"first {len(listed_canon)} byte-identically."
                    ),
                )
        return None

    def _fast_venue_or_publisher(self, ref: str, cand: str) -> VenueEquivalenceResult | None:
        """Return deterministic verdict for venue/publisher if possible;
        None if LLM needed for alias judgment."""
        r, c = self._canon(ref), self._canon(cand)
        if not r and not c:
            return VenueEquivalenceResult(overall="exact", reason="both empty")
        if r == c and r:
            return VenueEquivalenceResult(overall="exact", reason="byte-identical after normalization")
        if not r or not c:
            # One side missing — defer to caller (field_status will be
            # candidate_missing/reference_missing); classifier not authoritative.
            return VenueEquivalenceResult(
                overall="different",
                reason=("reference empty" if not r else "candidate empty"),
            )
        return None

    def classify(
        self,
        citation_authors: list[str],
        candidate_authors: list[str],
        citation_venue: str = "",
        candidate_venue: str = "",
        citation_publisher: str = "",
        candidate_publisher: str = "",
    ) -> "FieldClassificationResult":
        from .models import FieldClassificationResult  # local import to avoid cycles

        # Per-field fast paths
        authors_fast = self._fast_authors(citation_authors, candidate_authors)
        venue_fast = self._fast_venue_or_publisher(citation_venue, candidate_venue)
        publisher_fast = self._fast_venue_or_publisher(citation_publisher, candidate_publisher)

        # Try deterministic venue/publisher via existing normalize.py heuristic
        if venue_fast is None:
            from .normalize import venues_equivalent_heuristic
            if venues_equivalent_heuristic(citation_venue, candidate_venue):
                venue_fast = VenueEquivalenceResult(
                    overall="alias",
                    reason="matched via normalize.venues_equivalent_heuristic (acronym/truncation/alias)",
                )
        if publisher_fast is None:
            from .normalize import venues_equivalent_heuristic
            if venues_equivalent_heuristic(citation_publisher, candidate_publisher):
                publisher_fast = VenueEquivalenceResult(
                    overall="alias",
                    reason="matched via normalize.venues_equivalent_heuristic",
                )

        # If all three fully resolved, skip LLM entirely
        if authors_fast is not None and venue_fast is not None and publisher_fast is not None:
            return FieldClassificationResult(
                authors=authors_fast,
                venue=venue_fast,
                publisher=publisher_fast,
            )

        # Otherwise call LLM (it decides all three in one pass — cheaper than
        # individual calls, and provides consistent judgment across fields).
        prompt = FIELD_CLASSIFIER_PROMPT.format(
            citation_authors=list(citation_authors or []),
            candidate_authors=list(candidate_authors or []),
            citation_venue=citation_venue or "",
            candidate_venue=candidate_venue or "",
            citation_publisher=citation_publisher or "",
            candidate_publisher=candidate_publisher or "",
        )
        try:
            # Long biomedical author lists overrun the default budget; a truncated
            # response has no closing brace, so _call_bedrock returns {} silently
            # and every field defaults to p1_variant. Give the classifier room.
            parsed = _call_bedrock(self.client, self.model_id, prompt, max_tokens=2048)
        except Exception:
            # LLM failure → conservative fallback (preserve whatever fast
            # paths resolved, use p1_variant / different for the unresolved).
            return FieldClassificationResult(
                authors=authors_fast or AuthorVariantResult(
                    overall="p1_variant",
                    reason="classifier LLM failed; fallback to p1_variant",
                ),
                venue=venue_fast or VenueEquivalenceResult(
                    overall="different",
                    reason="classifier LLM failed; conservative fallback",
                ),
                publisher=publisher_fast or VenueEquivalenceResult(
                    overall="different",
                    reason="classifier LLM failed; conservative fallback",
                ),
            )

        def _parse_sub(raw: Any, allowed: set[str], fallback: str) -> tuple[str, str]:
            if not isinstance(raw, dict):
                # No parseable verdict for this field (typically a response
                # truncated at max_tokens on a long author list). Say so instead
                # of silently defaulting, so downstream short-circuits can tell
                # "the classifier judged this" from "the classifier never answered".
                return fallback, "UNRESOLVED: classifier returned no parseable verdict"
            overall = str(raw.get("overall", fallback)).strip().lower()
            if overall not in allowed:
                overall = fallback
            reason = str(raw.get("reason", "") or "").strip()

            # Post-parse self-consistency check: the LLM sometimes writes
            # `overall` before thinking through `reason`, then contradicts
            # itself ("Therefore, the correct verdict is h2_error, not exact").
            # Scan the reason for an explicit contradicting verdict phrase and
            # override overall when found.
            reason_lc = reason.lower()
            if reason_lc:
                import re as _re
                # Build per-label trigger patterns — only match inside an explicit
                # verdict-commit phrase to avoid catching incidental mentions
                # ("this is NOT h2_error" should not trigger h2_error).
                label_alternation = "|".join(_re.escape(lbl) for lbl in allowed)
                commit_patterns = [
                    rf"final verdict:\s*({label_alternation})\b",
                    rf"correct verdict is\s*({label_alternation})\b",
                    rf"therefore,? (?:the )?(?:correct )?verdict\s*(?:is|:)\s*({label_alternation})\b",
                    rf"→\s*({label_alternation})\b[^.]*$",  # last sentence ending with "→ X"
                    rf"verdict:\s*({label_alternation})\b",
                ]
                detected: str | None = None
                for pat in commit_patterns:
                    # Scan for LAST match (since self-corrections come late)
                    matches = list(_re.finditer(pat, reason_lc, flags=_re.IGNORECASE))
                    if matches:
                        detected = matches[-1].group(1).lower()
                        break
                if detected and detected in allowed and detected != overall:
                    reason = (
                        f"[override: LLM overall={overall!r} contradicted by "
                        f"reason → {detected!r}] {reason}"
                    )
                    overall = detected

            return overall, reason

        authors_overall, authors_reason = _parse_sub(
            parsed.get("authors"),
            {"exact", "r2_initial", "p1_variant", "h2_error"},
            "p1_variant",
        )
        venue_overall, venue_reason = _parse_sub(
            parsed.get("venue"),
            {"exact", "alias", "different"},
            "different",
        )
        publisher_overall, publisher_reason = _parse_sub(
            parsed.get("publisher"),
            {"exact", "alias", "different"},
            "different",
        )

        return FieldClassificationResult(
            authors=authors_fast or AuthorVariantResult(
                overall=authors_overall, reason=authors_reason,
            ),
            venue=venue_fast or VenueEquivalenceResult(
                overall=venue_overall, reason=venue_reason,
            ),
            publisher=publisher_fast or VenueEquivalenceResult(
                overall=publisher_overall, reason=publisher_reason,
            ),
            raw_llm_response=parsed,
        )


# Backward-compat alias so existing call sites/tests don't break.
BedrockAuthorVariantClassifier = BedrockFieldClassifier


# ===========================================================================
# ValidAgent
# ===========================================================================

VALID_AGENT_PROMPT = """You are a citation validity checker. Your job: determine if a citation matches a candidate record from a database or web search.

## Taxonomy

### Base rule (applies to ALL R-types):
To be REAL (valid=true), ALL of the following MUST hold:
  - title: match (after case normalization)
  - year: match
  - venue: match (after normalization/abbreviation/proceedings-prefix) OR both_missing OR candidate_missing

    USE YOUR WORLD KNOWLEDGE to recognize ALL of these as equivalent (match, not mismatch):

    1. Acronym ≡ full name — recognize the acronym regardless of whether it's in your
       training data. Rule: if the short form's letters match the initial letters of the
       full form's content words (skipping "of", "on", "the", "and", "for", "in"), treat as match.
       Examples: "JMLR" ≡ "Journal of Machine Learning Research",
                 "SICOMP" ≡ "SIAM Journal on Computing",
                 "COLT" ≡ "Conference on Learning Theory",
                 "AISTATS" ≡ "Artificial Intelligence and Statistics",
                 "STOC" ≡ "Symposium on Theory of Computing",
                 "UAI" ≡ "Uncertainty in Artificial Intelligence"

    2. Dot-truncated word abbreviations ≡ full name — when each dot-separated token is a
       prefix of the corresponding full word.
       Examples: "J. Mach. Learn. Res." ≡ "Journal of Machine Learning Research",
                 "Artif. Intell." ≡ "Artificial Intelligence",
                 "SIAM J. Comput." ≡ "SIAM Journal on Computing",
                 "IEEE Trans. Pattern Anal. Mach. Intell." ≡ "IEEE Transactions on Pattern Analysis and Machine Intelligence"

    3. Conference vs proceedings prefix:
       "NeurIPS" ≡ "Advances in Neural Information Processing Systems" ≡ "Proceedings of NeurIPS"
       "ICML" ≡ "International Conference on Machine Learning" ≡ "Proceedings of ICML"
       "AISTATS" ≡ "Proceedings of The 24th International Conference on Artificial Intelligence and Statistics"

    4. Conference-proceedings book titles (Springer/LNCS style):
       "Learning Theory and Kernel Machines" ≡ "COLT" (LNCS book title for COLT 2003 proceedings)

    5. Preprint server synonyms:
       "Preprint" ≡ "arXiv" ≡ "arXiv preprint" ≡ "CoRR" — all refer to the same arXiv preprint server.
       Many citation styles (ICML, NeurIPS bib templates) render arXiv-only papers with venue="Preprint".

    6. Publisher acronym ≡ full name (applies to publisher field):
       "ACM" ≡ "Association for Computing Machinery"
       "IEEE" ≡ "Institute of Electrical and Electronics Engineers"
       "PMLR" ≡ "Proceedings of Machine Learning Research"
       "Springer" ≡ "Springer Nature" ≡ "Springer-Verlag"
       "Elsevier" ≡ "Elsevier B.V." ≡ "Elsevier Ltd."
       "OUP" ≡ "Oxford University Press"
       "CUP" ≡ "Cambridge University Press"

    GENERAL PRINCIPLE: If you recognize both names as the same conference/journal/publisher
    (even if the automated field_status says "mismatch"), OVERRIDE it and treat as match.
    You have world knowledge the automated comparator doesn't.

    Only treat as MISMATCH when venues are genuinely different publishers/venues
    (e.g. "NeurIPS" vs "ICML", "Personal Blog" vs "NeurIPS", "ICLR" vs "arXiv").
  - authors: match (details vary by R-type below)

HARD RULES — if ANY of these is true, return valid=false immediately:
  - venue is MISMATCH → valid=false (e.g. "Personal Blog" vs "NeurIPS", "ICLR" vs "arXiv")
  - year is MISMATCH → valid=false
  - title is MISMATCH → valid=false

NOTE on authors=reordered_match:
  The automated comparison may report "reordered_match" due to name-format differences
  (e.g. "Wang, S" vs "Sicheng Wang" confusing the token matcher). Do NOT blindly trust it.
  Instead, compare the actual author lists yourself:
  - If the authors are the SAME people in the SAME order (just different name formats like
    "LastName, Initial" vs "FirstName LastName") → treat as match, valid=true
  - If the authors are genuinely reordered (different people in different positions) → valid=false

### REAL (valid=true) — base rules must ALL pass, PLUS:
- R1. Exact match: All fields match exactly (no normalization needed).
- R2. Format variant: Base rules pass after these DETERMINISTIC transformations ONLY:
  * Single-letter initial: "Gao Hao" → "G. Hao" (EXACTLY one letter) ✓ R2
  * Case change: "attention is all you need" vs "Attention Is All You Need" ✓ R2
  * Venue abbreviation: "Artif. Intell." vs "Artificial Intelligence" ✓ R2
  * NOTHING ELSE counts as R2 for author names.
- R3. Et al. abbreviation: Base rules pass, PLUS author list uses "et al."/"Others",
  all listed authors are correct AND in correct order.
  R3 ONLY relaxes author count — venue and year must STILL match.

### NOT VALID (valid=false) — give hint
- hint="likely_potential": Discrepancy requires WORLD KNOWLEDGE to explain:
  * Nickname: "Mike" vs "Michael", "Bob" vs "Robert", "Kate" vs "Katherine" → P1
  * Multi-letter truncation: "Edw." vs "Edward", "Séb." vs "Sébastien" → P1 (more than 1 letter)
  * Transliteration: "Wei Zhang" vs "William Zhang" → P1
  * Spelling variant / typo in a name: "Ulle" vs "Ullie", "Liam" vs "Liang" → P1
  * NOTE: last-first / "Surname, Given" notation alone is NOT P1. Decompose
    first; if the only first-name difference is a 1-char initial, that pair
    is r2_initial (e.g. "Sun, Y" ↔ "Yu Sun" → r2_initial, not P1).
- hint="likely_hallucinated": Discrepancies are real errors:
  * Title word changed, wrong author count, wrong year by >1, fabricated DOI, venue mismatch.

★★★ CRITICAL AUTHOR ROUTING RULES ★★★

Route by author discrepancy TYPE:

[A] SURNAMES ALIGN ONE-TO-ONE (same count, same order, same last names),
    only FIRST-NAME forms differ (nickname / truncation / typo / initial):
    → hint="likely_potential", issues=["author_name_variant"]  (P1)
    NEVER likely_hallucinated, NEVER H2.

[B] SURNAME SET CHANGES (author count differs, new surname appears/disappears,
    fabricated-looking full name not present on either side, "et al." not in
    citation but citation has fewer authors than candidate):
    → hint="likely_hallucinated", issues=["author_addition"/"author_deletion"/
      "author_fabrication"/"author_count_mismatch"]  (H2)
    This holds EVEN IF some individual first-name forms are legitimate R2/P1
    variants. The presence of a fabricated/missing author dominates.

[C] AUTHORS REORDERED (same set, different order, count matches):
    → hint="likely_hallucinated", issues=["author_reordered"]  (H2)

Do NOT let plausible first-name variants (DBLP "0003" suffix, "L. Zhao" vs
"Long Zhao", "Afouras T" vs "Triantafyllos Afouras") distract you from a
count/set mismatch. If you see Citation.surnames ≠ Candidate.surnames as a
set, or len(Citation.authors) ≠ len(Candidate.authors) without "et al.",
return likely_hallucinated regardless of other minor name variations.

Patterns that are ALWAYS likely_potential (P1), never likely_hallucinated:
  1. Nickname:                "Mike"   ← "Michael"
                              "Bob"    ← "Robert"
                              "Andy"   ← "Andrew"
                              "Nando"  ← "Fernando"  (Portuguese/Spanish diminutive)
                              "Tim"    ← "Timon"
                              "Dima"   ← "Dmitrii"   (Russian diminutive)
                              "Masa"   ← "Masashi"   (Japanese short form)
                              "Gi"     ← "Gihun"     (Korean short form)
  2. Multi-letter truncation: "Edw."   ← "Edward"    (N chars, dot-terminated)
                              "Shu"    ← "Shuang"    (no-dot truncation)
                              "Chu"    ← "Chujun"
                              "Sai"    ← "Sai Qian"  (dropped middle token)
                              "Zhong"  ← "Zhongtao"
                              "Vini"   ← "Vinayak"
  3. Single character typo:    "Ulle"   ↔ "Ullie"    (one-character off)
                              "Liam"   ↔ "Liang"
                              "Ily"    ↔ "Ilya"      (one-character off)
  4. (REMOVED — last-first + 1-char initial is r2_initial, not p1_variant.
      Decompose using "Surname, Given" rule; the 1-char rule then applies to
      the first-name pair just like any other r2 case.)
  5. Full vs initial + middle: "Masa Egi"  ↔ "M. Egi"    (still → R2 if Masa → M single-letter)
                               but "Masa Egi" ↔ "Masashi Egi" → P1 (Masa is truncation of Masashi)
  6. Transliteration:          "Wei"  ↔ "William"

KEY DISTINCTION for taxonomy (when valid=true):
  R2 = ONLY single-letter initial (exactly 1 character):
    "G. Hao" vs "Gao Hao" → R2 ✓ (G is one letter, rule-derivable)
    "S. Bubeck" vs "Sébastien Bubeck" → R2 ✓
    "D. D. Johnson" vs "Daniel D. Johnson" → R2 ✓
  P1 (valid=false, likely_potential) = EVERYTHING ELSE:
    ANY multi-letter or world-knowledge-requiring variant → NOT R2 → P1 via PotentialAgent

STEP 1 — count characters: Before claiming R2, count the characters
of the shorter (abbreviated) first-name form.
  - 1 character (with or without '.') → proceed with R2
  - 2 or more characters → NOT R2, return P1 via likely_potential

## Examples

Example 1 — R1 exact match:
Citation: Title="Attention Is All You Need" Authors=["Ashish Vaswani","Noam Shazeer"] Year=2017
Candidate: Title="Attention is All you Need" Authors=["Ashish Vaswani","Noam Shazeer"] Year=2017
Field: title=match, authors=match, year=match
→ {{"valid": true, "taxonomy": ["R1"], "hint": "", "issues": [], "reason": "All fields match."}}

Example 2 — R2 format variant (initial abbreviation):
Citation: Title="attention is all you need" Authors=["A. Vaswani","N. Shazeer"] Year=2017
Candidate: Title="Attention is All you Need" Authors=["Ashish Vaswani","Noam Shazeer"] Year=2017
Field: title=match, authors=partial_overlap (initials vs full names), year=match
→ {{"valid": true, "taxonomy": ["R2"], "hint": "", "issues": [], "reason": "A.=Ashish (first letter), N.=Noam (first letter). Rule-derivable abbreviation."}}

Example 3 — likely_potential (multi-letter truncation → NOT R2):
Citation: Title="Paper X" Authors=["Edw. Hu","Yel. Shen"] Year=2024
Candidate: Title="Paper X" Authors=["Edward Hu","Yelong Shen"] Year=2024
Field: title=match, authors=partial_overlap, year=match
→ {{"valid": false, "taxonomy": [], "hint": "likely_potential", "issues": ["author_name_variant"], "reason": "Edw=3 letters (not single-letter initial), Yel=3 letters. Multi-letter truncation → P1, not R2."}}

Example 4 — R3 et al. (venue must still match!):
Citation: Title="Attention Is All You Need" Authors=["Ashish Vaswani","Others"] Venue="NeurIPS" Year=2017
Candidate: Title="Attention is All you Need" Authors=["Ashish Vaswani","Noam Shazeer","Niki Parmar",...] Venue="NeurIPS" Year=2017
Field: title=match, authors=match (et al detected, listed author correct), venue=match, year=match
→ {{"valid": true, "taxonomy": ["R3"], "hint": "", "issues": [], "reason": "Et al. with correct listed author. Venue and year both match."}}

Example 4b — R3 et al. with NeurIPS proceedings-prefix venue (still MATCH):
Citation: Title="Language models are few-shot learners" Authors=["Tom Brown",...,"et al."] Venue="Advances in neural information processing systems" Year=2020
Candidate: Title="Language Models are Few-Shot Learners" Authors=["Tom B. Brown",...,"Dario Amodei"] Venue="Neural Information Processing Systems" Year=2020
Field: title=match, authors=match (et al), venue=match (NeurIPS proceedings ≡ NeurIPS), year=match
→ {{"valid": true, "taxonomy": ["R3"], "hint": "", "issues": [], "reason": "'Advances in NeurIPS' and 'NeurIPS' are the same venue (proceedings vs conference name)."}}

Example 5 — likely_potential (nickname, NOT rule-derivable):
Citation: Title="Large Language Models" Authors=["Mike Chen","Bob Smith"] Year=2024
Candidate: Title="Large Language Models" Authors=["Michael Chen","Robert Smith"] Year=2024
Field: title=match, authors=partial_overlap, year=match
→ {{"valid": false, "taxonomy": [], "hint": "likely_potential", "issues": ["author_name_variant"], "reason": "Mike≠initial of Michael (M≠Mi), Bob≠initial of Robert (B≠Bo). These are nicknames requiring world knowledge → P1."}}

Example 5b — likely_potential (multi-letter truncation, NOT hallucination):
Citation: Title="Paper Y" Authors=["Shu Yang","Xilin Chen"] Year=2024
Candidate: Title="Paper Y" Authors=["Shuang Yang","Xilin Chen"] Year=2024
Field: title=match, authors=mismatch, year=match
→ {{"valid": false, "taxonomy": [], "hint": "likely_potential", "issues": ["author_name_variant"], "reason": "'Shu' is a multi-letter truncation of 'Shuang' (Chinese short form, not single-letter initial). This is a name-variant P1, NOT a genuine author error."}}

Example 5c — likely_potential (foreign-language nickname):
Citation: Authors=["Nando Gama","Alejandro Ribeiro"] Year=2020
Candidate: Authors=["Fernando Gama","Alejandro Ribeiro"] Year=2020
Field: title=match, authors=mismatch, year=match
→ {{"valid": false, "taxonomy": [], "hint": "likely_potential", "issues": ["author_name_variant"], "reason": "'Nando' is a well-known Portuguese/Spanish diminutive of 'Fernando'. Requires world knowledge → P1, not H2."}}

Example 5d — likely_potential (spelling variant / one-char typo):
Citation: Authors=["Ullie Endriss"] Year=2021
Candidate: Authors=["Ulle Endriss"] Year=2021
Field: title=match, authors=mismatch, year=match
→ {{"valid": false, "taxonomy": [], "hint": "likely_potential", "issues": ["author_name_variant"], "reason": "'Ullie' vs 'Ulle' differ by one character — spelling variant of the same person's name. P1, not H2."}}

Example 5e — likely_potential (last-first format w/ initial):
Citation: Authors=["Triantafyllos Afouras","Oriol Vinyals"] Year=2020
Candidate: Authors=["Afouras T","Vinyals O"] Year=2020
Field: title=match, authors=mismatch, year=match
→ {{"valid": false, "taxonomy": [], "hint": "likely_potential", "issues": ["author_name_variant"], "reason": "Candidate uses 'LastName Initial' format (Afouras T = Triantafyllos Afouras). This is name-format variant → P1, NOT a fabricated author."}}

Example 6 — likely_hallucinated (title word substitution):
Citation: Title="Training novel language models to reason" Authors=["Charlie Chen"] Year=2024
Candidate: Title="Training large language models to reason" Authors=["Charlie Chen"] Year=2024
Field: title=mismatch, authors=match, year=match
→ {{"valid": false, "taxonomy": [], "hint": "likely_hallucinated", "issues": ["title_mismatch"], "reason": "Title word 'novel' differs from 'large' — word substitution."}}

Example 7 — NOT valid (candidate missing volume that reference provides):
Citation: Title="Paper X" Authors=["Alice"] Year=2020 Volume='29' Pages='100-200' Publisher='PMLR'
Candidate: Title="Paper X" Authors=["Alice"] Year=2020 Volume='' Pages='100-200' Publisher='PMLR'
Field: title=match, authors=match, year=match, volume=candidate_missing, pages=match, publisher=match
→ {{"valid": false, "taxonomy": [], "hint": "likely_potential", "issues": ["volume_missing"], "reason": "Reference has volume='29' but candidate does not provide it. All fields the reference provides must be verified by the candidate."}}

Example 8 — likely_hallucinated (author reordered):
Citation: Title="Paper X" Authors=["Bob","Alice","Carol"] Year=2024
Candidate: Title="Paper X" Authors=["Alice","Bob","Carol"] Year=2024
Field: title=match, authors=reordered_match, year=match
→ {{"valid": false, "taxonomy": [], "hint": "likely_hallucinated", "issues": ["author_reordered"], "reason": "Author order differs from published version."}}

Example 8 — likely_hallucinated (year error):
Citation: Title="Paper X" Authors=["Alice"] Year=2027
Candidate: Title="Paper X" Authors=["Alice"] Year=2025
Field: title=match, authors=match, year=mismatch
→ {{"valid": false, "taxonomy": [], "hint": "likely_hallucinated", "issues": ["year_mismatch"], "reason": "Year 2027 vs actual 2025, differs by >1."}}

Example 9 — likely_hallucinated (venue mismatch, even with et al.):
Citation: Title="Paper X" Authors=["Alice","Others"] Venue="Personal Blog" Year=2024
Candidate: Title="Paper X" Authors=["Alice","Bob","Carol"] Venue="NeurIPS" Year=2024
Field: title=match, authors=match (et al), venue=mismatch, year=match
→ {{"valid": false, "taxonomy": [], "hint": "likely_hallucinated", "issues": ["venue_mismatch"], "reason": "Venue 'Personal Blog' vs 'NeurIPS' is a mismatch. Venue mismatch blocks VALID even with et al."}}

## Now evaluate this citation:

Citation: Title='{citation_title}' Authors={citation_authors} Venue='{citation_venue}' Year={citation_year} Volume='{citation_volume}' Pages='{citation_pages}' Publisher='{citation_publisher}' Location='{citation_location}'
Candidate (from {connector}): Title='{candidate_title}' Authors={candidate_authors} Venue='{candidate_venue}' Year={candidate_year} Volume='{candidate_volume}' Pages='{candidate_pages}' Publisher='{candidate_publisher}' Location='{candidate_location}'

Field comparison:
{field_summary}
{web_evidence}
Author analysis (AUTHORITATIVE — use this verdict for the authors field, do not re-derive):
{author_analysis}
  - If overall == "exact": authors are byte-identical → R1.
  - If overall == "r2_initial": authors differ by single-letter initial only → R2 if other fields also align.
  - If overall == "p1_variant": same people but multi-letter / nickname / typo variant.
    The authors are still "match" for taxonomy/identity purposes (same paper),
    BUT this is NOT R2 — it is P1. If everything else matches, hint=likely_potential, issues=["author_name_variant"].
  - If overall == "h2_error": authors are genuinely wrong → likely_hallucinated, issues=["author_addition"|"author_deletion"|"author_count_mismatch"].
NOTE on volume/pages/publisher/location:
  - If both sides have a value and they DIFFER → issues should include the corresponding
    "_mismatch" and valid=false (e.g. volume_mismatch / pages_mismatch / publisher_mismatch
    / location_mismatch).
  - If the REFERENCE has a value but the CANDIDATE does not (candidate_missing) → valid=false.
    The candidate must verify all fields the reference provides.
  - If the REFERENCE is missing a value (reference_missing or both_missing) → OK, no
    verification needed for that field.
  - Page ranges with different separators ("1877-1901" vs "1877--1901" vs "pp. 1877–1901")
    are the SAME, treat as match.
  - Location (conference city) counts as a full field: if the reference provides it and
    the candidate's value differs, that IS a real discrepancy — do NOT dismiss as "optional".
TRUST THE FIELD_STATUS:
  The `field_status` above was produced by deterministic rule comparison plus
  the dedicated FieldClassifier (authors/venue/publisher). It is AUTHORITATIVE.
  Do not re-derive per-field verdicts — use field_status as given.
  Your job is to integrate field_status + author_analysis into a final verdict
  (valid/invalid, taxonomy, hint, issues, reason).

Return JSON only:
{{"valid": true/false, "taxonomy": [...], "hint": "likely_potential"/"likely_hallucinated"/"",
  "issues": [...], "reason": "..."}}"""


class BedrockValidAgent:
    def __init__(self, client, model_id: str) -> None:
        self.client = client
        self.model_id = model_id

    def evaluate(
        self,
        citation: CitationRecord,
        candidate: CandidateMatch,
        comparison: FieldComparisonResult,
        author_analysis: AuthorVariantResult | None = None,
    ) -> ValidAgentResult:
        fs = comparison.field_status if isinstance(comparison, FieldComparisonResult) else {}
        raw = candidate.raw_record if isinstance(candidate.raw_record, dict) else {}
        snippet = str(raw.get("search_result_text", "") or raw.get("raw_content", "") or "")[:2000]

        web_evidence = ""
        if snippet and candidate.connector in ("google_search", "web_search"):
            web_evidence = f"\nWeb search evidence:\n{snippet[:1500]}\n"

        if author_analysis:
            author_analysis_str = (
                f"  overall = {author_analysis.overall}\n"
                f"  reason  = {author_analysis.reason or '(none)'}"
            )
        else:
            author_analysis_str = "  (not provided — derive from citation/candidate authors yourself)"

        prompt = VALID_AGENT_PROMPT.format(
            citation_title=citation.title,
            citation_authors=citation.authors,
            citation_venue=citation.venue or "",
            citation_year=citation.year,
            citation_volume=citation.volume or "",
            citation_pages=citation.pages or "",
            citation_publisher=citation.publisher or "",
            citation_location=citation.location or "",
            connector=candidate.connector,
            candidate_title=candidate.title,
            candidate_authors=candidate.authors,
            candidate_venue=candidate.venue or "",
            candidate_year=candidate.year,
            candidate_volume=getattr(candidate, "volume", "") or "",
            candidate_pages=getattr(candidate, "pages", "") or "",
            candidate_publisher=getattr(candidate, "publisher", "") or "",
            candidate_location=str((candidate.raw_record or {}).get("location", "") or "")
                if isinstance(candidate.raw_record, dict) else "",
            field_summary=_field_summary(fs),
            web_evidence=web_evidence,
            author_analysis=author_analysis_str,
        )

        try:
            parsed = _call_bedrock(self.client, self.model_id, prompt)
        except Exception:
            return ValidAgentResult(
                is_valid=False, hint="likely_hallucinated",
                issues=["llm_error"], reason="LLM call failed",
                field_result=comparison, candidate=candidate,
            )

        is_valid = bool(parsed.get("valid", False))
        taxonomy = _parse_taxonomy(parsed.get("taxonomy", []))
        hint = str(parsed.get("hint", "likely_hallucinated")).strip()
        issues = parsed.get("issues", []) or []
        if isinstance(issues, str):
            issues = [issues]
        issues = _normalize_issue_names(issues)
        issues = _filter_llm_issues_by_field_status(issues, fs)
        reason = str(parsed.get("reason", "") or "").strip()

        return ValidAgentResult(
            is_valid=is_valid,
            taxonomy=taxonomy,
            hint=hint if not is_valid else "",
            issues=issues if not is_valid else [],
            reason=reason,
            field_result=comparison,
            candidate=candidate,
            raw_llm_response=parsed,
        )


# ===========================================================================
# PotentialAgent
# ===========================================================================

POTENTIAL_AGENT_PROMPT = """You are a citation discrepancy analyst. The citation was NOT a direct match. Your job: determine if the discrepancies are explainable.

## Taxonomy

### POTENTIAL (discrepancies are explainable)
- P1. Author name variant: Name is a plausible nickname/transliteration/spelling variant
  of the SAME PERSON. The surname sets must still align one-to-one (same count,
  same people, same order). Only the first-name form differs.
  Valid P1 examples: "Katherine" vs "Kate", "Mike" vs "Michael", "Wei Zhang" vs
  "William Zhang", "Nando" vs "Fernando", "Shu Yang" vs "Shuang Yang",
  "Afouras T" vs "Triantafyllos Afouras", "L. Zhao" vs "Long Zhao".
- P2. Non-academic source unverifiable: Citation references a non-academic source
  whose existence cannot be fully verified through structured databases.
- P3. Insufficient field evidence: Citation provides optional peripheral fields
  (volume, pages, publisher, location — ONLY these four) that cannot be
  confirmed or denied because no candidate source supplies them. The CORE
  identity fields (title + authors + year + venue) all match the candidate,
  so the paper is verifiably real — but one or more of volume/pages/
  publisher/location carry no external evidence either way.
  Signals: one or more of `{{volume,pages,publisher,location}}_candidate_missing`
  in the issues list while core fields match.
  Valid P3 examples:
    - Citation has volume="35" for a NeurIPS paper; DBLP/CrossRef/arXiv do
      not supply volume for NeurIPS → cannot verify. Other fields match → P3.
    - Citation lists publisher="PMLR" and location="Vancouver, Canada" for
      an ICML paper; no connector confirms these specific values → P3.
  Do NOT use P3 when:
    - Any core field (title / authors / year / venue) has a real mismatch.
    - A secondary field has an explicit contradicting value from a candidate
      (that is H6, not P3).
    - `doi_candidate_missing` or `arxiv_id_candidate_missing` is the only
      issue — DOI/arxiv_id fall under H5, not P3.

### HALLUCINATED (unexplained errors → escalate)
★★★ AUTHOR COUNT / SET MISMATCH IS ALWAYS HALLUCINATED (do NOT explain as P1) ★★★
These patterns ALWAYS escalate to HALLUCINATED, never P1:
  - Citation has MORE authors than candidate (added author) → H2
  - Citation has FEWER authors than candidate AND ref has no "et al." → H2
  - A surname in citation is absent from candidate (or vice versa) → H2
  - Authors reordered (same set, different order) → H2
DBLP disambiguation suffixes ("Ting Chen 0007") are NOT a reason to explain
away a missing author. The suffix applies to the existing author; a separate
added/removed author is still a real discrepancy → escalate.

If ANY discrepancy cannot be explained by P1/P2/P3, return HALLUCINATED.

## Examples

Example 1 — P1 (author name variant):
Issues: ["author_name_variant"]
Evidence: authors: [EXPLAINED] 'Edw.' is a prefix of 'Edward' — plausible abbreviation.
→ {{"label": "POTENTIAL", "taxonomy": ["P1"], "reason": "Author 'Edw.' is a plausible abbreviation of 'Edward'."}}

Example 2 — Escalate to HALLUCINATED (venue mismatch):
Issues: ["author_name_variant", "venue_mismatch"]
Evidence: authors: [EXPLAINED] prefix match. venue: [UNEXPLAINED] 'Personal Blog' vs 'NeurIPS'.
→ {{"label": "HALLUCINATED", "taxonomy": ["H3"], "reason": "Venue 'Personal Blog' cannot be explained as a variant of 'NeurIPS'."}}

Example 3 — Escalate to HALLUCINATED (author count mismatch = added author):
Issues: ["author_addition"]
Citation authors: ["Ting Chen", "Simon Kornblith", "Mohammad Norouzi", "Geoffrey Hinton", "Aria Lin"]
Candidate authors: ["Ting Chen 0007", "Simon Kornblith", "Mohammad Norouzi 0002", "Geoffrey E. Hinton"]
→ {{"label": "HALLUCINATED", "taxonomy": ["H2"], "reason": "Citation added a fabricated author 'Aria Lin' not in the candidate. DBLP suffixes '0007'/'0002' do not explain the extra author."}}

Example 4 — Escalate to HALLUCINATED (author deletion):
Issues: ["author_deletion"]
Citation authors: ["Tanmay Chavan"]
Candidate authors: ["Tanmay Chavan", "Aditya Kane"]
→ {{"label": "HALLUCINATED", "taxonomy": ["H2"], "reason": "Citation omits real author 'Aditya Kane'. No 'et al.' marker → this is a real deletion, not a citation abbreviation."}}

Example 5 — P3 (insufficient field evidence):
Issues: ["volume_candidate_missing", "publisher_candidate_missing", "location_candidate_missing"]
Citation: volume="35", publisher="PMLR", location="Vancouver, Canada" for an ICML paper
Candidate: title/authors/venue/year all match; volume/publisher/location all empty
Evidence: these fields cannot be verified — DBLP/CrossRef do not supply them for ICML.
→ {{"label": "POTENTIAL", "taxonomy": ["P3"], "reason": "Core fields (title/authors/venue/year) match. Secondary fields (volume/publisher/location) are absent from all candidates — cannot be confirmed or denied. Insufficient evidence to flag as hallucination."}}

## Now evaluate:

Citation: Title='{citation_title}' Authors={citation_authors} Venue='{citation_venue}' Year={citation_year} Location='{citation_location}'
Candidate: Title='{candidate_title}' Authors={candidate_authors} Venue='{candidate_venue}' Year={candidate_year} Location='{candidate_location}'

Issues from ValidAgent: {issues}
ValidAgent reason: {valid_reason}

Secondary evidence:
{evidence_lines}

Return JSON only:
{{"label": "POTENTIAL"/"HALLUCINATED", "taxonomy": ["P1"/"P2"/"P3"/"H1".."H6"], "reason": "..."}}"""


class BedrockPotentialAgent:
    def __init__(self, client, model_id: str) -> None:
        self.client = client
        self.model_id = model_id

    def evaluate(
        self,
        citation: CitationRecord,
        candidate: CandidateMatch,
        comparison: FieldComparisonResult,
        valid_hint: ValidAgentResult,
        evidence: list[DiscrepancyEvidence],
    ) -> DownstreamAgentResult:
        evidence_lines = []
        for ev in evidence:
            tag = "EXPLAINED" if ev.evidence_found else "UNEXPLAINED"
            evidence_lines.append(f"  {ev.field}: [{tag}] {ev.explanation[:120]}")

        prompt = POTENTIAL_AGENT_PROMPT.format(
            citation_title=citation.title,
            citation_authors=citation.authors,
            citation_venue=citation.venue or "",
            citation_year=citation.year,
            citation_location=citation.location or "",
            candidate_title=candidate.title,
            candidate_authors=candidate.authors,
            candidate_venue=candidate.venue or "",
            candidate_year=candidate.year,
            candidate_location=str((candidate.raw_record or {}).get("location", "") or "")
                if isinstance(candidate.raw_record, dict) else "",
            issues=valid_hint.issues,
            valid_reason=valid_hint.reason,
            evidence_lines="\n".join(evidence_lines) or "  (none gathered)",
        )

        try:
            parsed = _call_bedrock(self.client, self.model_id, prompt)
        except Exception:
            return DownstreamAgentResult(
                label=VerdictLabel.FAKE_REFERENCE,
                reason="PotentialAgent LLM failed",
            )

        label_str = str(parsed.get("label", "HALLUCINATED")).upper()
        if "POTENTIAL" in label_str:
            label = VerdictLabel.POTENTIAL_REFERENCE
        else:
            label = VerdictLabel.FAKE_REFERENCE

        return DownstreamAgentResult(
            label=label,
            taxonomy=_parse_taxonomy(parsed.get("taxonomy", [])),
            reason=str(parsed.get("reason", "") or "").strip(),
            escalate_to="hallucinated" if label == VerdictLabel.FAKE_REFERENCE else "",
            raw_llm_response=parsed,
        )


# ===========================================================================
# HallucinatedAgent
# ===========================================================================

HALLUCINATED_AGENT_PROMPT = """You are a citation error classifier. The citation has confirmed errors. Your job: list ALL errors found.

## CRITICAL RULE: Trust the field_status
The field comparison was done by an automated system with proper normalization (case, dashes,
abbreviations, etc.). If field_status says a field is "match", it IS correct — do NOT override
it based on cosmetic differences you see in the raw values. Only flag errors for fields where
field_status says "mismatch" or "candidate_missing".

## Taxonomy — list ALL that apply

- H1: Title error — title differs from real paper (word substitution, paraphrase, or fabrication).
  Example: "Attention Is What You Need" vs real "Attention Is All You Need"
  NOTE: If field_status says title=match, it IS a match. Capitalization differences
  (e.g. "Tree of thoughts" vs "Tree of Thoughts") are NOT title errors — they are
  already handled by normalization. Trust the field_status.
- H2: Author error — wrong authors: addition, deletion, or fabrication.
  H2 ONLY when the authors are DIFFERENT PEOPLE (new surname, missing surname,
  added/deleted person, or a clearly fabricated name unrelated to the candidate).
  NOTE: these are NOT errors (do NOT flag as H2):
    * Citation uses a TRUNCATION MARKER as the last author — any of
      "et al.", "et al", "Others", "and others", "et.al.", "etal", "...",
      "etc." — this is R3 (et al. abbreviation). The candidate having MORE
      authors than the citation is EXPECTED. Verify only that the listed
      (non-marker) authors appear in the candidate at matching positions.
      If they do → authors match, NOT H2. Count mismatch caused purely by
      the truncation marker is NEVER H2.
    * "LastName, F" vs "FirstName LastName" (e.g. "Guedj, B" vs "Benjamin Guedj") — format variant
    * Single-letter initial expansion (e.g. "J. Langford" vs "John Langford") — R2
    * Name order "First Last" vs "Last, First" — citation style difference
    * Nickname / diminutive: "Mike" ← "Michael", "Nando" ← "Fernando", "Andy" ← "Andrew",
      "Dima" ← "Dmitrii", "Masa" ← "Masashi", "Bob" ← "Robert" — same person, P1
    * Multi-letter truncation of first name: "Shu" ← "Shuang", "Edw." ← "Edward",
      "Nando" ← "Fernando", "Vini" ← "Vinayak", "Zhong" ← "Zhongtao" — same person, P1
    * Single-character spelling variant: "Ulle" ↔ "Ullie", "Liam" ↔ "Liang",
      "Ily" ↔ "Ilya" — most likely same person, P1
    * Transliteration / anglicized form: "Wei" ↔ "William" — same person, P1
    * Dropped middle name: "Sai Zhang" ↔ "Sai Qian Zhang" — same person, P1
    * Middle-name / middle-initial variations — ALL are same person, P1 (never H2):
        - Middle initial dropped:    "Dmitry P. Vetrov"   ↔ "Dmitry Vetrov"
        - Middle initial added:      "Dmitry Vetrov"      ↔ "Dmitry P. Vetrov"
        - Middle full name dropped:  "Aditya Manish Kane" ↔ "Aditya Kane"
        - Middle full name added:    "Aditya Kane"        ↔ "Aditya Manish Kane"
        - Middle initial ↔ full:     "Richard S. Zemel"   ↔ "Richard Samuel Zemel"
    * Surname-first + initial: "Afouras T" ↔ "Triantafyllos Afouras" — format variant + initial
  KEY PRINCIPLE: If the SURNAMES still align one-to-one (same people, same order,
  same count) and only the FIRST-NAME forms vary, this is P1 (name variant), not H2.
  H2 is reserved for changes to the SET of people (add/delete/fabricate/swap to
  a new surname).
- H3: Venue error — paper exists but at a GENUINELY different venue.
  NOTE: these are NOT errors (do NOT flag as H3):
    * Acronym ≡ full name: "ICML" ≡ "International Conference on Machine Learning"
    * Proceedings prefix: "Proceedings of the 36th ICML" ≡ "ICML"
    * Dot-abbreviated: "J. Mach. Learn. Res." ≡ "Journal of Machine Learning Research"
    * "NeurIPS" ≡ "NIPS" ≡ "Advances in Neural Information Processing Systems"
    * "AISTATS" ≡ "International Conference on Artificial Intelligence and Statistics"
    * "Preprint" ≡ "arXiv" ≡ "arXiv preprint" ≡ "CoRR" — all refer to the same arXiv preprint server
    USE YOUR WORLD KNOWLEDGE: if both names refer to the same conference/journal, it's NOT H3.
- H4: Year error — year is verifiably wrong.
  Example: Citation says 2027 but paper published 2025.
- H5: DOI/identifier error — DOI resolves to a different paper or doesn't exist.
- H6: Pages/volume/publisher/location error — page numbers, volume, publisher, or location are wrong OR missing from candidate.
  If the reference has volume/pages/publisher/location but candidate does not → H6.
  NOTE: these are NOT errors:
    * Dash format differences: "1025–1068" vs "1025-1068" — same page range
    * Roman numeral vs arabic: "II" vs "2" — same volume
    * Publisher acronym ≡ full name (USE YOUR WORLD KNOWLEDGE):
      - "ACM" ≡ "Association for Computing Machinery"
      - "IEEE" ≡ "Institute of Electrical and Electronics Engineers"
      - "PMLR" ≡ "Proceedings of Machine Learning Research"
      - "MIT Press" ≡ "The MIT Press"
      - "Springer" ≡ "Springer Nature" ≡ "Springer-Verlag"
      - "Elsevier" ≡ "Elsevier B.V." ≡ "Elsevier Ltd."
      - "OUP" ≡ "Oxford University Press"
      - "CUP" ≡ "Cambridge University Press"
      If both names refer to the same publisher, it's NOT H6 — even if field_status says "mismatch".

## Important: candidate_missing IS an error
If the reference provides a value for a field (e.g. volume, pages, publisher, location) but the candidate
does not have it, this means the candidate CANNOT verify that field → flag as H6.
You must ALWAYS return label="HALLUCINATED", never downgrade to "POTENTIAL".

## Examples

Example 1 — single error (title):
Citation: Title="Training novel language models" Authors=["Alice"] Year=2024
Candidate: Title="Training large language models" Authors=["Alice"] Year=2024
Field: title=mismatch, authors=match, year=match
→ {{"label": "HALLUCINATED", "taxonomy": ["H1"], "reason": "Title word 'novel' substituted for 'large'."}}

Example 2 — multiple errors:
Citation: Title="Paper X" Authors=["Alice","Bob","Carol"] Venue="ICML" Year=2027
Candidate: Title="Paper X" Authors=["Alice","Bob"] Venue="NeurIPS" Year=2025
Field: title=match, authors=count_mismatch, venue=mismatch, year=mismatch
→ {{"label": "HALLUCINATED", "taxonomy": ["H2","H3","H4"], "reason": "Author 'Carol' added; venue ICML vs NeurIPS; year 2027 vs 2025."}}

Example 3 — candidate_missing fields → H6:
Citation: Title="Paper X" Authors=["Alice"] Year=2024 Volume='33' Pages='100-200' Publisher='PMLR'
Candidate: Title="Paper X" Authors=["Alice"] Year=2024 Volume='' Pages='' Publisher=''
Field: title=match, authors=match, year=match, volume=candidate_missing, pages=candidate_missing, publisher=candidate_missing
→ {{"label": "HALLUCINATED", "taxonomy": ["H6"], "reason": "Reference provides volume=33, pages=100-200, publisher=PMLR but candidate cannot verify these fields."}}

## Now classify:

Citation: Title='{citation_title}' Authors={citation_authors} Venue='{citation_venue}' Year={citation_year} DOI='{citation_doi}' Pages='{citation_pages}' Location='{citation_location}'
Candidate: Title='{candidate_title}' Authors={candidate_authors} Venue='{candidate_venue}' Year={candidate_year} Location='{candidate_location}'

Field comparison:
{field_summary}

Issues from ValidAgent: {issues}
ValidAgent reason: {valid_reason}

Return JSON only. List ALL errors. Always return label="HALLUCINATED":
{{"label": "HALLUCINATED", "taxonomy": ["H1", "H4"], "reason": "..."}}"""


class BedrockHallucinatedAgent:
    def __init__(self, client, model_id: str) -> None:
        self.client = client
        self.model_id = model_id

    def evaluate(
        self,
        citation: CitationRecord,
        candidate: CandidateMatch,
        comparison: FieldComparisonResult,
        valid_hint: ValidAgentResult,
    ) -> DownstreamAgentResult:
        fs = comparison.field_status if isinstance(comparison, FieldComparisonResult) else {}

        prompt = HALLUCINATED_AGENT_PROMPT.format(
            citation_title=citation.title,
            citation_authors=citation.authors,
            citation_venue=citation.venue or "",
            citation_year=citation.year,
            citation_doi=citation.doi or "",
            citation_pages=citation.pages or "",
            citation_location=citation.location or "",
            candidate_title=candidate.title,
            candidate_authors=candidate.authors,
            candidate_venue=candidate.venue or "",
            candidate_year=candidate.year,
            candidate_location=str((candidate.raw_record or {}).get("location", "") or "")
                if isinstance(candidate.raw_record, dict) else "",
            field_summary=_field_summary(fs),
            issues=valid_hint.issues,
            valid_reason=valid_hint.reason,
        )

        try:
            parsed = _call_bedrock(self.client, self.model_id, prompt)
        except Exception:
            return DownstreamAgentResult(
                label=VerdictLabel.FAKE_REFERENCE,
                taxonomy=["H1"],
                reason="HallucinatedAgent LLM failed",
            )

        label_str = str(parsed.get("label", "HALLUCINATED")).upper()
        if "POTENTIAL" in label_str:
            label = VerdictLabel.POTENTIAL_REFERENCE
        else:
            label = VerdictLabel.FAKE_REFERENCE

        taxonomy = _parse_taxonomy(parsed.get("taxonomy", []))

        # Safety net: if FieldClassifier already said authors == p1_variant
        # (same people, soft name variant), the LLM should NOT emit H2. Drop
        # any H2 the LLM still emitted; if H2 was the only code, downgrade
        # the verdict to POTENTIAL.
        author_info = fs.get("authors") if isinstance(fs, dict) else None
        author_variant = (
            author_info.get("author_variant_type")
            if isinstance(author_info, dict) else None
        )
        if author_variant == "p1_variant" and "H2" in taxonomy:
            taxonomy = [t for t in taxonomy if t != "H2"]
            if not taxonomy:
                label = VerdictLabel.POTENTIAL_REFERENCE
                taxonomy = ["P1"]

        # Surname-typo safety net (RELAXED MODE ONLY): if LLM emits H2 but
        # the only author difference is a tiny surname OCR typo (≤2 chars,
        # given names match, not partial_overlap / reordered_match), drop
        # H2 and restore the verdict to VALID R1. This catches OCR errors
        # like "Schreck" vs "Schrek" or "Bugarra" vs "Bugrara".
        from .adjudicate import _RELAXED_FIELDS as _relaxed
        if _relaxed and "H2" in taxonomy:
            from .cascading_agents import _h2_is_minor_surname_typo
            ref_authors = list(citation.authors or [])
            cand_authors = list(candidate.authors or [])
            if _h2_is_minor_surname_typo(ref_authors, cand_authors, fs):
                taxonomy = [t for t in taxonomy if t != "H2"]
                if not taxonomy:
                    label = VerdictLabel.VALID
                    taxonomy = ["R1"]

        return DownstreamAgentResult(
            label=label,
            taxonomy=taxonomy,
            reason=str(parsed.get("reason", "") or "").strip(),
            raw_llm_response=parsed,
        )
