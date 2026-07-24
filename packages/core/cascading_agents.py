"""Cascading 3-Agent Citation Verification System.

Pipeline: ValidAgent → PotentialAgent → HallucinatedAgent

ValidAgent: Is this citation verifiably REAL? (rule-based + LLM)
  YES → REAL (done)
  NO → hint ("likely_potential" or "likely_hallucinated") → route downstream

PotentialAgent: Are the discrepancies explainable? (evidence + LLM)
  POTENTIAL → done
  HALLUCINATED → escalate to HallucinatedAgent

HallucinatedAgent: Classify ALL errors. (LLM)
  HALLUCINATED → done (with full error list)
  POTENTIAL → rare downgrade
"""
from __future__ import annotations

import re
from typing import Any, Protocol

from .models import (
    AuthorVariantResult,
    CandidateMatch,
    CitationRecord,
    CitationVerdict,
    DiscrepancyEvidence,
    DownstreamAgentResult,
    ExtractionQuality,
    FieldClassificationResult,
    FieldComparisonResult,
    ValidAgentResult,
    VenueEquivalenceResult,
    VerdictLabel,
    candidate_match_to_dict,
)


# ---------------------------------------------------------------------------
# Agent Protocols
# ---------------------------------------------------------------------------

class ValidAgentProtocol(Protocol):
    def evaluate(
        self,
        citation: CitationRecord,
        candidate: CandidateMatch,
        comparison: FieldComparisonResult,
        author_analysis: AuthorVariantResult | None = None,
    ) -> ValidAgentResult: ...


class FieldClassifierProtocol(Protocol):
    def classify(
        self,
        citation_authors: list[str],
        candidate_authors: list[str],
        citation_venue: str = "",
        candidate_venue: str = "",
        citation_publisher: str = "",
        candidate_publisher: str = "",
    ) -> FieldClassificationResult: ...


# Backward-compat alias
AuthorVariantClassifierProtocol = FieldClassifierProtocol


class PotentialAgentProtocol(Protocol):
    def evaluate(
        self,
        citation: CitationRecord,
        candidate: CandidateMatch,
        comparison: FieldComparisonResult,
        valid_hint: ValidAgentResult,
        evidence: list[DiscrepancyEvidence],
    ) -> DownstreamAgentResult: ...


class HallucinatedAgentProtocol(Protocol):
    def evaluate(
        self,
        citation: CitationRecord,
        candidate: CandidateMatch,
        comparison: FieldComparisonResult,
        valid_hint: ValidAgentResult,
    ) -> DownstreamAgentResult: ...


# ---------------------------------------------------------------------------
# Rule-based hint computation
# ---------------------------------------------------------------------------

_WEB_CONNECTORS = {"google_search", "web_search", "searxng_search"}

_TAXONOMY_CODE_RE = re.compile(r"^[HPR]\d+$")


def _author_variant_pairs(ref_authors: list[str], cand_authors: list[str]) -> list[tuple[str, str]]:
    """Pairwise diff of two author lists; return positions whose names are
    not byte-identical after lower+whitespace normalization.

    Used to make P1 verdicts informative ("which author is the variant?").
    Returns up to 5 (ref, cand) pairs to keep the reason text compact.
    """
    if not ref_authors or not cand_authors:
        return []

    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", str(s or "")).strip().lower()

    out: list[tuple[str, str]] = []
    for r, c in zip(ref_authors, cand_authors):
        if _norm(r) != _norm(c):
            out.append((str(r), str(c)))
            if len(out) >= 5:
                break
    return out


def _format_variant_pairs(pairs: list[tuple[str, str]]) -> str:
    if not pairs:
        return ""
    return "; ".join(f"'{r}' ↔ '{c}'" for r, c in pairs)


def _clean_conflicts(items: list[str] | None) -> list[str]:
    """Strip taxonomy codes (H1/P3/R2/...) from a conflicts list.

    The conflicts field is meant to carry field-level descriptors
    ('author_name_variant', 'pages_candidate_missing', etc.); a taxonomy
    code leaking in is a sign that an upstream stage reused issues for
    cross-agent communication. Filter defensively at every emit site so
    one stray pollution does not show up in the report.
    """
    if not items:
        return []
    return [it for it in items if not _TAXONOMY_CODE_RE.match(str(it))]


def _enrich_web_candidate_via_url(
    candidate: CandidateMatch,
    tavily_api_key: str = "",
    extractor_agent: Any = None,
) -> CandidateMatch:
    """Fetch URL page via direct HTTP GET and use ExtractorAgent LLM to extract structured fields.

    No Tavily fallback — web search already returned raw_content via Tavily,
    so the first ExtractorAgent pass already used that. This step only adds
    value when direct HTTP GET returns HTML with richer metadata (e.g. meta tags,
    BibTeX blocks) than the Tavily markdown.
    """
    from dataclasses import replace as _replace

    url = (candidate.url or "").strip()
    if not url or extractor_agent is None:
        return candidate

    # Skip URLs that are clearly not academic pages
    skip_domains = {"youtube.com", "twitter.com", "x.com", "github.com", "slideslive.com"}
    try:
        from urllib.parse import urlsplit
        host = urlsplit(url).hostname or ""
        host = host.lower().lstrip("www.")
        if any(host.endswith(d) for d in skip_domains):
            return candidate
    except Exception:
        pass

    # Direct HTTP GET only — no Tavily fallback (already have raw_content from search)
    content = ""
    try:
        from packages.connectors.url_direct import URLDirectConnector
        from packages.connectors.base import RequestPolicy as _RP
        connector = URLDirectConnector()
        content = connector._request_text(
            url, {}, _RP(timeout_s=5.0),
            headers={"Accept": "text/html,application/xhtml+xml"},
        )
    except Exception:
        return candidate

    if not content or not content.strip():
        return candidate

    # Use ExtractorAgent LLM to extract structured fields from the page content.
    # Build a temporary candidate with URL page content as raw_content,
    # then extract and merge results back — URL page values override snippet values
    # when they differ (URL page is more authoritative).
    try:
        from packages.core.extractor_agent import EXTRACTOR_PROMPT, _call_bedrock
        import re as _re, json as _json

        prompt = EXTRACTOR_PROMPT.format(content=content)
        parsed = _call_bedrock(extractor_agent.client, extractor_agent.model_id, prompt)
        if not parsed:
            return candidate
    except Exception:
        return candidate

    new_title = str(parsed.get("title", "") or "").strip()
    new_authors = parsed.get("authors") or []
    if isinstance(new_authors, str):
        new_authors = [a.strip() for a in new_authors.split(",") if a.strip()]
    new_venue = str(parsed.get("venue", "") or "").strip()
    raw_year = parsed.get("year")
    new_year = int(raw_year) if raw_year is not None else None
    new_doi = str(parsed.get("doi", "") or "").strip()
    new_pages = str(parsed.get("pages", "") or "").strip()
    new_volume = str(parsed.get("volume", "") or "").strip()
    new_publisher = str(parsed.get("publisher", "") or "").strip()

    # Merge: URL page extraction overrides when values differ.
    # If URL page didn't extract a field (empty), keep existing value.
    def _pick(old: str, new: str) -> str:
        if not new:
            return old  # URL page didn't extract → keep existing
        return new      # URL page has value → use it (override even if different)

    def _pick_list(old: list, new: list) -> list:
        if not new:
            return old
        return new

    updated_raw = dict(candidate.raw_record) if isinstance(candidate.raw_record, dict) else {}
    updated_raw["url_direct_enrichment"] = parsed

    return _replace(
        candidate,
        title=_pick(candidate.title, new_title),
        authors=_pick_list(candidate.authors, new_authors),
        venue=_pick(candidate.venue, new_venue),
        year=new_year if new_year is not None else candidate.year,
        doi=_pick(candidate.doi, new_doi),
        pages=_pick(candidate.pages, new_pages),
        volume=_pick(candidate.volume, new_volume),
        publisher=_pick(candidate.publisher, new_publisher),
        raw_record=updated_raw,
    )


def compute_hint(field_result: FieldComparisonResult) -> tuple[str, list[str]]:
    """Compute routing hint + issues from field comparison. No LLM needed."""
    fs = field_result.field_status
    issues: list[str] = []
    hint = "likely_hallucinated"  # default

    title_s = fs.get("title", {}).get("status", "")
    year_s = fs.get("year", {}).get("status", "")
    author_s = fs.get("authors", {}).get("status", "")
    venue_s = fs.get("venue", {}).get("status", "")
    doi_s = fs.get("doi", {}).get("status", "")
    pages_s = fs.get("pages", {}).get("status", "")
    volume_s = fs.get("volume", {}).get("status", "")
    publisher_s = fs.get("publisher", {}).get("status", "")
    location_s = fs.get("location", {}).get("status", "")

    # Title
    if title_s == "mismatch":
        issues.append("title_mismatch")

    # Authors
    if author_s == "count_mismatch":
        issues.append("author_count_mismatch")
    elif author_s in ("partial_overlap", "reordered_match"):
        issues.append("author_name_variant" if author_s == "partial_overlap" else "author_reordered")
    elif author_s == "mismatch":
        issues.append("author_mismatch")

    # Year — must match exactly. Any mismatch is a hard error.
    if year_s == "mismatch":
        issues.append("year_mismatch")
    elif year_s == "candidate_missing":
        issues.append("year_candidate_missing")

    # Venue — arXiv is NOT a valid variant of any conference/journal.
    # candidate_missing counts as a hallucination signal: by taxonomy, only
    # H6-class fields (pages/volume/publisher/location) are allowed to be
    # missing and still go to POTENTIAL. Core fields cannot be silently
    # absent; HallucinatedAgent decides whether it's H3 vs acceptable.
    if venue_s == "mismatch":
        issues.append("venue_mismatch")
    elif venue_s == "candidate_missing":
        issues.append("venue_candidate_missing")

    # DOI
    if doi_s == "mismatch":
        issues.append("doi_mismatch")
    elif doi_s == "candidate_missing":
        issues.append("doi_candidate_missing")

    # Pages
    if pages_s == "mismatch":
        issues.append("pages_mismatch")
    elif pages_s == "candidate_missing":
        issues.append("pages_candidate_missing")
    # Volume
    if volume_s == "mismatch":
        issues.append("volume_mismatch")
    elif volume_s == "candidate_missing":
        issues.append("volume_candidate_missing")
    # Publisher
    if publisher_s == "mismatch":
        issues.append("publisher_mismatch")
    elif publisher_s == "candidate_missing":
        issues.append("publisher_candidate_missing")
    # Location
    if location_s == "mismatch":
        issues.append("location_mismatch")
    elif location_s == "candidate_missing":
        issues.append("location_candidate_missing")

    # Determine hint from issues
    potential_signals = {
        "author_name_variant",
    }
    hallucinated_signals = {
        "title_mismatch", "author_count_mismatch", "author_mismatch", "author_reordered",
        "year_mismatch", "venue_mismatch", "doi_mismatch", "pages_mismatch",
        "volume_mismatch", "publisher_mismatch", "location_mismatch",
        # candidate_missing where reference provides a value → skip ValidAgent, go to HallucinatedAgent.
        # H6-class fields (pages/volume/publisher/location) route here but may
        # be downgraded to P3 later by the `_only_candidate_missing_issues`
        # fallback. Core fields (venue/year/doi) must not be silently absent
        # per taxonomy — HallucinatedAgent decides H3/H4/H5.
        "venue_candidate_missing", "year_candidate_missing", "doi_candidate_missing",
        "pages_candidate_missing", "volume_candidate_missing", "publisher_candidate_missing",
        "location_candidate_missing",
    }

    has_potential = any(i in potential_signals for i in issues)
    has_hallucinated = any(i in hallucinated_signals for i in issues)

    if has_hallucinated:
        hint = "likely_hallucinated"
    elif has_potential:
        hint = "likely_potential"
    elif not issues:
        hint = "likely_potential"  # no obvious issues → might be format variant

    return hint, issues


# ---------------------------------------------------------------------------
# ValidAgent (rule-based part)
# ---------------------------------------------------------------------------

def _has_format_variant(field_status: dict[str, Any]) -> bool:
    """Detect R2 (format variant) after _is_direct_valid already passes.

    Distinguishes R1 (exact or case-only difference) from R2 (structural
    difference):
    - authors: initial expansion ("J. Smith" vs "John Smith"), dropped middle name
    - venue: acronym vs full name ("NeurIPS" vs "Advances in Neural ...")
    - title: word-level differences (punctuation, spacing, added/removed words)
    - pages: "3770-3776" vs "pp. 3770-3776", "3770--3776", etc.

    Case-only differences are treated as R1 (trivial formatting from different
    data sources, not an intentional user-side variant).

    Returns True if any match-status field has a structural (non-case-only)
    difference between raw reference and raw candidate.
    """
    def _case_insensitive_eq(a: str, b: str) -> bool:
        return a.strip().lower() == b.strip().lower()

    for fname in ("title", "authors", "venue", "pages"):
        meta = field_status.get(fname) or {}
        if meta.get("status") != "match":
            continue
        ref_val = meta.get("reference")
        cand_val = meta.get("candidate")
        if isinstance(ref_val, list) or isinstance(cand_val, list):
            ref_list = [str(x or "") for x in (ref_val or [])]
            cand_list = [str(x or "") for x in (cand_val or [])]
            if len(ref_list) != len(cand_list):
                return True
            for r, c in zip(ref_list, cand_list):
                if not _case_insensitive_eq(r, c):
                    return True
        else:
            if not _case_insensitive_eq(str(ref_val or ""), str(cand_val or "")):
                return True
    return False


def rule_based_valid_check(
    citation: CitationRecord,
    candidate: CandidateMatch,
    build_comparison_func,
    is_direct_valid_func,
) -> ValidAgentResult:
    """Rule-based validation. If direct_valid passes, return REAL immediately."""
    comparison = build_comparison_func(citation, candidate, list(candidate.conflicts))
    # Use recomputed conflicts from field_status (authoritative) instead of
    # the stale quick-match conflicts from build_candidate_match()
    conflicts = list(comparison.get("conflicts", []))
    fs = comparison["field_status"]
    has_et_al = fs.get("authors", {}).get("has_et_al", False)
    # Sync candidate.conflicts so downstream (reports, guards) see authoritative version
    candidate.conflicts = conflicts

    field_result = FieldComparisonResult(
        field_status=fs,
        signal="",
        has_et_al=has_et_al,
        comparison=comparison,
        conflicts=conflicts,
    )

    # Fast path: direct valid
    if is_direct_valid_func(fs, conflicts):
        ref_authors = fs.get("authors", {}).get("reference", []) or []
        cand_authors = fs.get("authors", {}).get("candidate", []) or []

        # Strict author gate: the rule engine only accepts author lists that
        # are byte-identical (after whitespace collapse + case-insensitive
        # compare). Anything else — reordered, initial expansion, nickname,
        # truncation, dropped middle name — is deferred to LLM ValidAgent
        # which can distinguish R2 (format variant) from P1 (name variant)
        # from H2 (author error).
        def _author_canon(a):
            return " ".join(str(a or "").strip().lower().split())
        ref_canon = [_author_canon(a) for a in ref_authors]
        cand_canon = [_author_canon(a) for a in cand_authors]
        if ref_canon != cand_canon:
            field_result.signal = "soft_discrepancy"
            return ValidAgentResult(
                is_valid=False,
                taxonomy=[],
                hint="likely_potential",
                issues=["author_name_variant"],
                reason=(
                    "Author list not byte-identical — rule engine defers to "
                    "LLM ValidAgent to classify as R2 / P1 / H2."
                ),
                field_result=field_result,
                candidate=candidate,
            )

        is_web = candidate.connector in _WEB_CONNECTORS
        if has_et_al:
            taxonomy = ["R3"]
        elif _has_format_variant(fs):
            taxonomy = ["R2"]
        else:
            taxonomy = ["R1"]

        field_result.signal = "direct_valid"
        return ValidAgentResult(
            is_valid=True,
            taxonomy=taxonomy,
            hint="",
            issues=[],
            reason="All core fields match.",
            field_result=field_result,
            candidate=candidate,
        )

    # Not direct valid → compute hint
    hint, issues = compute_hint(field_result)
    field_result.signal = "soft_discrepancy" if hint == "likely_potential" else "hard_mismatch"

    return ValidAgentResult(
        is_valid=False,
        taxonomy=[],
        hint=hint,
        issues=issues,
        reason=f"Not valid. Issues: {issues}",
        field_result=field_result,
        candidate=candidate,
    )


# ---------------------------------------------------------------------------
# Post-validation guards for LLM VALID results
# ---------------------------------------------------------------------------

# Fields to skip in the "reference field missing from all candidates" check.
# pages and volume are excluded for now — will be handled separately later.
_SKIP_FIELDS_FOR_MISSING_CHECK = {"pages", "volume", "url"}


def _has_multi_letter_truncation(
    citation: CitationRecord, candidate: CandidateMatch,
) -> tuple[bool, str]:
    """Return (detected, description) for multi-letter author truncation (P2, not R2).

    R2 allows ONLY single-letter initials (e.g. "C. Zou" vs "Chang Zou").
    Multi-letter truncation like "Cha. Zou" vs "Chang Zou" is P2.

    Handles name-order differences (e.g. "Cha. Zou" vs "Zou, Chang") by
    matching each ref token against the best candidate token (unordered).

    The description names the specific author pair that triggered the rule,
    e.g. "'Josh Achiam' (citation) vs 'Joshua Achiam' (candidate): 'josh' is
    a multi-letter prefix of 'joshua'".
    """
    from .normalize import normalize_text
    import re as _re

    ref_authors = list(citation.authors or [])
    cand_authors = list(candidate.authors or [])
    if not ref_authors or not cand_authors or len(ref_authors) != len(cand_authors):
        return False, ""

    def _clean_parts(name: str) -> list[str]:
        parts = normalize_text(str(name)).split()
        # Strip DBLP numeric suffixes (e.g. "0001")
        parts = [_re.sub(r"\s*\d{4,}$", "", p).strip() for p in parts]
        return [p for p in parts if p]

    def _is_multi_letter_trunc(short: str, long: str) -> bool:
        """True if short is a multi-letter prefix of long (not single-letter)."""
        s = short.rstrip(".")
        l = long.rstrip(".")
        return len(s) > 1 and len(l) > len(s) and l.startswith(s)

    for r, c in zip(ref_authors, cand_authors):
        r_parts = _clean_parts(r)
        c_parts = _clean_parts(c)

        # For each ref part, find the best matching candidate part
        remaining_cparts = list(c_parts)
        for rp in r_parts:
            # 1) Try exact match first
            if rp in remaining_cparts:
                remaining_cparts.remove(rp)
                continue
            # 2) Try single-letter initial (R2 — allowed)
            matched = False
            for cp in remaining_cparts:
                if (len(rp) == 1 and cp.startswith(rp)) or (len(cp) == 1 and rp.startswith(cp)):
                    remaining_cparts.remove(cp)
                    matched = True
                    break
            if matched:
                continue
            # 3) Check multi-letter truncation (P2)
            for cp in remaining_cparts:
                if _is_multi_letter_trunc(rp, cp):
                    return True, (
                        f"'{r}' (citation) vs '{c}' (candidate): "
                        f"'{rp}' is a multi-letter prefix of '{cp}' "
                        f"(R2 only allows single-letter initials)"
                    )
                if _is_multi_letter_trunc(cp, rp):
                    return True, (
                        f"'{r}' (citation) vs '{c}' (candidate): "
                        f"'{cp}' is a multi-letter prefix of '{rp}' "
                        f"(R2 only allows single-letter initials)"
                    )

    return False, ""


def _h2_is_minor_surname_typo(
    citation_authors: list[str],
    candidate_authors: list[str],
    field_status: dict | None = None,
    threshold: int = 2,
) -> bool:
    """True iff the H2 verdict can plausibly be downgraded back to VALID R1
    because the only author difference is a tiny surname OCR typo.

    Conditions (ALL must hold):
      1. Authors field status is NOT 'partial_overlap' or 'reordered_match'
         (those indicate genuinely different/swapped people, not a typo).
      2. Both author lists are non-empty and the same length (post-et-al strip).
      3. For every position-aligned (ref, cand) pair, the surname edit
         distance is <= `threshold` characters AND given names match.
    """
    if field_status is not None:
        a = field_status.get("authors") or {}
        if isinstance(a, dict):
            st = a.get("status")
            if st in ("partial_overlap", "reordered_match"):
                return False

    import re as _re_h2

    def _et_al(s: str) -> bool:
        return bool(_re_h2.fullmatch(
            r"\s*(?:et\s*al\.?|and\s+others|others)\s*",
            s or "", _re_h2.IGNORECASE,
        ))

    def _split_name(name: str) -> tuple[str, str]:
        """Return (given, surname) lowercase. Handle 'Surname, Given' form."""
        n = (name or "").strip()
        if not n:
            return "", ""
        if "," in n:
            parts = [p.strip() for p in n.split(",", 1)]
            if len(parts) == 2 and parts[0] and parts[1]:
                return parts[1].lower(), parts[0].lower()
        toks = _re_h2.split(r"\s+", n)
        toks = [t for t in toks if t]
        if not toks:
            return "", ""
        if len(toks) == 1:
            return "", toks[0].lower()
        return " ".join(toks[:-1]).lower(), toks[-1].lower()

    # Use the local Levenshtein from adjudicate to avoid circular imports.
    from .adjudicate import _edit_distance

    ref = [a for a in (citation_authors or []) if a and not _et_al(a)]
    cand = [a for a in (candidate_authors or []) if a and not _et_al(a)]
    if not ref or not cand or len(ref) != len(cand):
        return False

    any_typo = False
    for r, c in zip(ref, cand):
        rg, rs = _split_name(r)
        cg, cs = _split_name(c)
        if not rs or not cs:
            return False
        # Surname edit distance must be small
        sdist = _edit_distance(rs, cs, cap=threshold + 1)
        if sdist > threshold:
            return False
        # Given names must be byte-equal after lowercasing (no nickname / variant tolerance)
        if rg != cg:
            return False
        if sdist > 0:
            any_typo = True
    # Require at least one pair to actually have a non-zero typo (otherwise
    # the verdict would already be R1 and this function shouldn't matter).
    return any_typo


def _has_reference_field_missing_from_all(
    citation: CitationRecord,
    candidates: list[CandidateMatch],
    build_comparison_func,
) -> list[str]:
    """Return list of fields that exist in the citation but are missing in ALL candidates.

    If the citation provides a value for a field (e.g. venue, year, doi) and no
    candidate has that field at all, this is suspicious — the field cannot be
    verified.  We exclude pages/volume for now.
    """
    # Collect which reference fields have values
    ref_fields_with_value: set[str] = set()
    for fname in ("venue", "year", "doi"):
        val = getattr(citation, fname, None)
        if val and str(val).strip():
            ref_fields_with_value.add(fname)

    if not ref_fields_with_value:
        return []

    # For each field, check if ANY candidate has it (not candidate_missing)
    fields_found_in_candidates: set[str] = set()
    for cand in candidates:
        for fname in ref_fields_with_value:
            cand_val = getattr(cand, fname, None)
            if cand_val and str(cand_val).strip():
                fields_found_in_candidates.add(fname)

    missing_fields = sorted(ref_fields_with_value - fields_found_in_candidates)
    return missing_fields


def _venue_is_arxiv(venue: str) -> bool:
    """Return True if venue normalizes to arXiv/CoRR (preprint server)."""
    from .normalize import normalize_venue
    return normalize_venue(venue) == "arxiv"


def _venue_is_conference_or_journal(venue: str) -> bool:
    """Return True if venue normalizes to a known conference or journal."""
    from .normalize import normalize_venue, VENUE_ALIASES
    norm = normalize_venue(venue)
    # Everything in VENUE_ALIASES that is NOT arxiv is a conference/journal
    known_venues = {v for v in VENUE_ALIASES.values() if v != "arxiv"}
    return norm in known_venues


def _has_venue_arxiv_vs_conference(citation: CitationRecord, candidate: CandidateMatch) -> tuple[bool, str]:
    """Detect when citation says arXiv but candidate is at a conference, or vice versa.

    arXiv preprint is NOT a valid format variant (R2) of any conference/journal.
    This is at minimum a version difference (P1) or venue error (H4).
    Returns (detected, description).
    """
    ref_venue = (citation.venue or "").strip()
    cand_venue = (candidate.venue or "").strip()
    if not ref_venue or not cand_venue:
        return (False, "")

    ref_is_arxiv = _venue_is_arxiv(ref_venue)
    cand_is_arxiv = _venue_is_arxiv(cand_venue)

    if ref_is_arxiv and not cand_is_arxiv and _venue_is_conference_or_journal(cand_venue):
        from .normalize import normalize_venue
        return (True, f"Citation venue '{ref_venue}' is arXiv but candidate is published at '{cand_venue}' ({normalize_venue(cand_venue)})")
    if cand_is_arxiv and not ref_is_arxiv and _venue_is_conference_or_journal(ref_venue):
        from .normalize import normalize_venue
        return (True, f"Citation venue '{ref_venue}' ({normalize_venue(ref_venue)}) but candidate is only on arXiv")

    return (False, "")


def _has_cross_candidate_year_contradiction(
    citation: CitationRecord,
    candidate: CandidateMatch,
    all_candidates: list[CandidateMatch],
) -> tuple[bool, str]:
    """Detect when the current candidate has missing year but other candidates
    show a year mismatch with the citation.

    If >=2 other candidates have a concrete year that disagrees with the citation,
    we should not trust a candidate with missing year as VALID.
    """
    cite_year = str(citation.year or "").strip()
    cand_year = str(candidate.year or "").strip()
    if not cite_year:
        return (False, "")
    # Only trigger if the winning candidate has no year
    if cand_year and cand_year != "None":
        return (False, "")

    contradicting = 0
    actual_year = None
    for c in all_candidates:
        cy = str(c.year or "").strip()
        if cy and cy != "None" and cy != cite_year:
            contradicting += 1
            actual_year = cy

    if contradicting >= 2:
        return (True, f"Citation year={cite_year} but {contradicting} candidates say year={actual_year}; winning candidate has no year")

    return (False, "")


def _has_structured_conflict_on_missing_fields(
    candidate: CandidateMatch,
    field_result: "FieldComparisonResult",
    prior_evaluations: list[dict],
) -> tuple[bool, str, list[str]]:
    """Detect when the winning candidate has candidate_missing fields, but prior
    structured sources found errors on those same fields.

    E.g. web_search has authors=candidate_missing, but dblp/crossref found
    authors=reordered_match.  The web_search VALID should inherit that conflict.
    """
    if not prior_evaluations:
        return (False, "", [])

    fs = field_result.field_status if field_result else {}
    # Which fields are candidate_missing in the winning candidate?
    missing_fields: set[str] = set()
    for fname, info in fs.items():
        if isinstance(info, dict) and info.get("status") == "candidate_missing":
            missing_fields.add(fname)

    if not missing_fields:
        return (False, "", [])

    # Check prior evaluations from structured (non-web) sources
    _CONFLICT_STATUSES = {"mismatch", "reordered_match", "count_mismatch", "partial_overlap"}
    _FIELD_TO_H = {
        "title": "H1",
        "authors": "H2",
        "venue": "H3",
        "year": "H4",
        "doi": "H5",
        "pages": "H6",
        "volume": "H6",
        "publisher": "H6",
        "location": "H6",
    }

    inherited_issues: list[str] = []
    inherited_descs: list[str] = []

    for ev in prior_evaluations:
        connector = ev.get("connector", "")
        if connector in _WEB_CONNECTORS:
            continue  # only inherit from structured sources
        prior_fs = ev.get("comparison", {}).get("field_status", {})
        for fname in missing_fields:
            prior_info = prior_fs.get(fname, {})
            if isinstance(prior_info, dict) and prior_info.get("status") in _CONFLICT_STATUSES:
                h_code = _FIELD_TO_H.get(fname, "")
                if h_code and h_code not in inherited_issues:
                    inherited_issues.append(h_code)
                    inherited_descs.append(
                        f"{fname}={prior_info['status']} from {connector} "
                        f"(ref={str(prior_info.get('reference',''))[:40]} "
                        f"cand={str(prior_info.get('candidate',''))[:40]})"
                    )

    if inherited_issues:
        desc = "; ".join(inherited_descs)
        return (True, f"Winning candidate missing {sorted(missing_fields)}, but structured sources found conflicts: {desc}", inherited_issues)

    return (False, "", [])


def _post_validate_valid_result(
    citation: CitationRecord,
    candidate: CandidateMatch,
    all_candidates: list[CandidateMatch],
    taxonomy: list[str],
    field_result: "FieldComparisonResult",
    build_comparison_func=None,
    prior_evaluations: list[dict] | None = None,
) -> tuple[bool, str, list[str], VerdictLabel]:
    """Post-validation guard after rule-based or LLM declares VALID.

    Returns (should_override, reason, override_taxonomy, override_verdict).
    override_verdict: FAKE_REFERENCE for hard evidence (H*), POTENTIAL_REFERENCE for soft (P*).
    """
    # Guard 0: Non-academic venue detection
    # If citation claims a non-academic venue but candidate is from an academic source,
    # the venue claim is suspicious → P2
    _NON_ACADEMIC_VENUES = {"personal blog", "blog", "blog post", "tweet", "twitter",
                            "github", "medium", "substack", "personal website", "website"}
    ref_venue_lower = (citation.venue or "").strip().lower()
    if any(marker in ref_venue_lower for marker in _NON_ACADEMIC_VENUES):
        if candidate.connector not in _WEB_CONNECTORS:
            return (
                True,
                f"Guard 0 (P2): citation venue '{citation.venue}' is non-academic but matched academic paper from {candidate.connector}.",
                ["P2"],
                VerdictLabel.POTENTIAL_REFERENCE,
            )

    # Guard 1: Multi-letter author truncation → P1, not R2
    has_trunc, trunc_desc = _has_multi_letter_truncation(citation, candidate)
    if has_trunc:
        return (
            True,
            f"Guard 1 (P1): author name has multi-letter truncation. {trunc_desc}",
            ["P1"],
            VerdictLabel.POTENTIAL_REFERENCE,
        )

    # Guard 2: arXiv vs conference/journal venue mismatch → H3 (venue error)
    venue_issue, venue_desc = _has_venue_arxiv_vs_conference(citation, candidate)
    if venue_issue:
        return (
            True,
            f"Guard 2 (H3): {venue_desc}. Venue mismatch.",
            ["H3"],
            VerdictLabel.FAKE_REFERENCE,
        )

    # Guard 3: Cross-candidate year contradiction → hard evidence
    year_issue, year_desc = _has_cross_candidate_year_contradiction(
        citation, candidate, all_candidates,
    )
    if year_issue:
        return (
            True,
            f"Guard 3 (H4): {year_desc}. Cannot trust VALID from candidate with missing year.",
            ["H4"],
            VerdictLabel.FAKE_REFERENCE,
        )

    # Guard 3b: Year contradiction from any prior candidate for web search winners.
    # Only triggers when the WINNER's own year is unreliable (missing or mismatch).
    # If the winner itself confirms a matching year, prior contradictions are noise.
    if candidate.connector in _WEB_CONNECTORS and prior_evaluations:
        cite_year = str(citation.year or "").strip()
        winner_year_status = ""
        if field_result and field_result.field_status:
            winner_year_status = (
                field_result.field_status.get("year", {}).get("status", "")
            )
        # Skip Guard 3b if winner already independently confirmed the year
        winner_year_unreliable = winner_year_status in ("candidate_missing", "mismatch", "")
        if cite_year and winner_year_unreliable:
            for ev in (prior_evaluations or []):
                conn = ev.get("connector", "")
                yr_status = ev.get("comparison", {}).get("field_status", {}).get("year", {}).get("status")
                if yr_status == "mismatch":
                    actual_year = ev.get("comparison", {}).get("field_status", {}).get("year", {}).get("candidate", "")
                    return (
                        True,
                        f"Guard 3b (H4): prior candidate {conn} says year={actual_year} but citation says {cite_year} (winner year status={winner_year_status!r}).",
                        ["H4"],
                        VerdictLabel.FAKE_REFERENCE,
                    )

    # Guard 4: Inherit structured source conflicts on candidate_missing fields
    conflict_issue, conflict_desc, conflict_tax = _has_structured_conflict_on_missing_fields(
        candidate, field_result, prior_evaluations or [],
    )
    if conflict_issue:
        # H-codes → FAKE, P-codes → POTENTIAL
        has_h = any(t.startswith("H") for t in conflict_tax)
        override_verdict = VerdictLabel.FAKE_REFERENCE if has_h else VerdictLabel.POTENTIAL_REFERENCE
        return (
            True,
            f"Guard 4 ({','.join(conflict_tax)}): {conflict_desc}",
            conflict_tax,
            override_verdict,
        )

    # Guard 5: Reference field present but missing from ALL candidates.
    # Skip if the current winning candidate already has these fields matched
    # (enrichment may have filled them after the original candidates list was built).
    fs = field_result.field_status if field_result else {}
    missing = _has_reference_field_missing_from_all(
        citation, all_candidates, build_comparison_func,
    )
    if missing:
        # Check if the winning candidate actually covers these fields
        still_missing = [
            f for f in missing
            if fs.get(f, {}).get("status") not in ("match", "both_missing", "reference_missing")
        ]
        if still_missing:
            return (
                True,
                f"Guard 5 (P2): reference field(s) {still_missing} not found in any candidate.",
                ["P2"],
                VerdictLabel.POTENTIAL_REFERENCE,
            )

    # Guard 6: Reference provides H6-class fields (volume/pages/publisher/location)
    # that the winning candidate can't verify → override LLM's VALID to P3
    # (insufficient field evidence). Enforces the ValidAgent prompt rule
    # "candidate_missing → valid=false" that LLM sometimes ignores when the
    # rest of the record matches.
    _h6_fields = ("volume", "pages", "publisher", "location")
    _p3_missing = []
    for f in _h6_fields:
        ref_val = getattr(citation, f, None)
        if not (ref_val and str(ref_val).strip()):
            continue  # reference didn't provide the field, nothing to verify
        if fs.get(f, {}).get("status") == "candidate_missing":
            _p3_missing.append(f)
    if _p3_missing:
        return (
            True,
            f"Guard 6 (P3): winning candidate cannot verify reference-provided "
            f"field(s) {_p3_missing}. Downgraded from LLM-VALID to P3.",
            ["P3"],
            VerdictLabel.POTENTIAL_REFERENCE,
        )

    return (False, "", [], VerdictLabel.VALID)


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------

def select_best_candidate(valid_results: list[ValidAgentResult]) -> ValidAgentResult | None:
    """Pick the best failing candidate for downstream routing."""
    if not valid_results:
        return None

    # Prefer: most matching fields, structured over web, potential over hallucinated.
    # Two corrections to a plain match count:
    #   * only fields the REFERENCE actually provides count, so a connector cannot
    #     win by returning extras the citation never claimed;
    #   * candidate_missing on a reference-provided field is penalised, so a record
    #     that simply omits the decisive field cannot outrank one that carries it.
    def score(r: ValidAgentResult) -> tuple:
        fs = r.field_result.field_status if r.field_result else {}
        match_count = 0
        missing_penalty = 0
        for v in fs.values():
            if not isinstance(v, dict):
                continue
            status = v.get("status")
            reference_has = bool(v.get("reference"))
            if status == "match":
                if reference_has:
                    match_count += 1
            elif status == "candidate_missing" and reference_has:
                missing_penalty += 1
        is_structured = 1 if r.candidate and r.candidate.connector not in _WEB_CONNECTORS else 0
        is_potential = 1 if r.hint == "likely_potential" else 0
        return (is_structured, match_count - missing_penalty, is_potential)

    return max(valid_results, key=score)


# ---------------------------------------------------------------------------
# Cascading Orchestrator
# ---------------------------------------------------------------------------

def _apply_url_cap(
    verdict: CitationVerdict,
    url_status: str,
    url_issues: list[str],
    url_type: str = "",
) -> CitationVerdict:
    """If URL check found issues, cap VALID based on URL type.

    url_type: "doi" | "arxiv" | "other"
    - DOI/arXiv 404 → H6/FAKE (academic identifiers must resolve)
    - Other URL 404 → P2/POTENTIAL (URLs can go stale)
    """
    if url_status not in ("mismatch", "not_found") or verdict.verdict != VerdictLabel.VALID:
        return verdict

    # DOI or arXiv identifier not found → H6 (identifier error)
    if url_status == "not_found" and url_type in ("doi", "arxiv"):
        return CitationVerdict(
            citation_id=verdict.citation_id,
            verdict=VerdictLabel.FAKE_REFERENCE,
            evidence_sources=verdict.evidence_sources,
            conflicts=["identifier_not_found"],
            adjudication_reason=f"URL check: {url_type.upper()} resolves to 404. Identifier does not exist (H5). {verdict.adjudication_reason}",
            matched_candidate=verdict.matched_candidate,
            reference_snapshot=verdict.reference_snapshot,
            comparison=verdict.comparison,
            candidate_evaluations=verdict.candidate_evaluations,
            taxonomy_subtype=["H5"],
            needs_human_review=True,
            secondary_evidence=verdict.secondary_evidence,
        )

    # Other cases (mismatch, or non-academic URL not found) → P2
    return CitationVerdict(
        citation_id=verdict.citation_id,
        verdict=VerdictLabel.POTENTIAL_REFERENCE,
        evidence_sources=verdict.evidence_sources,
        conflicts=url_issues or ["url_issue"],
        adjudication_reason=f"URL check: {url_status}. Capped to P2. {verdict.adjudication_reason}",
        matched_candidate=verdict.matched_candidate,
        reference_snapshot=verdict.reference_snapshot,
        comparison=verdict.comparison,
        candidate_evaluations=verdict.candidate_evaluations,
        taxonomy_subtype=["P2"],
        needs_human_review=True,
        secondary_evidence=verdict.secondary_evidence,
    )


# Titles that indicate a redirect/CAPTCHA/error page, not a real resolved title
_PHASE0_JUNK_TITLES = {
    "redirecting", "just a moment", "not found", "access denied",
    "page not found", "error", "captcha", "verify you are human",
    "please wait", "loading", "forbidden", "unauthorized",
}

# Domains that are deterministic scholar identifiers — URL alone uniquely
# identifies a paper. Mismatch on these → hard FAKE (treated like arxiv/doi).
_SCHOLAR_DOMAINS = {
    "arxiv.org",
    "aclanthology.org",
    "openreview.net",
    "ieeexplore.ieee.org",
    "dl.acm.org",
    "link.springer.com",
    "proceedings.mlr.press",
    "proceedings.neurips.cc",
    "papers.nips.cc",
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov/pmc",
}


def _classify_url_type(url: str) -> str:
    """Classify a URL into 'arxiv' | 'scholar' | 'other'."""
    u = url.lower()
    if "arxiv.org" in u or "arxiv:" in u:
        return "arxiv"
    if any(d in u for d in _SCHOLAR_DOMAINS):
        return "scholar"
    return "other"


def phase0_url_check(citation: CitationRecord) -> tuple[str, list[str], str, list[dict]]:
    """Phase 0: Direct URL/DOI verification BEFORE orchestrator query.

    Probes citation.url and citation.doi via URLDirectConnector (with Tavily fallback).
    Returns (url_match_status, url_issues, url_type_triggered, fetched_records).

    - url_match_status: "" | "match" | "mismatch" | "not_found" | "blocked"
    - url_issues: list of H codes detected (e.g. ["H1", "H3"])
    - url_type_triggered: "" | "doi" | "arxiv" | "other"
    - fetched_records: raw url_direct records (can be added to candidates)
    """
    from packages.connectors.url_direct import URLDirectConnector
    from packages.connectors.base import RequestPolicy as _RP
    from .normalize import normalize_title, similarity
    import os as _os

    url_match_status = ""
    url_issues: list[str] = []
    url_type_triggered = ""
    fetched_records: list[dict] = []

    citation_url = (citation.url or "").strip()
    citation_doi = (citation.doi or "").strip()
    citation_arxiv_id = (citation.arxiv_id or "").strip()
    if not (citation_url or citation_doi or citation_arxiv_id):
        return url_match_status, url_issues, url_type_triggered, fetched_records

    # Resolve Tavily key from env or config
    _tavily_key = _os.getenv("TAVILY_API_KEY", "")
    if not _tavily_key:
        try:
            from apps.pdf_checker.config import load_pdf_checker_config as _load_cfg
            _tavily_key = (_load_cfg().connectors.tavily_api_key or "").strip()
        except Exception:
            pass

    url_connector = URLDirectConnector(tavily_api_key=_tavily_key)
    url_policy = _RP(timeout_s=15)

    urls_to_try: list[tuple[str, str]] = []
    if citation_doi:
        urls_to_try.append((f"https://doi.org/{citation_doi}", "doi"))
    if citation_url:
        # Normalize arxiv PDF URLs to abs URLs so we can extract HTML metadata
        # and run version expansion. Examples:
        #   https://arxiv.org/pdf/2106.09685.pdf  → https://arxiv.org/abs/2106.09685
        #   https://arxiv.org/pdf/2106.09685v2    → https://arxiv.org/abs/2106.09685v2
        normalized_url = citation_url
        _u_lower = citation_url.lower()
        if "arxiv.org/pdf/" in _u_lower:
            import re as _re
            normalized_url = _re.sub(
                r"(arxiv\.org)/pdf/", r"\1/abs/", citation_url, flags=_re.IGNORECASE,
            )
            # Strip trailing .pdf if present
            if normalized_url.lower().endswith(".pdf"):
                normalized_url = normalized_url[:-4]
        urls_to_try.append((normalized_url, _classify_url_type(normalized_url)))
    elif (citation.arxiv_id or "").strip():
        # No url but has arxiv_id → construct arxiv URL
        import re as _re
        arxiv_id = (citation.arxiv_id or "").strip()
        arxiv_id = _re.sub(r"^arxiv:", "", arxiv_id, flags=_re.IGNORECASE)
        urls_to_try.append((f"https://arxiv.org/abs/{arxiv_id}", "arxiv"))

    if not urls_to_try:
        return url_match_status, url_issues, url_type_triggered, fetched_records

    # Build extractor agent for LLM-based field extraction from raw content
    _extractor = None
    try:
        from apps.pdf_checker.config import load_pdf_checker_config as _load_cfg2
        _cfg2 = _load_cfg2()
        if _cfg2.verification_llm.enabled and _cfg2.verification_llm.provider == "bedrock":
            from citation_verification_demo import _build_bedrock_client
            from .extractor_agent import BedrockExtractorAgent
            _bc = _build_bedrock_client(
                region=_cfg2.verification_llm.bedrock.region,
                bearer_token=_cfg2.verification_llm.bedrock.bearer_token,
            )
            _extractor = BedrockExtractorAgent(_bc, str(_cfg2.verification_llm.bedrock.model_id))
    except Exception:
        pass

    for try_url, _url_type in urls_to_try:
        try:
            probe = CitationRecord(citation_id="_url_check", url=try_url)
            fetched = url_connector.search(probe, url_policy)
            if not fetched:
                url_match_status = "not_found"
                url_type_triggered = _url_type
                break

            resolved = fetched[0]
            raw_content = str(resolved.get("raw_content", "") or "").strip()

            # Use ExtractorAgent LLM to extract structured fields from raw content
            if _extractor and raw_content:
                try:
                    from .extractor_agent import EXTRACTOR_PROMPT, _call_bedrock
                    prompt = EXTRACTOR_PROMPT.format(content=raw_content[:3000])
                    parsed = _call_bedrock(_extractor.client, _extractor.model_id, prompt)
                    if parsed:
                        resolved["title"] = str(parsed.get("title", "") or "").strip()
                        resolved["authors"] = parsed.get("authors") or []
                        resolved["venue"] = str(parsed.get("venue", "") or "").strip()
                        raw_year = parsed.get("year")
                        resolved["year"] = int(raw_year) if raw_year is not None else None
                        resolved["doi"] = str(parsed.get("doi", "") or "").strip()
                        resolved["pages"] = str(parsed.get("pages", "") or "").strip()
                        resolved["volume"] = str(parsed.get("volume", "") or "").strip()
                        resolved["publisher"] = str(parsed.get("publisher", "") or "").strip()
                        resolved["_extraction_method"] = "extractor_agent_llm"
                except Exception:
                    pass

            resolved_title = str(resolved.get("title", "") or "").strip()

            if resolved_title.lower().strip() in _PHASE0_JUNK_TITLES:
                url_match_status = "blocked"
                continue

            # If extraction couldn't get a title, treat as blocked
            if not resolved_title:
                url_match_status = "blocked"
                continue

            # Compare resolved fields with citation
            issues: list[str] = []
            ref_title_norm = normalize_title(citation.title)
            cand_title_norm = normalize_title(resolved_title)
            # Scholar URLs (arxiv/doi/scholar) → require exact title match
            # Other URLs → 0.75 similarity threshold
            if _url_type in ("arxiv", "doi", "scholar"):
                if ref_title_norm != cand_title_norm:
                    issues.append("H1")
            else:
                sim = similarity(ref_title_norm, cand_title_norm)
                if sim < 0.75:
                    issues.append("H1")
            resolved_year = resolved.get("year")
            if citation.year and resolved_year:
                try:
                    # Publisher landing pages advertise the online-first date while
                    # citations use the issue year, so a +/-1 delta is a formatting
                    # artifact, not evidence of a different paper. Only flag H4 when
                    # the gap is >= 2 years.
                    if abs(int(citation.year) - int(resolved_year)) >= 2:
                        issues.append("H4")
                except (TypeError, ValueError):
                    pass
            # Venue is NOT compared here: Extractor LLM often returns the long
            # CrossRef-style venue name while citations use DBLP-style acronyms
            # (e.g. "BlackboxNLP@EMNLP"). Venue differences are handled later
            # by ValidAgent / field comparison against structured connector
            # candidates, which have proper venue normalization.

            # Save the record so caller can add to candidates
            resolved["_resolved_from"] = try_url
            fetched_records.append(resolved)

            # arXiv version expansion: if this is an arxiv URL, fetch version
            # history and per-version metadata. Title/authors/year may differ
            # across versions (e.g. AutoGen v1 has 10 authors, v2 has 14).
            if _url_type == "arxiv":
                version_records = _expand_arxiv_versions(
                    try_url, resolved, url_policy,
                )
                for vr in version_records:
                    fetched_records.append(vr)

                # Cross-version full match: clear ALL issues only if there
                # exists ONE version where every checked field matches the
                # citation simultaneously. Field-by-field stripping (any
                # version matches title, any matches year) is too lenient
                # because the matches might come from different versions.
                if version_records and issues:
                    cite_year_int = None
                    if citation.year:
                        try:
                            cite_year_int = int(citation.year)
                        except (TypeError, ValueError):
                            cite_year_int = None

                    for vr in version_records:
                        v_title = str(vr.get("title", "") or "").strip()
                        v_year = vr.get("year")
                        title_match = (
                            v_title and
                            normalize_title(v_title) == ref_title_norm
                        )
                        year_match = (
                            cite_year_int is None
                            or (v_year is not None and int(v_year) == cite_year_int)
                        )
                        # All checked fields must match for this version
                        if title_match and year_match:
                            issues = []  # this version fully matches the citation
                            break

            if issues:
                url_match_status = "mismatch"
                url_issues = issues
                url_type_triggered = _url_type
            else:
                url_match_status = "match"
            break

        except RuntimeError as e:
            err_msg = str(e).lower()
            if "404" in err_msg:
                url_match_status = "not_found"
                url_type_triggered = _url_type
                break
            if any(code in err_msg for code in ["403", "429", "ssl", "certificate"]):
                url_match_status = "blocked"
                continue
            continue
        except Exception:
            continue

    return url_match_status, url_issues, url_type_triggered, fetched_records


def _expand_arxiv_versions(
    try_url: str,
    base_record: dict,
    policy,
) -> list[dict]:
    """For an arxiv URL, fetch per-version metadata (title/authors/year) for each version.

    Uses ArxivConnector.fetch_per_version_records() which hits arxiv.org/abs/{id}{vN}
    for each version and parses the version-specific HTML meta tags. This is the
    only reliable way to get per-version author lists (e.g. AutoGen v1 has 10 authors,
    v2 has 14 — different lists, not just different years).
    """
    from packages.connectors.arxiv import ArxivConnector
    import re as _re

    arxiv_id_raw = ArxivConnector._clean_arxiv_id(try_url)
    if not arxiv_id_raw:
        return []
    base_id = _re.sub(r"v\d+$", "", arxiv_id_raw)
    if not base_id:
        return []

    arxiv_conn = ArxivConnector()
    try:
        records = arxiv_conn.fetch_per_version_records(base_id, policy)
    except Exception:
        return []

    # Tag each record with _resolved_from for evidence trace
    for r in records:
        r["_resolved_from"] = r.get("url", "")
    return records


def _llm_classify_url_academic(url: str) -> bool:
    """Use LLM to classify whether a URL is academic/scholarly or not.

    Returns True if the URL points to an academic resource (journal, conference,
    preprint server, university, research lab, etc.).
    """
    import json as _json, re as _re
    try:
        import os as _os
        from apps.pdf_checker.config import load_pdf_checker_config
        cfg = load_pdf_checker_config()
        if not cfg.verification_llm.enabled:
            return False
        from citation_verification_demo import _build_bedrock_client
        client = _build_bedrock_client(
            region=cfg.verification_llm.bedrock.region,
            bearer_token=cfg.verification_llm.bedrock.bearer_token,
        )
        prompt = (
            f"Classify this URL as academic or non-academic. Reply with JSON only.\n\n"
            f"Mark academic = TRUE for any of:\n"
            f"  - Journal / conference / workshop / preprint hosts: arxiv.org, "
            f"aclanthology.org, openreview.net, ieeexplore.ieee.org, dl.acm.org, "
            f"link.springer.com, sciencedirect.com, nature.com, science.org, "
            f"plos.org, biorxiv.org, ssrn.com, ojs.aaai.org, papers.nips.cc, "
            f"proceedings.mlr.press, jmlr.org, semanticscholar.org\n"
            f"  - DOI redirects: doi.org/*, dx.doi.org/*\n"
            f"  - University domains (.edu, .ac.<cc>) for paper / thesis / tech report pages\n"
            f"  - Research lab pages reporting peer-reviewed work\n\n"
            f"Mark academic = FALSE only when the URL clearly points to:\n"
            f"  - GitHub repo / Gitlab repo / Bitbucket\n"
            f"  - Personal blog (medium.com, substack, wordpress, blog.*)\n"
            f"  - Social media (x.com, twitter.com, mastodon, threads)\n"
            f"  - Company marketing or product page (e.g., openai.com/blog, "
            f"anthropic.com/news, mistral.ai, ai.meta.com/blog)\n"
            f"  - News outlet (nytimes, theverge, theregister, etc.)\n"
            f"  - Forum (lesswrong, alignmentforum, reddit, hackernews)\n\n"
            f"If the URL is ambiguous or you cannot recognise the domain, default to TRUE (academic).\n\n"
            f"URL: {url}\n\n"
            f'Return: {{"academic": true/false, "reason": "..."}}'
        )
        response = client.converse(
            modelId=str(cfg.verification_llm.bedrock.model_id),
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"temperature": 0, "maxTokens": 128},
        )
        text = response["output"]["message"]["content"][0]["text"]
        m = _re.search(r"\{.*\}", text, _re.DOTALL)
        if m:
            parsed = _json.loads(m.group(0))
            return bool(parsed.get("academic", False))
    except Exception:
        pass
    # Fallback: use domain heuristic
    return any(d in url.lower() for d in _SCHOLAR_DOMAINS)


def _llm_classify_citation_academic(citation: CitationRecord) -> bool:
    """Use LLM to classify whether a citation (without URL) is academic or not.

    Returns True if the citation refers to an academic paper/report.
    Returns False for blog posts, product pages, API docs, announcements, etc.
    """
    import json as _json, re as _re
    title = (citation.title or "").strip()
    authors = ", ".join(citation.authors or [])
    venue = (citation.venue or "").strip()
    year = citation.year
    raw_text = (getattr(citation, "raw_text", None) or "").strip()
    if not title:
        return True  # can't classify without title, assume academic
    try:
        from apps.pdf_checker.config import load_pdf_checker_config
        cfg = load_pdf_checker_config()
        if not cfg.verification_llm.enabled:
            return True
        from citation_verification_demo import _build_bedrock_client
        client = _build_bedrock_client(
            region=cfg.verification_llm.bedrock.region,
            bearer_token=cfg.verification_llm.bedrock.bearer_token,
        )
        prompt = (
            f"Classify this citation as academic or non-academic. Reply with JSON only.\n\n"
            f"Look at the VENUE field first; it is the most reliable signal.\n\n"
            f"Mark academic = TRUE when the venue contains any of these patterns "
            f"(case-insensitive substring match):\n"
            f"  - 'Proceedings', 'Journal', 'Conference', 'Symposium', 'Workshop', "
            f"'Trans.', 'Transactions'\n"
            f"  - Conference / journal acronyms: NeurIPS, ICML, ICLR, AAAI, EMNLP, "
            f"NAACL, ACL, COLING, CVPR, ICCV, ECCV, KDD, SIGIR, WWW, UAI, COLT, "
            f"AISTATS, JMLR, PMLR, TACL, IJCAI\n"
            f"  - Publisher names: IEEE, ACM, Springer, Elsevier, Nature, Science, "
            f"PLOS, Wiley, Oxford, Cambridge\n"
            f"  - Preprint marker: 'arXiv', 'arxiv preprint', 'bioRxiv', 'SSRN'\n"
            f"  - Thesis / report: 'PhD thesis', 'Master thesis', 'Tech report', "
            f"'Technical Report'\n"
            f"A garbled or fragmentary venue does not change the verdict — if any "
            f"academic marker above appears, the citation is academic.\n\n"
            f"Mark academic = FALSE only when the venue or raw_text clearly indicates:\n"
            f"  - A blog (e.g., 'OpenAI blog', 'Anthropic blog', 'AI Alignment Forum', "
            f"'Meta AI Research Blog', 'Mistral.ai', 'Substack', 'Medium')\n"
            f"  - A GitHub / Gitlab / package repository\n"
            f"  - Social media (Twitter / X post, tweet)\n"
            f"  - News article from a non-academic outlet\n"
            f"  - Company product page or API documentation\n"
            f"  - **Web-citation markers in raw_text** (very reliable signal):\n"
            f"      * 'Retrieved <date> from <url>' / 'Retrieved from <url>'\n"
            f"      * 'Accessed <date>' / 'Accessed: <date>' / 'Last accessed <date>'\n"
            f"      * '[Online]. Available: <url>'\n"
            f"      * Any 'visited on'/'viewed on' + date + URL pattern.\n"
            f"    These are APA/Chicago/CSE/IEEE style markers used for webpages,\n"
            f"    not academic papers — when present, return academic=FALSE.\n\n"
            f"If the venue is empty or you cannot decide, default to TRUE (academic).\n"
            f"This is a hallucinated-citation benchmark — do NOT use 'fields look "
            f"garbled / suspicious' as a signal; that is a separate hallucination "
            f"check and is not your job here.\n\n"
            f"Title: {title}\n"
            f"Authors: {authors}\n"
            f"Venue: {venue}\n"
            f"Year: {year}\n"
            f"Raw text: {raw_text[:500]}\n\n"
            f'Return: {{"academic": true/false, "reason": "..."}}'
        )
        response = client.converse(
            modelId=str(cfg.verification_llm.bedrock.model_id),
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"temperature": 0, "maxTokens": 128},
        )
        text = response["output"]["message"]["content"][0]["text"]
        m = _re.search(r"\{.*\}", text, _re.DOTALL)
        if m:
            parsed = _json.loads(m.group(0))
            return bool(parsed.get("academic", True))
    except Exception:
        pass
    return True  # default: assume academic


def phase0_decide_verdict(
    citation: CitationRecord,
    url_match_status: str,
    url_issues: list[str],
    url_type_triggered: str,
    evidence_sources: list[str] | None = None,
    fetched_records: list[dict] | None = None,
) -> CitationVerdict | None:
    """Decide whether Phase 0 alone is enough to return a final verdict.

    New logic:
    - match → return None (continue normal pipeline with fetched records as candidates)
    - mismatch → return None (continue normal pipeline — let agents decide)
    - not_found → LLM classifies URL:
        - academic URL → FAKE_REFERENCE H5 (URL/identifier error)
        - non-academic URL → POTENTIAL_REFERENCE P2
    - blocked / "" → return None (continue normal flow)
    """
    # blocked / "" → continue normal pipeline
    if url_match_status not in ("match", "mismatch", "not_found"):
        return None
    if not url_type_triggered and url_match_status != "match":
        return None

    # match → check if ALL reference fields are covered by the fetched record.
    # If yes, early-exit VALID. If not, continue to normal pipeline for full verification.
    if url_match_status == "match" and fetched_records:
        from .adjudicate import _build_comparison, _is_direct_valid, _reference_snapshot
        from .matching import build_candidate_match

        for rec in fetched_records:
            cand = build_candidate_match(citation, "url_direct", rec)
            comparison = _build_comparison(citation, cand, [])
            fs = comparison["field_status"]
            conflicts = comparison.get("conflicts", [])
            if _is_direct_valid(fs, conflicts):
                return CitationVerdict(
                    citation_id=citation.citation_id,
                    verdict=VerdictLabel.VALID,
                    evidence_sources=evidence_sources or ["url_direct"],
                    conflicts=[],
                    adjudication_reason=(
                        f"Phase 0: URL {rec.get('_resolved_from', rec.get('url', ''))} "
                        f"resolved and all reference fields match. Early VALID."
                    ),
                    matched_candidate={"connector": "url_direct", **rec},
                    reference_snapshot=_reference_snapshot(citation),
                    comparison=comparison,
                    candidate_evaluations=[{
                        "connector": "url_direct",
                        "candidate": rec,
                        "verdict": "VALID",
                        "reason": "Phase 0 full field match",
                        "comparison": comparison,
                    }],
                    taxonomy_subtype=["R1"],
                    needs_human_review=False,
                )
        # No fetched record fully matches → continue normal pipeline
        return None

    evidence_sources = evidence_sources or ["url_direct"]

    from .adjudicate import _reference_snapshot

    # Build candidate_evaluations from Phase 0 fetched records
    phase0_evaluations: list[dict] = []
    for rec in (fetched_records or []):
        phase0_evaluations.append({
            "connector": "url_direct",
            "candidate": rec,
            "verdict": f"PHASE0_{url_match_status.upper()}",
            "reason": f"Phase 0 URL fetch from {rec.get('_resolved_from', rec.get('url', ''))}",
        })

    # Determine if URL is academic.
    # First check url_type_triggered (already classified during Phase 0 fetch).
    # Only fall back to LLM if url_type is "other" (unknown).
    citation_url = (citation.url or "").strip()
    if url_type_triggered in ("arxiv", "doi", "scholar"):
        is_academic = True
    elif citation_url:
        is_academic = _llm_classify_url_academic(citation_url)
    else:
        is_academic = False

    if is_academic:
        # A "mismatch" verdict is only trustworthy when the scraped record carries
        # real corroborating metadata. Landing pages often yield just a title plus
        # an online-first date, which is too thin to declare "different paper";
        # those cases fall through to the normal pipeline instead of hard-failing.
        if url_match_status == "mismatch":
            _best_corroboration = 0
            for _rec in (fetched_records or []):
                _best_corroboration = max(_best_corroboration, sum(
                    1 for _f in ("authors", "venue", "volume", "pages")
                    if (_rec or {}).get(_f)
                ))
            if _best_corroboration < 2:
                return None
        # Academic URL: mismatch or not_found → FAKE H5 (identifier inconsistency)
        taxonomy = ["H5"]
        if url_match_status == "mismatch":
            reason = (
                f"Phase 0: academic identifier ({url_type_triggered}) resolved but "
                f"fields don't match citation. Identifier points to a different paper (H5)."
            )
        else:  # not_found
            reason = (
                f"Phase 0: academic identifier ({url_type_triggered}) not reachable. "
                f"Identifier does not resolve (H5)."
            )
        return CitationVerdict(
            citation_id=citation.citation_id,
            verdict=VerdictLabel.FAKE_REFERENCE,
            evidence_sources=evidence_sources,
            conflicts=taxonomy,
            adjudication_reason=reason,
            reference_snapshot=_reference_snapshot(citation),
            candidate_evaluations=phase0_evaluations,
            taxonomy_subtype=taxonomy,
            needs_human_review=True,
        )

    # Non-academic URL: mismatch or not_found → P2
    if url_match_status == "mismatch":
        reason = (
            f"Phase 0: non-academic URL resolved but fields don't match citation "
            f"({citation_url}). Source unverifiable (P2)."
        )
    else:  # not_found
        reason = (
            f"Phase 0: non-academic URL not reachable ({citation_url}). "
            f"Source unverifiable (P2)."
        )
    return CitationVerdict(
        citation_id=citation.citation_id,
        verdict=VerdictLabel.POTENTIAL_REFERENCE,
        evidence_sources=evidence_sources,
        conflicts=list(url_issues) if url_issues else ["url_unverified"],
        adjudication_reason=reason,
        reference_snapshot=_reference_snapshot(citation),
        candidate_evaluations=phase0_evaluations,
        taxonomy_subtype=["P2"],
        needs_human_review=True,
    )


# Domains / venue keywords that signal a non-academic citation source.
_NON_ACADEMIC_URL_DOMAINS = {
    "github.com", "gitlab.com", "bitbucket.org",
    "cran.r-project.org", "r-project.org",
    "pypi.org", "npmjs.com",
    "huggingface.co",
    "docs.python.org", "docs.pydantic.dev",
    "medium.com", "substack.com",
    "twitter.com", "x.com",
    "alignmentforum.org", "lesswrong.com",
    "linkedin.com",
    "youtube.com", "youtu.be",
    "stackoverflow.com", "stackexchange.com",
    "reddit.com",
    "kaggle.com",
    "wikipedia.org",
    "openai.com/blog", "anthropic.com/news", "ai.meta.com/blog",
    # Music / video / streaming — the connector occasionally matches
    # citation titles to song / podcast metadata via web_search.
    "spotify.com", "open.spotify.com",
    "music.apple.com", "podcasts.apple.com",
    "soundcloud.com",
    "vimeo.com",
    # Crypto / DeFi product pages (citation often points at protocol homepages)
    "makerdao.com", "uniswap.org", "aave.com", "curve.fi",
    "ethereum.org", "bitcoin.org", "solana.com",
}

_NON_ACADEMIC_VENUE_KEYWORDS = {
    "github", "gitlab", "blog", "tweet", "twitter",
    "personal blog", "personal website", "website",
    "package", "library", "software", "tool",
    "documentation", "docs",
    "medium", "substack",
    "openai blog", "anthropic blog",
    "spotify", "soundcloud", "youtube", "vimeo",
    "linkedin", "reddit", "stackoverflow", "stack overflow",
    "wikipedia", "wiki",
}

_NON_ACADEMIC_RAW_TEXT_KEYWORDS = {
    "r package", "python package", "npm package",
    "github repository", "gitlab repository",
    "blog post", "tweet",
}


# Web-citation patterns: APA/Chicago/CSE-style "retrieved <date> from <url>"
# or "accessed <date>" indicate the citation points at a webpage / online
# resource, not an academic paper. These are reliable enough to short-circuit
# directly to non-academic without further checks.
import re as _re_web
_NON_ACADEMIC_RAW_TEXT_PATTERNS = [
    # "retrieved <date> from <url>" / "retrieved from <url>"
    _re_web.compile(r"\bretrieved\b[^.\n]*\bfrom\b", _re_web.IGNORECASE),
    # "accessed <date>" / "accessed: <date>" / "accessed on <date>"
    _re_web.compile(
        r"\baccessed\b\s*(?::|on)?\s*"
        r"(?:\d{1,2}[-/.\s]\d{1,2}[-/.\s]\d{2,4}|"     # 12/03/2024
        r"\d{4}[-/.\s]\d{1,2}[-/.\s]\d{1,2}|"           # 2024-12-03
        r"(?:january|february|march|april|may|june|july|august|"
        r"september|october|november|december|jan\.?|feb\.?|mar\.?|apr\.?|"
        r"may\.?|jun\.?|jul\.?|aug\.?|sep\.?|oct\.?|nov\.?|dec\.?)\b"  # March 2024 / Mar. 2024
        r")",
        _re_web.IGNORECASE,
    ),
    # "last accessed" variant (common in LaTeX bibs)
    _re_web.compile(r"\blast\s+accessed\b", _re_web.IGNORECASE),
    # "online" + URL pattern (e.g., "[Online]. Available: https://...")
    _re_web.compile(r"\[\s*online\s*\].{0,40}?\bavailable\b", _re_web.IGNORECASE),
]


def _matches_web_citation_pattern(raw_text: str) -> bool:
    """True if raw_text contains a web-citation marker (retrieved-from /
    accessed-date / [Online] Available)."""
    if not raw_text:
        return False
    return any(p.search(raw_text) for p in _NON_ACADEMIC_RAW_TEXT_PATTERNS)


# Brand / organization names that, when used as the SOLE author of a
# citation, signal a corporate blog / product page / model card etc., not
# an academic paper. e.g. citation = ['Huggingface'] (brand) means it is
# pointing at huggingface.co content, not a peer-reviewed paper.
_BRAND_AUTHOR_NAMES = {
    "huggingface", "hugging face",
    "openai", "anthropic", "google", "deepmind", "google deepmind",
    "meta", "meta ai", "microsoft", "microsoft research", "ibm", "amazon",
    "nvidia", "apple", "tesla",
    "github", "gitlab",
}


def _authors_are_brand_only(authors: list[str]) -> bool:
    """True iff every listed author is a brand/org name (not a person)."""
    if not authors:
        return False
    cleaned = [str(a or "").strip().lower() for a in authors if str(a or "").strip()]
    if not cleaned:
        return False
    return all(a in _BRAND_AUTHOR_NAMES for a in cleaned)


def _is_title_only_citation(citation: "CitationRecord") -> bool:
    """True if the citation has ONLY a title (and maybe a year), with empty
    authors / venue / url / doi / arxiv_id. Such citations are almost
    always pointing at a product / protocol / website / brand page rather
    than an academic paper. e.g.:
        title='MakerDAO' authors=[] venue='' url='' year=2025
    """
    title = (citation.title or "").strip()
    if not title:
        return False
    has_authors = bool([a for a in (citation.authors or []) if str(a or "").strip()])
    has_venue = bool((citation.venue or "").strip())
    has_url = bool((citation.url or "").strip())
    has_doi = bool((citation.doi or "").strip())
    has_arxiv = bool((citation.arxiv_id or "").strip())
    return not (has_authors or has_venue or has_url or has_doi or has_arxiv)


_ACADEMIC_VENUE_KEYWORDS = {
    "proceedings", "journal", "conference", "symposium", "workshop",
    "trans.", "transactions",
    "neurips", "icml", "iclr", "aaai", "emnlp", "naacl", "acl", "coling",
    "cvpr", "iccv", "eccv", "kdd", "sigir", "www", "uai", "colt",
    "aistats", "jmlr", "pmlr", "tacl", "ijcai",
    "ieee", "acm", "springer", "elsevier", "nature", "science", "plos",
    "wiley", "oxford", "cambridge",
    "arxiv", "bioarxiv", "biorxiv", "ssrn",
    "phd thesis", "master thesis", "tech report", "technical report",
}


def _venue_is_academic(venue: str) -> bool:
    """Cheap rule: venue contains a strong academic marker (Proceedings /
    Journal / Conference / Symposium / Workshop / Trans. / publisher name /
    venue acronym). When this fires, skip the non-academic rule and the LLM
    fallback — the citation is academic regardless of any 'software' /
    'package' / 'tool' substrings that would otherwise trip false positives
    on names like 'Journal of Statistical Software'."""
    return any(kw in venue for kw in _ACADEMIC_VENUE_KEYWORDS)


_ACADEMIC_IDENTIFIER_RES = (
    re.compile(r"\barxiv\s*:\s*\d{4}\.\d{4,5}", re.IGNORECASE),
    re.compile(r"\barxiv\.org/(?:abs|pdf)/\d{4}\.\d{4,5}", re.IGNORECASE),
    re.compile(r"\b10\.\d{4,9}/[\w./\-]+"),
)


def is_non_academic_citation(citation: CitationRecord) -> bool:
    """Detect whether a citation refers to a non-academic source.

    Signals (checked in order):
      0. raw_text contains a valid arXiv ID or DOI → academic, short-circuit
         (overrides "[Online]. Available: ..." IEEE-format markers and similar
         web-citation patterns that would otherwise mis-flag IEEE-style arXiv
         references as non-academic).
      1. Venue contains an academic marker (Proceedings / Journal / IEEE /
         ACM / arXiv / NeurIPS / ICML / ...) → academic, short-circuit.
      2. URL domain in non-academic list (fast rule).
      3. Venue contains non-academic keywords (fast rule).
      4. raw_text mentions package / library / blog (fast rule).
      5. LLM classification via URL or citation metadata (if rules silent).
    """
    raw = getattr(citation, "raw_text", None) or ""
    if raw and any(rx.search(raw) for rx in _ACADEMIC_IDENTIFIER_RES):
        return False  # academic short-circuit via identifier presence

    venue = (citation.venue or "").lower().strip()
    if venue and _venue_is_academic(venue):
        return False  # academic short-circuit

    url = (citation.url or "").lower()
    if any(d in url for d in _NON_ACADEMIC_URL_DOMAINS):
        return True

    if venue:
        if any(kw in venue for kw in _NON_ACADEMIC_VENUE_KEYWORDS):
            return True

    raw_lower = raw.lower()
    if raw_lower:
        if any(kw in raw_lower for kw in _NON_ACADEMIC_RAW_TEXT_KEYWORDS):
            return True
        # Web-citation patterns: "retrieved <date> from <url>" / "accessed
        # <date>" / "[Online]. Available: ..." — APA/Chicago/CSE style
        # markers for online resources, not academic papers.
        if _matches_web_citation_pattern(raw):
            return True

    # Brand-only author rule: when the citation lists a company / lab name as
    # the sole author (and venue gave no academic signal above), it is almost
    # always pointing at a corporate blog / product page / model card.
    if _authors_are_brand_only(list(citation.authors or [])):
        return True

    # Title-only rule: when the citation has ONLY a title (no authors, venue,
    # url, doi, arxiv_id), it is almost always a product / protocol / website
    # reference (e.g. "MakerDAO", "Bitcoin", "Uniswap") rather than a paper.
    if _is_title_only_citation(citation):
        return True

    # Fallback: LLM classification.
    # If URL exists, classify by URL. Otherwise classify by title/authors/venue
    # (and raw_text — see _llm_classify_citation_academic for the prompt).
    citation_url = (citation.url or "").strip()
    if citation_url:
        if not _llm_classify_url_academic(citation_url):
            return True
    else:
        # No URL — classify by citation metadata
        if not _llm_classify_citation_academic(citation):
            return True

    return False


def downgrade_non_academic_fake(
    citation: CitationRecord, verdict: CitationVerdict,
) -> CitationVerdict:
    """If verdict is FAKE_REFERENCE but the citation is non-academic, downgrade to P2.

    Non-academic citations (R packages, GitHub repos, blog posts, tweets) often
    have inconsistent / sparse metadata, so a strict FAKE judgment is too harsh.
    Treat them as POTENTIAL/P2 (unverifiable) instead.
    """
    if verdict.verdict != VerdictLabel.FAKE_REFERENCE:
        return verdict

    # Also check the matched candidate's URL/venue: when the verifier's best
    # match landed on a LinkedIn / GitHub / blog page even though the citation
    # itself had no URL, the citation is effectively pointing at non-academic
    # content (the candidate is the closest thing the connectors found, and
    # it's a blog).
    cand_non_academic = False
    mc = verdict.matched_candidate or {}
    cand_url = (mc.get("url") or "").lower()
    cand_venue = (mc.get("venue") or "").lower()
    if any(d in cand_url for d in _NON_ACADEMIC_URL_DOMAINS):
        cand_non_academic = True
    elif cand_venue and any(kw in cand_venue for kw in _NON_ACADEMIC_VENUE_KEYWORDS):
        cand_non_academic = True
    elif cand_venue in {"linkedin", "youtube", "reddit", "stackoverflow"}:
        cand_non_academic = True

    if not (is_non_academic_citation(citation) or cand_non_academic):
        return verdict
    return CitationVerdict(
        citation_id=verdict.citation_id,
        verdict=VerdictLabel.POTENTIAL_REFERENCE,
        evidence_sources=verdict.evidence_sources,
        conflicts=verdict.conflicts,
        adjudication_reason=(
            f"Downgraded from FAKE to POTENTIAL because citation is non-academic "
            f"(software/blog/tweet). Original reason: {verdict.adjudication_reason}"
        ),
        matched_candidate=verdict.matched_candidate,
        reference_snapshot=verdict.reference_snapshot,
        comparison=verdict.comparison,
        candidate_evaluations=verdict.candidate_evaluations,
        taxonomy_subtype=["P2"],
        needs_human_review=True,
        secondary_evidence=verdict.secondary_evidence,
    )


def verify_single_candidate(
    citation: CitationRecord,
    candidate: CandidateMatch,
    sibling_candidates: list[CandidateMatch],
    prior_evaluations: list[dict[str, Any]],
    valid_agent: ValidAgentProtocol | None,
    extractor_agent: Any,
    tavily_api_key: str,
    evidence_sources: list[str],
    url_match_status: str,
    url_issues: list[str],
    url_type_triggered: str,
    build_comparison_func=None,
    is_direct_valid_func=None,
    author_classifier: AuthorVariantClassifierProtocol | None = None,
) -> tuple[CitationVerdict | None, dict[str, Any], ValidAgentResult | None]:
    """Run rule-based + LLM ValidAgent verification on a single candidate.

    This encapsulates the per-candidate logic from cascading_adjudicate's
    Phase 1 loop so callers can invoke it per-connector in parallel.

    Returns (verdict, eval_dict, valid_result):
        - verdict: clean VALID CitationVerdict if this candidate passes; else None
        - eval_dict: always populated — the entry to append to all_evaluations
        - valid_result: the ValidAgentResult if NOT_VALID (for Phase 2 best-candidate
          selection); None when the candidate produced a VALID or was overridden by guard
    """
    from .adjudicate import _build_comparison, _is_direct_valid

    build_comp = build_comparison_func or _build_comparison
    is_valid_func = is_direct_valid_func or _is_direct_valid

    # Web candidate enrichment (Extractor LLM on snippet/raw_content, then URL fetch)
    if candidate.connector in _WEB_CONNECTORS and not candidate.title and extractor_agent:
        try:
            candidate = extractor_agent.extract(candidate)
        except Exception:
            pass
    if candidate.connector in _WEB_CONNECTORS and candidate.url and extractor_agent:
        candidate = _enrich_web_candidate_via_url(
            candidate, tavily_api_key=tavily_api_key, extractor_agent=extractor_agent,
        )

    # Rule-based check
    result = rule_based_valid_check(citation, candidate, build_comp, is_valid_func)

    # Rule-based VALID → guard
    if result.is_valid:
        override, override_reason, override_tax, override_verdict = _post_validate_valid_result(
            citation, candidate, sibling_candidates, result.taxonomy,
            result.field_result, build_comp, prior_evaluations=prior_evaluations,
        )
        if override:
            eval_dict = {
                "connector": candidate.connector,
                "candidate": candidate_match_to_dict(candidate),
                "verdict": override_verdict.value,
                "taxonomy": override_tax,
                "reason": override_reason,
                "comparison": result.field_result.comparison if result.field_result else {},
            }
            return None, eval_dict, None
        # Clean VALID
        eval_dict = {
            "connector": candidate.connector,
            "candidate": candidate_match_to_dict(candidate),
            "verdict": "VALID",
            "taxonomy": result.taxonomy,
            "reason": result.reason,
            "comparison": result.field_result.comparison if result.field_result else {},
        }
        verdict = _apply_url_cap(CitationVerdict(
            citation_id=citation.citation_id,
            verdict=VerdictLabel.VALID,
            evidence_sources=evidence_sources,
            conflicts=[],
            adjudication_reason=result.reason,
            matched_candidate=candidate_match_to_dict(candidate),
            reference_snapshot=result.field_result.comparison.get("reference", {}),
            comparison=result.field_result.comparison,
            candidate_evaluations=prior_evaluations + [eval_dict],
            taxonomy_subtype=result.taxonomy,
            needs_human_review=False,
        ), url_match_status, url_issues, url_type_triggered)
        return verdict, eval_dict, None

    # Unified field pre-classification: one LLM call produces authoritative
    # verdicts for authors (exact/r2_initial/p1_variant/h2_error), venue
    # (exact/alias/different), and publisher (exact/alias/different).
    # ValidAgent consumes these instead of re-deriving them, and R4 coverage
    # + field_status routing use them (p1_variant NOT a match for R4; venue
    # alias IS a match for downstream). When the classifier's venue verdict
    # is "alias" and rule-level status was "mismatch", we upgrade status to
    # "match" (because aliases ARE semantically equivalent).
    author_analysis: AuthorVariantResult | None = None
    field_classification: FieldClassificationResult | None = None
    if author_classifier and result.field_result is not None:
        try:
            field_classification = author_classifier.classify(
                citation.authors or [], candidate.authors or [],
                citation_venue=citation.venue or "",
                candidate_venue=candidate.venue or "",
                citation_publisher=citation.publisher or "",
                candidate_publisher=getattr(candidate, "publisher", "") or "",
            )
            author_analysis = field_classification.authors
            fs = (result.field_result.field_status or {})

            # Stamp authors verdict
            authors_entry = fs.setdefault("authors", {})
            if isinstance(authors_entry, dict):
                authors_entry["author_variant_type"] = field_classification.authors.overall
                authors_entry["author_variant_reason"] = field_classification.authors.reason
                # Mirror the venue/publisher behavior: when the LLM
                # classifier concludes "exact" or "r2_initial" (a pure
                # formatting variant), promote status from mismatch back
                # to match so HallucinatedAgent does not see a
                # contradictory (status=mismatch, variant=exact) signal
                # and mechanically flag H2 against the classifier's own
                # conclusion ("...so per rules we flag as H2" cases).
                if (field_classification.authors.overall in ("exact", "r2_initial")
                        and authors_entry.get("status") == "mismatch"):
                    authors_entry["_rule_status"] = authors_entry.get("status")
                    authors_entry["status"] = "match"
                    authors_entry["_classifier_corrected"] = True

            # Stamp venue verdict; upgrade status to match when classifier says alias/exact
            venue_entry = fs.setdefault("venue", {})
            if isinstance(venue_entry, dict):
                venue_entry["venue_variant_type"] = field_classification.venue.overall
                venue_entry["venue_variant_reason"] = field_classification.venue.reason
                if (field_classification.venue.overall in ("exact", "alias")
                        and venue_entry.get("status") == "mismatch"):
                    venue_entry["_rule_status"] = venue_entry.get("status")
                    venue_entry["status"] = "match"
                    venue_entry["_classifier_corrected"] = True

            # Stamp publisher verdict; upgrade status similarly
            pub_entry = fs.setdefault("publisher", {})
            if isinstance(pub_entry, dict):
                pub_entry["publisher_variant_type"] = field_classification.publisher.overall
                pub_entry["publisher_variant_reason"] = field_classification.publisher.reason
                if (field_classification.publisher.overall in ("exact", "alias")
                        and pub_entry.get("status") == "mismatch"):
                    pub_entry["_rule_status"] = pub_entry.get("status")
                    pub_entry["status"] = "match"
                    pub_entry["_classifier_corrected"] = True

            # Recompute hint/issues since venue/publisher status may have changed
            new_hint, new_issues = compute_hint(result.field_result)
            result.hint = new_hint
            result.issues = new_issues

            # Stash classifier summary + raw response on the comparison dict so
            # it travels into eval_dict / final JSON report for diagnostics.
            if result.field_result.comparison is not None:
                result.field_result.comparison["_field_classification"] = {
                    "authors": {
                        "overall": field_classification.authors.overall,
                        "reason": field_classification.authors.reason,
                    },
                    "venue": {
                        "overall": field_classification.venue.overall,
                        "reason": field_classification.venue.reason,
                    },
                    "publisher": {
                        "overall": field_classification.publisher.overall,
                        "reason": field_classification.publisher.reason,
                    },
                    "raw_llm_response": field_classification.raw_llm_response,
                }
        except Exception as exc:
            author_analysis = None
            field_classification = None
            if result.field_result is not None and result.field_result.comparison is not None:
                result.field_result.comparison["_field_classification"] = {
                    "error": f"{type(exc).__name__}: {exc}",
                }

    # Rule-based NOT_VALID → call LLM ValidAgent to integrate field_status +
    # author_analysis into a final verdict. Per-field classification is
    # already authoritative from the rule comparator + FieldClassifier;
    # ValidAgent just picks taxonomy/hint/issues from that state.
    if valid_agent and not result.is_valid:
        try:
            llm_result = valid_agent.evaluate(
                citation, candidate, result.field_result,
                author_analysis=author_analysis,
            )
            if llm_result.is_valid:
                override, override_reason, override_tax, override_verdict = _post_validate_valid_result(
                    citation, candidate, sibling_candidates, llm_result.taxonomy,
                    result.field_result, build_comp, prior_evaluations=prior_evaluations,
                )
                if override:
                    eval_dict = {
                        "connector": candidate.connector,
                        "candidate": candidate_match_to_dict(candidate),
                        "verdict": override_verdict.value,
                        "taxonomy": override_tax,
                        "reason": f"LLM said VALID but overridden: {override_reason}",
                        "comparison": result.field_result.comparison if result.field_result else {},
                    }
                    return None, eval_dict, None
                # Clean LLM VALID
                eval_dict = {
                    "connector": candidate.connector,
                    "candidate": candidate_match_to_dict(candidate),
                    "verdict": "VALID",
                    "taxonomy": llm_result.taxonomy,
                    "reason": llm_result.reason,
                    "comparison": result.field_result.comparison if result.field_result else {},
                }
                verdict = _apply_url_cap(CitationVerdict(
                    citation_id=citation.citation_id,
                    verdict=VerdictLabel.VALID,
                    evidence_sources=evidence_sources,
                    conflicts=[],
                    adjudication_reason=f"ValidAgent LLM: {llm_result.reason}",
                    matched_candidate=candidate_match_to_dict(candidate),
                    reference_snapshot=result.field_result.comparison.get("reference", {}),
                    comparison=result.field_result.comparison,
                    candidate_evaluations=prior_evaluations + [eval_dict],
                    taxonomy_subtype=llm_result.taxonomy,
                    needs_human_review=False,
                ), url_match_status, url_issues, url_type_triggered)
                return verdict, eval_dict, None
            # LLM NOT_VALID → use its hint/issues/taxonomy/reason directly.
            result = llm_result
        except Exception:
            pass

    eval_dict = {
        "connector": candidate.connector,
        "candidate": candidate_match_to_dict(candidate),
        "verdict": "NOT_VALID",
        "hint": result.hint,
        "issues": result.issues,
        "reason": result.reason,
        "comparison": result.field_result.comparison if result.field_result else {},
    }
    return None, eval_dict, result


def cascading_phase2_only(
    citation: CitationRecord,
    candidates: list[CandidateMatch],
    all_valid_results: list[ValidAgentResult],
    all_evaluations: list[dict[str, Any]],
    evidence_sources: list[str],
    secondary_verifier: Any = None,
    potential_agent: PotentialAgentProtocol | None = None,
    hallucinated_agent: HallucinatedAgentProtocol | None = None,
    url_match_status: str = "",
    url_issues: list[str] | None = None,
    url_type_triggered: str = "",
) -> CitationVerdict:
    """Run Phase 2 adjudication on pre-computed Phase 1 results.

    Assumes the caller has already iterated over candidates using
    verify_single_candidate and collected all_valid_results + all_evaluations.
    This function handles: R4 cross-candidate coverage check, best-candidate
    selection, Phase 2a (PotentialAgent), Phase 2b (HallucinatedAgent), plus
    R4/P3 fallbacks.
    """
    from .adjudicate import _reference_snapshot

    if url_issues is None:
        url_issues = []

    # No candidates at all → H1 / INSUFFICIENT
    if not candidates:
        return CitationVerdict(
            citation_id=citation.citation_id,
            verdict=VerdictLabel.FAKE_REFERENCE,
            evidence_sources=evidence_sources,
            conflicts=["no_candidate"],
            adjudication_reason="No matching paper found.",
            reference_snapshot=_reference_snapshot(citation),
            taxonomy_subtype=["H1"],
            candidate_evaluations=all_evaluations,
            needs_human_review=True,
        )

    # R4 cross-candidate coverage tracking.
    # Only candidates that are "essentially the same paper" (title + authors +
    # year all match the reference) contribute to coverage. This filters out
    # noise candidates (e.g. connectors returning semantically unrelated records)
    # that would otherwise artificially inflate coverage.
    _trackable_fields = (
        "title", "authors", "venue", "year", "volume", "pages", "publisher",
        "location", "doi", "arxiv_id",
    )
    ref_fields_with_value: set[str] = set()
    for fname in _trackable_fields:
        val = getattr(citation, fname, None)
        if val and str(val).strip():
            ref_fields_with_value.add(fname)

    def _authors_authoritative_match(fs: dict) -> bool:
        """For R4 purposes, authors are authoritatively 'match' only when the
        AuthorVariantClassifier said exact or r2_initial. P1 variants
        (nickname / multi-letter truncation / typo) MUST NOT count as match
        for cross-candidate coverage — otherwise P1 mutations get upgraded
        to VALID R4 incorrectly.

        Falls back to plain status==match when classifier verdict absent
        (preserves legacy behavior when the classifier is not configured).
        """
        info = fs.get("authors", {})
        if not isinstance(info, dict):
            return False
        if info.get("status") != "match":
            return False
        variant = info.get("author_variant_type")
        if variant is None:
            return True  # no classifier verdict → trust status
        return variant in ("exact", "r2_initial")

    def _qualifies_for_p3(ev: dict[str, Any]) -> bool:
        """A candidate qualifies for R4 coverage if title + authors match
        (or LLM judged the candidate VALID). Year is intentionally NOT
        required here — preprint/published versions of the same paper can
        report different years yet be the same work. The per-field coverage
        check still tracks year separately: if any qualifying candidate
        reports year=mismatch, year ends up in `contradicted_fields` and
        R4 does not fire (so H4 mutations are still caught).

        For authors specifically, "match" means classifier said exact or
        r2_initial — p1_variant does NOT qualify (variant requires
        interpretation, not authoritative identity proof).
        """
        if ev.get("verdict") == "VALID":
            return True
        fs = ev.get("comparison", {}).get("field_status", {})
        title_ok = fs.get("title", {}).get("status") == "match"
        authors_ok = _authors_authoritative_match(fs)
        return bool(title_ok and authors_ok)

    # A field is "covered" only if at least one qualifying candidate reports
    # match AND no qualifying candidate reports mismatch. This prevents a
    # single noisy candidate (e.g. web_search extractor parroting back the
    # query-injected metadata) from overriding multiple candidates that
    # explicitly contradict the reference on that field.
    matched_fields_across_candidates: set[str] = set()
    contradicted_fields_across_candidates: set[str] = set()
    for ev in all_evaluations:
        if not _qualifies_for_p3(ev):
            continue
        fs = ev.get("comparison", {}).get("field_status", {})
        for fname in ref_fields_with_value:
            info = fs.get(fname, {})
            if not isinstance(info, dict):
                continue
            st = info.get("status")
            if fname == "authors":
                # For R4 coverage the authors field is authoritatively "match"
                # only when classifier verdict is exact / r2_initial.
                if _authors_authoritative_match(fs):
                    matched_fields_across_candidates.add(fname)
                elif st == "mismatch":
                    contradicted_fields_across_candidates.add(fname)
                continue
            if st == "match":
                matched_fields_across_candidates.add(fname)
            elif st == "mismatch":
                contradicted_fields_across_candidates.add(fname)

    covered_fields = matched_fields_across_candidates - contradicted_fields_across_candidates
    # Default rule: every reference field with value must be covered.
    # Relaxed rule: title OR authors covered is enough for the title/authors
    # group; year and venue still must be individually covered if present.
    from .adjudicate import _RELAXED_FIELDS as _relaxed
    if _relaxed:
        ta_in_ref = ref_fields_with_value & {"title", "authors"}
        ta_covered = bool(ta_in_ref) and bool(ta_in_ref & covered_fields)
        gating = ref_fields_with_value & {"year", "venue"}
        gating_covered = gating.issubset(covered_fields)
        all_fields_covered = (
            bool(ref_fields_with_value)
            and (ta_covered or not ta_in_ref)
            and gating_covered
        )
    else:
        all_fields_covered = (
            bool(ref_fields_with_value)
            and ref_fields_with_value.issubset(covered_fields)
        )

    best = select_best_candidate(all_valid_results)
    if best is None:
        if all_fields_covered:
            return CitationVerdict(
                citation_id=citation.citation_id,
                verdict=VerdictLabel.VALID,
                evidence_sources=evidence_sources,
                conflicts=[],
                adjudication_reason=(
                    f"R4: every reference field {sorted(ref_fields_with_value)} "
                    f"was independently verified by at least one qualifying candidate "
                    f"(title+authors+year matching), though no single candidate matched "
                    f"all fields cleanly. Cross-candidate coverage."
                ),
                reference_snapshot=_reference_snapshot(citation),
                candidate_evaluations=all_evaluations,
                taxonomy_subtype=["R4"],
                needs_human_review=False,
            )
        return CitationVerdict(
            citation_id=citation.citation_id,
            verdict=VerdictLabel.FAKE_REFERENCE,
            evidence_sources=evidence_sources,
            conflicts=[],
            adjudication_reason="No candidates matched.",
            taxonomy_subtype=["H1"],
            candidate_evaluations=all_evaluations,
            needs_human_review=True,
        )

    best_candidate = best.candidate
    best_field_result = best.field_result
    best_comparison = best_field_result.comparison if best_field_result else {}

    # Cross-candidate hard-evidence early gate: if any reference field was
    # contradicted by some qualifying candidate and NEVER matched by any, that
    # is stronger evidence of fabrication than whatever best's hint says. Emit
    # FAKE + the H-code for those fields BEFORE Phase 2a/2b can route us to a
    # softer verdict (P3 / PotentialAgent).
    _FIELD_TO_HCODE = {
        "title": "H1",
        "authors": "H2",
        "venue": "H3",
        "year": "H4",
        "doi": "H5",
        "arxiv_id": "H5",
        "pages": "H6",
        "volume": "H6",
        "publisher": "H6",
        "location": "H6",
    }
    _unverified_mismatch = (
        contradicted_fields_across_candidates - matched_fields_across_candidates
    )
    _cross_h_codes: list[str] = []
    for _f in sorted(_unverified_mismatch):
        _h = _FIELD_TO_HCODE.get(_f)
        if _h and _h not in _cross_h_codes:
            _cross_h_codes.append(_h)
    _best_fs_early = (best_field_result.field_status if best_field_result else {}) or {}
    _early_core_match = all(
        _best_fs_early.get(f, {}).get("status") == "match"
        for f in ("title", "authors", "year")
        if _best_fs_early.get(f, {}).get("status") is not None
    ) and any(
        _best_fs_early.get(f, {}).get("status") == "match"
        for f in ("title", "authors", "year")
    )
    if _cross_h_codes and _early_core_match and not all_fields_covered:
        # Surname-typo safety net for H2 (RELAXED MODE ONLY): if the only
        # contradicted field is authors AND the diff is just a small surname
        # OCR typo, downgrade to VALID R1 instead of emitting H2.
        from .adjudicate import _RELAXED_FIELDS as _relaxed
        if _relaxed and _cross_h_codes == ["H2"] and "authors" in _unverified_mismatch:
            best_fs_authors = (best_field_result.field_status if best_field_result else {}).get("authors") or {}
            if _h2_is_minor_surname_typo(
                list(citation.authors or []),
                list((best_candidate.authors if best_candidate else []) or []),
                {"authors": best_fs_authors},
            ):
                # Suppress the H2 cross-candidate gate; let the normal flow
                # produce a VALID verdict.
                _cross_h_codes = []

    if _cross_h_codes and _early_core_match and not all_fields_covered:
        # Collect ref vs cand evidence per unverified field so the reason is
        # actionable instead of just "field X contradicted".
        _details: list[str] = []
        for _f in sorted(_unverified_mismatch):
            _ref_val = ""
            _per_conn: list[str] = []
            for _ev in all_evaluations:
                _info = (_ev.get("comparison", {}) or {}).get("field_status", {}).get(_f, {})
                if not isinstance(_info, dict):
                    continue
                if _info.get("status") == "mismatch":
                    if not _ref_val:
                        _ref_val = str(_info.get("reference", ""))[:80]
                    _conn = _ev.get("connector", "?")
                    _cand = str(_info.get("candidate", ""))[:80]
                    _per_conn.append(f"{_conn}={_cand!r}")
            if _per_conn:
                _details.append(f"{_f}: citation={_ref_val!r} ≠ {', '.join(_per_conn)}")
            else:
                _details.append(f"{_f}: contradicted (no per-connector detail)")
        _detail_str = "; ".join(_details)
        all_evaluations.append({
            "agent": "CrossCandidateHardEvidence",
            "verdict": "FAKE_REFERENCE",
            "taxonomy": _cross_h_codes,
            "reason": (
                f"Cross-candidate hard evidence: {_detail_str}. "
                f"Upgraded from potential soft path to FAKE."
            ),
        })
        return CitationVerdict(
            citation_id=citation.citation_id,
            verdict=VerdictLabel.FAKE_REFERENCE,
            evidence_sources=evidence_sources,
            conflicts=list(_unverified_mismatch),
            adjudication_reason=(
                f"CrossCandidate ({','.join(_cross_h_codes)}): {_detail_str}."
            ),
            matched_candidate=candidate_match_to_dict(best_candidate),
            reference_snapshot=best_comparison.get("reference", {}),
            comparison=best_comparison,
            candidate_evaluations=all_evaluations,
            taxonomy_subtype=_cross_h_codes,
            needs_human_review=False,
        )

    # P-code short-circuits: trust FieldClassifier when it gives a clean soft
    # signal. Bypass PotentialAgent (LLM, overly strict on nicknames) and
    # HallucinatedAgent (would default to H1 with no signal) to avoid them
    # turning POTENTIAL into FAKE incorrectly.
    #
    # "match-equivalent" = {match, both_missing, reference_missing}. This is
    # stricter than allowing candidate_missing — a candidate that lacks the
    # field doesn't count as "match-equivalent", only as "unverifiable".
    _sc_fs = (best_field_result.field_status or {}) if best_field_result else {}
    def _sc_status(f):
        e = _sc_fs.get(f)
        return e.get("status", "") if isinstance(e, dict) else ""

    _MATCH_EQ = ("match", "both_missing", "reference_missing")

    _sc_author_variant = ""
    _sc_authors_entry = _sc_fs.get("authors")
    if isinstance(_sc_authors_entry, dict):
        _sc_author_variant = _sc_authors_entry.get("author_variant_type", "")

    # P1 short-circuit: classifier says authors = p1_variant AND every other
    # field the reference provides is match-equivalent (match / both_missing /
    # reference_missing — NOT candidate_missing or mismatch). PotentialAgent
    # often over-rejects less-obvious nicknames ("Moe" vs "Mohit"); trust the
    # classifier and emit POTENTIAL/P1 directly.
    _p1_non_author_fields = (
        "title", "venue", "year", "doi", "arxiv_id",
        "pages", "volume", "publisher", "location",
    )
    _p1_non_author_match_eq = all(
        _sc_status(f) in _MATCH_EQ for f in _p1_non_author_fields
    )
    # An UNRESOLVED classification means the classifier never produced a verdict
    # (truncated response); p1_variant is only its default. Short-circuiting on it
    # turns a non-answer into a POTENTIAL verdict, so fall through to the agent
    # chain instead and let a real judgement be made.
    _sc_author_reason = ""
    if isinstance(_sc_authors_entry, dict):
        _sc_author_reason = str(_sc_authors_entry.get("author_variant_reason", "") or "")
    _sc_author_unresolved = _sc_author_reason.startswith("UNRESOLVED")
    if (_sc_author_variant == "p1_variant" and _p1_non_author_match_eq
            and not _sc_author_unresolved):
        # Name the specific author(s) that flagged as p1_variant so
        # reviewers see which name(s) are unverifiable instead of a
        # generic "author name variant" message.
        _ref_authors = best_comparison.get("reference", {}).get("authors", []) or []
        _cand_authors = list(getattr(best_candidate, "authors", []) or [])
        _variant_pairs = _author_variant_pairs(_ref_authors, _cand_authors)
        _variant_desc = _format_variant_pairs(_variant_pairs)
        _reason_suffix = f" Variant author(s): {_variant_desc}." if _variant_desc else ""
        all_evaluations.append({
            "agent": "P1_short_circuit",
            "verdict": "POTENTIAL",
            "taxonomy": ["P1"],
            "reason": "Short-circuit: classifier authors=p1_variant; every other reference-provided field is match-equivalent.",
            "variant_authors": [{"reference": r, "candidate": c} for r, c in _variant_pairs],
        })
        return CitationVerdict(
            citation_id=citation.citation_id,
            verdict=VerdictLabel.POTENTIAL_REFERENCE,
            evidence_sources=evidence_sources,
            conflicts=["author_name_variant"],
            adjudication_reason=(
                "P1 short-circuit: FieldClassifier judged authors = p1_variant "
                "and every other field the reference provides is match-equivalent. "
                "Trust classifier; mark as POTENTIAL/P1." + _reason_suffix
            ),
            matched_candidate=candidate_match_to_dict(best_candidate),
            reference_snapshot=best_comparison.get("reference", {}),
            comparison=best_comparison,
            candidate_evaluations=all_evaluations,
            taxonomy_subtype=["P1"],
            needs_human_review=True,
        )

    # P3 short-circuit: authors effectively match (classifier exact / r2_initial
    # or raw match), every non-peripheral reference-provided field is match-
    # equivalent, AND at least one peripheral field (pages/volume/publisher/
    # location) is candidate_missing — with the other peripherals staying
    # match-equivalent-or-candidate_missing (no peripheral mismatch). Bypass the
    # HallucinatedAgent → P3 fallback chain and emit POTENTIAL/P3 directly.
    _sc_authors_effectively_match = (
        _sc_status("authors") == "match"
        or _sc_author_variant in ("exact", "r2_initial")
    )
    _p3_non_peri_fields = ("title", "venue", "year", "doi", "arxiv_id")
    _p3_non_peri_match_eq = all(
        _sc_status(f) in _MATCH_EQ for f in _p3_non_peri_fields
    )
    _p3_peri_fields = ("pages", "volume", "publisher", "location")
    _p3_peri_clean = all(
        _sc_status(f) in _MATCH_EQ + ("candidate_missing",)
        for f in _p3_peri_fields
    )
    def _peri_corroborated(field: str) -> bool:
        """True if some OTHER candidate already verified this peripheral field.

        The best candidate is chosen by an unweighted match count, so a connector
        that omits a peripheral field can win over one that supplies it. Without
        this check the P3 reason ("not verifiable by any candidate") can be false:
        a sibling candidate held a matching value. A fabricated peripheral cannot
        be corroborated by definition, so genuine P3 citations are unaffected.
        """
        for _r in all_valid_results:
            if _r is best or not getattr(_r, "field_result", None):
                continue
            _fs = (_r.field_result.field_status or {}).get(field) or {}
            if _fs.get("status") == "match":
                return True
        return False

    missing_peri = [
        f for f in _p3_peri_fields
        if _sc_status(f) == "candidate_missing" and not _peri_corroborated(f)
    ]
    _p3_has_peri_cm = bool(missing_peri)
    if (
        _sc_authors_effectively_match
        and _p3_non_peri_match_eq
        and _p3_peri_clean
        and _p3_has_peri_cm
    ):
        all_evaluations.append({
            "agent": "P3_short_circuit",
            "verdict": "POTENTIAL",
            "taxonomy": ["P3"],
            "reason": (
                f"Short-circuit: authors effectively match, every non-peripheral "
                f"reference field is match-equivalent, peripherals "
                f"{missing_peri} are candidate_missing."
            ),
        })
        return CitationVerdict(
            citation_id=citation.citation_id,
            verdict=VerdictLabel.POTENTIAL_REFERENCE,
            evidence_sources=evidence_sources,
            conflicts=[f"{f}_candidate_missing" for f in missing_peri],
            adjudication_reason=(
                f"P3 short-circuit: all non-peripheral reference-provided fields "
                f"match-equivalent, peripheral field(s) {missing_peri} not "
                f"verifiable by any candidate. POTENTIAL/P3."
            ),
            matched_candidate=candidate_match_to_dict(best_candidate),
            reference_snapshot=best_comparison.get("reference", {}),
            comparison=best_comparison,
            candidate_evaluations=all_evaluations,
            taxonomy_subtype=["P3"],
            needs_human_review=True,
        )

    # Core-identity gate: if the best candidate is missing a core identity
    # field that the reference provides (title / authors / venue / year /
    # doi), the candidate cannot plausibly be the cited paper. Skip Phase 2a
    # entirely and route to HallucinatedAgent / H-code, regardless of what
    # upstream LLM agents may have hinted. P3 (peripheral unverifiable) is
    # reserved for pages / volume / publisher / location only.
    _CORE_IDENTITY_FIELDS_GATE = ("title", "authors", "venue", "year", "doi")
    _gate_fs = (best.field_result.field_status if best and best.field_result else {}) or {}
    _gate_missing = [
        f for f in _CORE_IDENTITY_FIELDS_GATE
        if _gate_fs.get(f, {}).get("status") == "candidate_missing"
    ]
    if _gate_missing and best.hint == "likely_potential":
        all_evaluations.append({
            "agent": "CoreFieldGate",
            "verdict": "ESCALATED_TO_HALLUCINATED",
            "taxonomy": [],
            "reason": (
                f"Best candidate is missing core identity field(s) the reference "
                f"provides ({_gate_missing}); routing to HallucinatedAgent "
                f"instead of PotentialAgent."
            ),
        })
        best.hint = "likely_hallucinated"
        best.issues = list(set(best.issues + [f"{f}_candidate_missing" for f in _gate_missing]))

    # Phase 2a: likely_potential → PotentialAgent
    if best.hint == "likely_potential":
        sec_evidence: list[DiscrepancyEvidence] = []
        if secondary_verifier and best_field_result:
            try:
                sec_evidence = secondary_verifier.gather_evidence(
                    citation, best_candidate,
                    best_field_result.field_status, best_field_result.conflicts,
                )
            except Exception:
                sec_evidence = []

        if potential_agent:
            try:
                downstream = potential_agent.evaluate(
                    citation, best_candidate, best_field_result, best, sec_evidence,
                )
            except Exception as exc:
                downstream = DownstreamAgentResult(
                    label=VerdictLabel.FAKE_REFERENCE,
                    reason=f"PotentialAgent error: {exc}",
                )
        else:
            all_explained = sec_evidence and all(e.evidence_found for e in sec_evidence)
            if all_explained:
                downstream = DownstreamAgentResult(
                    label=VerdictLabel.POTENTIAL_REFERENCE,
                    taxonomy=["P1"],
                    reason="All discrepancies explained by evidence.",
                )
            else:
                downstream = DownstreamAgentResult(
                    label=VerdictLabel.FAKE_REFERENCE,
                    reason="Some discrepancies unexplained.",
                )

        if downstream.label == VerdictLabel.POTENTIAL_REFERENCE:
            all_evaluations.append({
                "agent": "PotentialAgent",
                "verdict": "POTENTIAL",
                "taxonomy": downstream.taxonomy,
                "reason": downstream.reason,
            })
            return CitationVerdict(
                citation_id=citation.citation_id,
                verdict=VerdictLabel.POTENTIAL_REFERENCE,
                evidence_sources=evidence_sources,
                conflicts=_clean_conflicts(best.issues),
                adjudication_reason=f"PotentialAgent: {downstream.reason}",
                matched_candidate=candidate_match_to_dict(best_candidate),
                reference_snapshot=best_comparison.get("reference", {}),
                comparison=best_comparison,
                candidate_evaluations=all_evaluations,
                taxonomy_subtype=downstream.taxonomy,
                secondary_evidence=sec_evidence,
                needs_human_review=True,
            )
        # Escalate to HallucinatedAgent. Track PotentialAgent's taxonomy
        # findings on a separate slot so they do not pollute best.issues
        # (which is downstream-fed as the verdict's conflicts list, where
        # field-level tags like 'author_name_variant' are appropriate but
        # taxonomy codes like 'H1' are not).
        best.hint = "likely_hallucinated"
        if not hasattr(best, "escalated_taxonomy"):
            best.escalated_taxonomy = []
        best.escalated_taxonomy = list(set((best.escalated_taxonomy or []) + (downstream.taxonomy or [])))
        all_evaluations.append({
            "agent": "PotentialAgent",
            "verdict": "ESCALATED_TO_HALLUCINATED",
            "taxonomy": downstream.taxonomy,
            "reason": downstream.reason,
        })

    # Phase 2b: likely_hallucinated → HallucinatedAgent
    if hallucinated_agent:
        try:
            downstream = hallucinated_agent.evaluate(
                citation, best_candidate, best_field_result, best,
            )
        except Exception as exc:
            downstream = DownstreamAgentResult(
                label=VerdictLabel.FAKE_REFERENCE,
                reason=f"HallucinatedAgent error: {exc}",
            )
    else:
        taxonomy = []
        for issue in best.issues:
            if "title" in issue: taxonomy.append("H1")
            elif "author" in issue: taxonomy.append("H2")
            elif "venue" in issue: taxonomy.append("H3")
            elif "year" in issue: taxonomy.append("H4")
            elif "doi" in issue: taxonomy.append("H5")
            elif "pages" in issue: taxonomy.append("H6")
            elif "volume" in issue: taxonomy.append("H6")
            elif "publisher" in issue: taxonomy.append("H6")
            elif "location" in issue: taxonomy.append("H6")
        downstream = DownstreamAgentResult(
            label=VerdictLabel.FAKE_REFERENCE,
            taxonomy=list(set(taxonomy)) or ["H1"],
            reason=f"Hallucinated based on issues: {best.issues}",
        )

    if downstream.label == VerdictLabel.POTENTIAL_REFERENCE:
        verdict_label = VerdictLabel.POTENTIAL_REFERENCE
    else:
        verdict_label = VerdictLabel.FAKE_REFERENCE

    # R4 upgrade: HallucinatedAgent said FAKE, but every reference field was
    # independently matched by a qualifying candidate → cross-candidate
    # coverage counts as REAL (R4), overriding the FAKE verdict.
    if verdict_label == VerdictLabel.FAKE_REFERENCE and all_fields_covered:
        all_evaluations.append({
            "agent": "R4_upgrade",
            "verdict": "VALID",
            "taxonomy": ["R4"],
            "reason": (
                f"HallucinatedAgent said FAKE but every reference field "
                f"{sorted(ref_fields_with_value)} was independently matched by a "
                f"qualifying candidate (title+authors+year matching). Upgraded to R4."
            ),
        })
        return CitationVerdict(
            citation_id=citation.citation_id,
            verdict=VerdictLabel.VALID,
            evidence_sources=evidence_sources,
            conflicts=_clean_conflicts(best.issues if best else []),
            adjudication_reason=(
                f"R4: every reference field {sorted(ref_fields_with_value)} "
                f"was independently verified by at least one qualifying candidate "
                f"(title+authors+year matching), though no single candidate matched "
                f"all fields cleanly. Cross-candidate coverage."
            ),
            matched_candidate=candidate_match_to_dict(best_candidate),
            reference_snapshot=best_comparison.get("reference", {}),
            comparison=best_comparison,
            candidate_evaluations=all_evaluations,
            taxonomy_subtype=["R4"],
            needs_human_review=False,
        )

    # P3 fallback: FAKE + only candidate_missing issues → P3.
    # Guard: require that the best candidate's core fields (title, authors, year)
    # genuinely match the reference. Candidates with core fields all `candidate_missing`
    # (e.g. empty web_search records) must not trigger this fallback.
    _only_candidate_missing_issues = {
        "pages_candidate_missing", "volume_candidate_missing",
        "publisher_candidate_missing", "location_candidate_missing",
    }
    _best_fs = (best.field_result.field_status if best and best.field_result else {}) or {}
    _core_fields_match = all(
        _best_fs.get(f, {}).get("status") == "match"
        for f in ("title", "authors", "year")
        if _best_fs.get(f, {}).get("status") is not None
    ) and any(
        _best_fs.get(f, {}).get("status") == "match"
        for f in ("title", "authors", "year")
    )

    # Cross-candidate hard-evidence upgrade: BEFORE P3 fallback, check if any
    # field was contradicted by some qualifying candidate but NEVER matched
    # by any. If so, that's cross-candidate proof of fabrication — emit FAKE
    # with the H-code for those fields, instead of P3 "insufficient evidence".
    #
    # This catches the common structural bug where crossref returns the real
    # publisher (→ mismatch for that candidate) but best is DBLP (publisher
    # empty → candidate_missing), and P3 would otherwise swallow the evidence.
    _FIELD_TO_HCODE = {
        "title": "H1",
        "authors": "H2",
        "venue": "H3",
        "year": "H4",
        "doi": "H5",
        "arxiv_id": "H5",
        "pages": "H6",
        "volume": "H6",
        "publisher": "H6",
        "location": "H6",
    }
    unverified_mismatch_fields = (
        contradicted_fields_across_candidates - matched_fields_across_candidates
    )
    cross_candidate_h_codes: list[str] = []
    for f in sorted(unverified_mismatch_fields):
        h = _FIELD_TO_HCODE.get(f)
        if h and h not in cross_candidate_h_codes:
            cross_candidate_h_codes.append(h)
    if (
        verdict_label == VerdictLabel.FAKE_REFERENCE
        and not all_fields_covered
        and cross_candidate_h_codes
        and best
        and _core_fields_match
    ):
        # Build per-field ref vs cand evidence so the reason is actionable.
        details_2: list[str] = []
        for f in sorted(unverified_mismatch_fields):
            ref_val_2 = ""
            per_conn_2: list[str] = []
            for ev in all_evaluations:
                info = (ev.get("comparison", {}) or {}).get("field_status", {}).get(f, {})
                if not isinstance(info, dict):
                    continue
                if info.get("status") == "mismatch":
                    if not ref_val_2:
                        ref_val_2 = str(info.get("reference", ""))[:80]
                    conn = ev.get("connector", "?")
                    cand_val = str(info.get("candidate", ""))[:80]
                    per_conn_2.append(f"{conn}={cand_val!r}")
            if per_conn_2:
                details_2.append(f"{f}: citation={ref_val_2!r} ≠ {', '.join(per_conn_2)}")
            else:
                details_2.append(f"{f}: contradicted (no per-connector detail)")
        detail_str_2 = "; ".join(details_2)
        all_evaluations.append({
            "agent": "CrossCandidateHardEvidence",
            "verdict": "FAKE_REFERENCE",
            "taxonomy": cross_candidate_h_codes,
            "reason": (
                f"Cross-candidate hard evidence: {detail_str_2}. "
                f"Upgraded from P3 to FAKE."
            ),
        })
        return CitationVerdict(
            citation_id=citation.citation_id,
            verdict=VerdictLabel.FAKE_REFERENCE,
            evidence_sources=evidence_sources,
            conflicts=_clean_conflicts(best.issues if best else []),
            adjudication_reason=(
                f"CrossCandidate ({','.join(cross_candidate_h_codes)}): {detail_str_2}."
            ),
            matched_candidate=candidate_match_to_dict(best_candidate),
            reference_snapshot=best_comparison.get("reference", {}),
            comparison=best_comparison,
            candidate_evaluations=all_evaluations,
            taxonomy_subtype=cross_candidate_h_codes,
            needs_human_review=False,
        )
    if (
        verdict_label == VerdictLabel.FAKE_REFERENCE
        and best
        and best.issues
        and set(best.issues) <= _only_candidate_missing_issues
        and _core_fields_match
    ):
        all_evaluations.append({
            "agent": "P3_insufficient_evidence",
            "verdict": "POTENTIAL",
            "taxonomy": ["P3"],
            "reason": (
                f"All core fields match but candidate cannot verify "
                f"{sorted(best.issues)}. Insufficient evidence to confirm or deny — "
                f"downgraded to P3 for human review."
            ),
        })
        return CitationVerdict(
            citation_id=citation.citation_id,
            verdict=VerdictLabel.POTENTIAL_REFERENCE,
            evidence_sources=evidence_sources,
            conflicts=_clean_conflicts(best.issues),
            adjudication_reason=(
                f"P3: all core fields match but {sorted(best.issues)} cannot be "
                f"verified because no candidate source provides them. "
                f"Needs human review."
            ),
            matched_candidate=candidate_match_to_dict(best_candidate),
            reference_snapshot=best_comparison.get("reference", {}),
            comparison=best_comparison,
            candidate_evaluations=all_evaluations,
            taxonomy_subtype=["P3"],
            needs_human_review=True,
        )

    all_evaluations.append({
        "agent": "HallucinatedAgent",
        "verdict": downstream.label.value,
        "taxonomy": downstream.taxonomy,
        "reason": downstream.reason,
    })

    return CitationVerdict(
        citation_id=citation.citation_id,
        verdict=verdict_label,
        evidence_sources=evidence_sources,
        conflicts=_clean_conflicts(best.issues if best else []),
        adjudication_reason=f"HallucinatedAgent: {downstream.reason}",
        matched_candidate=candidate_match_to_dict(best_candidate),
        reference_snapshot=best_comparison.get("reference", {}),
        comparison=best_comparison,
        candidate_evaluations=all_evaluations,
        taxonomy_subtype=downstream.taxonomy,
        needs_human_review=True,
    )


def cascading_adjudicate(
    citation: CitationRecord,
    candidates: list[CandidateMatch],
    evidence_traces: list[Any],
    extraction_quality: ExtractionQuality = ExtractionQuality.UNKNOWN,
    valid_agent: ValidAgentProtocol | None = None,
    potential_agent: PotentialAgentProtocol | None = None,
    hallucinated_agent: HallucinatedAgentProtocol | None = None,
    secondary_verifier: Any = None,
    extractor_agent: Any = None,
    source_paper_title: str = "",
    build_comparison_func=None,
    is_direct_valid_func=None,
    url_match_status: str = "",
    url_issues: list[str] | None = None,
    url_type_triggered: str = "",
    tavily_api_key: str = "",
    author_classifier: AuthorVariantClassifierProtocol | None = None,
) -> CitationVerdict:
    """Cascading 3-agent verification (legacy sequential entry point).

    Thin wrapper: runs verify_single_candidate per candidate serially, then
    hands off to cascading_phase2_only. Kept for backward compatibility.
    New streaming entry point lives in verifier.py.
    """
    from .adjudicate import _build_comparison, _is_direct_valid, _reference_snapshot

    build_comp = build_comparison_func or _build_comparison
    is_valid_func = is_direct_valid_func or _is_direct_valid

    evidence_sources = sorted({
        t.connector for t in evidence_traces if hasattr(t, 'connector') and t.error is None
    })

    if url_issues is None:
        url_issues = []

    # No candidates at all
    if not candidates:
        source_errors = [t for t in evidence_traces if hasattr(t, 'error') and t.error]
        if source_errors and len(source_errors) == len(evidence_traces):
            return CitationVerdict(
                citation_id=citation.citation_id,
                verdict=VerdictLabel.INSUFFICIENT_EVIDENCE,
                evidence_sources=evidence_sources,
                conflicts=["no_candidate"],
                adjudication_reason="All sources failed.",
                taxonomy_subtype=[],
                needs_human_review=True,
            )
        return CitationVerdict(
            citation_id=citation.citation_id,
            verdict=VerdictLabel.FAKE_REFERENCE,
            evidence_sources=evidence_sources,
            conflicts=["no_candidate"],
            adjudication_reason="No matching paper found.",
            reference_snapshot=_reference_snapshot(citation),
            taxonomy_subtype=["H1"],
            needs_human_review=True,
        )

    # Phase 1: iterate candidates serially using verify_single_candidate
    all_valid_results: list[ValidAgentResult] = []
    all_evaluations: list[dict[str, Any]] = []

    for candidate in candidates:
        verdict_or_none, eval_dict, vresult = verify_single_candidate(
            citation=citation,
            candidate=candidate,
            sibling_candidates=candidates,
            prior_evaluations=all_evaluations,
            valid_agent=valid_agent,
            extractor_agent=extractor_agent,
            tavily_api_key=tavily_api_key,
            evidence_sources=evidence_sources,
            url_match_status=url_match_status,
            url_issues=url_issues,
            url_type_triggered=url_type_triggered,
            build_comparison_func=build_comp,
            is_direct_valid_func=is_valid_func,
            author_classifier=author_classifier,
        )
        all_evaluations.append(eval_dict)
        if verdict_or_none is not None:
            return verdict_or_none
        if vresult is not None:
            all_valid_results.append(vresult)

    # Phase 2 (cross-candidate coverage + best + 2a + 2b + R4/P3 fallbacks)
    return cascading_phase2_only(
        citation=citation,
        candidates=candidates,
        all_valid_results=all_valid_results,
        all_evaluations=all_evaluations,
        evidence_sources=evidence_sources,
        secondary_verifier=secondary_verifier,
        potential_agent=potential_agent,
        hallucinated_agent=hallucinated_agent,
        url_match_status=url_match_status,
        url_issues=url_issues,
        url_type_triggered=url_type_triggered,
    )


