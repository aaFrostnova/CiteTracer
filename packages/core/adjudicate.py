from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from typing import Protocol


# Test mode: when CITATION_CHECKER_RELAXED_FIELDS=1, the verifier masks out
# all reference fields except title/authors/year/venue at the citation input
# stage. Downstream comparison logic stays untouched: it just sees an empty
# reference for the masked fields, so no mismatch can fire on those.
# In relaxed mode we also (1) accept title/author mismatches as match if
# normalized strings differ by <=_RELAXED_CHAR_TOLERANCE chars, and
# (2) loosen R4 coverage so that title OR authors being matched is enough
# (year/venue still gate normally).
_RELAXED_FIELDS = os.getenv("CITATION_CHECKER_RELAXED_FIELDS", "0").strip() in {"1", "true", "yes", "on"}
_RELAXED_CHAR_TOLERANCE = 3


def _edit_distance(a: str, b: str, cap: int = 8) -> int:
    """Levenshtein distance, capped (returns cap+1 if exceeded). O(len(a)*len(b))."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > cap:
        return cap + 1
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * lb
        row_min = curr[0]
        for j, cb in enumerate(b, 1):
            ins = curr[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            curr[j] = min(ins, dele, sub)
            if curr[j] < row_min:
                row_min = curr[j]
        if row_min > cap:
            return cap + 1
        prev = curr
    return prev[-1]


def _within_relaxed_tolerance(a: str, b: str) -> bool:
    """True if normalized strings differ by <= _RELAXED_CHAR_TOLERANCE chars."""
    return _edit_distance(a or "", b or "", cap=_RELAXED_CHAR_TOLERANCE) <= _RELAXED_CHAR_TOLERANCE

from .models import (
    CandidateMatch,
    CitationRecord,
    CitationVerdict,
    DiscrepancyEvidence,
    EvidenceTrace,
    ExtractionQuality,
    VerdictLabel,
    candidate_match_to_dict,
    canonical_verdict_label,
)
from .normalize import normalize_arxiv_id, normalize_title, normalize_venue, similarity


class LLMResolver(Protocol):
    def review(
        self,
        citation: CitationRecord,
        candidates: list[CandidateMatch],
        conflicts: list[str],
        proposed_verdict: VerdictLabel,
        secondary_evidence: list[DiscrepancyEvidence] | None = None,
    ) -> dict[str, Any] | None:
        """
        Optional LLM second-opinion for ambiguous/fake references.
        Expected keys: label_override, note.
        """


class SecondaryVerifierProtocol(Protocol):
    def gather_evidence(
        self,
        citation: CitationRecord,
        candidate: CandidateMatch,
        field_status: dict[str, dict[str, Any]],
        conflicts: list[str],
    ) -> list[DiscrepancyEvidence]: ...


@dataclass
class AdjudicationPolicy:
    pass


def _attempt_upgrade(
    secondary_evidence: list[DiscrepancyEvidence],
) -> tuple[VerdictLabel, str]:
    """Upgrade from FAKE to POTENTIAL only if ALL discrepancies have positive evidence."""
    if not secondary_evidence:
        return VerdictLabel.FAKE_REFERENCE, "No secondary evidence gathered."

    unexplained = [ev for ev in secondary_evidence if not ev.evidence_found]
    if unexplained:
        fields = ", ".join(ev.field for ev in unexplained)
        details = "; ".join(f"{ev.field}: {ev.explanation}" for ev in unexplained)
        return (
            VerdictLabel.FAKE_REFERENCE,
            f"Discrepancies in [{fields}] not explained by evidence. {details}",
        )

    explained = "; ".join(f"{ev.field}: {ev.explanation}" for ev in secondary_evidence)
    return (
        VerdictLabel.POTENTIAL_REFERENCE,
        f"All discrepancies explained by positive evidence. {explained}",
    )


def _classify_taxonomy(
    verdict: VerdictLabel,
    field_status: dict[str, dict[str, Any]],
    candidate_connector: str,
    has_et_al: bool,
    has_candidates: bool,
) -> list[str]:
    """Classify the verdict into one or more taxonomy subtypes (R1-R4, P1-P3,
    H1-H6). Returns a list because a single FAKE verdict can be backed by
    multiple field errors at once (e.g. an H4 mutation that also breaks venue
    is both H4 and H3). The list is ordered by code id so callers and
    evaluators get a stable canonical ordering.
    """
    if verdict == VerdictLabel.VALID:
        if has_et_al:
            return ["R3"]
        return ["R1"]

    if verdict == VerdictLabel.POTENTIAL_REFERENCE:
        author_status = field_status.get("authors", {}).get("status", "")
        url_status = field_status.get("url", {}).get("status", "")
        if (candidate_connector in {"google_search", "web_search"}
                and url_status == "mismatch"):
            return ["P2"]
        if author_status == "partial_overlap":
            return ["P1"]
        return ["P1"]

    if verdict == VerdictLabel.FAKE_REFERENCE:
        if not has_candidates:
            return ["H1"]
        title_status = field_status.get("title", {}).get("status", "")
        author_status = field_status.get("authors", {}).get("status", "")
        venue_status = field_status.get("venue", {}).get("status", "")
        year_status = field_status.get("year", {}).get("status", "")
        doi_status = field_status.get("doi", {}).get("status", "")
        pages_status = field_status.get("pages", {}).get("status", "")
        volume_status = field_status.get("volume", {}).get("status", "")
        publisher_status = field_status.get("publisher", {}).get("status", "")
        location_status = field_status.get("location", {}).get("status", "")

        codes: list[str] = []
        if title_status == "mismatch":
            codes.append("H1")
        if author_status in ("count_mismatch", "mismatch"):
            codes.append("H2")
        if venue_status == "mismatch":
            codes.append("H3")
        if year_status == "mismatch":
            codes.append("H4")
        if doi_status == "mismatch":
            codes.append("H5")
        if (pages_status == "mismatch"
                or volume_status == "mismatch"
                or publisher_status == "mismatch"
                or location_status == "mismatch"):
            codes.append("H6")

        return codes or ["H1"]

    return []


def _norm_text(value: Any) -> str:
    text = " ".join(str(value or "").strip().lower().split())
    return text.strip(" .,:;!?\"'()[]{}")


def _norm_title(value: Any) -> str:
    return normalize_title(str(value or ""))


def _norm_identifier(value: Any) -> str:
    return _norm_text(value).rstrip(".,;")


def _norm_pages(value: Any) -> str:
    """Normalize page ranges: '1877--1901' / '1877—1901' / 'pp. 1877-1901' → '1877-1901'.

    Also handles ACM article-number format: '766:1-766:28' → '1-28'.
    The pattern is <article_num>:<start>-<article_num>:<end>.
    """
    import re as _re
    text = str(value or "").lower().strip()
    if not text:
        return ""
    text = text.replace("pp.", "").replace("pp", "").strip()
    # Unify all dash variants to ASCII hyphen
    for sep in ("\u2013", "\u2014", "\u2212", "--", "—", "–"):
        text = text.replace(sep, "-")
    text = " ".join(text.split())
    # Collapse whitespace around the hyphen so "12 - 22" ≡ "12-22".
    text = _re.sub(r"\s*-\s*", "-", text)
    text = text.strip(" .,:;")
    # Biomedical e-locators: sources disagree on the "e" prefix for the same
    # article ("e123158" in CrossRef/PubMed vs "123158" in Europe PMC). Drop a
    # leading e before an all-digit locator so the two forms compare equal.
    if _re.fullmatch(r"e\d+", text):
        text = text[1:]
    # ACM article-number format: "766:1-766:28" → "1-28"
    # Pattern: <num>:<start> - <num>:<end>  (same article number on both sides)
    m = _re.match(r"^(\d+):(\d+)\s*-\s*(\d+):(\d+)$", text)
    if m and m.group(1) == m.group(3):
        text = f"{m.group(2)}-{m.group(4)}"
    # Also handle single-sided: "29:1" → "1" (JMLR style, less common in ranges)
    m2 = _re.match(r"^\d+:(\d+)\s*-\s*\d+:(\d+)$", text)
    if m2:
        text = f"{m2.group(1)}-{m2.group(2)}"
    # Collapse degenerate single-page ranges so "139-139" matches "139".
    # Single-page citations are sometimes rendered as a same-start/same-end
    # range by BibTeX styles ("pages={139--139}") and the candidate side
    # usually stores the bare page number. Without this collapse we get a
    # pages_mismatch on what is actually the same page.
    m3 = _re.match(r"^(\d+[a-z]?)\s*-\s*(\d+[a-z]?)$", text)
    if m3 and m3.group(1) == m3.group(2):
        text = m3.group(1)
    return text


_ROMAN_NUMERAL_MAP = {
    "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8,
    "ix": 9, "x": 10, "xi": 11, "xii": 12, "xiii": 13, "xiv": 14, "xv": 15,
    "xvi": 16, "xvii": 17, "xviii": 18, "xix": 19, "xx": 20,
}


def _norm_volume(value: Any) -> str:
    """Normalize volume: handle roman numerals and 'vol.' prefix.

    Examples: 'II' → '2', 'vol. 5' → '5', 'Volume 12' → '12'
    """
    text = str(value or "").strip().lower()
    if not text:
        return ""
    import re as _re
    # Discard arXiv-id-shaped junk: 'abs/2209.01174' is an arXiv identifier
    # misrouted into the volume field by some citation parsers, not a real
    # volume number.
    if _re.match(r"^abs/\d", text):
        return ""
    # Strip common prefixes
    text = _re.sub(r"^(vol(?:ume)?\.?\s*)", "", text).strip()
    text = text.strip(" .,:;")
    # Convert roman numeral to arabic if applicable
    if text in _ROMAN_NUMERAL_MAP:
        return str(_ROMAN_NUMERAL_MAP[text])
    return text


def _norm_arxiv_id(value: Any) -> str:
    return normalize_arxiv_id(str(value or ""))


def _norm_venue(value: Any) -> str:
    return normalize_venue(str(value or ""))


def _norm_url(value: Any) -> str:
    return _norm_identifier(value).rstrip("/")


def _to_author_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, list):
        return len(value) == 0
    return False


def _compare_scalar(reference: Any, candidate: Any, normalizer) -> dict[str, Any]:
    if _is_missing(reference) and _is_missing(candidate):
        status = "both_missing"
    elif _is_missing(reference):
        # Reference made no claim about this field, so there is no
        # disagreement to flag. Treat as match. (Was "reference_missing"
        # historically, but downstream LLM agents kept flagging H3/H4/H5/H6
        # off this status against the field_status rules; the masked-field
        # extreme/relaxed modes also rely on this.) candidate_missing is
        # still a real signal — citation makes a claim, candidate cannot
        # verify — so that path stays.
        status = "match"
    elif _is_missing(candidate):
        status = "candidate_missing"
    else:
        status = "match" if normalizer(reference) == normalizer(candidate) else "mismatch"
    return {
        "status": status,
        "reference": reference,
        "candidate": candidate,
    }


def _compare_venue(reference: Any, candidate: Any) -> dict[str, Any]:
    """Venue comparison with containment, abbreviation, and similarity fallback."""
    from .normalize import venues_equivalent_heuristic

    if _is_missing(reference) and _is_missing(candidate):
        status = "both_missing"
    elif _is_missing(reference):
        # Reference made no claim → no disagreement to flag, treat as match.
        status = "match"
    elif _is_missing(candidate):
        status = "candidate_missing"
    else:
        ref_norm = _norm_venue(reference)
        cand_norm = _norm_venue(candidate)
        if ref_norm == cand_norm:
            status = "match"
        elif ref_norm and cand_norm and (cand_norm in ref_norm or ref_norm in cand_norm):
            status = "match"
        elif venues_equivalent_heuristic(str(reference), str(candidate)):
            # General acronym / word-truncation heuristic
            # (e.g. "JMLR" ≡ "Journal of Machine Learning Research",
            #       "J. Mach. Learn. Res." ≡ "Journal of Machine Learning Research")
            status = "match"
        elif ref_norm and cand_norm and similarity(ref_norm, cand_norm) >= 0.7:
            status = "match"
        else:
            status = "mismatch"
    return {
        "status": status,
        "reference": reference,
        "candidate": candidate,
    }


def _compare_year(reference: Any, candidate: Any, candidate_version_years: list[Any] | None = None) -> dict[str, Any]:
    version_years: list[int] = []
    for item in candidate_version_years or []:
        try:
            year_value = int(str(item).strip())
        except (TypeError, ValueError):
            continue
        if year_value not in version_years:
            version_years.append(year_value)

    if _is_missing(reference) and _is_missing(candidate) and not version_years:
        status = "both_missing"
    elif _is_missing(reference):
        # Reference made no year claim → no disagreement, treat as match.
        status = "match"
    elif _is_missing(candidate) and not version_years:
        status = "candidate_missing"
    else:
        reference_text = str(reference)
        candidate_matches = str(candidate) == reference_text if not _is_missing(candidate) else False
        version_matches = reference_text in {str(year) for year in version_years}
        status = "match" if candidate_matches or version_matches else "mismatch"

    return {
        "status": status,
        "reference": reference,
        "candidate": candidate,
        "candidate_version_years": version_years,
    }


_ET_AL_MARKERS = {"others", "et al", "et al.", "and others", "and others."}


def _is_et_al_marker(name: str) -> bool:
    return _norm_text(name) in _ET_AL_MARKERS


def _compare_authors(reference: list[str], candidate: list[str]) -> dict[str, Any]:
    """Author comparison — strict byte-identical (after whitespace + case
    normalization). Any non-exact difference (typos, initials, reorder,
    nickname, dropped middle name, count mismatch) is reported as `mismatch`.

    Exception: et al. markers in reference match if every listed ref author
    matches the candidate's corresponding prefix positionally (for R3 support).
    """
    ref = _to_author_list(reference)
    cand = _to_author_list(candidate)

    # Detect "et al." / "Others" in reference and filter them out
    has_et_al = any(_is_et_al_marker(name) for name in ref)
    ref_actual = [name for name in ref if not _is_et_al_marker(name)]

    import re as _re
    def _norm_author_name(name: str) -> str:
        """Normalize author name: lowercase + strip DBLP disambiguation suffix
        + collapse whitespace. Does NOT remove dots or do token-level matching —
        keeps the full string for strict equality comparison."""
        text = str(name or "").strip().lower()
        # Strip DBLP-style disambiguation suffixes (e.g. "mark chen 0003" → "mark chen")
        text = _re.sub(r"\s+\d{4,}$", "", text).strip()
        return " ".join(text.split())  # collapse whitespace

    ref_norm = [_norm_author_name(name) for name in ref_actual]
    cand_norm = [_norm_author_name(name) for name in cand]

    if not ref_norm and not cand_norm:
        status = "both_missing"
    elif not ref_norm:
        # Reference made no author claim → no disagreement, treat as match.
        status = "match"
    elif not cand_norm:
        status = "candidate_missing"
    elif ref_norm == cand_norm:
        # Strict byte-identical position-wise match
        status = "match"
    elif (has_et_al
          and len(ref_norm) <= len(cand_norm)
          and cand_norm[: len(ref_norm)] == ref_norm):
        # R3 special case: reference uses et al. and its listed authors byte-match
        # the prefix of the candidate's author list.
        status = "match"
    else:
        status = "mismatch"

    overlap = sum(1 for i in range(min(len(ref_norm), len(cand_norm))) if ref_norm[i] == cand_norm[i])

    return {
        "status": status,
        "reference": ref,
        "candidate": cand,
        "overlap_count": overlap,
        "reference_count": len(ref_actual),
        "candidate_count": len(cand),
        "has_et_al": has_et_al,
    }


def _reference_snapshot(citation: CitationRecord) -> dict[str, Any]:
    return {
        "title": citation.title,
        "authors": list(citation.authors),
        "venue": citation.venue,
        "year": citation.year,
        "volume": citation.volume,
        "pages": citation.pages,
        "publisher": citation.publisher,
        "location": citation.location,
        "doi": citation.doi,
        "arxiv_id": citation.arxiv_id,
        "url": citation.url,
    }


def _candidate_snapshot(candidate: CandidateMatch | None) -> dict[str, Any]:
    if candidate is None:
        return {}
    return {
        "connector": candidate.connector,
        "title": candidate.title,
        "authors": list(candidate.authors),
        "venue": candidate.venue,
        "year": candidate.year,
        "volume": candidate.volume,
        "pages": candidate.pages,
        "publisher": candidate.publisher,
        "doi": candidate.doi,
        "arxiv_id": candidate.arxiv_id,
        "url": candidate.url,
    }


def _build_comparison(
    citation: CitationRecord,
    candidate: CandidateMatch | None,
    conflicts: list[str],
) -> dict[str, Any]:
    reference = _reference_snapshot(citation)
    candidate_data = _candidate_snapshot(candidate)
    candidate_version_years = []
    if candidate is not None and isinstance(candidate.raw_record, dict):
        raw_years = candidate.raw_record.get("version_years")
        if isinstance(raw_years, list):
            candidate_version_years = raw_years
    # Prefer candidate.* (populated by build_candidate_match from connector records),
    # fall back to raw_record for fields some connectors may only stash there.
    cand_raw = candidate.raw_record if candidate is not None and isinstance(candidate.raw_record, dict) else {}

    def _from_candidate(field_name: str) -> str:
        value = ""
        if candidate is not None:
            value = str(getattr(candidate, field_name, "") or "")
        if not value:
            value = str(cand_raw.get(field_name, "") or "")
        return value

    candidate_core = {
        "title": candidate_data.get("title", ""),
        "authors": candidate_data.get("authors", []),
        "venue": candidate_data.get("venue", ""),
        "year": candidate_data.get("year"),
        "volume": _from_candidate("volume"),
        "pages": _from_candidate("pages"),
        "publisher": _from_candidate("publisher"),
        "location": str(cand_raw.get("location", "") or ""),
        "doi": candidate_data.get("doi", ""),
        "arxiv_id": candidate_data.get("arxiv_id", ""),
        "url": candidate_data.get("url", ""),
    }

    # arxiv/CoRR records carry noisy peripheral values that reflect the preprint
    # version, not the published conference/journal version. Example: arxiv's
    # publisher is always "arXiv", volume looks like "abs/2305.13617". Comparing
    # these against a citation's real journal publisher / volume produces
    # spurious `mismatch` signals. Null them out so they read as candidate_missing
    # instead (their NON-peripheral fields — title/authors/year — are still valid).
    _peri_venue = (candidate_core.get("venue") or "").strip().lower()
    _peri_pub = (candidate_core.get("publisher") or "").strip().lower()
    _is_arxiv_record = (
        _peri_venue in ("arxiv", "corr", "arxiv preprint")
        or _peri_pub in ("arxiv", "corr", "arxiv.org", "arxiv preprint")
    )
    if _is_arxiv_record:
        candidate_core["volume"] = ""
        candidate_core["pages"] = ""
        candidate_core["publisher"] = ""
        candidate_core["location"] = ""

    field_status = {
        "title": _compare_scalar(reference["title"], candidate_core["title"], _norm_title),
        "authors": _compare_authors(reference["authors"], candidate_core["authors"]),
        "venue": _compare_scalar(reference["venue"], candidate_core["venue"], _norm_title),
        "year": _compare_year(reference["year"], candidate_core["year"], candidate_version_years),
        "volume": _compare_scalar(reference.get("volume", ""), candidate_core["volume"], _norm_volume),
        "pages": _compare_scalar(reference.get("pages", ""), candidate_core["pages"], _norm_pages),
        "publisher": _compare_scalar(reference.get("publisher", ""), candidate_core["publisher"], _norm_text),
        "location": _compare_scalar(reference.get("location", ""), candidate_core["location"], _norm_text),
        "doi": _compare_scalar(reference["doi"], candidate_core["doi"], _norm_identifier),
        "arxiv_id": _compare_scalar(reference["arxiv_id"], candidate_core["arxiv_id"], _norm_arxiv_id),
        "url": _compare_scalar(reference["url"], candidate_core["url"], _norm_url),
    }

    # Relaxed mode: if title or authors look "almost equal" (<=3 char edit
    # distance after normalization), upgrade mismatch → match. Only applies
    # to title/authors; year/venue still gate strictly.
    if _RELAXED_FIELDS:
        t_info = field_status["title"]
        if t_info.get("status") == "mismatch":
            ref_t = _norm_title(t_info.get("reference"))
            cand_t = _norm_title(t_info.get("candidate"))
            if ref_t and cand_t and _within_relaxed_tolerance(ref_t, cand_t):
                t_info["status"] = "match"
                t_info["relaxed_match"] = True
        a_info = field_status["authors"]
        if a_info.get("status") == "mismatch":
            ref_a = _to_author_list(a_info.get("reference"))
            cand_a = _to_author_list(a_info.get("candidate"))
            ref_actual = [n for n in ref_a if not _is_et_al_marker(n)]
            if ref_actual and cand_a and len(ref_actual) <= len(cand_a):
                # Position-wise: every ref author within tolerance of corresponding cand author.
                ok = all(
                    _within_relaxed_tolerance(
                        _norm_text(ref_actual[i]), _norm_text(cand_a[i])
                    )
                    for i in range(len(ref_actual))
                )
                if ok:
                    a_info["status"] = "match"
                    a_info["relaxed_match"] = True

    # Recompute conflicts from field_status (authoritative) instead of sticking
    # with the stale quick-match conflicts from build_candidate_match().
    # The quick-match phase uses simple string overlap which can differ from
    # the token-aware _compare_authors / _compare_venue / etc.
    recomputed_conflicts: list[str] = []
    _mismatch_statuses = {"mismatch"}
    _field_to_conflict = {
        "title": "title_mismatch",
        "authors": "author_mismatch",
        "venue": "venue_mismatch",
        "year": "year_mismatch",
        "doi": "doi_mismatch",
        "arxiv_id": "arxiv_id_mismatch",
        "volume": "volume_mismatch",
        "pages": "pages_mismatch",
        "publisher": "publisher_mismatch",
    }
    for fname, conflict_name in _field_to_conflict.items():
        status = field_status.get(fname, {}).get("status", "")
        if status in _mismatch_statuses:
            recomputed_conflicts.append(conflict_name)
    # authors has extra statuses
    author_status = field_status.get("authors", {}).get("status", "")
    if author_status == "reordered_match" and "author_reordered" not in recomputed_conflicts:
        recomputed_conflicts.append("author_reordered")
    elif author_status == "count_mismatch" and "author_count_mismatch" not in recomputed_conflicts:
        recomputed_conflicts.append("author_count_mismatch")

    summary_pairs = [f"{name}:{meta['status']}" for name, meta in field_status.items()]
    return {
        "connector": candidate_data.get("connector", ""),
        "conflicts": recomputed_conflicts,
        "reference": reference,
        "candidate": candidate_core,
        "field_status": field_status,
        "summary": ", ".join(summary_pairs),
    }


def _status_in(meta: dict[str, Any], accepted: set[str]) -> bool:
    return str(meta.get("status", "") or "") in accepted


def _has_soft_discrepancies(field_status: dict[str, dict[str, Any]], conflicts: list[str]) -> bool:
    title_status = field_status["title"]["status"]
    year_status = field_status["year"]["status"]
    if title_status == "mismatch" or year_status == "mismatch":
        return False

    review_statuses = {
        "mismatch",
        "partial_overlap",
        "reordered_match",
        "candidate_missing",
        "reference_missing",
    }
    for field_name in ("authors", "venue", "doi", "pages", "publisher", "location", "volume"):
        if field_name in field_status and _status_in(field_status[field_name], review_statuses):
            return True

    review_conflicts = {
        "author_mismatch",
        "venue_mismatch",
        "doi_mismatch",
        "arxiv_id_mismatch",
    }
    return any(conflict in review_conflicts for conflict in conflicts)


def _is_direct_valid(field_status: dict[str, dict[str, Any]], conflicts: list[str]) -> bool:
    if conflicts:
        return False
    # Core fields: title + authors + year must match
    if field_status["title"]["status"] != "match":
        return False
    if field_status["year"]["status"] != "match":
        return False
    if field_status["authors"]["status"] != "match":
        return False
    # All optional fields: if reference has a value, candidate must also have it and match.
    # - both_missing / reference_missing → reference didn't provide it, OK to skip
    # - candidate_missing → reference HAS a value but candidate doesn't → NOT valid
    for f in ("venue", "doi", "pages", "publisher", "location", "volume"):
        if f in field_status:
            if not _status_in(field_status[f], {"match", "both_missing", "reference_missing"}):
                return False
    return True


def _candidate_has_no_structured_data(field_status: dict[str, dict[str, Any]]) -> bool:
    """True when candidate returned no useful structured fields (e.g. raw web_search snippet)."""
    return all(
        field_status.get(f, {}).get("status") == "candidate_missing"
        for f in ("title", "authors", "year")
    )


# Core identity fields. If a candidate is missing ANY of these while the
# reference provides one, the candidate cannot plausibly be the cited paper:
# absence of the title, authors, venue, year, or DOI is a hard FAKE signal,
# not a soft discrepancy. Peripheral fields (volume, pages, publisher,
# location) are kept out because they correspond to the P3 unverifiable-
# peripheral case. arxiv_id is also excluded because most non-arXiv
# connectors (DBLP, Crossref, etc.) legitimately do not carry it, so
# treating its absence as FAKE would over-penalise real-paper matches from
# those sources.
_CORE_IDENTITY_FIELDS = ("title", "authors", "venue", "year", "doi")


def _core_fields_candidate_missing(field_status: dict[str, dict[str, Any]]) -> list[str]:
    """Return the subset of core identity fields where the reference has a
    value but the candidate does not."""
    return [
        f for f in _CORE_IDENTITY_FIELDS
        if field_status.get(f, {}).get("status") == "candidate_missing"
    ]


def _evaluate_candidate(
    citation: CitationRecord,
    candidate: CandidateMatch,
    policy: AdjudicationPolicy,
) -> tuple[VerdictLabel, str, list[str], dict[str, Any], bool, bool]:
    """Returns (verdict, reason, conflicts, comparison, should_run_llm_recheck, needs_secondary_verification)."""
    conflicts = list(candidate.conflicts)
    comparison = _build_comparison(citation, candidate, conflicts)
    field_status = comparison["field_status"]
    should_run_llm_recheck = False
    needs_secondary_verification = False

    if _is_direct_valid(field_status, conflicts):
        verdict = VerdictLabel.VALID
        reason = "All core metadata fields match, so the citation is treated as a direct valid match."
    elif field_status["title"]["status"] == "mismatch" or field_status["year"]["status"] == "mismatch":
        verdict = VerdictLabel.FAKE_REFERENCE
        reason = "Title or year mismatches the current candidate."
        should_run_llm_recheck = True
    elif field_status["authors"].get("status") == "count_mismatch":
        ref_count = field_status["authors"].get("reference_count", "?")
        cand_count = field_status["authors"].get("candidate_count", "?")
        verdict = VerdictLabel.FAKE_REFERENCE
        reason = (
            f"Author count mismatch without et al.: citation lists {ref_count} "
            f"specific authors but candidate has {cand_count} (H3a)."
        )
        should_run_llm_recheck = True
    elif (core_missing := _core_fields_candidate_missing(field_status)):
        verdict = VerdictLabel.FAKE_REFERENCE
        reason = (
            f"Candidate is missing core identity field(s) the reference provides "
            f"({', '.join(core_missing)}); the matched record cannot be the cited paper."
        )
        should_run_llm_recheck = True
    elif _has_soft_discrepancies(field_status, conflicts):
        verdict = VerdictLabel.POTENTIAL_REFERENCE
        reason = "Only soft metadata discrepancies were found; mark as potential unless stronger evidence disproves the match."
        should_run_llm_recheck = True
        needs_secondary_verification = True
    else:
        verdict = VerdictLabel.FAKE_REFERENCE
        reason = "No positive evidence of match."
        should_run_llm_recheck = True

    return verdict, reason, conflicts, comparison, should_run_llm_recheck, needs_secondary_verification


def adjudicate(
    citation: CitationRecord,
    candidates: list[CandidateMatch],
    evidence: list[EvidenceTrace],
    extraction_quality: ExtractionQuality = ExtractionQuality.UNKNOWN,
    llm_resolver: LLMResolver | None = None,
    policy: AdjudicationPolicy | None = None,
    secondary_verifier: SecondaryVerifierProtocol | None = None,
) -> CitationVerdict:
    policy = policy or AdjudicationPolicy()
    evidence_sources = sorted({trace.connector for trace in evidence if trace.error is None})
    source_errors = [trace for trace in evidence if trace.error]

    if not candidates:
        if source_errors and len(source_errors) == len(evidence):
            verdict_label = VerdictLabel.INSUFFICIENT_EVIDENCE
            reason = "All sources failed or timed out, unable to establish ground truth."
        else:
            verdict_label = VerdictLabel.INSUFFICIENT_EVIDENCE
            reason = "No matching paper was found in queried sources, but absence alone is insufficient to prove fabrication."
        return CitationVerdict(
            citation_id=citation.citation_id,
            verdict=verdict_label,
            evidence_sources=evidence_sources,
            conflicts=["no_candidate"],
            adjudication_reason=reason,
            matched_candidate=None,
            reference_snapshot=_reference_snapshot(citation),
            comparison=_build_comparison(citation, None, ["no_candidate"]),
            candidate_evaluations=[],
            taxonomy_subtype=["H1"] if verdict_label == VerdictLabel.FAKE_REFERENCE else [],  # H1: no candidates found
            llm_recheck_reason="",
            needs_human_review=True,
            extraction_quality=extraction_quality,
        )

    preferred_fallback: CitationVerdict | None = None
    candidate_evaluations: list[dict[str, Any]] = []
    # Track if any structured (non-web-search) candidate found a hard mismatch.
    # If so, web_search LLM cannot override to VALID.
    _WEB_CONNECTORS = {"google_search", "web_search"}
    has_structured_hard_mismatch = False

    for candidate in candidates:

        verdict, reason, conflicts, comparison, should_run_llm_recheck, needs_secondary = _evaluate_candidate(
            citation=citation,
            candidate=candidate,
            policy=policy,
        )

        # When candidate has no structured data (web_search with empty fields),
        # skip secondary verification (nothing to compare) and go straight to LLM.
        # LLM verdict is capped at POTENTIAL_REFERENCE for such candidates.
        no_structured = _candidate_has_no_structured_data(comparison["field_status"])

        sec_evidence: list[DiscrepancyEvidence] = []
        if needs_secondary and secondary_verifier and not no_structured:
            try:
                sec_evidence = secondary_verifier.gather_evidence(
                    citation, candidate, comparison["field_status"], conflicts,
                )
            except Exception:
                sec_evidence = []
            upgraded_verdict, upgrade_reason = _attempt_upgrade(sec_evidence)
            verdict = upgraded_verdict
            reason = upgrade_reason

        # Track if secondary verification upgraded to POTENTIAL (e.g. author name variant)
        was_upgraded_to_potential = (
            verdict == VerdictLabel.POTENTIAL_REFERENCE and sec_evidence
        )

        # LLM resolver with secondary evidence context
        llm_note = ""
        llm_taxonomy = ""
        if llm_resolver and should_run_llm_recheck:
            try:
                review = llm_resolver.review(
                    citation, [candidate], conflicts, verdict,
                    secondary_evidence=sec_evidence or None,
                ) or {}
            except Exception as exc:  # noqa: BLE001 - reviewer failure should not crash adjudication
                review = {"note": f"LLM review failed: {exc}"}
            override = review.get("label_override")
            if override:
                overridden = canonical_verdict_label(override)
                fs = comparison["field_status"]
                is_structured = candidate.connector not in _WEB_CONNECTORS

                # Guard 1: structured source with title mismatch or author count_mismatch
                # cannot be overridden to VALID by LLM.
                has_hard_field_mismatch = (
                    fs.get("title", {}).get("status") == "mismatch"
                    or fs.get("authors", {}).get("status") == "count_mismatch"
                )
                if overridden == VerdictLabel.VALID and is_structured and has_hard_field_mismatch:
                    pass  # Keep original verdict

                # Guard 2: secondary verification upgraded to POTENTIAL (e.g. author name variant)
                # LLM cannot override back to VALID — the discrepancy was real but explainable.
                elif overridden == VerdictLabel.VALID and was_upgraded_to_potential:
                    pass  # Keep POTENTIAL

                else:
                    verdict = overridden
            llm_taxonomy = str(review.get("taxonomy", "") or "").strip()
            llm_note = str(review.get("note", "")).strip()

        if llm_note:
            reason = f"{reason} LLM disambiguation: {llm_note}"

        candidate_evaluations.append(
            {
                "connector": candidate.connector,
                "candidate": candidate_match_to_dict(candidate),
                "verdict": canonical_verdict_label(verdict).value,
                "reason": reason,
                "llm_recheck_reason": llm_note,
                "comparison": comparison,
                "secondary_evidence": [
                    {"field": ev.field, "type": ev.discrepancy_type,
                     "found": ev.evidence_found, "explanation": ev.explanation}
                    for ev in sec_evidence
                ],
            }
        )

        # Prefer LLM taxonomy when present; otherwise fall back to the
        # rule-based classifier, which now returns ALL detected codes (e.g.
        # ["H3", "H4"] when both venue and year are wrong) instead of a
        # single priority winner.
        if llm_note and llm_taxonomy:
            # llm_taxonomy is a comma- or space-separated string from the
            # LLM (e.g. "H3" or "H3,H4"); split into a list.
            taxonomy_list = [
                t.strip() for t in llm_taxonomy.replace(",", " ").split()
                if t.strip()
            ]
        else:
            taxonomy_list = _classify_taxonomy(
                verdict=verdict,
                field_status=comparison["field_status"],
                candidate_connector=candidate.connector,
                has_et_al=comparison["field_status"].get("authors", {}).get("has_et_al", False),
                has_candidates=True,
            )

        current = CitationVerdict(
            citation_id=citation.citation_id,
            verdict=verdict,
            evidence_sources=evidence_sources,
            conflicts=conflicts,
            adjudication_reason=reason,
            matched_candidate=candidate_match_to_dict(candidate),
            reference_snapshot=_reference_snapshot(citation),
            comparison=comparison,
            candidate_evaluations=list(candidate_evaluations),
            llm_recheck_reason=llm_note,
            taxonomy_subtype=taxonomy_list,
            needs_human_review=(verdict != VerdictLabel.VALID),
            extraction_quality=extraction_quality,
            secondary_evidence=sec_evidence,
        )

        # Track hard mismatches from structured sources (title mismatch or author count mismatch)
        if (verdict == VerdictLabel.FAKE_REFERENCE
                and candidate.connector not in _WEB_CONNECTORS):
            fs = comparison["field_status"]
            # Structured source found the paper but with field mismatches.
            # This blocks web_search from overriding to VALID (at most POTENTIAL).
            title_ok = fs.get("title", {}).get("status") in ("match", "candidate_missing", "both_missing")
            if title_ok:
                # Title matches but other fields don't → paper exists but citation has errors
                has_structured_hard_mismatch = True

        if verdict == VerdictLabel.VALID:
            # If a structured source already found title mismatch or author count mismatch,
            # web_search VALID cannot override — downgrade to FAKE.
            if has_structured_hard_mismatch and candidate.connector in _WEB_CONNECTORS:
                current = CitationVerdict(
                    citation_id=citation.citation_id,
                    verdict=VerdictLabel.POTENTIAL_REFERENCE,
                    evidence_sources=evidence_sources,
                    conflicts=["structured_mismatch_caps_web_search"],
                    adjudication_reason=(
                        f"Structured database found the paper but with field discrepancies. "
                        f"Web search confirms existence but verdict capped at POTENTIAL. {reason}"
                    ),
                    matched_candidate=candidate_match_to_dict(candidate),
                    reference_snapshot=_reference_snapshot(citation),
                    comparison=comparison,
                    candidate_evaluations=list(candidate_evaluations),
                    llm_recheck_reason=llm_note,
                    needs_human_review=True,
                    extraction_quality=extraction_quality,
                    secondary_evidence=sec_evidence,
                )
            else:
                return current

        if preferred_fallback is None:
            preferred_fallback = current
            continue

        priority = {
            VerdictLabel.POTENTIAL_REFERENCE: 3,
            VerdictLabel.FAKE_REFERENCE: 2,
            VerdictLabel.INSUFFICIENT_EVIDENCE: 1,
        }
        if priority.get(verdict, 0) > priority.get(preferred_fallback.verdict, 0):
            preferred_fallback = current

    if preferred_fallback is not None:
        return preferred_fallback

    best = candidates[0]
    return CitationVerdict(
        citation_id=citation.citation_id,
        verdict=VerdictLabel.FAKE_REFERENCE,
        evidence_sources=evidence_sources,
        conflicts=list(best.conflicts),
        adjudication_reason="No candidate provided sufficient positive evidence.",
        matched_candidate=candidate_match_to_dict(best),
        reference_snapshot=_reference_snapshot(citation),
        comparison=_build_comparison(citation, best, list(best.conflicts)),
        candidate_evaluations=list(candidate_evaluations),
        llm_recheck_reason="",
        needs_human_review=True,
        extraction_quality=extraction_quality,
    )
