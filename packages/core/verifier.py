from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from typing import Any

from packages.connectors.base import ConnectorOrchestrator, ConnectorResult, RequestPolicy

from .adjudicate import AdjudicationPolicy, LLMResolver, SecondaryVerifierProtocol, adjudicate, canonical_verdict_label
from .agents import ExistenceJudge, StructuredJudge, multi_agent_adjudicate
from .cascading_agents import (
    ValidAgentProtocol, PotentialAgentProtocol, HallucinatedAgentProtocol,
    cascading_adjudicate,
    cascading_phase2_only,
    verify_single_candidate,
    phase0_url_check,
    phase0_decide_verdict,
    downgrade_non_academic_fake,
    is_non_academic_citation,
)
from .matching import CandidateMatch, build_candidate_match, collect_candidates
from .models import CheckReport, CitationRecord, CitationVerdict, EvidenceTrace, ExtractionQuality, VerdictLabel
from .normalize import extract_identifier, normalize_title, similarity
from .report import build_summary, compute_risk_score

# Connectors treated as "web search" (non-structured, need ExtractorAgent).
# These are deferred to Phase 1b after structured connectors exhaust.
_WEB_CONNECTOR_NAMES = {"web_search", "google_search"}

# Threshold for keeping a web_search record as a candidate. Records whose
# title is wholly different from the citation title are discarded so the
# verdict cannot escalate to FAKE based on noise from Tavily / similar
# keyword-based hits. Override via env var when needed.
import os as _os_filter
_WEBSEARCH_TITLE_MIN_SIM = float(
    _os_filter.getenv("CITATION_CHECKER_WEBSEARCH_TITLE_THRESHOLD", "0.8")
)


def _web_search_record_titles_match(citation_title: str, candidate_title: str) -> bool:
    """Decide whether a web_search candidate is plausibly the same paper.

    Pass when:
      - normalized titles are equal
      - one normalized title is a substring of the other (truncation case)
      - SequenceMatcher similarity >= threshold

    Reject (extra strict) when both titles contain a colon and the part
    BEFORE the colon (the paper's distinctive name, e.g. "Trace2TAP" vs
    "AutoTap") is wholly different. This catches the "shared keyword phrase
    but different paper" failure mode that bare similarity misses.
    """
    if not candidate_title or not citation_title:
        return True  # cannot decide; let downstream judge it.
    nt_cite = normalize_title(citation_title)
    nt_cand = normalize_title(candidate_title)
    if not nt_cite or not nt_cand:
        return True
    if nt_cite == nt_cand:
        return True
    if nt_cite in nt_cand or nt_cand in nt_cite:
        return True
    # Colon-prefix paper-name guard: if both titles have a colon, require
    # the lead names to share enough surface form. "Trace2TAP" vs "AutoTap"
    # have similarity ≈ 0.4, which fails this guard regardless of how much
    # of the subtitle matches.
    if ":" in citation_title and ":" in candidate_title:
        cite_lead = citation_title.split(":", 1)[0].strip()
        cand_lead = candidate_title.split(":", 1)[0].strip()
        if cite_lead and cand_lead:
            lead_sim = similarity(cite_lead, cand_lead)
            if lead_sim < 0.55:
                return False
            # Lead name is essentially identical (same distinctive paper name);
            # subtitle drift is normal across DBs/citations, keep regardless of
            # the global similarity threshold.
            if lead_sim >= 0.9:
                return True
    return similarity(citation_title, candidate_title) >= _WEBSEARCH_TITLE_MIN_SIM


def _filter_websearch_records_by_title(
    citation: CitationRecord, records: list[dict]
) -> list[dict]:
    """Drop web_search records whose title clearly does not match the
    citation. Returns the kept subset; preserves order."""
    cite_title = (citation.title or "").strip()
    if not cite_title:
        return list(records)
    return [
        r for r in records
        if _web_search_record_titles_match(cite_title, str(r.get("title") or ""))
    ]

# Connectors demoted out of the parallel structured tier (currently empty —
# arxiv used to live here because of its global lock + min_interval_s = 3.0,
# but is now back in the parallel pool).
_DEMOTED_CONNECTOR_NAMES: set[str] = set()

# Connectors that return full given names for a DOI-exact lookup, in priority
# order. Used only to enrich initials-only candidates from other connectors.
_AUTHORITY_CONNECTOR_NAMES = ("crossref", "europepmc")


def _is_initials_token(given: str) -> bool:
    """A given-name token that is just initials: 'Y', 'Y.', 'YB', 'Y.B.'."""
    core = given.replace(".", "").replace("-", "").strip()
    return bool(core) and len(core) <= 3 and core.isupper() and core.isalpha()


_COLLECTIVE_MARKERS = (
    "group", "consortium", "investigator", "collaboration", "collaborator",
    "network", "team", "study", "committee", "trial", "authors", "society",
)


def _is_collective_row(name: str) -> bool:
    low = str(name).lower()
    return any(m in low for m in _COLLECTIVE_MARKERS)


def _authors_initials_only(authors: list[str]) -> bool:
    """True if EVERY personal author's given part is initials-only (NLM style).

    'An Y' / 'Zheng Z' -> True; 'Yuanyuan An' / 'Yua An' -> False. Collective /
    single-token rows are skipped rather than counted as personal names.
    """
    seen_personal = False
    for a in authors or []:
        parts = str(a).split()
        if len(parts) < 2 or _is_collective_row(a):
            continue  # collective or single-token row, not a person
        seen_personal = True
        # NLM writes "Family I" (initials last); western writes "Given Family".
        first, last = parts[0], parts[-1]
        if not (_is_initials_token(last) or _is_initials_token(first)):
            return False
    return seen_personal


@dataclass
class VerifyConfig:
    report_version: str = "1.0"
    default_extraction_quality: ExtractionQuality = ExtractionQuality.UNKNOWN


class CitationVerifier:
    def __init__(
        self,
        orchestrator: ConnectorOrchestrator,
        policy: AdjudicationPolicy | None = None,
        llm_resolver: LLMResolver | None = None,
        config: VerifyConfig | None = None,
        secondary_verifier: SecondaryVerifierProtocol | None = None,
        structured_judge: StructuredJudge | None = None,
        existence_judge: ExistenceJudge | None = None,
        # Cascading agents
        valid_agent: ValidAgentProtocol | None = None,
        potential_agent: PotentialAgentProtocol | None = None,
        hallucinated_agent: HallucinatedAgentProtocol | None = None,
        extractor_agent: Any = None,
        author_classifier: Any = None,
        verified_ref_cache: Any = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.policy = policy or AdjudicationPolicy()
        self.llm_resolver = llm_resolver
        self.config = config or VerifyConfig()
        self.secondary_verifier = secondary_verifier
        self.structured_judge = structured_judge
        self.existence_judge = existence_judge
        self.valid_agent = valid_agent
        self.potential_agent = potential_agent
        self.hallucinated_agent = hallucinated_agent
        self.extractor_agent = extractor_agent
        self.author_classifier = author_classifier
        self.verified_ref_cache = verified_ref_cache
        # Phase-level timing log: append-only list of per-citation traces.
        # Each entry: {citation_id, total_ms, cache_hit, phase_0_ms, phase_1a_ms,
        # phase_15_ms, phase_1b_ms, phase_2_ms, exit_at, verdict}
        import threading as _threading
        self._timing_log: list[dict] = []
        self._timing_lock = _threading.Lock()

    _CACHEABLE_TAXONOMY = {"R1", "R3"}

    def _cache_if_valid(self, citation: CitationRecord, verdict: CitationVerdict) -> None:
        """Cache the verified reference if verdict is VALID with R1/R3."""
        if self.verified_ref_cache is None:
            return
        taxonomy = set(verdict.taxonomy_subtype or [])
        if verdict.verdict == VerdictLabel.VALID and taxonomy & self._CACHEABLE_TAXONOMY:
            from .models import citation_verdict_to_dict
            self.verified_ref_cache.set(citation, citation_verdict_to_dict(verdict))

    def _result_to_trace(self, citation: CitationRecord, result: ConnectorResult) -> EvidenceTrace:
        return EvidenceTrace(
            connector=result.connector,
            query={
                "title": citation.title,
                "year": citation.year,
                "doi": citation.doi,
                "arxiv_id": citation.arxiv_id,
                "url": citation.url,
            },
            latency_ms=result.latency_ms,
            cache_hit=result.cache_hit,
            source_health=result.source_health,
            candidates_count=len(result.records),
            error=result.error,
        )

    @staticmethod
    def _enrich_citation_from_url(citation: CitationRecord) -> CitationRecord:
        url = str(citation.url or "").strip()
        # Step 1: extract doi/arxiv_id from URL
        if url:
            extracted_doi, extracted_arxiv_id = extract_identifier(url)
        else:
            extracted_doi, extracted_arxiv_id = "", ""

        new_doi = (citation.doi or "").strip().lower() or extracted_doi
        new_arxiv_id = (citation.arxiv_id or "").strip().lower() or extracted_arxiv_id

        # Step 2: if citation has arxiv_id but no venue, infer venue="arXiv"
        # (avoids LLM HallucinatedAgent flagging empty venue as H4 for arxiv papers)
        new_venue = citation.venue
        if new_arxiv_id and not (citation.venue or "").strip():
            new_venue = "arXiv"

        # Skip rebuilding if nothing actually changed
        unchanged = (
            new_doi == (citation.doi or "").strip().lower()
            and new_arxiv_id == (citation.arxiv_id or "").strip().lower()
            and new_venue == citation.venue
        )
        if unchanged:
            return citation

        return replace(
            citation,
            doi=new_doi,
            arxiv_id=new_arxiv_id,
            venue=new_venue,
        )

    _WEB_CONNECTORS = {"google_search", "web_search"}

    # Titles that indicate a redirect/CAPTCHA/error page, not a real resolved title
    _JUNK_TITLES = {
        "redirecting", "just a moment", "not found", "404", "access denied",
        "page not found", "error", "captcha", "verify you are human",
        "please wait", "loading", "forbidden", "unauthorized",
    }

    def _check_identifier_mismatch(
        self, citation: CitationRecord, url_direct_records: list[dict],
    ) -> CitationVerdict | None:
        """When citation provides DOI/arXiv/URL and url_direct resolved them,
        check if the resolved title matches. If not, ask LLM to judge whether
        the resolved content and the citation refer to the same work."""
        if not citation.title:
            return None
        has_identifier = bool(
            (citation.doi or "").strip()
            or (citation.arxiv_id or "").strip()
            or (citation.url or "").strip()
        )
        if not has_identifier or not url_direct_records:
            return None

        citation_title_norm = normalize_title(citation.title)

        # Filter junk titles and check for match
        valid_resolved = []
        for r in url_direct_records:
            resolved_title = str(r.get("title", "") or "").strip()
            if not resolved_title or resolved_title.lower().strip() in self._JUNK_TITLES:
                continue
            resolved_title_norm = normalize_title(resolved_title)
            if not resolved_title_norm:
                continue
            valid_resolved.append(r)
            if similarity(citation_title_norm, resolved_title_norm) >= 0.75:
                return None  # Title matches, no mismatch

        # All junk → skip
        if not valid_resolved:
            return None

        # Title doesn't match — ask LLM to judge
        if self.llm_resolver:
            best = valid_resolved[0]
            resolved_info = (
                f"The citation's identifier resolved to a page with:\n"
                f"  Title: {best.get('title', '')}\n"
                f"  Authors: {best.get('authors', [])}\n"
                f"  Venue: {best.get('venue', '')}\n"
                f"  Year: {best.get('year')}\n"
                f"  URL: {best.get('_resolved_from', '')}\n"
            )
            # Build a minimal CandidateMatch for LLM review
            from .matching import build_candidate_match
            cand = build_candidate_match(citation, "url_direct", best)
            try:
                review = self.llm_resolver.review(
                    citation, [cand], ["identifier_title_mismatch"],
                    VerdictLabel.FAKE_REFERENCE,
                ) or {}
            except Exception:
                review = {}

            override = review.get("label_override")
            llm_note = str(review.get("note", "")).strip()

            if override:
                verdict = canonical_verdict_label(override)
            else:
                verdict = VerdictLabel.FAKE_REFERENCE

            reason = (
                f"Citation identifier resolved to: '{best.get('title', '')[:80]}' "
                f"(from {best.get('_resolved_from', '')}). "
            )
            if llm_note:
                reason += f"LLM review: {llm_note}"

            if verdict == VerdictLabel.VALID:
                return None  # LLM says it's the same work — continue normal flow

            return CitationVerdict(
                citation_id=citation.citation_id,
                verdict=verdict,
                evidence_sources=["url_direct"],
                conflicts=["identifier_title_mismatch"],
                adjudication_reason=reason,
                reference_snapshot={
                    "title": citation.title,
                    "authors": list(citation.authors),
                    "venue": citation.venue,
                    "year": citation.year,
                    "doi": citation.doi,
                    "arxiv_id": citation.arxiv_id,
                    "url": citation.url,
                },
                llm_recheck_reason=llm_note,
                needs_human_review=True,
                extraction_quality=ExtractionQuality.UNKNOWN,
            )

        # No LLM available — skip, let normal flow handle it
        return None


    def _resolve_tavily_key(self) -> str:
        import os as _os
        key = _os.getenv("TAVILY_API_KEY", "")
        if key:
            return key
        try:
            from apps.pdf_checker.config import load_pdf_checker_config as _load_cfg
            return (_load_cfg().connectors.tavily_api_key or "").strip()
        except Exception:
            return ""

    def _doi_authority_authors(self, citation: CitationRecord) -> list[str] | None:
        """Full-given-name author list from a DOI-exact authority connector.

        Returns None unless the citation has a DOI and a crossref/europepmc
        DOI-exact lookup yields authors that actually carry full given names
        (so we never overwrite good initials with more initials). Runs once per
        citation, before the streaming race, so the same authority is available
        to every per-connector unit regardless of which one wins.
        """
        doi = (citation.doi or "").strip()
        if not doi:
            return None
        by_name = {c.name: c for c in self.orchestrator.connectors}
        for name in _AUTHORITY_CONNECTOR_NAMES:
            conn = by_name.get(name)
            if conn is None:
                continue
            try:
                records = conn.search(citation, self.orchestrator.policy)
            except Exception:
                continue
            for rec in records or []:
                rec_doi = str(rec.get("doi", "") or "").strip().lower()
                authors = rec.get("authors") or []
                if rec_doi == doi.lower() and authors and \
                        not _authors_initials_only(authors):
                    return [str(a) for a in authors]
        return None

    def _run_connector_unit(
        self,
        citation: CitationRecord,
        connector: Any,
        records_override: list[dict] | None,
        url_match_status: str,
        url_issues: list[str],
        url_type_triggered: str,
        tavily_api_key: str,
        authority_authors: list[str] | None = None,
    ) -> dict[str, Any]:
        """Query one connector then verify its candidates (unit of parallelism).

        If ``records_override`` is provided, skip the connector.search() call and
        treat those records as already-fetched (used for Phase 0 url_direct).

        Returns a dict with keys:
            kind: "VALID" | "NOT_VALID" | "ERROR"
            verdict: CitationVerdict if kind == VALID, else None
            candidates: list[CandidateMatch]
            evaluations: list[dict]
            valid_results: list[ValidAgentResult]  (for NOT_VALID Phase 2 selection)
            trace: EvidenceTrace
        """
        import time as _time

        conn_name = connector.name if connector is not None else "url_direct"
        started = _time.perf_counter()
        error: str | None = None

        if records_override is not None:
            records = list(records_override)
        else:
            try:
                records = connector.search(citation, self.orchestrator.policy)
            except Exception as exc:
                records = []
                error = str(exc)

        # Drop web_search records whose title clearly is not the same paper.
        # Without this guard, Tavily's keyword-based hits (e.g. "AutoTap" for
        # a "Trace2TAP" query) would be passed to the HallucinatedAgent and
        # produce a FAKE verdict on a totally different paper. After dropping,
        # if no records remain the citation falls through to its
        # INSUFFICIENT_EVIDENCE / POTENTIAL bucket naturally.
        if records and conn_name in _WEB_CONNECTOR_NAMES:
            records = _filter_websearch_records_by_title(citation, records)

        latency_ms = (_time.perf_counter() - started) * 1000
        trace = EvidenceTrace(
            connector=conn_name,
            query={
                "title": citation.title, "year": citation.year, "doi": citation.doi,
                "arxiv_id": citation.arxiv_id, "url": citation.url,
            },
            latency_ms=latency_ms,
            cache_hit=False,
            source_health=1.0,
            candidates_count=len(records),
            error=error,
        )

        if error or not records:
            return {
                "kind": "ERROR" if error else "NOT_VALID",
                "verdict": None,
                "candidates": [],
                "evaluations": [],
                "valid_results": [],
                "trace": trace,
            }

        cands = collect_candidates(citation, {conn_name: records})
        # Full-given-name enrichment. Connectors such as PubMed return NLM-style
        # initials ("An Y"), under which a legitimate initial ("Y. An") and a
        # truncated name ("Yua An") are indistinguishable — the first VALID wins
        # the streaming race and the full-name source never gets adjudicated.
        # When a DOI-exact authority list is available, overwrite the authors of
        # an initials-only same-paper candidate with the full names so the
        # adjudicator can tell R2 from P1 inside this very unit.
        if authority_authors and cands:
            for cand in cands:
                if _authors_initials_only(cand.authors) and \
                        len(cand.authors) == len(authority_authors):
                    cand.authors = list(authority_authors)
        evals: list[dict[str, Any]] = []
        vresults = []

        for cand in cands:
            verdict_or_none, eval_dict, vresult = verify_single_candidate(
                citation=citation,
                candidate=cand,
                sibling_candidates=cands,
                prior_evaluations=evals,
                valid_agent=self.valid_agent,
                extractor_agent=self.extractor_agent,
                tavily_api_key=tavily_api_key,
                evidence_sources=[conn_name],  # patched by caller on VALID
                url_match_status=url_match_status,
                url_issues=url_issues,
                url_type_triggered=url_type_triggered,
                author_classifier=getattr(self, "author_classifier", None),
            )
            evals.append(eval_dict)
            if verdict_or_none is not None:
                return {
                    "kind": "VALID",
                    "verdict": verdict_or_none,
                    "candidates": cands,
                    "evaluations": evals,
                    "valid_results": vresults,
                    "trace": trace,
                }
            if vresult is not None:
                vresults.append(vresult)

        return {
            "kind": "NOT_VALID",
            "verdict": None,
            "candidates": cands,
            "evaluations": evals,
            "valid_results": vresults,
            "trace": trace,
        }

    def _verify_streaming(
        self,
        citation: CitationRecord,
        phase0_records: list[dict],
        url_match_status: str,
        url_issues: list[str],
        url_type_triggered: str,
        extraction_quality: ExtractionQuality,
        source_paper_title: str,
        _trace: dict | None = None,
    ) -> CitationVerdict:
        import time as _time
        if _trace is None:
            _trace = {}
        """Parallel per-connector verify with early VALID short-circuit.

        Flow:
          1a. Phase 0 url_direct records (reuse) + all structured connectors run as
              independent (query+verify) units in parallel. First unit returning
              VALID wins; remaining futures are cancelled.
          1b. If all structured units NOT_VALID → run web_search unit(s). First
              VALID wins.
          2.  Otherwise → cascading_phase2_only on accumulated candidates.
        """
        tavily_key = self._resolve_tavily_key()
        authority_authors = self._doi_authority_authors(citation)

        all_candidates: list[CandidateMatch] = []
        all_evaluations: list[dict[str, Any]] = []
        all_valid_results: list = []
        all_evidence: list[EvidenceTrace] = []

        def _patch_and_return_valid(verdict: CitationVerdict) -> CitationVerdict:
            """Patch verdict.evidence_sources with all connectors that completed so far."""
            completed = sorted({t.connector for t in all_evidence if t.error is None})
            if verdict.evidence_sources and verdict.evidence_sources[0] not in completed:
                completed = sorted(set(completed) | set(verdict.evidence_sources))
            verdict.evidence_sources = completed
            return verdict

        # --- Phase 1a.0: synthetic unit for Phase 0 url_direct records ---
        if phase0_records:
            unit = self._run_connector_unit(
                citation=citation,
                connector=None,
                records_override=phase0_records,
                url_match_status=url_match_status,
                url_issues=url_issues,
                url_type_triggered=url_type_triggered,
                tavily_api_key=tavily_key,
                authority_authors=authority_authors,
            )
            all_evidence.append(unit["trace"])
            if unit["kind"] == "VALID":
                return _patch_and_return_valid(unit["verdict"])
            all_candidates.extend(unit["candidates"])
            all_evaluations.extend(unit["evaluations"])
            all_valid_results.extend(unit["valid_results"])

        # --- Phase 1a: parallel structured connectors ---
        structured = [
            c for c in self.orchestrator.connectors
            if c.name not in _WEB_CONNECTOR_NAMES
            and c.name not in _DEMOTED_CONNECTOR_NAMES
        ]
        demoted = [c for c in self.orchestrator.connectors if c.name in _DEMOTED_CONNECTOR_NAMES]
        web = [c for c in self.orchestrator.connectors if c.name in _WEB_CONNECTOR_NAMES]

        _t_p1a = _time.perf_counter()
        if structured:
            max_workers = min(self.orchestrator.max_workers, len(structured)) or 1
            pool = ThreadPoolExecutor(max_workers=max_workers)
            futures = {
                pool.submit(
                    self._run_connector_unit,
                    citation, c, None,
                    url_match_status, url_issues, url_type_triggered,
                    tavily_key, authority_authors,
                ): c
                for c in structured
            }
            early_verdict: CitationVerdict | None = None
            try:
                for fut in as_completed(futures):
                    try:
                        unit = fut.result()
                    except Exception:
                        continue
                    all_evidence.append(unit["trace"])
                    if unit["kind"] == "VALID":
                        early_verdict = unit["verdict"]
                        break
                    all_candidates.extend(unit["candidates"])
                    all_evaluations.extend(unit["evaluations"])
                    all_valid_results.extend(unit["valid_results"])
            finally:
                # Do NOT wait for running futures — they may be stuck in slow
                # connector retries (e.g. 429 backoff on openalex). cancel_futures
                # stops not-yet-started ones; running ones are orphaned.
                pool.shutdown(wait=False, cancel_futures=True)
            _trace["phase_1a_ms"] = (_time.perf_counter() - _t_p1a) * 1000
            if early_verdict is not None:
                _trace["exit_at"] = "phase_1a_VALID"
                return _patch_and_return_valid(early_verdict)
        else:
            _trace["phase_1a_ms"] = (_time.perf_counter() - _t_p1a) * 1000

        # --- Phase 1.5: demoted structured connectors (sequential) ---
        # These connectors (e.g. arXiv) self-throttle aggressively, so running
        # them in parallel with the rest of Phase 1a serializes the whole pool
        # behind their lock. Run them only after Phase 1a returned no VALID.
        _t_p15 = _time.perf_counter()
        for dem_conn in demoted:
            unit = self._run_connector_unit(
                citation=citation,
                connector=dem_conn,
                records_override=None,
                url_match_status=url_match_status,
                url_issues=url_issues,
                url_type_triggered=url_type_triggered,
                tavily_api_key=tavily_key,
                authority_authors=authority_authors,
            )
            all_evidence.append(unit["trace"])
            if unit["kind"] == "VALID":
                _trace["phase_15_ms"] = (_time.perf_counter() - _t_p15) * 1000
                _trace["exit_at"] = "phase_15_VALID"
                return _patch_and_return_valid(unit["verdict"])
            all_candidates.extend(unit["candidates"])
            all_evaluations.extend(unit["evaluations"])
            all_valid_results.extend(unit["valid_results"])
        _trace["phase_15_ms"] = (_time.perf_counter() - _t_p15) * 1000

        # --- Phase 1b: web_search fallback (sequential) ---
        _t_p1b = _time.perf_counter()
        for web_conn in web:
            unit = self._run_connector_unit(
                citation=citation,
                connector=web_conn,
                records_override=None,
                url_match_status=url_match_status,
                url_issues=url_issues,
                url_type_triggered=url_type_triggered,
                tavily_api_key=tavily_key,
                authority_authors=authority_authors,
            )
            all_evidence.append(unit["trace"])
            if unit["kind"] == "VALID":
                _trace["phase_1b_ms"] = (_time.perf_counter() - _t_p1b) * 1000
                _trace["exit_at"] = "phase_1b_VALID"
                return _patch_and_return_valid(unit["verdict"])
            all_candidates.extend(unit["candidates"])
            all_evaluations.extend(unit["evaluations"])
            all_valid_results.extend(unit["valid_results"])
        _trace["phase_1b_ms"] = (_time.perf_counter() - _t_p1b) * 1000

        # --- Phase 2: cascading on accumulated non-VALID candidates ---
        _t_p2 = _time.perf_counter()
        evidence_sources = sorted({
            t.connector for t in all_evidence if t.error is None
        })

        # No candidates from any source → H1/INSUFFICIENT handling
        if not all_candidates:
            source_errors = [t for t in all_evidence if t.error]
            if source_errors and len(source_errors) == len(all_evidence):
                _trace["phase_2_ms"] = (_time.perf_counter() - _t_p2) * 1000
                _trace["exit_at"] = "no_candidates_INSUFFICIENT"
                return CitationVerdict(
                    citation_id=citation.citation_id,
                    verdict=VerdictLabel.INSUFFICIENT_EVIDENCE,
                    evidence_sources=evidence_sources,
                    conflicts=["no_candidate"],
                    adjudication_reason="All sources failed.",
                    taxonomy_subtype=[],
                    needs_human_review=True,
                )

        _r = cascading_phase2_only(
            citation=citation,
            candidates=all_candidates,
            all_valid_results=all_valid_results,
            all_evaluations=all_evaluations,
            evidence_sources=evidence_sources,
            secondary_verifier=self.secondary_verifier,
            potential_agent=self.potential_agent,
            hallucinated_agent=self.hallucinated_agent,
            url_match_status=url_match_status,
            url_issues=url_issues,
            url_type_triggered=url_type_triggered,
        )
        _trace["phase_2_ms"] = (_time.perf_counter() - _t_p2) * 1000
        _trace["exit_at"] = "phase_2_end"
        return _r

    def verify_citation(
        self,
        citation: CitationRecord,
        extraction_quality: ExtractionQuality | None = None,
        source_paper_title: str = "",
    ):
        import time as _time
        # Extreme-relaxed mode: same cascading logic as strict mode, but
        # mask EVERY reference field except title + authors. Downstream agents
        # see "reference didn't provide this" for venue/year/doi/etc., so they
        # only judge title + authors mismatch (H1/H2 + their P-counterparts).
        # Web search query also auto-narrows to title+authors via
        # CITATION_CHECKER_EXTREME_RELAXED env var (see google_search.py).
        from .extreme_relaxed import is_enabled as _extreme_enabled
        from .adjudicate import _RELAXED_FIELDS as _relaxed
        if _extreme_enabled():
            from dataclasses import replace as _dc_replace
            citation = _dc_replace(
                citation,
                venue="", year=None,
                doi="", arxiv_id="", url="", volume="", pages="",
                publisher="", location="",
            )
        elif _relaxed:
            # Relaxed-fields test mode: mask peripheral fields but keep
            # venue + year so the cascade can still flag H3/H4.
            from dataclasses import replace as _dc_replace
            citation = _dc_replace(
                citation,
                doi="", arxiv_id="", url="", volume="", pages="",
                publisher="", location="",
            )
        _t_total = _time.perf_counter()
        _trace: dict = {
            "citation_id": citation.citation_id,
            "cache_hit": False,
            "phase_0_ms": 0.0,
            "phase_1a_ms": 0.0,
            "phase_15_ms": 0.0,
            "phase_1b_ms": 0.0,
            "phase_2_ms": 0.0,
            "exit_at": "?",
            "total_ms": 0.0,
            "verdict": "?",
        }
        def _record(verdict_obj=None, exit_at: str = "?"):
            _trace["exit_at"] = exit_at
            _trace["total_ms"] = (_time.perf_counter() - _t_total) * 1000
            if verdict_obj is not None:
                v = getattr(verdict_obj, "verdict", None)
                _trace["verdict"] = getattr(v, "value", str(v))
            with self._timing_lock:
                self._timing_log.append(_trace)
        # Two-tier verified reference cache lookup:
        # Tier 1 (exact): full-field key. If an identically-formatted citation
        #   was verified before, return its verdict as-is — the taxonomy
        #   (R1/R2/R3) is already correct for this exact form.
        # Tier 2 (loose): title+year key. Some form of this paper was verified
        #   before — use its matched_candidate as a synthetic connector result
        #   and re-run verify_single_candidate so taxonomy is recomputed for
        #   the current citation's formatting.
        if self.verified_ref_cache is not None:
            exact = self.verified_ref_cache.get_exact(citation)
            if exact is not None:
                from .models import citation_verdict_from_dict
                _trace["cache_hit"] = True
                v = citation_verdict_from_dict(exact, citation.citation_id)
                _record(v, "cache_exact")
                return v

            loose = self.verified_ref_cache.get_loose(citation)
            if loose is not None and loose.get("matched_candidate"):
                candidate = build_candidate_match(citation, "cache", loose["matched_candidate"])
                verdict_or_none, _eval, _vr = verify_single_candidate(
                    citation=citation,
                    candidate=candidate,
                    sibling_candidates=[candidate],
                    prior_evaluations=[],
                    valid_agent=self.valid_agent,
                    extractor_agent=self.extractor_agent,
                    tavily_api_key=self._resolve_tavily_key(),
                    evidence_sources=["cache"],
                    url_match_status="",
                    url_issues=[],
                    url_type_triggered="",
                    author_classifier=getattr(self, "author_classifier", None),
                )
                if verdict_or_none is not None:
                    _trace["cache_hit"] = True
                    _record(verdict_or_none, "cache_loose")
                    return verdict_or_none
                # Re-validation failed → fall through to the full pipeline

        citation = self._enrich_citation_from_url(citation)
        extraction_quality = extraction_quality or self.config.default_extraction_quality

        # Phase 0: URL/DOI direct verification (BEFORE orchestrator query).
        # Resolves citation.url and citation.doi via URLDirectConnector + Tavily fallback.
        # Returns status signal AND any fetched records to merge into candidates.
        _t_p0 = _time.perf_counter()
        url_match_status, url_issues, url_type_triggered, phase0_records = phase0_url_check(citation)
        _trace["phase_0_ms"] = (_time.perf_counter() - _t_p0) * 1000

        # Phase 0 early-exit:
        # - Scholar URL (arxiv/doi) mismatch or not_found → FAKE_REFERENCE
        # - Other URL mismatch or not_found → POTENTIAL_REFERENCE / P2
        phase0_verdict = phase0_decide_verdict(
            citation, url_match_status, url_issues, url_type_triggered,
            fetched_records=phase0_records,
        )
        if phase0_verdict is not None:
            v = downgrade_non_academic_fake(citation, phase0_verdict)
            _record(v, "phase0_verdict")
            return v

        # If identifiers resolve to a clearly different paper → immediate FAKE
        # (but downgrade to P2 if the citation is non-academic)
        id_mismatch = self._check_identifier_mismatch(citation, phase0_records)
        if id_mismatch:
            v = downgrade_non_academic_fake(citation, id_mismatch)
            _record(v, "id_mismatch")
            return v

        # Non-academic short-circuit: skip cascade entirely. Connector cascade
        # against blogs/repos/tools just retrieves unrelated content and the
        # HallucinatedAgent then reasons spuriously (e.g. "Cityly" matched
        # against an unrelated CiteAudit paper). The downstream downgrade
        # always rewrites that to P2 anyway, so save the connector hits and
        # the LLM call.
        if is_non_academic_citation(citation):
            v = CitationVerdict(
                citation_id=citation.citation_id,
                verdict=VerdictLabel.POTENTIAL_REFERENCE,
                evidence_sources=[],
                conflicts=["non_academic_source"],
                adjudication_reason=(
                    "Non-academic source (blog/repo/tool); skipped connector "
                    "cascade because academic databases cannot verify it. "
                    "Marked POTENTIAL/P2."
                ),
                matched_candidate=None,
                reference_snapshot={},
                comparison={},
                candidate_evaluations=[],
                taxonomy_subtype=["P2"],
                needs_human_review=True,
                extraction_quality=extraction_quality,
                secondary_evidence=[],
            )
            _record(v, "non_academic")
            return v

        # Build Phase 0 evaluation record for the report (always included)
        _phase0_eval: dict | None = None
        if phase0_records or url_match_status:
            _phase0_eval = {
                "connector": "url_direct (Phase 0)",
                "verdict": f"PHASE0_{url_match_status.upper()}" if url_match_status else "PHASE0_SKIPPED",
                "url_type": url_type_triggered,
                "issues": url_issues,
                "candidates": phase0_records,
                "reason": (
                    f"Phase 0: url_match_status={url_match_status}, "
                    f"url_type={url_type_triggered}, issues={url_issues}"
                ),
            }

        # Use cascading 3-agent flow if configured (preferred): streaming
        # parallel per-connector verify units with Phase 2 fallback.
        if self.valid_agent:
            verdict = self._verify_streaming(
                citation=citation,
                phase0_records=phase0_records,
                url_match_status=url_match_status,
                url_issues=url_issues,
                url_type_triggered=url_type_triggered,
                extraction_quality=extraction_quality,
                source_paper_title=source_paper_title,
                _trace=_trace,
            )
            verdict = downgrade_non_academic_fake(citation, verdict)
            if _phase0_eval:
                verdict.candidate_evaluations.insert(0, _phase0_eval)
            self._cache_if_valid(citation, verdict)
            _record(verdict, _trace.get("exit_at", "streaming_end"))
            if _extreme_enabled():
                from .extreme_relaxed import apply_filters as _extreme_filter
                verdict = _extreme_filter(verdict)
            return verdict

        # --- Legacy path: collect all connector results then run adjudicate ---
        connector_results = self.orchestrator.query(citation, max_connectors=None)
        # Filter web_search records by title similarity (same guard as the
        # streaming path) so far-off Tavily hits never reach the judges.
        for _r in connector_results:
            if _r.connector in _WEB_CONNECTOR_NAMES and _r.records:
                _r.records = _filter_websearch_records_by_title(citation, _r.records)
        records = {result.connector: result.records for result in connector_results}
        if phase0_records:
            existing = records.get("url_direct", [])
            existing_urls = {str(r.get("url", "")) for r in existing}
            for r in phase0_records:
                if str(r.get("url", "")) not in existing_urls:
                    existing.append(r)
            records["url_direct"] = existing

        candidates = collect_candidates(citation, records)
        evidence = [self._result_to_trace(citation, result) for result in connector_results]

        # Use parallel multi-agent flow if judges are configured
        if self.structured_judge or self.existence_judge:
            verdict = multi_agent_adjudicate(
                citation=citation,
                candidates=candidates,
                evidence_traces=evidence,
                extraction_quality=extraction_quality,
                structured_judge=self.structured_judge,
                existence_judge=self.existence_judge,
                secondary_verifier=self.secondary_verifier,
                source_paper_title=source_paper_title,
            )
            if _phase0_eval:
                verdict.candidate_evaluations.insert(0, _phase0_eval)
            self._cache_if_valid(citation, verdict)
            if _extreme_enabled():
                from .extreme_relaxed import apply_filters as _extreme_filter
                verdict = _extreme_filter(verdict)
            return verdict

        # Fallback: legacy single-LLM adjudication
        verdict = adjudicate(
            citation=citation,
            candidates=candidates,
            evidence=evidence,
            extraction_quality=extraction_quality,
            llm_resolver=self.llm_resolver,
            policy=self.policy,
            secondary_verifier=self.secondary_verifier,
        )
        if _phase0_eval:
            verdict.candidate_evaluations.insert(0, _phase0_eval)
        self._cache_if_valid(citation, verdict)
        if _extreme_enabled():
            from .extreme_relaxed import apply_filters as _extreme_filter
            verdict = _extreme_filter(verdict)
        return verdict

    def verify_paper(
        self,
        paper_id: str,
        pipeline_type: str,
        citations: list[CitationRecord],
        extraction_quality_map: dict[str, ExtractionQuality] | None = None,
        metadata: dict | None = None,
        show_progress: bool = False,
    ) -> CheckReport:
        extraction_quality_map = extraction_quality_map or {}
        verdicts = []
        total = len(citations)
        for idx, citation in enumerate(citations, start=1):
            if show_progress:
                print(
                    f"[verify] citation {idx}/{total} start id={citation.citation_id} title={citation.title[:120]!r}",
                    flush=True,
                )
            verdict = self.verify_citation(
                citation,
                extraction_quality=extraction_quality_map.get(citation.citation_id, self.config.default_extraction_quality),
                source_paper_title=paper_id,
            )
            verdicts.append(verdict)
            if show_progress:
                print(
                    f"[verify] citation {idx}/{total} result id={citation.citation_id} "
                    f"verdict={verdict.verdict.value} reason={verdict.adjudication_reason}",
                    flush=True,
                )
            if show_progress and total > 0:
                width = 28
                filled = min(width, int(width * idx / total))
                bar = "#" * filled + "-" * (width - filled)
                sys.stdout.write(f"\r[verify] [{bar}] {idx}/{total}")
                sys.stdout.flush()
        if show_progress and total > 0:
            sys.stdout.write("\n")
            sys.stdout.flush()
        summary = build_summary(verdicts)
        risk_score = compute_risk_score(verdicts)
        requires_human_review = any(verdict.needs_human_review for verdict in verdicts)
        return CheckReport(
            report_version=self.config.report_version,
            paper_id=paper_id,
            pipeline_type=pipeline_type,
            citations=verdicts,
            summary=summary,
            risk_score=risk_score,
            requires_human_review=requires_human_review,
            metadata=metadata or {},
        )
