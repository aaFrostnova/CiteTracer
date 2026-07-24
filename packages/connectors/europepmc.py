from __future__ import annotations

from typing import Any

from packages.core.models import CitationRecord

from .base import BaseConnector, RequestPolicy


class EuropePMCConnector(BaseConnector):
    name = "europepmc"
    ttl_s = 60 * 60 * 24

    def search(self, citation: CitationRecord, policy: RequestPolicy) -> list[dict[str, Any]]:
        query = citation.doi or citation.title or citation.raw_text
        if not query:
            return []
        payload = self._request_json(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            {
                "query": query,
                "resultType": "core",
                "pageSize": 5,
                "format": "json",
            },
            policy,
        )
        results = payload.get("resultList", {}).get("result", []) or []
        return [self._normalize_item(item) for item in results]

    @staticmethod
    def _normalize_item(item: dict[str, Any]) -> dict[str, Any]:
        # resultType=core returns BOTH an NLM-abbreviated "authorString"
        # ("Basu M, Mahapatra E") and a structured "authorList" carrying full given
        # names ("Mukta Basu") plus collectiveName rows for consortium authors.
        # Prefer the structured list: the abbreviated form destroys the given-name
        # evidence needed to tell a legitimate initial from a truncated name, and
        # splitting it on commas silently turns a collective name into an author.
        authors: list[str] = []
        for a in ((item.get("authorList") or {}).get("author") or []):
            first, last = (a.get("firstName") or "").strip(), (a.get("lastName") or "").strip()
            if first and last:
                authors.append(f"{first} {last}")
            elif a.get("collectiveName"):
                authors.append(str(a["collectiveName"]).strip())
            elif a.get("fullName"):
                authors.append(str(a["fullName"]).strip())
        if not authors:
            author_string = str(item.get("authorString", "") or "").strip()
            if author_string:
                authors = [p.strip().rstrip(".") for p in author_string.split(",") if p.strip()]
        doi = str(item.get("doi", "") or "").strip().lower()
        pmcid = str(item.get("pmcid", "") or "").strip()
        pmid = str(item.get("pmid", "") or "").strip()
        url = ""
        if pmcid:
            url = f"https://europepmc.org/article/PMC/{pmcid}"
        elif pmid:
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        journal_info = item.get("journalInfo") or {}
        journal_volume = str(item.get("journalVolume", "") or journal_info.get("volume", "") or "")
        return {
            "title": str(item.get("title", "") or ""),
            "authors": authors,
            # resultType=core carries the journal name under journalInfo.journal,
            # not as a top-level "journalTitle" (that only exists for resultType=lite).
            # Reading the wrong key left venue empty on every core-mode hit.
            "venue": str(
                item.get("journalTitle", "")
                or (journal_info.get("journal") or {}).get("title", "")
                or (journal_info.get("journal") or {}).get("isoabbreviation", "")
                or ""
            ),
            "year": _safe_int(item.get("pubYear")),
            "doi": doi,
            "arxiv_id": "",
            "url": url,
            "volume": journal_volume,
            "pages": str(item.get("pageInfo", "") or ""),
            "publisher": "",
        }


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
