"""Cached online event retrieval for paper experiments.

The retriever is intentionally small and dependency-light. It records every
query and response into the experiment directory so explanations can cite a
stable local evidence cache even when online search is unavailable later.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from agents.evidence_verifier import EvidenceVerifier


@dataclass
class RetrievedEvidence:
    query: str
    url: str
    title: str
    snippet: str
    retrieved_at: str
    cache_path: str
    source: str = "online_search"
    status: str = "ok"
    event_key: str = ""
    accepted: bool = False
    accepted_reason: str = ""
    rejected_reason: str = ""
    relevance_score: float = 0.0
    source_time_status: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class EventRetrievalAgent:
    """Retrieve NYC event evidence and cache raw search artifacts."""

    CACHE_SCHEMA_VERSION = "event_search_v2"

    def __init__(
        self,
        cache_dir: str | Path,
        user_agent: str = "Mozilla/5.0 (compatible; 0206moment-event-forecasting/1.0)",
        timeout_s: float = 8.0,
        enabled: bool = True,
        search_backend: str = "bing_rss",
        verifier: EvidenceVerifier | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.user_agent = user_agent
        self.timeout_s = float(timeout_s)
        self.enabled = bool(enabled)
        self.search_backend = search_backend
        self.verifier = verifier

    def retrieve_for_forecast(
        self,
        target_date: str,
        station_names: Iterable[str],
        max_results: int = 5,
    ) -> List[RetrievedEvidence]:
        query = self._build_query(target_date, station_names)
        return self.search(query, max_results=max_results)

    def retrieve_for_events(
        self,
        events: Sequence[object],
        target_date: str,
        station_names: Iterable[str],
        max_results_per_event: int = 4,
    ) -> List[RetrievedEvidence]:
        """Retrieve evidence for each structured event.

        Queries are event-driven, not station-list driven, so the cache can be
        traced back to the concrete event that triggered forecast calibration.
        """
        rows: List[RetrievedEvidence] = []
        for event in events:
            event_dict = self._event_to_dict(event)
            event_key = self._event_key(event_dict)
            query = self._build_event_query(event_dict, target_date)
            event_rows = self.search(query, max_results=max_results_per_event, event_key=event_key)
            verifier = self.verifier or EvidenceVerifier(require_source_time=True, anchor_time=target_date)
            verified = verifier.verify([row.to_dict() for row in event_rows], event_dict)
            event_rows = [self._row_from_verified(row) for row in verified]
            rows.extend(event_rows)
        if not rows:
            rows = self.retrieve_for_forecast(target_date, station_names, max_results=max_results_per_event)
        return rows

    def search(self, query: str, max_results: int = 5, event_key: str = "") -> List[RetrievedEvidence]:
        key = hashlib.sha256(f"{self.CACHE_SCHEMA_VERSION}|{query}".encode("utf-8")).hexdigest()[:16]
        cache_path = self.cache_dir / f"{key}.json"
        if cache_path.is_file():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if not self.enabled or payload.get("status") == "ok":
                return [RetrievedEvidence(**row) for row in payload.get("evidence", [])]

        retrieved_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rows: List[RetrievedEvidence] = []
        raw_html = ""
        status = "disabled"
        if self.enabled:
            errors = []
            for backend in self._backend_order():
                try:
                    url = self._search_url(query, backend)
                    req = Request(url, headers={"User-Agent": self.user_agent})
                    with urlopen(req, timeout=self.timeout_s) as resp:
                        raw_html = resp.read().decode("utf-8", errors="replace")
                    if backend == "bing_rss":
                        rows = self._parse_bing_rss(raw_html, query, retrieved_at, cache_path, max_results)
                    else:
                        rows = self._parse_duckduckgo(raw_html, query, retrieved_at, cache_path, max_results)
                    if rows:
                        status = "ok"
                        break
                    errors.append(f"{backend}: no_results")
                except Exception as exc:
                    errors.append(f"{backend}: {type(exc).__name__}: {exc}")
            if not rows:
                status = "error: " + " | ".join(errors) if errors else "error: no search backend attempted"

        if not rows:
            rows = [
                RetrievedEvidence(
                    query=query,
                    url="",
                    title="No online evidence retrieved",
                    snippet=status,
                    retrieved_at=retrieved_at,
                    cache_path=str(cache_path),
                    status=status,
                    event_key=event_key,
                )
            ]
        else:
            rows = [RetrievedEvidence(**{**row.to_dict(), "event_key": event_key}) for row in rows]

        payload = {
            "schema_version": self.CACHE_SCHEMA_VERSION,
            "query": query,
            "retrieved_at": retrieved_at,
            "status": status,
            "raw_html_sha256": hashlib.sha256(raw_html.encode("utf-8")).hexdigest() if raw_html else None,
            "evidence": [row.to_dict() for row in rows],
        }
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if raw_html:
            (self.cache_dir / f"{key}.html").write_text(raw_html, encoding="utf-8", errors="ignore")
        return rows

    def _backend_order(self) -> List[str]:
        choices = [self.search_backend, "bing_rss", "duckduckgo_html"]
        out = []
        for choice in choices:
            if choice and choice not in out:
                out.append(choice)
        return out

    @staticmethod
    def _search_url(query: str, backend: str) -> str:
        encoded = quote_plus(query)
        if backend == "bing_rss":
            return f"https://www.bing.com/search?format=rss&q={encoded}"
        return f"https://duckduckgo.com/html/?q={encoded}"

    @staticmethod
    def _build_query(target_date: str, station_names: Iterable[str]) -> str:
        station_text = " ".join(list(station_names)[:4])
        return f'NYC major events near subway stations {station_text} on {target_date}'

    @staticmethod
    def _event_to_dict(event: object) -> dict:
        if hasattr(event, "model_dump"):
            return event.model_dump()
        if isinstance(event, dict):
            return dict(event)
        return {
            key: getattr(event, key)
            for key in ["title", "event_time", "venue_name", "location", "event_type", "channel_name"]
            if hasattr(event, key)
        }

    @staticmethod
    def _event_key(event: dict) -> str:
        return "|".join([
            str(event.get("title", ""))[:60],
            str(event.get("event_time", ""))[:16],
            str(event.get("location") or event.get("venue_name") or "")[:60],
        ])

    @staticmethod
    def _build_event_query(event: dict, target_date: str) -> str:
        title = str(event.get("title") or "")
        title = re.sub(r"[_()|,/]+", " ", title)
        title = re.sub(r"\s+", " ", title).strip()[:80]
        venue = str(event.get("venue_name") or event.get("location") or "")
        venue = venue.split("|")[0]
        venue = re.sub(r"[_()|,/]+", " ", venue)
        venue = re.sub(r"\b[A-Z]\d{3,}\b", " ", venue)
        venue = re.sub(r"\s+", " ", venue).strip()[:90]
        event_time = str(event.get("event_time") or target_date)
        date = event_time[:10]
        parts = []
        if title:
            parts.append(f'"{title}"')
        if "tsq" in title.lower() or "times sq" in venue.lower():
            parts.append('"Times Square"')
        if venue:
            parts.append(f'"{venue}"')
        parts.extend(["NYC", date, "event"])
        return " ".join(part for part in parts if part).strip()

    @staticmethod
    def _row_from_verified(row: dict) -> RetrievedEvidence:
        allowed = {field.name for field in RetrievedEvidence.__dataclass_fields__.values()}
        data = {k: v for k, v in row.items() if k in allowed}
        return RetrievedEvidence(**data)

    @staticmethod
    def _parse_duckduckgo(
        raw_html: str,
        query: str,
        retrieved_at: str,
        cache_path: Path,
        max_results: int,
    ) -> List[RetrievedEvidence]:
        rows: List[RetrievedEvidence] = []
        blocks = re.findall(r'<a rel="nofollow" class="result__a" href="([^"]+)".*?>(.*?)</a>.*?<a class="result__snippet".*?>(.*?)</a>', raw_html, flags=re.S)
        for url, title, snippet in blocks[:max_results]:
            rows.append(
                RetrievedEvidence(
                    query=query,
                    url=html.unescape(re.sub(r"&amp;", "&", url)),
                    title=EventRetrievalAgent._clean_html(title),
                    snippet=EventRetrievalAgent._clean_html(snippet),
                    retrieved_at=retrieved_at,
                    cache_path=str(cache_path),
                )
            )
        return rows

    @staticmethod
    def _parse_bing_rss(
        raw_xml: str,
        query: str,
        retrieved_at: str,
        cache_path: Path,
        max_results: int,
    ) -> List[RetrievedEvidence]:
        rows: List[RetrievedEvidence] = []
        try:
            root = ElementTree.fromstring(raw_xml)
        except ElementTree.ParseError:
            return rows
        for item in root.findall("./channel/item")[:max_results]:
            title = item.findtext("title") or ""
            link = item.findtext("link") or ""
            description = item.findtext("description") or ""
            if not title and not link:
                continue
            rows.append(
                RetrievedEvidence(
                    query=query,
                    url=link,
                    title=EventRetrievalAgent._clean_html(title),
                    snippet=EventRetrievalAgent._clean_html(description),
                    retrieved_at=retrieved_at,
                    cache_path=str(cache_path),
                    source="bing_rss",
                )
            )
        return rows

    @staticmethod
    def _clean_html(text: str) -> str:
        text = re.sub(r"<.*?>", " ", text, flags=re.S)
        text = html.unescape(text)
        return re.sub(r"\s+", " ", text).strip()
