"""Cost-gated Qwen-Plus summary research for major event explanations."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from agents.evidence_verifier import EvidenceVerifier


TIER_SCORE = {"A": 4, "B": 3, "C": 2, "D": 1}


def _event_dict(event: object) -> dict:
    if hasattr(event, "model_dump"):
        return event.model_dump()
    if isinstance(event, dict):
        return dict(event)
    out = {}
    for key in [
        "title",
        "event_time",
        "location",
        "venue_name",
        "event_type",
        "event_category",
        "impact_tier",
        "relevance_score",
        "channel_name",
        "station_complex_id",
        "distance_m",
    ]:
        if hasattr(event, key):
            out[key] = getattr(event, key)
    return out


def _parse_dt(value: str) -> Optional[datetime]:
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(value)[:19], fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def _physical_event_key(event: dict) -> str:
    title = re.sub(r"\s+", " ", str(event.get("title", "") or "").strip().lower())
    when = str(event.get("event_time", "") or "")[:16]
    where = re.sub(r"\s+", " ", str(event.get("venue_name") or event.get("location") or "").strip().lower())[:80]
    return "|".join([title, when, where])


def _relevance_score(event: dict) -> float:
    for key in ["relevance_score", "confidence", "score"]:
        value = event.get(key)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except Exception:
            pass
    confidence = str(event.get("impact_confidence", "") or "").lower()
    return {"high": 0.9, "medium": 0.7, "low": 0.35}.get(confidence, 0.0)


def _station_relevant(event: dict, max_distance_m: float) -> bool:
    if event.get("channel_name") or event.get("station_complex_id"):
        return True
    try:
        return float(event.get("distance_m")) <= max_distance_m
    except Exception:
        return False


def _overlaps_forecast_window(event: dict, target_date: str, horizon_hours: int) -> bool:
    start = _parse_dt(target_date)
    ev_time = _parse_dt(str(event.get("event_time", "") or ""))
    if start is None or ev_time is None:
        return False
    end = start + timedelta(hours=int(horizon_hours))
    return start <= ev_time < end


def _to_plain_data(obj: object) -> object:
    """Convert SDK response objects into plain dict/list structures."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _to_plain_data(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain_data(v) for v in obj]
    if hasattr(obj, "model_dump"):
        try:
            return _to_plain_data(obj.model_dump())
        except Exception:
            pass
    if hasattr(obj, "to_dict"):
        try:
            return _to_plain_data(obj.to_dict())
        except Exception:
            pass
    out = {}
    for key in ["output", "search_info", "search_results", "choices", "message", "content"]:
        if hasattr(obj, key):
            out[key] = _to_plain_data(getattr(obj, key))
    return out or str(obj)


def _iter_search_info_nodes(data: object) -> Iterable[dict]:
    if isinstance(data, dict):
        if isinstance(data.get("search_info"), dict):
            yield data["search_info"]
        if isinstance(data.get("search_results"), list):
            yield data
        for value in data.values():
            yield from _iter_search_info_nodes(value)
    elif isinstance(data, list):
        for item in data:
            yield from _iter_search_info_nodes(item)


def extract_dashscope_search_results(response: object) -> List[dict]:
    """Extract DashScope native `search_info.search_results` rows.

    DashScope's native API returns citation-quality search sources through
    `output.search_info.search_results` when `search_options.enable_source` is
    enabled. The OpenAI-compatible path may not expose this field, but tests and
    cached payloads can still pass SDK-like response objects here.
    """
    data = _to_plain_data(response)
    rows: List[dict] = []
    seen = set()
    for node in _iter_search_info_nodes(data):
        for raw in node.get("search_results", []) or []:
            if not isinstance(raw, dict):
                continue
            url = str(raw.get("url") or raw.get("link") or "").strip()
            title = str(raw.get("title") or raw.get("name") or "").strip()
            snippet = str(raw.get("snippet") or raw.get("summary") or raw.get("content") or "").strip()
            if not url:
                continue
            key = (url, title, snippet[:80])
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "source_agent": "qwen_plus_native_search",
                    "search_index": raw.get("index", len(rows) + 1),
                }
            )
    return rows


def select_qwenplus_events(
    events: Sequence[object],
    target_date: str,
    horizon_hours: int,
    max_events: int = 3,
    min_score: float = 0.75,
    max_distance_m: float = 800.0,
) -> List[dict]:
    """Select high-value unique events worth a paid Qwen-Plus search call."""
    candidates = []
    seen = set()
    for raw_event in events:
        event = _event_dict(raw_event)
        tier = str(event.get("impact_tier", "C") or "C").upper()
        if tier not in ("A", "B"):
            continue
        score = _relevance_score(event)
        if score < min_score:
            continue
        if not _station_relevant(event, max_distance_m=max_distance_m):
            continue
        if not _overlaps_forecast_window(event, target_date, horizon_hours):
            continue
        key = _physical_event_key(event)
        if key in seen:
            continue
        seen.add(key)
        event["relevance_score"] = score
        event["physical_event_key"] = key
        candidates.append(event)

    candidates.sort(
        key=lambda ev: (
            TIER_SCORE.get(str(ev.get("impact_tier", "C")).upper(), 0),
            float(ev.get("relevance_score", 0.0)),
            bool(ev.get("channel_name") or ev.get("station_complex_id")),
            str(ev.get("title", "")),
        ),
        reverse=True,
    )
    return candidates[: max(0, int(max_events))]


@dataclass
class EvidenceResearchResult:
    accepted: List[dict] = field(default_factory=list)
    rejected: List[dict] = field(default_factory=list)
    summaries: List[dict] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)

    def to_sources(self) -> List[dict]:
        # The formal paper workflow is summary-only. URL evidence rows are a
        # legacy diagnostic and are not surfaced as forecast-time sources.
        return []

    def to_dict(self) -> dict:
        return asdict(self)


class EvidenceResearchAgent:
    """Runs cached Qwen-Plus summary generation only for selected events."""

    CACHE_SCHEMA_VERSION = "summary_only_v1"

    def __init__(
        self,
        cache_dir: str | Path,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        cache_only: bool = False,
        force_refresh: bool = False,
        verifier: Optional[EvidenceVerifier] = None,
        client_factory: Optional[Callable[..., object]] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.model = model or os.environ.get("LLM_MODEL", "qwen-plus")
        self.cache_only = bool(cache_only)
        self.force_refresh = bool(force_refresh)
        self.verifier = verifier or EvidenceVerifier()
        self.client_factory = client_factory

    def research_events(
        self,
        events: Sequence[object],
        target_date: str,
        station_names: Iterable[str],
        max_events: int = 3,
        min_score: float = 0.75,
        horizon_hours: int = 192,
    ) -> EvidenceResearchResult:
        selected = select_qwenplus_events(
            events,
            target_date,
            horizon_hours,
            max_events=max_events,
            min_score=min_score,
        )
        result = EvidenceResearchResult(stats={
            "selected_events": len(selected),
            "api_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "accepted_evidence": 0,
            "rejected_evidence": 0,
            "summary_count": 0,
            "summary_used_for_explanation": 0,
        })
        for event in selected:
            payload, cache_hit = self._load_or_call(event, target_date, list(station_names)[:6])
            if cache_hit:
                result.stats["cache_hits"] += 1
            else:
                result.stats["cache_misses"] += 1
                if payload.get("status") == "ok":
                    result.stats["api_calls"] += 1
            if payload.get("model_assisted_summary"):
                result.summaries.append({
                    "event_key": event.get("physical_event_key", ""),
                    "summary": payload.get("model_assisted_summary", ""),
                    "event_summary": payload.get("event_summary", payload.get("model_assisted_summary", "")),
                    "event_relevance_to_station": payload.get("event_relevance_to_station", ""),
                    "expected_ridership_effect": payload.get("expected_ridership_effect", ""),
                    "uncertainty": payload.get("uncertainty", ""),
                    "summary_used_for_explanation": bool(payload.get("summary_used_for_explanation", True)),
                    "source_agent": "qwen_plus",
                })
        result.stats["accepted_evidence"] = len(result.accepted)
        result.stats["rejected_evidence"] = len(result.rejected)
        result.stats["summary_count"] = len(result.summaries)
        result.stats["summary_used_for_explanation"] = sum(1 for row in result.summaries if row.get("summary_used_for_explanation"))
        return result

    def _cache_path(self, event: dict, target_date: str) -> Path:
        key = hashlib.sha256(
            f"{self.CACHE_SCHEMA_VERSION}|{_physical_event_key(event)}|{target_date}|{self.model}".encode("utf-8")
        ).hexdigest()[:16]
        return self.cache_dir / f"{key}.json"

    def _load_or_call(self, event: dict, target_date: str, station_names: List[str]) -> tuple[dict, bool]:
        path = self._cache_path(event, target_date)
        if path.is_file() and not self.force_refresh:
            return json.loads(path.read_text(encoding="utf-8")), True
        if self.cache_only or not self.api_key:
            payload = self._payload(event, status="cache_miss_no_api_call", evidence=[], raw_response="")
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return payload, False

        raw = ""
        evidence: List[dict] = []
        summary = ""
        parsed: dict = {}
        status = "ok"
        try:
            response, raw = self._call_model(event, target_date, station_names)
            parsed = self._parse_json(raw) if raw else {}
            summary = str(parsed.get("event_summary") or parsed.get("model_assisted_summary") or "")
        except Exception as exc:
            status = f"error: {type(exc).__name__}: {exc}"
        payload = self._payload(event, status=status, evidence=evidence, raw_response=raw, summary=summary, parsed=parsed)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload, False

    def _call_model(self, event: dict, target_date: str, station_names: List[str]) -> tuple[object, str]:
        if self.client_factory is None and "dashscope.aliyuncs.com" in self.base_url:
            response = self._call_dashscope_generation(event, target_date, station_names)
            raw = self._extract_message_content(response)
            return response, raw
        client = self._client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Return strict JSON only. Do not include API keys or secrets."},
                {"role": "user", "content": self._prompt(event, target_date, station_names)},
            ],
            temperature=0.1,
            max_tokens=1200,
            extra_body={
                "enable_search": True,
                "search_options": {"forced_search": True, "search_strategy": "turbo"},
            },
        )
        return response, self._extract_message_content(response)

    def _call_dashscope_generation(self, event: dict, target_date: str, station_names: List[str]) -> dict:
        body = {
            "model": self.model,
            "input": {
                "messages": [
                    {"role": "system", "content": "Return strict JSON only. Do not include API keys or secrets."},
                    {"role": "user", "content": self._prompt(event, target_date, station_names)},
                ]
            },
            "parameters": {
                "temperature": 0.1,
                "max_tokens": 1200,
                "enable_search": True,
                "search_options": {
                    "forced_search": True,
                    "search_strategy": "turbo",
                    "intention_options": {
                        "prompt_intervene": "Use web search only to summarize the named NYC public event, venue, date, crowd, and transit relevance. Do not return URL citations."
                    },
                },
                "result_format": "message",
            },
        }
        req = urllib.request.Request(
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=45) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _client(self):
        if self.client_factory is not None:
            return self.client_factory(api_key=self.api_key, base_url=self.base_url)
        from openai import OpenAI

        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    @staticmethod
    def _parse_json(raw: str) -> dict:
        text = (raw or "").strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end >= start:
            text = text[start : end + 1]
        return json.loads(text)

    @staticmethod
    def _extract_message_content(response: object) -> str:
        data = _to_plain_data(response)
        if isinstance(data, dict):
            try:
                return str(data["choices"][0]["message"]["content"] or "")
            except Exception:
                pass
            try:
                return str(data["output"]["choices"][0]["message"]["content"] or "")
            except Exception:
                pass
        try:
            return str(response.choices[0].message.content or "")
        except Exception:
            return ""

    @staticmethod
    def _prompt(event: dict, target_date: str, station_names: List[str]) -> str:
        title = str(event.get("title", "") or "")
        event_time = str(event.get("event_time", "") or "")
        location = str(event.get("location") or event.get("venue_name") or "")
        return f"""Summarize this NYC public event for forecast-time subway ridership explanation.

Return strict JSON with exactly these fields:
- event_summary: one concise sentence describing the event and venue.
- event_relevance_to_station: why the event could matter for the nearby station/channel list.
- expected_ridership_effect: likely ridership direction/timing, stated cautiously.
- uncertainty: what remains uncertain and why bounded correction or abstention may be needed.
- summary_used_for_explanation: true.

Do not return URLs, titles, snippets, citation markers, or an evidence list.
If web search is available, use it only to improve the summary; do not expose sources.

Event query context:
"{title}" "{event_time[:10]}" "{location[:120]}" NYC official event

Event:
{json.dumps(event, ensure_ascii=False, indent=2)}

Forecast target date: {target_date}
Nearby station channels: {station_names}

Do not include dictionary/translation pages. Do not invent URLs.
"""

    def _payload(self, event: dict, status: str, evidence: List[dict], raw_response: str, summary: str = "", parsed: dict | None = None) -> dict:
        parsed = dict(parsed or {})
        event_summary = str(parsed.get("event_summary") or parsed.get("model_assisted_summary") or summary or "")
        relevance = str(parsed.get("event_relevance_to_station") or "")
        effect = str(parsed.get("expected_ridership_effect") or "")
        uncertainty = str(parsed.get("uncertainty") or "")
        used = bool(parsed.get("summary_used_for_explanation", True if event_summary else False))
        combined = event_summary
        if relevance or effect or uncertainty:
            pieces = [event_summary]
            if relevance:
                pieces.append(f"Station relevance: {relevance}")
            if effect:
                pieces.append(f"Expected ridership effect: {effect}")
            if uncertainty:
                pieces.append(f"Uncertainty: {uncertainty}")
            combined = " ".join(piece for piece in pieces if piece).strip()
        return {
            "provider": "qwen_plus",
            "model": self.model,
            "base_url_host": self.base_url.split("//")[-1].split("/")[0],
            "event_key": _physical_event_key(event),
            "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "schema_version": self.CACHE_SCHEMA_VERSION,
            "status": status,
            "model_assisted_summary": combined,
            "event_summary": event_summary,
            "event_relevance_to_station": relevance,
            "expected_ridership_effect": effect,
            "uncertainty": uncertainty,
            "summary_used_for_explanation": used,
            "raw_response_sha256": hashlib.sha256((raw_response or "").encode("utf-8")).hexdigest() if raw_response else None,
            "evidence": [],
        }
