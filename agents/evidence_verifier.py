"""Evidence relevance checks for event-aware forecast explanations."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence
from urllib.parse import urlparse


LOW_VALUE_DOMAINS = {
    "iciba.com",
    "baike.baidu.com",
    "dictionary.cambridge.org",
    "tingclass.net",
    "ddooo.com",
    "m.ddooo.com",
    "jingyan.baidu.com",
    "zhidao.baidu.com",
    "thermofisher.cn",
    "instrument.com.cn",
    "speedtest.mybroadband.co.za",
    "minhaconexao.com.br",
}

LOW_VALUE_TITLE_PATTERNS = [
    "是什么意思",
    "translation",
    "dictionary",
    "翻译",
    "读音",
    "用法",
    "例句",
    "软件下载",
    "下载",
    "测速",
    "speed test",
]

GENERIC_EVENT_WORDS = {
    "event",
    "events",
    "nyc",
    "new",
    "york",
    "city",
    "subway",
    "crowd",
    "transit",
    "ridership",
    "plaza",
    "parade",
    "street",
}


def _parse_dt(value: object) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text[:19], fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None

EVENT_FIELD_GROUPS = {
    "title": ["title"],
    "date": ["event_time"],
    "location": ["location", "venue_name"],
    "category": ["event_type", "event_category"],
}


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z0-9]+", text or "") if len(t) >= 3]


def _domain(url: str) -> str:
    host = urlparse(url or "").netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _event_keywords(event: dict) -> List[str]:
    text = " ".join(
        str(event.get(k, "") or "")
        for k in ["title", "event_time", "location", "venue_name", "event_type", "event_category"]
    )
    out = []
    for token in _tokens(text):
        if token in GENERIC_EVENT_WORDS:
            continue
        if token not in out:
            out.append(token)
    return out[:16]


def _field_tokens(event: dict, keys: Sequence[str]) -> List[str]:
    text = " ".join(str(event.get(k, "") or "") for k in keys)
    out = []
    for token in _tokens(text):
        if token in GENERIC_EVENT_WORDS:
            continue
        if token not in out:
            out.append(token)
    return out


class EvidenceVerifier:
    """Rejects low-value or irrelevant external evidence before citation."""

    def __init__(
        self,
        min_relevance_score: float = 0.34,
        min_field_matches: int = 2,
        require_date_match: bool = True,
        require_source_time: bool = False,
        anchor_time: object = None,
    ):
        self.min_relevance_score = float(min_relevance_score)
        self.min_field_matches = int(min_field_matches)
        self.require_date_match = bool(require_date_match)
        self.require_source_time = bool(require_source_time)
        self.anchor_time = _parse_dt(anchor_time)

    def verify(self, rows: Sequence[dict], event: dict) -> List[Dict]:
        keywords = _event_keywords(event)
        verified: List[Dict] = []
        for row in rows:
            item = dict(row)
            title = str(item.get("title", "") or "")
            snippet = str(item.get("snippet", "") or "")
            url = str(item.get("url", "") or "")
            reasons = []
            if not url:
                reasons.append("missing_url")
            domain = _domain(url)
            if any(domain == d or domain.endswith("." + d) for d in LOW_VALUE_DOMAINS):
                reasons.append("low_value_domain")
            low_text = f"{title} {snippet}".lower()
            if any(p.lower() in low_text for p in LOW_VALUE_TITLE_PATTERNS):
                reasons.append("dictionary_or_translation_page")

            evidence_tokens = set(_tokens(f"{title} {snippet} {url}"))
            matched = [kw for kw in keywords if kw in evidence_tokens]
            score = len(matched) / max(len(keywords), 1)
            item["relevance_score"] = round(score, 4)
            item["matched_keywords"] = matched[:8]
            field_matches = []
            for group, keys in EVENT_FIELD_GROUPS.items():
                tokens = _field_tokens(event, keys)
                if tokens and any(token in evidence_tokens for token in tokens):
                    field_matches.append(group)
            item["field_matches"] = field_matches
            item["field_match_count"] = len(field_matches)
            if score < self.min_relevance_score:
                reasons.append("low_keyword_overlap")
            if len(field_matches) < self.min_field_matches:
                reasons.append("insufficient_field_match")
            if self.require_date_match and "date" not in field_matches:
                if "insufficient_field_match" not in reasons:
                    reasons.append("insufficient_field_match")
                reasons.append("missing_date_match")
            source_time = _parse_dt(
                item.get("published_at")
                or item.get("publish_time")
                or item.get("date_published")
                or item.get("source_time")
            )
            if self.require_source_time:
                if source_time is None:
                    item["source_time_status"] = "unknown"
                    reasons.append("source_time_unknown")
                elif self.anchor_time is not None and source_time > self.anchor_time:
                    item["source_time_status"] = "after_anchor"
                    reasons.append("source_time_after_anchor")
                else:
                    item["source_time_status"] = "known_before_anchor"
            elif source_time is not None and self.anchor_time is not None:
                item["source_time_status"] = "known_before_anchor" if source_time <= self.anchor_time else "after_anchor"
            item["accepted"] = not reasons
            item["accepted_reason"] = "keyword overlap and source URL passed" if item["accepted"] else ""
            item["rejected_reason"] = "; ".join(reasons)
            verified.append(item)
        return verified

    @staticmethod
    def accepted(rows: Iterable[dict]) -> List[dict]:
        return [dict(row) for row in rows if row.get("accepted")]
