import json
import tempfile
import unittest
from pathlib import Path


class EvidenceQwenPlusTests(unittest.TestCase):
    def test_dashscope_native_search_results_are_extracted_as_evidence(self):
        from agents.evidence_research_agent import extract_dashscope_search_results

        response = {
            "output": {
                "search_info": {
                    "search_results": [
                        {
                            "title": "NYC DOT Plaza Program Event",
                            "url": "https://www.nyc.gov/html/dot/html/pedestrians/plaza-events.shtml",
                            "snippet": "Public event information for Herald Square plaza on May 19, 2023.",
                        }
                    ]
                }
            }
        }

        rows = extract_dashscope_search_results(response)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["url"], "https://www.nyc.gov/html/dot/html/pedestrians/plaza-events.shtml")
        self.assertEqual(rows[0]["source_agent"], "qwen_plus_native_search")

    def test_openai_compatible_response_keeps_summary_only(self):
        from agents.evidence_research_agent import EvidenceResearchAgent

        class FakeResponse:
            def __init__(self):
                msg = type("Msg", (), {})()
                choice = type("Choice", (), {})()
                msg.content = json.dumps({"event_summary": "YELL at Herald Square", "evidence": []})
                choice.message = msg
                self.choices = [choice]

            def model_dump(self):
                return {
                    "output": {
                        "search_info": {
                            "search_results": [
                                {
                                    "title": "YELL Herald Square event listing",
                                    "url": "https://www.timessquarenyc.org/whats-happening/yell-herald-square",
                                    "snippet": "YELL public event at Herald Square in NYC on May 19, 2023.",
                                }
                            ]
                        }
                    }
                }

        class FakeClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        return FakeResponse()

        with tempfile.TemporaryDirectory() as td:
            agent = EvidenceResearchAgent(
                cache_dir=Path(td),
                api_key="sk-unit-test-secret",
                client_factory=lambda **_: FakeClient(),
            )
            event = {
                "title": "YELL",
                "event_time": "2023-05-19 09:00:00",
                "location": "Herald Square",
                "impact_tier": "A",
                "relevance_score": 0.9,
                "channel_name": "A013__49_St",
            }

            result = agent.research_events([event], "2023-05-18 10:00:00", ["A013__49_St"], max_events=1)

            self.assertEqual(result.accepted, [])
            self.assertEqual(result.rejected, [])
            self.assertEqual(len(result.summaries), 1)
            self.assertIn("YELL at Herald Square", result.summaries[0]["event_summary"])

    def test_gate_keeps_only_major_relevant_unique_events(self):
        from agents.evidence_research_agent import select_qwenplus_events

        events = [
            {
                "title": "YELL",
                "event_time": "2023-05-19 09:00:00",
                "location": "Herald Square",
                "impact_tier": "A",
                "relevance_score": 0.91,
                "channel_name": "A013__49_St",
                "distance_m": 240.0,
            },
            {
                "title": "YELL",
                "event_time": "2023-05-19 09:00:00",
                "location": "Herald Square",
                "impact_tier": "A",
                "relevance_score": 0.90,
                "channel_name": "N056__50_St",
                "distance_m": 497.0,
            },
            {
                "title": "Routine setup",
                "event_time": "2023-05-19 10:00:00",
                "location": "NYC",
                "impact_tier": "C",
                "relevance_score": 0.99,
                "channel_name": "A013__49_St",
                "distance_m": 100.0,
            },
            {
                "title": "Distant parade",
                "event_time": "2023-05-19 10:00:00",
                "location": "Queens",
                "impact_tier": "A",
                "relevance_score": 0.95,
                "distance_m": 2500.0,
            },
        ]

        selected = select_qwenplus_events(events, "2023-05-18 10:00:00", 192, max_events=3, min_score=0.75)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["title"], "YELL")
        self.assertEqual(selected[0]["channel_name"], "A013__49_St")

    def test_qwenplus_cache_and_key_redaction(self):
        from agents.evidence_research_agent import EvidenceResearchAgent

        class FakeClient:
            calls = 0

            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        FakeClient.calls += 1
                        self = type("Response", (), {})()
                        msg = type("Msg", (), {})()
                        choice = type("Choice", (), {})()
                        msg.content = json.dumps(
                            {
                                "event_summary": "YELL at Herald Square",
                                "evidence": [
                                    {
                                        "title": "NYC Plaza Event Permit",
                                        "url": "https://example.org/yell",
                                        "snippet": "YELL plaza event at Herald Square on May 19, 2023.",
                                    }
                                ],
                            }
                        )
                        choice.message = msg
                        self.choices = [choice]
                        return self

        with tempfile.TemporaryDirectory() as td:
            agent = EvidenceResearchAgent(
                cache_dir=Path(td),
                api_key="sk-unit-test-secret",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                model="qwen-plus",
                client_factory=lambda **_: FakeClient(),
            )
            event = {
                "title": "YELL",
                "event_time": "2023-05-19 09:00:00",
                "location": "Herald Square",
                "impact_tier": "A",
                "relevance_score": 0.9,
                "channel_name": "A013__49_St",
            }
            first = agent.research_events([event], "2023-05-18 10:00:00", ["A013__49_St"], max_events=1)
            second = agent.research_events([event], "2023-05-18 10:00:00", ["A013__49_St"], max_events=1)

            self.assertEqual(FakeClient.calls, 1)
            self.assertEqual(len(first.accepted), 0)
            self.assertEqual(len(first.rejected), 0)
            self.assertEqual(len(first.summaries), 1)
            self.assertEqual(len(second.accepted), 0)
            self.assertEqual(len(second.rejected), 0)
            self.assertEqual(len(second.summaries), 1)
            self.assertIn("YELL at Herald Square", first.summaries[0]["summary"])
            all_text = "\n".join(p.read_text(encoding="utf-8") for p in Path(td).glob("*.json"))
            self.assertNotIn("sk-unit-test-secret", all_text)
            self.assertIn("qwen_plus", all_text)
            self.assertIn("summary_only_v", all_text)

    def test_qwenplus_summary_only_ignores_url_evidence(self):
        from agents.evidence_research_agent import EvidenceResearchAgent

        class FakeClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        prompt = kwargs["messages"][1]["content"]
                        self = type("Response", (), {})()
                        msg = type("Msg", (), {})()
                        choice = type("Choice", (), {})()
                        msg.content = json.dumps(
                            {
                                "event_summary": "TSQ LIVE is a Times Square plaza event.",
                                "event_relevance_to_station": "The event is adjacent to Times Sq-42 St channels.",
                                "expected_ridership_effect": "Demand may increase near arrival and departure periods.",
                                "uncertainty": "Magnitude is uncertain and should be bounded.",
                                "summary_used_for_explanation": True,
                                "evidence": [
                                    {
                                        "title": "Should be ignored",
                                        "url": "https://example.org/ignored",
                                        "snippet": "URL rows are not part of the formal summary-only workflow.",
                                    }
                                ],
                            }
                        )
                        choice.message = msg
                        self.choices = [choice]
                        self._prompt = prompt
                        return self

        with tempfile.TemporaryDirectory() as td:
            agent = EvidenceResearchAgent(
                cache_dir=Path(td),
                api_key="sk-unit-test-secret",
                client_factory=lambda **_: FakeClient(),
            )
            event = {
                "title": "TSQ LIVE: 45/46 Plaza Programming",
                "event_time": "2023-06-20 17:00:00",
                "location": "Times Square",
                "impact_tier": "A",
                "relevance_score": 0.9,
                "channel_name": "N060__Times_Sq_42_St",
            }

            result = agent.research_events([event], "2023-06-19 10:00:00", ["N060__Times_Sq_42_St"], max_events=1)

            self.assertEqual(result.accepted, [])
            self.assertEqual(result.rejected, [])
            self.assertEqual(result.stats["accepted_evidence"], 0)
            self.assertEqual(result.stats["rejected_evidence"], 0)
            self.assertEqual(result.stats["summary_count"], 1)
            summary = result.summaries[0]
            self.assertTrue(summary["summary_used_for_explanation"])
            self.assertIn("Times Square plaza", summary["event_summary"])
            self.assertIn("Magnitude is uncertain", summary["uncertainty"])
            cache_text = "\n".join(p.read_text(encoding="utf-8") for p in Path(td).glob("*.json"))
            self.assertIn('"evidence": []', cache_text)
            self.assertIn("summary_only", cache_text)

    def test_qwenplus_force_refresh_bypasses_existing_cache(self):
        from agents.evidence_research_agent import EvidenceResearchAgent

        class FakeClient:
            calls = 0

            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        FakeClient.calls += 1
                        self = type("Response", (), {})()
                        msg = type("Msg", (), {})()
                        choice = type("Choice", (), {})()
                        msg.content = json.dumps(
                            {
                                "event_summary": f"fresh summary {FakeClient.calls}",
                                "evidence": [
                                    {
                                        "title": "TSQ LIVE Plaza Programming",
                                        "url": "https://example.org/tsq-live",
                                        "snippet": "TSQ LIVE event at Times Square on June 20, 2023.",
                                    }
                                ],
                            }
                        )
                        choice.message = msg
                        self.choices = [choice]
                        return self

        event = {
            "title": "TSQ LIVE: 45/46 Plaza Programming",
            "event_time": "2023-06-20 17:00:00",
            "location": "Times Square",
            "impact_tier": "A",
            "relevance_score": 0.9,
            "channel_name": "N060__Times_Sq_42_St",
        }
        with tempfile.TemporaryDirectory() as td:
            cached = EvidenceResearchAgent(
                cache_dir=Path(td),
                api_key="sk-unit-test-secret",
                client_factory=lambda **_: FakeClient(),
            )
            cached.research_events([event], "2023-06-19 10:00:00", ["N060__Times_Sq_42_St"], max_events=1)
            cached.research_events([event], "2023-06-19 10:00:00", ["N060__Times_Sq_42_St"], max_events=1)
            self.assertEqual(FakeClient.calls, 1)

            refreshed = EvidenceResearchAgent(
                cache_dir=Path(td),
                api_key="sk-unit-test-secret",
                force_refresh=True,
                client_factory=lambda **_: FakeClient(),
            )
            refreshed.research_events([event], "2023-06-19 10:00:00", ["N060__Times_Sq_42_St"], max_events=1)
            self.assertEqual(FakeClient.calls, 2)

    def test_evidence_verifier_rejects_irrelevant_and_url_missing_sources(self):
        from agents.evidence_verifier import EvidenceVerifier

        event = {"title": "YELL", "event_time": "2023-05-19 09:00:00", "location": "Herald Square"}
        rows = [
            {"title": "yell是什么意思", "url": "https://dict.example/yell", "snippet": "dictionary definition"},
            {"title": "YELL at Herald Square", "url": "", "snippet": "summary without source URL"},
            {
                "title": "NYC Plaza Event Permit",
                "url": "https://example.org/permit",
                "snippet": "YELL plaza event at Herald Square on May 19, 2023.",
            },
        ]

        verified = EvidenceVerifier().verify(rows, event)

        accepted = [r for r in verified if r["accepted"]]
        rejected = [r for r in verified if not r["accepted"]]
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["url"], "https://example.org/permit")
        self.assertEqual(len(rejected), 2)

    def test_citation_verifier_requires_url_and_multi_field_match(self):
        from agents.evidence_verifier import EvidenceVerifier

        event = {
            "title": "TSQ LIVE: 45/46 Plaza Programming",
            "event_time": "2023-06-20 17:00:00",
            "location": "43/44 Broadway Pedestrian Plaza Times Square",
            "venue_name": "Times Square Plaza",
            "channel_name": "A013__49_St",
        }
        rows = [
            {
                "title": "TSQ LIVE Plaza Programming in Times Square",
                "url": "https://www.timessquarenyc.org/whats-happening/tsq-live",
                "snippet": "TSQ LIVE programming at Times Square plaza near Broadway on June 20, 2023.",
            },
            {
                "title": "TSQ LIVE music video",
                "url": "https://example.com/music",
                "snippet": "A generic TSQ page with no date or Times Square venue information.",
            },
            {
                "title": "TSQ LIVE Plaza Programming",
                "url": "",
                "snippet": "Times Square June 20 event but no URL.",
            },
        ]

        verified = EvidenceVerifier(min_relevance_score=0.20).verify(rows, event)
        accepted = [r for r in verified if r["accepted"]]
        rejected = [r for r in verified if not r["accepted"]]

        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["url"], "https://www.timessquarenyc.org/whats-happening/tsq-live")
        self.assertGreaterEqual(accepted[0]["field_match_count"], 2)
        self.assertEqual(len(rejected), 2)
        self.assertTrue(any("insufficient_field_match" in r["rejected_reason"] for r in rejected))

    def test_source_time_gate_rejects_post_anchor_or_unknown_sources_when_required(self):
        from agents.evidence_verifier import EvidenceVerifier

        event = {
            "title": "TSQ LIVE: 45/46 Plaza Programming",
            "event_time": "2023-06-20 17:00:00",
            "location": "Times Square",
        }
        rows = [
            {
                "title": "TSQ LIVE Plaza Programming in Times Square",
                "url": "https://www.timessquarenyc.org/tsq-live",
                "snippet": "TSQ LIVE event at Times Square on June 20, 2023.",
                "published_at": "2023-06-18 12:00:00",
            },
            {
                "title": "TSQ LIVE Plaza Programming in Times Square",
                "url": "https://news.example.com/recap",
                "snippet": "TSQ LIVE event at Times Square on June 20, 2023.",
                "published_at": "2023-06-21 12:00:00",
            },
            {
                "title": "TSQ LIVE Plaza Programming in Times Square",
                "url": "https://unknown.example.com/tsq",
                "snippet": "TSQ LIVE event at Times Square on June 20, 2023.",
            },
        ]

        verified = EvidenceVerifier(
            min_relevance_score=0.20,
            require_source_time=True,
            anchor_time="2023-06-19 10:00:00",
        ).verify(rows, event)

        accepted = [r for r in verified if r["accepted"]]
        rejected = [r for r in verified if not r["accepted"]]
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["source_time_status"], "known_before_anchor")
        self.assertEqual(len(rejected), 2)
        self.assertTrue(any("source_time_after_anchor" in r["rejected_reason"] for r in rejected))
        self.assertTrue(any("source_time_unknown" in r["rejected_reason"] for r in rejected))

    def test_event_retrieval_gate_rejects_low_value_unrelated_sources(self):
        from agents.event_retrieval_agent import EventRetrievalAgent

        event = {
            "title": "Violife Cheese",
            "event_time": "2023-06-22 08:00:00",
            "location": "Times Square",
        }
        row = {
            "query": "q",
            "url": "https://www.ddooo.com/",
            "title": "多多软件站-提供绿色软件下载",
            "snippet": "软件下载平台",
            "retrieved_at": "2026-06-08T00:00:00Z",
            "cache_path": "cache.json",
            "status": "ok",
        }

        retriever = EventRetrievalAgent("/tmp/event-retrieval-unit", enabled=False)
        verified = retriever.verifier or __import__("agents.evidence_verifier", fromlist=["EvidenceVerifier"]).EvidenceVerifier(
            require_source_time=True,
            anchor_time="2023-06-19 10:00:00",
        )
        rows = verified.verify([row], event)

        self.assertFalse(rows[0]["accepted"])
        self.assertIn("low_value_domain", rows[0]["rejected_reason"])


if __name__ == "__main__":
    unittest.main()
