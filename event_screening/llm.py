from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional

from pydantic import ValidationError

from event_screening.models import MajorEventDecision, PhysicalEvent

VALID_TYPES = "sports / concert / festival / parade / service_disruption / conference / ceremony / street_event / other / none"
LLM_BACKEND_NAMES = {"hf_api", "isolated_transformers"}


class RunnerUnavailable(RuntimeError):
    """Raised when a requested LLM backend cannot be used safely."""


def build_llm_prompt(event: PhysicalEvent, traffic_evidence: Dict[str, Any]) -> str:
    affected = ", ".join(event.affected_channels[:12])
    evidence_json = json.dumps(traffic_evidence, ensure_ascii=False, indent=2)
    return f"""You are a NYC subway ridership analyst screening permit/event records.

Decide whether this physical event is a truly large event likely to affect subway ridership.
Do not classify an event as major only because its permit event_type is "Special Event".
Do not keep routine youth sports, farmers markets, filming/setup/breakdown, lawn/park closures, private parties, or tiny gatherings unless traffic evidence clearly shows an abnormal subway ridership change.
Use both your general knowledge of NYC venues/events and the provided traffic evidence.

Event:
- id: {event.event_id}
- title: {event.title}
- event_time: {event.event_time}
- event_type: {event.event_type}
- location: {event.location}
- duplicate_station_matches: {event.duplicate_count}
- min_distance_m: {event.min_distance_m}
- affected_channels: {affected}
- content: {event.content[:1200]}

Traffic evidence:
{evidence_json}

Respond ONLY as JSON with exactly these fields:
{{
  "is_major_event": boolean,
  "major_event_type": one of [{VALID_TYPES}],
  "crowd_scale": one of ["none", "small", "medium", "large", "mega"],
  "transit_impact_likelihood": number from 0 to 1,
  "expected_direction": one of ["increase", "decrease", "neutral", "unknown"],
  "affected_scope": one of ["none", "station", "multi_station", "borough", "citywide", "unknown"],
  "traffic_evidence_used": boolean,
  "keep_for_modeling": boolean,
  "reason": short string
}}
"""


def _extract_json(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        return match.group(0)
    return text


def parse_llm_decision(text: str) -> MajorEventDecision:
    try:
        payload = json.loads(_extract_json(text))
        decision = MajorEventDecision(**payload)
        decision.raw_response = text
        decision.keep_for_modeling = decision.modeling_eligible() if decision.keep_for_modeling else False
        return decision
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
        return MajorEventDecision(needs_review=True, reason=f"invalid_llm_output: {type(exc).__name__}", raw_response=text)


class HeuristicLLMRunner:
    """Deterministic fallback runner with the same schema as the LLM output."""

    def classify(self, event: PhysicalEvent, traffic_evidence: Dict[str, Any], rule_category: str = "other") -> MajorEventDecision:
        text = " ".join([event.title, event.event_type, event.location, event.content]).lower()
        z = abs(float(traffic_evidence.get("z_score", 0.0) or 0.0))
        actual_sum = float(traffic_evidence.get("actual_sum", 0.0) or 0.0)
        low_volume = bool(traffic_evidence.get("low_volume_warning", False))
        noise = any(k in text for k in [
            "youth", "little league", "soccer - non", "soccer -regulation", "softball", "farmers market",
            "greenmarket", "filming", "setup", "breakdown", "lawn closure", "party", "picnic", "miscellaneous",
        ])
        pro_sports = any(k in text for k in ["yankees", "mets", "knicks", "nets", "rangers", "islanders", "liberty"])
        major_venue = any(k in text for k in ["yankee stadium", "barclays", "madison square garden", "msg", "citi field", "usta"])
        concert_like = any(k in text for k in ["concert", "tour", "music festival", "live nation"])
        likelihood = 0.10
        event_type = "none"
        scale = "small"
        direction = "unknown"
        scope = "station"
        reason = "heuristic fallback classification"
        if rule_category == "service_disruption" or any(k in text for k in ["service disruption", "station closure", "closed"]):
            event_type = "service_disruption"; scale = "large"; direction = "decrease"; likelihood = 0.75; reason += "; explicit disruption"
        elif any(k in text for k in ["parade", "marathon"]):
            event_type = "parade"; scale = "mega"; direction = "increase"; likelihood = 0.80; scope = "multi_station"; reason += "; parade/marathon"
        elif major_venue and (pro_sports or concert_like or " vs " in text or "game" in text):
            event_type = "sports" if (pro_sports or " vs " in text or "game" in text) else "concert"
            scale = "large"; direction = "increase"; likelihood = 0.72; reason += "; major venue/pro event"
        elif z >= 4.0 and actual_sum >= 5000 and not low_volume and not noise:
            event_type = "other"; scale = "large"; direction = "increase"; likelihood = 0.66; reason += "; strong traffic anomaly"
        elif noise:
            likelihood = 0.20; reason += "; ordinary permit/noise demoted"
        decision = MajorEventDecision(
            is_major_event=likelihood >= 0.65,
            major_event_type=event_type,
            crowd_scale=scale if likelihood >= 0.65 else "none",
            transit_impact_likelihood=likelihood,
            expected_direction=direction,
            affected_scope=scope if likelihood >= 0.65 else "none",
            traffic_evidence_used=bool(traffic_evidence.get("available", False)),
            keep_for_modeling=likelihood >= 0.65 and (scale in {"large", "mega"} or event_type == "service_disruption"),
            reason=reason,
        )
        return decision


class HFApiRunner:
    """Hugging Face serverless/provider API runner.

    This runner never silently falls back to heuristic labels. If the API output
    is invalid after one retry, the returned decision is marked needs_review.
    """

    def __init__(
        self,
        model_name: str,
        token: Optional[str] = None,
        token_env: str = "HF_TOKEN",
        endpoint: Optional[str] = None,
        max_new_tokens: int = 256,
        timeout: int = 60,
        request_fn: Optional[Any] = None,
    ):
        self.model_name = model_name
        self.token = token or os.environ.get(token_env)
        if not self.token:
            raise RunnerUnavailable(f"{token_env} is not set")
        self.endpoint = endpoint or f"https://api-inference.huggingface.co/models/{model_name}"
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout
        self.request_fn = request_fn

    def _request(self, prompt: str) -> str:
        request_fn = self.request_fn
        if request_fn is None:
            import requests
            request_fn = requests.post
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": 0.0,
                "return_full_text": False,
            },
        }
        response = request_fn(self.endpoint, headers=headers, json=payload, timeout=self.timeout)
        if hasattr(response, "raise_for_status"):
            response.raise_for_status()
        data = response.json() if hasattr(response, "json") else response
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                return str(first.get("generated_text") or first.get("summary_text") or first.get("text") or "")
            return str(first)
        if isinstance(data, dict):
            if "generated_text" in data:
                return str(data["generated_text"])
            if "choices" in data and data["choices"]:
                choice = data["choices"][0]
                message = choice.get("message", {}) if isinstance(choice, dict) else {}
                return str(message.get("content") or choice.get("text") or "")
            if "error" in data:
                raise RunnerUnavailable(f"Hugging Face API error: {data['error']}")
        return str(data)

    def classify(self, event: PhysicalEvent, traffic_evidence: Dict[str, Any], rule_category: str = "other") -> MajorEventDecision:
        prompt = build_llm_prompt(event, traffic_evidence)
        text = self._request(prompt)
        parsed = parse_llm_decision(text)
        if not parsed.needs_review:
            return parsed
        retry_prompt = prompt + "\nYour previous answer was invalid. Return one valid JSON object only."
        retry_text = self._request(retry_prompt)
        retry = parse_llm_decision(retry_text)
        if retry.needs_review:
            retry.reason = "invalid_hf_api_output_after_retry"
            retry.raw_response = retry_text
        return retry


class IsolatedTransformersRunner:
    """Run local Transformers from an isolated venv, not from the MOMENT env."""

    def __init__(
        self,
        model_name: str,
        env_dir: str = "/root/autodl-tmp/llm_screening_env",
        hf_cache: str = "/root/autodl-tmp/hf_cache",
        max_new_tokens: int = 256,
        timeout: int = 180,
        require_ready: bool = True,
    ):
        self.model_name = model_name
        self.env_dir = env_dir
        self.hf_cache = hf_cache
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout
        if require_ready and not os.path.exists(self.python_path):
            raise RunnerUnavailable(
                "isolated transformers env is not ready; prepare it with: "
                + " && ".join(self.install_commands())
            )

    @property
    def python_path(self) -> str:
        return os.path.join(self.env_dir, "bin", "python")

    def install_commands(self) -> List[str]:
        pip = os.path.join(self.env_dir, "bin", "pip")
        return [
            f"python3 -m venv {self.env_dir}",
            f"mkdir -p {self.hf_cache}",
            f"{pip} install --upgrade 'torch>=2.1' 'transformers>=4.48,<5' accelerate safetensors sentencepiece protobuf requests",
        ]

    def _generate(self, prompt: str) -> str:
        script = r'''
import os
import sys

model_name = sys.argv[1]
max_new_tokens = int(sys.argv[2])
prompt = sys.stdin.read()
os.environ.setdefault("HF_HOME", sys.argv[3])
os.environ.setdefault("HF_HUB_CACHE", sys.argv[3])
os.environ.setdefault("TRANSFORMERS_CACHE", sys.argv[3])

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True)
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
print(tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True))
'''
        env = os.environ.copy()
        env["HF_HOME"] = self.hf_cache
        env["HF_HUB_CACHE"] = self.hf_cache
        env["TRANSFORMERS_CACHE"] = self.hf_cache
        proc = subprocess.run(
            [self.python_path, "-c", script, self.model_name, str(self.max_new_tokens), self.hf_cache],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=self.timeout,
            env=env,
        )
        if proc.returncode != 0:
            raise RunnerUnavailable(proc.stderr.strip() or "isolated transformers generation failed")
        return proc.stdout

    def classify(self, event: PhysicalEvent, traffic_evidence: Dict[str, Any], rule_category: str = "other") -> MajorEventDecision:
        prompt = build_llm_prompt(event, traffic_evidence)
        text = self._generate(prompt)
        parsed = parse_llm_decision(text)
        if not parsed.needs_review:
            return parsed
        retry_text = self._generate(prompt + "\nYour previous answer was invalid. Return one valid JSON object only.")
        retry = parse_llm_decision(retry_text)
        if retry.needs_review:
            retry.reason = "invalid_isolated_transformers_output_after_retry"
            retry.raw_response = retry_text
        return retry


class TransformersRunner:
    def __init__(self, model_name: str, device: str = "auto", max_new_tokens: int = 256):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map=device, trust_remote_code=True)
        self.max_new_tokens = max_new_tokens

    def classify(self, event: PhysicalEvent, traffic_evidence: Dict[str, Any], rule_category: str = "other") -> MajorEventDecision:
        prompt = build_llm_prompt(event, traffic_evidence)
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        gen = self.tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        parsed = parse_llm_decision(gen)
        if parsed.needs_review:
            return HeuristicLLMRunner().classify(event, traffic_evidence, rule_category)
        return parsed


class VLLMRunner:
    def __init__(self, model_name: str, max_new_tokens: int = 256):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=model_name, trust_remote_code=True, dtype="bfloat16")
        self.sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    def classify(self, event: PhysicalEvent, traffic_evidence: Dict[str, Any], rule_category: str = "other") -> MajorEventDecision:
        prompt = build_llm_prompt(event, traffic_evidence)
        outputs = self.llm.generate([prompt], self.sampling)
        text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
        parsed = parse_llm_decision(text)
        if parsed.needs_review:
            return HeuristicLLMRunner().classify(event, traffic_evidence, rule_category)
        return parsed


def make_runner(backend: str, model_name: str):
    backend = backend.lower()
    if backend == "heuristic":
        return HeuristicLLMRunner(), "heuristic"
    if backend in {"auto", "hf_api"}:
        try:
            return HFApiRunner(model_name), "hf_api"
        except Exception as exc:
            if backend == "hf_api":
                raise
            print(f"[event_screening] hf_api unavailable, falling back: {exc}", flush=True)
    if backend in {"auto", "isolated_transformers"}:
        try:
            return IsolatedTransformersRunner(model_name), "isolated_transformers"
        except Exception as exc:
            if backend == "isolated_transformers":
                raise
            print(f"[event_screening] isolated_transformers unavailable, falling back: {exc}", flush=True)
    if backend == "transformers":
        return TransformersRunner(model_name), "transformers"
    if backend == "vllm":
        raise RunnerUnavailable("vLLM backend is disabled for safe event screening; use hf_api or isolated_transformers")
    return HeuristicLLMRunner(), "heuristic"
