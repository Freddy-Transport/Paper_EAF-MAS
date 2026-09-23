#!/usr/bin/env python3
"""Generate/promote skills only through actual validation explanation replays."""
import argparse
from copy import deepcopy
import json
import hashlib
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.audit_inputs import load_bundle, new_output_dir, digest, verify_adapter_provenance, configuration_digest
from agents.autoskill_memory import build_replay_experience, evolve_skills
from agents.event_adapter import load_adapter
from agents.formal_workflow import run_numerical
from agents.skillbench import LocalJSONGenerator, evaluate_modes, METRICS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "config", "adapter", "output", "model"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    args = p.parse_args()
    manifest, records, memory = load_bundle(args.manifest, "val")
    config = json.loads(Path(args.config).read_text())
    adapter, adapter_config = load_adapter(args.adapter, config.get("device", "cpu"))
    verify_adapter_provenance(adapter_config, manifest, config)
    generator = LocalJSONGenerator(args.base_url, args.model)
    out = new_output_dir(args.output)
    by_id, experiences = {}, []
    for r in records:
        context, state = run_numerical(r, memory, config, adapter, adapter_config)
        by_id[r["request_id"]] = (r, context, state)
        experience = build_replay_experience(dict(request={"date": r["forecast_origin"]},
            evidence={"structured_future_events": r["events"], "structured_events": r["events"],
                      "historical_event_cases": context["historical_cases"]}, decision=state,
            numerical={"raw_forecast": r["raw_forecast"], "adjusted_forecast": state["final_adjusted_forecast"]}),
            split="val", experience_id=r["request_id"])
        experiences.append(experience)

    def replay(skill, supplied):
        mode = "no_skill" if skill is None else "full_autoskill"
        library = []
        if skill is not None:
            candidate = deepcopy(skill)
            candidate["promotion_status"] = {"promoted": True, "reason": "validation execution only"}
            library = [candidate]
        before, after, scored, audits, counts = {}, {}, [], [], []
        for e in supplied:
            ident = e["experience_id"]
            r, context, state = by_id[ident]
            before[ident] = digest(state)
            row = evaluate_modes(r, state, context["historical_cases"], library, generator, modes=[mode])[0]
            with (out/"validation_replays.jsonl").open("a") as f:
                f.write(json.dumps(dict(candidate_id=skill["skill_id"] if skill else None, **row), allow_nan=False)+"\n")
            if row["generation_status"] != "success":
                raise RuntimeError("Validation generation failed; incomplete replay cannot promote skills")
            after[ident] = row["numerical_state_digest"]
            scored.append(row["quality"])
            matched = context["matched"]
            audits.append({k: float(v[matched].mean()) if matched.any() else 0 for k,v in context["audit"].items()})
            counts.append(len(context["historical_cases"]))
        metrics = {key: float(np.mean([q[key] for q in scored])) for key in METRICS
                   if all(q.get(key) is not None for q in scored)}
        for key in ("source_validity_score", "geo_consistency_score", "temporal_alignment_score", "semantic_consistency_score"):
            metrics[key] = float(np.mean([a[key] for a in audits]))
        metrics["historical_memory_count"] = sum(counts)
        return dict(metrics=metrics, replay_ids=list(before), valid_sample_count=len(before),
                    numerical_digests_before=before, numerical_digests_after=after)

    result = evolve_skills(experiences, split="val", replay_evaluator=replay)
    result["input_manifest_digest"] = digest(manifest)
    result["model_identity"] = manifest["model_identity"]
    result["configuration_digest"] = configuration_digest(config)
    result["adapter_sha256"] = hashlib.sha256((Path(args.adapter)/"event_adapter.pt").read_bytes()).hexdigest()
    (out/"promoted_library.json").write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
