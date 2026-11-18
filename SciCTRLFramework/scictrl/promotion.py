"""Promote current, audited whole-task rewrites to final Agent-facing tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .task_output import private_control_record


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def promote_rewritten_tasks(
    cases: list[dict[str, Any]],
    rewritten: list[dict[str, Any]],
    decision_document: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    """Write only rewrites whose source and diagnostic decision are current."""

    by_case = {case["case_id"]: case for case in cases}
    decisions = {
        item["case_id"]: item for item in decision_document.get("decisions", [])
    }
    admitted_public: list[dict[str, Any]] = []
    admitted_private: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for candidate in rewritten:
        internal_id = candidate["case_id"]
        case = by_case.get(internal_id)
        decision = decisions.get(internal_id)
        reasons: list[str] = []
        if case is None:
            reasons.append("unknown_internal_case")
        else:
            if candidate.get("source_schema_version") != case["schema_version"]:
                reasons.append("stale_schema")
            if candidate.get("source_flawed_hash") != case["provenance"]["flawed_hash"]:
                reasons.append("stale_source_hash")
        if not decision or not decision.get("diagnostic_pass"):
            reasons.append("surface_diagnostic_not_passed")
        if not decision or decision.get("source_current") is not True:
            reasons.append("source_freshness_not_passed")

        surface_task = candidate.get("surface_task")
        research_brief = candidate.get("research_brief")
        if not isinstance(surface_task, str) or not surface_task.strip():
            reasons.append("missing_surface_task")
        if not isinstance(research_brief, str) or not research_brief.strip():
            reasons.append("missing_research_brief")
        if case is not None and isinstance(surface_task, str) and isinstance(research_brief, str):
            surface_blob = (surface_task + "\n" + research_brief).lower()
            forbidden_surface_tokens = (
                "target_slot",
                "flaw_id",
                "gold_state",
                "accepted_patch",
                "canonical_reference_patch",
                "repair direction",
                "causal target",
                "allowed slots",
                "target slots",
                "editable slot",
                "potentially dependent artifacts",
                "control_degree",
                case["case_id"].lower(),
                *(
                    item["opportunity_id"].lower()
                    for item in case["oracle"]["revision_opportunities"]
                ),
            )
            if any(token in surface_blob for token in forbidden_surface_tokens):
                reasons.append("surface_leaks_private_control_metadata")

        if reasons:
            rejected.append({"internal_case_id": internal_id, "reasons": reasons})
            continue

        assert case is not None
        admitted_public.append(
            {
                "schema_version": case["schema_version"],
                "task_id": case["public_case_id"],
                "layer": case["layer"],
                "domain": case["domain"],
                "domain_name": case["domain_name"],
                "research_brief": research_brief.strip(),
                "task": surface_task.strip(),
            }
        )
        admitted_private.append(private_control_record(case))

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "admitted_modified_tasks.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in admitted_public),
        encoding="utf-8",
    )
    (output_root / "admitted_private_contracts.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in admitted_private),
        encoding="utf-8",
    )
    manifest = {
        "input_candidates": len(rewritten),
        "admitted_tasks": len(admitted_public),
        "rejected_or_stale": len(rejected),
        "rejections": rejected,
        "public_file": "admitted_modified_tasks.jsonl",
        "private_file": "admitted_private_contracts.jsonl",
        "meaning": "Only these tasks are final Agent-facing whole-task rewrites.",
    }
    (output_root / "admission_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal-cases", type=Path, required=True)
    parser.add_argument("--rewritten-candidates", type=Path, required=True)
    parser.add_argument("--diagnostic-decisions", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = promote_rewritten_tasks(
        _read_jsonl(args.internal_cases),
        _read_jsonl(args.rewritten_candidates),
        json.loads(args.diagnostic_decisions.read_text(encoding="utf-8")),
        args.output_root,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
