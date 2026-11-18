"""Materialize public revision tasks and private opportunity contracts."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable


def public_task_record(case: dict[str, Any]) -> dict[str, Any]:
    """Return the modified task consumed by an end-to-end AI Scientist.

    This canonical surface contains a genuine, validated revision opportunity
    but no direction label, causal target, gold state, or repair oracle.
    """

    return {
        "schema_version": case["schema_version"],
        "task_id": case["public_case_id"],
        "layer": case["layer"],
        "domain": case["domain"],
        "domain_name": case["domain_name"],
        "surface_status": "canonical_revision_task_pending_whole_task_rewrite",
        "task_input": copy.deepcopy(case["agent_input"]),
    }


def private_control_record(case: dict[str, Any]) -> dict[str, Any]:
    """Return evaluator-only constraints paired with a public modified task."""

    return {
        "schema_version": case["schema_version"],
        "task_id": case["public_case_id"],
        "internal_case_id": case["case_id"],
        "source_flawed_hash": case["provenance"]["flawed_hash"],
        "gold_project_id": case["provenance"]["gold_project_id"],
        "control_certificate": copy.deepcopy(
            case["oracle"]["control_certificate"]
        ),
    }


def validate_output_pair(
    case: dict[str, Any],
    public_task: dict[str, Any],
    private_contract: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if public_task["task_id"] != private_contract["task_id"]:
        errors.append("public/private task IDs differ")
    if public_task["task_id"] != case["public_case_id"]:
        errors.append("output task ID differs from case public ID")

    idea = (
        public_task["task_input"]["state"]["idea"]
        if case["layer"] == "layer1_state"
        else public_task["task_input"]["idea"]
    )
    if idea != case["flawed_state"]:
        errors.append("public task does not contain the modified scientific state")

    public_blob = json.dumps(public_task, ensure_ascii=False).lower()
    forbidden_keys = (
        "modification_constraint",
        "root_cardinality",
        "allowed_root_slots",
        "revision_opportunity_contract",
        "control_certificate",
        "certificate_hash",
        "opportunity_count",
        "causal_targets",
        "accepted_repair_equivalence_classes",
        "repair_acceptance",
        "potentially_dependent_artifacts",
        "allowed_slots",
        "target_slots",
        "repair_direction",
        "editable_slot",
        "internal_case_id",
        "gold_project_id",
        "source_flawed_hash",
    )
    if any(key in public_blob for key in forbidden_keys):
        errors.append("public task leaks private control metadata")
    if case["case_id"].lower() in public_blob:
        errors.append("public task leaks semantic internal case ID")
    for opportunity in case["oracle"]["revision_opportunities"]:
        if opportunity["opportunity_id"].lower() in public_blob:
            errors.append(
                "public task leaks opportunity id "
                f"{opportunity['opportunity_id']}"
            )
    return errors


def materialize_task_outputs(
    cases: Iterable[dict[str, Any]], output_root: Path
) -> dict[str, Any]:
    cases = list(cases)
    public_rows = [public_task_record(case) for case in cases]
    private_rows = [private_control_record(case) for case in cases]
    errors = [
        {"task_id": public["task_id"], "errors": pair_errors}
        for case, public, private in zip(cases, public_rows, private_rows)
        if (pair_errors := validate_output_pair(case, public, private))
    ]
    if errors:
        raise RuntimeError("Task output validation failed: " + json.dumps(errors))

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "canonical_modified_tasks.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in public_rows),
        encoding="utf-8",
    )
    (output_root / "private_control_contracts.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in private_rows),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": cases[0]["schema_version"] if cases else None,
        "canonical_modified_tasks": len(public_rows),
        "private_control_contracts": len(private_rows),
        "public_file": "canonical_modified_tasks.jsonl",
        "private_file": "private_control_contracts.jsonl",
        "public_meaning": (
            "Agent-facing scientific tasks containing validated revision "
            "opportunities; no repair direction is prescribed."
        ),
        "surface_status": "canonical candidates; M3 whole-task rewrite and admission still required",
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
