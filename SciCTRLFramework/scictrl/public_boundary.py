"""Fail-closed structural checks for Agent-visible SciCTRL cases."""

from __future__ import annotations

import re
from typing import Any

from .models import SCHEMA_VERSION


LAYER1_ARTIFACT_INVENTORY = (
    "research_idea",
    "research_plan",
    "run_manifest",
    "implementation",
    "raw_results",
    "analysis",
    "scientific_claims",
    "audit_log",
)


FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "control_degree",
        "gold_state",
        "flawed_state",
        "oracle",
        "evidence_oracle",
        "provenance",
        "control_certificate",
        "potentially_dependent_artifacts",
        "allowed_slots",
        "target_slots",
        "repair_direction",
        "editable_slot",
        "available_actions",
        "canonical_reference_patch",
        "repair_acceptance",
        "causal_targets",
    }
)

ROOT_KEYS = frozenset(
    {"schema_version", "case_id", "layer", "domain", "domain_name", "agent_input"}
)
AGENT_INPUT_KEYS = frozenset(
    {
        "task",
        "fixed_constraints",
        "available_tools",
        "task_contract",
        "research_interface",
        "required_output",
        "state",
        "idea",
        "environment_entrypoint",
    }
)
LAYER1_STATE_KEYS = frozenset(
    {
        "checkpoint_stage",
        "idea",
        "experiment_summary",
        "observations",
        "artifact_inventory",
    }
)


def _walk_keys(value: Any, path: str = "") -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            current = f"{path}.{key}" if path else key
            found.append((key, current))
            found.extend(_walk_keys(nested, current))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            found.extend(_walk_keys(nested, f"{path}[{index}]"))
    return found


def validate_public_boundary(public_case: Any) -> list[str]:
    """Return structural leakage/schema errors for one public case."""

    if not isinstance(public_case, dict):
        return ["public_case_not_object"]
    errors: list[str] = []
    if set(public_case) != ROOT_KEYS:
        errors.append("public_root_keys_not_exact")
    if public_case.get("schema_version") != SCHEMA_VERSION:
        errors.append("public_schema_version_stale")
    if not re.fullmatch(r"scictrl-[a-f0-9]{16}", str(public_case.get("case_id", ""))):
        errors.append("public_case_id_not_opaque")

    leaked = sorted(
        path
        for key, path in _walk_keys(public_case)
        if key in FORBIDDEN_PUBLIC_KEYS
    )
    if leaked:
        errors.append("private_or_scope_hint_keys:" + ",".join(leaked))

    agent_input = public_case.get("agent_input")
    if not isinstance(agent_input, dict):
        return errors + ["agent_input_not_object"]
    if not set(agent_input).issubset(AGENT_INPUT_KEYS):
        errors.append("agent_input_has_unknown_keys")
    common_required = {
        "task",
        "fixed_constraints",
        "available_tools",
        "task_contract",
        "research_interface",
        "required_output",
    }
    if not common_required.issubset(agent_input):
        errors.append("agent_input_missing_common_contract")

    layer = public_case.get("layer")
    if layer == "layer1_state":
        if (
            "state" not in agent_input
            or "idea" in agent_input
            or "environment_entrypoint" in agent_input
        ):
            errors.append("layer1_view_not_isolated")
        state = agent_input.get("state")
        if not isinstance(state, dict) or set(state) != LAYER1_STATE_KEYS:
            errors.append("layer1_state_keys_not_exact")
        elif state.get("artifact_inventory") != list(LAYER1_ARTIFACT_INVENTORY):
            errors.append("layer1_artifact_inventory_not_canonical")
    elif layer == "layer2_idea":
        if (
            "idea" not in agent_input
            or "environment_entrypoint" not in agent_input
            or "state" in agent_input
        ):
            errors.append("layer2_view_not_isolated")
    else:
        errors.append("unknown_layer")

    interface = agent_input.get("research_interface")
    required_interface = {"operations", "request_schema", "budget", "feedback_policy"}
    if not isinstance(interface, dict) or set(interface) != required_interface:
        errors.append("research_interface_contract_not_exact")
    return errors
