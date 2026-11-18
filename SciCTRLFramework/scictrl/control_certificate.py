"""Executable matched-control certificates for scientific revision cases.

A certificate states exactly what is controlled by the benchmark constructor:
the held-constant conditions, scientific intervention, violated obligation,
causal root, propagation surface, protected task identity, accepted repair
family, evidence, and hidden checks. It does *not* restrict the Agent to
editing only the causal root.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any


def _leaf_paths(document: dict[str, Any], prefix: str = "") -> set[str]:
    paths: set[str] = set()
    for key, value in document.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            paths.update(_leaf_paths(value, path))
        else:
            paths.add(path)
    return paths


def _set_path(document: dict[str, Any], path: str, value: Any) -> None:
    cursor: Any = document
    parts = path.split(".")
    for part in parts[:-1]:
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _get_path(document: dict[str, Any], path: str) -> Any:
    cursor: Any = document
    for part in path.split("."):
        cursor = cursor[part]
    return cursor


def _digest(value: dict[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_control_certificate(case: dict[str, Any]) -> dict[str, Any]:
    """Build the evaluator-owned control proof for one internal case."""

    oracle = case["oracle"]
    root_positions = sorted(oracle["causal_targets"])
    downstream_positions = sorted(oracle["propagation_obligations"])
    protected_semantic = sorted(oracle["task_identity_invariants"])
    protected_contract = sorted(
        f"fixed_constraint::{key}" for key in oracle["fixed_commitments"]
    )
    semantic_slots = _leaf_paths(case["gold_state"])
    free_positions = sorted(
        semantic_slots
        - set(root_positions)
        - set(protected_semantic)
        - set(downstream_positions)
    )
    controlled_positions = sorted(oracle["construction_preserved_slots"])
    controlled_values = {
        path: _get_path(case["gold_state"], path)
        for path in controlled_positions
    }

    opportunity_rows: list[dict[str, Any]] = []
    dependency_edges: list[dict[str, str]] = []
    for opportunity in oracle["revision_opportunities"]:
        target = opportunity["causal_target"]
        evidence_ids = sorted(
            evidence_id
            for evidence_id, spec in case["evidence_oracle"].items()
            if spec["supports_slot"] == target
        )
        propagation = sorted(opportunity["propagation_targets"])
        opportunity_rows.append(
            {
                "opportunity_id": opportunity["opportunity_id"],
                "analysis_tag": opportunity["analysis_tag"],
                "violated_scientific_obligations": list(
                    opportunity["violated_obligations"]
                ),
                "causal_root": target,
                "evidence_ids": evidence_ids,
                "propagation_targets": propagation,
                "repair_contract_id": oracle["opportunity_contracts"][target][
                    "contract_id"
                ],
                "causal_justification": opportunity["causal_justification"],
            }
        )
        dependency_edges.extend(
            {
                "source": target,
                "target": dependent,
                "relation": "repair_invalidates_downstream_artifact",
            }
            for dependent in propagation
        )

    certificate = {
        "certificate_version": "2.0",
        "case_id": case["case_id"],
        "assurance_level": "structural_executable",
        "control_semantics": (
            "The benchmark controls the causal repair obligation and acceptance "
            "tests; it does not expose or enforce an Agent-side allowed-edit list."
        ),
        "uniqueness_scope": (
            "Validated against declared repair equivalence classes, competing "
            "non-repairs, shortcuts, and task-identity changes; not a proof over "
            "all possible open-ended scientific interventions."
        ),
        "matched_pair_control": {
            "definition": (
                "Gold and flawed states are a matched pair: every declared "
                "semantic position is held constant except the registered "
                "scientific intervention positions."
            ),
            "controlled_semantic_positions": controlled_positions,
            "controlled_semantic_values_hash": _digest(controlled_values),
            "intervention_positions": root_positions,
            "gold_state_hash": case["provenance"]["gold_hash"],
            "flawed_state_hash": case["provenance"]["flawed_hash"],
            "fixed_public_conditions": {
                "fixed_constraints": copy.deepcopy(
                    case["agent_input"]["fixed_constraints"]
                ),
                "available_tools": list(
                    case["agent_input"]["available_tools"]
                ),
                "task_contract": copy.deepcopy(
                    case["agent_input"]["task_contract"]
                ),
                "research_interface": copy.deepcopy(
                    case["agent_input"]["research_interface"]
                ),
                "required_output": copy.deepcopy(
                    case["agent_input"]["required_output"]
                ),
            },
            "agent_action_policy": (
                "Open research actions within protected task identity; no "
                "Agent-visible target-slot or allowed-edit list."
            ),
        },
        "scientific_interventions": opportunity_rows,
        "semantic_position_roles": {
            "root_positions": root_positions,
            "downstream_positions": downstream_positions,
            "protected_semantic_positions": protected_semantic,
            "protected_contract_positions": protected_contract,
            "free_positions": free_positions,
            "dependency_edges": dependency_edges,
        },
        "repair_acceptance": copy.deepcopy(oracle["opportunity_contracts"]),
        "construction_reference_patch": copy.deepcopy(
            oracle["canonical_reference_patch"]
        ),
        "evidence_control": {
            "required_evidence_ids": list(oracle["evidence_obligations"]),
            "private_recipe_count": len(case["evidence_oracle"]),
            "private_recipe_hashes": {
                evidence_id: _digest(spec["probe_recipe"])
                for evidence_id, spec in case["evidence_oracle"].items()
            },
            "generic_operations": list(
                case["agent_input"]["research_interface"]["operations"]
            ),
            "request_budget": copy.deepcopy(
                case["agent_input"]["research_interface"]["budget"]
            ),
            "task_specific_action_menu_exposed": False,
        },
        "task_identity_control": {
            "semantic_invariants": protected_semantic,
            "fixed_commitments": copy.deepcopy(oracle["fixed_commitments"]),
        },
        "evaluation_control": {
            "propagation_obligations": downstream_positions,
            "verification_obligations": list(
                oracle["verification_obligations"]
            ),
            "forbidden_shortcuts": list(oracle["forbidden_shortcuts"]),
            "hidden_tests": list(oracle["hidden_tests"]),
            "no_revision_expected": bool(oracle["no_revision_expected"]),
            "additional_nonidentity_edits": (
                "allowed_but_counted_against_revision_focus"
            ),
        },
    }
    if "executable_control" in oracle:
        certificate["assurance_level"] = "execution_backed_fixture"
        certificate["executable_control"] = copy.deepcopy(
            oracle["executable_control"]
        )
    certificate["certificate_hash"] = _digest(certificate)
    return certificate


def validate_control_certificate(case: dict[str, Any]) -> dict[str, Any]:
    """Validate certificate integrity, coverage, position roles, and closure."""

    certificate = case["oracle"].get("control_certificate")
    if not isinstance(certificate, dict):
        return {
            "case_id": case["case_id"],
            "valid": False,
            "checks": {"certificate_present": False},
            "errors": ["control certificate is missing"],
        }

    required_sections = {
        "certificate_version",
        "case_id",
        "control_semantics",
        "matched_pair_control",
        "scientific_interventions",
        "semantic_position_roles",
        "repair_acceptance",
        "construction_reference_patch",
        "evidence_control",
        "task_identity_control",
        "evaluation_control",
        "certificate_hash",
    }
    missing_sections = sorted(required_sections - set(certificate))
    nested_requirements = {
        "matched_pair_control": {
            "controlled_semantic_positions",
            "controlled_semantic_values_hash",
            "intervention_positions",
            "gold_state_hash",
            "flawed_state_hash",
            "fixed_public_conditions",
        },
        "semantic_position_roles": {
            "root_positions",
            "downstream_positions",
            "protected_semantic_positions",
            "free_positions",
        },
        "evidence_control": {"required_evidence_ids", "private_recipe_hashes"},
        "evaluation_control": {
            "propagation_obligations",
            "verification_obligations",
            "forbidden_shortcuts",
        },
    }
    malformed_sections = []
    for name, required in nested_requirements.items():
        section = certificate.get(name)
        if not isinstance(section, dict) or not required.issubset(section):
            malformed_sections.append(name)
    if missing_sections or malformed_sections:
        errors = [f"missing_section::{name}" for name in missing_sections]
        errors.extend(f"malformed_section::{name}" for name in malformed_sections)
        return {
            "case_id": case.get("case_id"),
            "valid": False,
            "checks": {"certificate_structure_valid": False},
            "errors": errors,
        }

    oracle = case["oracle"]
    positions = certificate["semantic_position_roles"]
    roots = set(positions["root_positions"])
    downstream = set(positions["downstream_positions"])
    protected = set(positions["protected_semantic_positions"])
    free = set(positions["free_positions"])
    all_semantic = _leaf_paths(case["gold_state"])
    known_downstream = downstream & all_semantic

    unhashed = copy.deepcopy(certificate)
    stored_hash = unhashed.pop("certificate_hash", None)
    opportunity_targets = {
        row["causal_root"] for row in certificate["scientific_interventions"]
    }
    evidence_ids = set(certificate["evidence_control"]["required_evidence_ids"])

    repaired = copy.deepcopy(case["flawed_state"])
    for path, value in certificate["construction_reference_patch"].items():
        _set_path(repaired, path, value)

    matched = certificate["matched_pair_control"]
    controlled_positions = set(matched["controlled_semantic_positions"])
    expected_controlled_positions = set(
        oracle["construction_preserved_slots"]
    )
    gold_controlled_values = {
        path: _get_path(case["gold_state"], path)
        for path in controlled_positions
    }
    flawed_controlled_values = {
        path: _get_path(case["flawed_state"], path)
        for path in controlled_positions
    }
    expected_public_conditions = {
        "fixed_constraints": case["agent_input"]["fixed_constraints"],
        "available_tools": case["agent_input"]["available_tools"],
        "task_contract": case["agent_input"]["task_contract"],
        "research_interface": case["agent_input"]["research_interface"],
        "required_output": case["agent_input"]["required_output"],
    }

    checks = {
        "hash_valid": stored_hash == _digest(unhashed),
        "matched_control_positions_exact": (
            controlled_positions == expected_controlled_positions
        ),
        "matched_control_values_equal": (
            gold_controlled_values == flawed_controlled_values
        ),
        "matched_control_values_hash_valid": (
            matched["controlled_semantic_values_hash"]
            == _digest(gold_controlled_values)
        ),
        "intervention_positions_exact": (
            set(matched["intervention_positions"]) == roots
        ),
        "paired_state_hashes_exact": (
            matched["gold_state_hash"] == case["provenance"]["gold_hash"]
            and matched["flawed_state_hash"]
            == case["provenance"]["flawed_hash"]
        ),
        "fixed_public_conditions_exact": (
            matched["fixed_public_conditions"] == expected_public_conditions
        ),
        "root_count_matches": len(roots) == case["control_degree"],
        "root_matches_oracle": roots == set(oracle["causal_targets"]),
        "opportunities_cover_roots": opportunity_targets == roots,
        "root_protected_disjoint": not roots & protected,
        "semantic_roles_disjoint": not (
            (roots & free) or (protected & free) or (known_downstream & free)
        ),
        "semantic_roles_cover_task": (
            roots | protected | known_downstream | free
        ) == all_semantic,
        "repair_contracts_cover_roots": set(
            certificate["repair_acceptance"]
        ) == roots,
        "reference_patch_covers_roots": set(
            certificate["construction_reference_patch"]
        ) == roots,
        "reference_repair_closes": repaired == case["gold_state"],
        "evidence_exact": evidence_ids == set(case["evidence_oracle"]),
        "evidence_recipes_bound": certificate["evidence_control"][
            "private_recipe_hashes"
        ]
        == {
            evidence_id: _digest(spec["probe_recipe"])
            for evidence_id, spec in case["evidence_oracle"].items()
        },
        "each_opportunity_has_evidence": all(
            row["evidence_ids"]
            for row in certificate["scientific_interventions"]
        ) if roots else True,
        "accepted_equivalence_classes": all(
            len(contract.get("accepted_operations", [])) >= 2
            for contract in certificate["repair_acceptance"].values()
        ),
        "competing_nonrepairs_declared": all(
            len(contract.get("competing_nonrepairs", [])) >= 2
            for contract in certificate["repair_acceptance"].values()
        ),
        "fixed_commitments_exact": (
            certificate["task_identity_control"]["fixed_commitments"]
            == oracle["fixed_commitments"]
        ),
        "generic_interface_only": (
            certificate["evidence_control"][
                "task_specific_action_menu_exposed"
            ]
            is False
            and "available_actions" not in case["agent_input"]
        ),
        "verification_exact": set(
            certificate["evaluation_control"]["verification_obligations"]
        ) == set(oracle["verification_obligations"]),
        "shortcuts_exact": set(
            certificate["evaluation_control"]["forbidden_shortcuts"]
        ) == set(oracle["forbidden_shortcuts"]),
    }
    if "executable_control" in oracle:
        checks["executable_control_exact"] = (
            certificate.get("assurance_level") == "execution_backed_fixture"
            and certificate.get("executable_control")
            == oracle["executable_control"]
        )
    errors = [name for name, passed in checks.items() if not passed]
    return {
        "case_id": case["case_id"],
        "certificate_hash": stored_hash,
        "valid": not errors,
        "checks": checks,
        "errors": errors,
    }
