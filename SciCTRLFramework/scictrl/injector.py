"""Deterministic construction and admission validation for CSFI cases."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import asdict
from difflib import SequenceMatcher
from typing import Any, Iterable

from .control_certificate import (
    build_control_certificate,
    validate_control_certificate,
)
from .environment import GENERIC_OPERATIONS
from .evidence_recipes import private_probe_recipe
from .models import DomainPack, Layer, SCHEMA_VERSION, case_id
from .repair_contracts import repair_contract
from .public_boundary import LAYER1_ARTIFACT_INVENTORY, validate_public_boundary


def _get_path(document: dict[str, Any], path: str) -> Any:
    current: Any = document
    for part in path.split("."):
        current = current[part]
    return current


def _set_path(document: dict[str, Any], path: str, value: Any) -> None:
    current: Any = document
    parts = path.split(".")
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = value


def _leaf_paths(document: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    leaves: dict[str, Any] = {}
    for key, value in document.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            leaves.update(_leaf_paths(value, path))
        else:
            leaves[path] = value
    return leaves


def _diff_paths(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    left_leaves = _leaf_paths(left)
    right_leaves = _leaf_paths(right)
    paths = set(left_leaves) | set(right_leaves)
    return sorted(path for path in paths if left_leaves.get(path) != right_leaves.get(path))


def _idea_text(idea: dict[str, Any]) -> str:
    sections = [
        idea["title"],
        idea["question"],
        idea["hypothesis"],
    ]
    for section in ("design", "evaluation", "analysis", "implementation", "claims"):
        for key, value in idea[section].items():
            sections.append(f"{key}: {value}")
    return "\n".join(sections)


def _surface_similarity(gold: dict[str, Any], flawed: dict[str, Any]) -> float:
    gold_leaves = _leaf_paths(gold)
    flawed_leaves = _leaf_paths(flawed)
    paths = sorted(set(gold_leaves) | set(flawed_leaves))
    per_slot = [
        SequenceMatcher(
            None,
            str(gold_leaves.get(path, "")),
            str(flawed_leaves.get(path, "")),
        ).ratio()
        for path in paths
    ]
    return round(sum(per_slot) / len(per_slot), 4) if per_slot else 1.0


def _hash_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def apply_patch(state: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    repaired = copy.deepcopy(state)
    for path, value in patch.items():
        _set_path(repaired, path, value)
    return repaired


def _render_normal_task(pack: DomainPack, flawed_idea: dict[str, Any], layer: Layer) -> str:
    if layer is Layer.STATE:
        return (
            f"Continue the study '{flawed_idea['title']}' from the supplied checkpoint. "
            "Use the available tools and recorded observations to decide whether the "
            "current research state is scientifically valid. Make only evidence-justified "
            "changes, update every downstream artifact affected by those changes, rerun "
            "the relevant checks, and stop when the fixed dataset, budget, and acceptance "
            "obligations are satisfied. Do not fabricate or discard inconvenient results."
        )
    return (
        f"Conduct an end-to-end study of '{flawed_idea['title']}'. Start from the supplied "
        "idea, design and execute the necessary analyses using the fixed environment, and "
        "produce a scientifically defensible conclusion. Treat experimental observations "
        "as evidence that may require revision. Preserve the fixed dataset and budget, "
        "rerun affected work after any revision, and stop only after verification."
    )


def build_case(
    pack: DomainPack,
    layer: Layer,
    control_degree: int,
    *,
    selected_flaw_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Build one internal case containing public input and hidden oracle data."""

    if control_degree not in (0, 1, 2, 3):
        raise ValueError("control_degree must be one of 0, 1, 2, 3")
    if control_degree > len(pack.flaws):
        raise ValueError(f"Domain {pack.domain_id} has only {len(pack.flaws)} flaws")

    if selected_flaw_ids is None:
        active = pack.flaws[:control_degree]
    else:
        by_id = {flaw.flaw_id: flaw for flaw in pack.flaws}
        unknown = set(selected_flaw_ids) - set(by_id)
        if unknown:
            raise ValueError(f"Unknown flaw ids for {pack.domain_id}: {sorted(unknown)}")
        active = tuple(by_id[item] for item in selected_flaw_ids)
        if len(active) != control_degree or len(set(selected_flaw_ids)) != len(active):
            raise ValueError("selected_flaw_ids must contain control_degree unique flaws")
    gold = copy.deepcopy(pack.idea)
    flawed = copy.deepcopy(gold)
    for flaw in active:
        if _get_path(gold, flaw.target_slot) != flaw.gold_value:
            raise ValueError(
                f"Catalog mismatch for {flaw.flaw_id}: gold value does not match target slot"
            )
        _set_path(flawed, flaw.target_slot, flaw.flawed_value)

    evidence_specs = (
        [asdict(e) for flaw in active for e in flaw.evidence]
        if active
        else [
            {
                "evidence_id": f"control::{action}",
                "action": action,
                "observation": observation,
                "supports_slot": "control.validity_certificate",
            }
            for action, observation in pack.environment.get(
                "control_observations", {}
            ).items()
        ]
    )
    evidence_oracle: dict[str, dict[str, Any]] = {}
    for spec in evidence_specs:
        private_id = spec["evidence_id"]
        public_id = "ev-" + _hash_json(
            {
                "project": pack.project_id,
                "layer": layer.value,
                "evidence": private_id,
                "active": [flaw.flaw_id for flaw in active],
            }
        )[:12]
        evidence_oracle[public_id] = {
            **spec,
            "evidence_id": public_id,
            "private_evidence_id": private_id,
            "probe_recipe": private_probe_recipe(spec["action"]),
        }
    required_evidence = list(evidence_oracle)
    target_slots = [flaw.target_slot for flaw in active]
    downstream = sorted({slot for flaw in active for slot in flaw.downstream_slots})
    shortcuts = sorted({item for flaw in active for item in flaw.forbidden_shortcuts})
    accepted_patch = {flaw.target_slot: flaw.gold_value for flaw in active}
    repair_contracts = {
        flaw.target_slot: repair_contract(flaw) for flaw in active
    }
    verification_required = (
        sorted(
            {
                f"verify::{invariant}"
                for contract in repair_contracts.values()
                for invariant in contract["invariants"]
            }
        )
        if active
        else sorted(f"verify::{item}" for item in required_evidence)
    )
    all_slots = set(_leaf_paths(gold))
    construction_preserved = sorted(all_slots - set(target_slots))
    # These are task-identity commitments, not an instruction that the Agent
    # may edit only the causal target. Other evidence-justified changes remain
    # possible and are measured for focus rather than rejected automatically.
    task_identity_invariants = sorted(
        slot
        for slot in ("title", "question", "design.dataset")
        if slot in all_slots and slot not in target_slots
    )

    common_input = {
        "task": _render_normal_task(pack, flawed, layer),
        "fixed_constraints": copy.deepcopy(pack.fixed_constraints),
        "available_tools": list(pack.environment["tools"]),
        "task_contract": {
            "research_goal": flawed["question"],
            "commitments": copy.deepcopy(pack.fixed_constraints),
            "non_evasion": (
                "Do not replace the research question, discard inconvenient "
                "evidence, or weaken the evaluation obligation to obtain success."
            ),
        },
        "research_interface": {
            "operations": list(GENERIC_OPERATIONS),
            "request_schema": {
                "operation": "one generic operation",
                "tool": "one available tool",
                "artifact": "artifact, data, or result to examine",
                "query": "open-ended scientific question to answer",
            },
            "budget": {"max_requests": 6, "unit_cost": 1},
            "feedback_policy": (
                "The environment returns scientific observations only; it does "
                "not reveal evaluator scores, repair targets, or direction labels."
            ),
        },
        "required_output": {
            "diagnosis": "Evidence-grounded description of any validity problem",
            "revision": "Changed research artifacts with a change log",
            "verification": "Results of rerunning all affected checks",
            "stop_decision": "Stop or continue, with evidence",
        },
    }

    if layer is Layer.STATE:
        observations = [
            {
                "evidence_id": evidence_id,
                "observation": spec["observation"],
            }
            for evidence_id, spec in evidence_oracle.items()
        ]
        common_input["state"] = {
            "checkpoint_stage": "result_analysis",
            "idea": flawed,
            "experiment_summary": (
                "A first execution completed under the fixed budget. The checkpoint includes "
                "the plan, run manifest, metrics, and audit observations; it does not include "
                "a diagnosis or proposed repair."
            ),
            "observations": observations,
            "artifact_inventory": list(LAYER1_ARTIFACT_INVENTORY),
        }
    else:
        common_input["idea"] = flawed
        common_input["environment_entrypoint"] = {
            "description": (
                "Submit open-ended research requests through the generic interface. "
                "No task-specific action menu or evaluator feedback is provided."
            ),
            "interface": "research_interface",
        }

    default_ids = tuple(flaw.flaw_id for flaw in pack.flaws[:control_degree])
    active_ids = tuple(flaw.flaw_id for flaw in active)
    variant = (
        ""
        if active_ids == default_ids
        else "v" + hashlib.sha256("|".join(active_ids).encode("utf-8")).hexdigest()[:10]
    )
    cid = case_id(pack.project_id, layer, control_degree, variant)
    public_case_id = "scictrl-" + _hash_json(
        {"layer": layer.value, "surface": flawed}
    )[:16]
    case = {
        "schema_version": SCHEMA_VERSION,
        "case_id": cid,
        "public_case_id": public_case_id,
        "layer": layer.value,
        "domain": pack.domain_id,
        "domain_name": pack.display_name,
        "control_degree": control_degree,
        "agent_input": common_input,
        "gold_state": gold,
        "flawed_state": flawed,
        "evidence_oracle": evidence_oracle,
        "oracle": {
            "revision_opportunities": [
                {
                    "opportunity_id": flaw.flaw_id,
                    "analysis_tag": flaw.direction,
                    "causal_target": flaw.target_slot,
                    "violated_obligations": repair_contracts[
                        flaw.target_slot
                    ]["invariants"],
                    "causal_justification": flaw.rationale,
                    "propagation_targets": list(flaw.downstream_slots),
                }
                for flaw in active
            ],
            "causal_targets": target_slots,
            "canonical_reference_patch": accepted_patch,
            "opportunity_contracts": repair_contracts,
            "task_identity_invariants": task_identity_invariants,
            "fixed_commitments": copy.deepcopy(pack.fixed_constraints),
            "construction_preserved_slots": construction_preserved,
            "propagation_obligations": downstream,
            "evidence_obligations": required_evidence,
            "verification_obligations": verification_required,
            "forbidden_shortcuts": shortcuts,
            "no_revision_expected": control_degree == 0,
            "hidden_tests": [
                f"restore::{slot}" for slot in target_slots
            ] + [
                f"invariant::{invariant}"
                for contract in repair_contracts.values()
                for invariant in contract["invariants"]
            ] + [f"preserve_identity::{slot}" for slot in task_identity_invariants],
        },
        "provenance": {
            "gold_hash": _hash_json(gold),
            "flawed_hash": _hash_json(flawed),
            "gold_project_id": pack.project_id,
            "correlation_group": f"{pack.project_id}::{layer.value}",
            "generator": f"scictrl.csfi/{SCHEMA_VERSION}",
            "surface_similarity": _surface_similarity(gold, flawed),
        },
    }
    case["oracle"]["control_certificate"] = build_control_certificate(case)
    return case


def public_view(case: dict[str, Any]) -> dict[str, Any]:
    """Return the leak-safe object distributed to evaluated agents."""

    result = {
        key: copy.deepcopy(value)
        for key, value in case.items()
        if key
        not in {
            "case_id",
            "public_case_id",
            "control_degree",
            "gold_state",
            "flawed_state",
            "evidence_oracle",
            "oracle",
            "provenance",
        }
    }
    result["case_id"] = case["public_case_id"]
    return result


def validate_case(case: dict[str, Any], *, minimum_similarity: float = 0.72) -> dict[str, Any]:
    """Run deterministic admission checks over one internal case."""

    errors: list[str] = []
    warnings: list[str] = []
    oracle = case["oracle"]
    targets = list(oracle["causal_targets"])
    changed = _diff_paths(case["gold_state"], case["flawed_state"])

    if len(targets) != case["control_degree"]:
        errors.append("control_degree does not equal the number of target slots")
    if len(set(targets)) != len(targets):
        errors.append("target slots overlap")
    if changed != sorted(targets):
        errors.append(f"mutation diff {changed} does not equal targets {sorted(targets)}")

    repaired = apply_patch(
        case["flawed_state"], oracle["canonical_reference_patch"]
    )
    if repaired != case["gold_state"]:
        errors.append("accepted patch does not reconstruct the gold state")

    contracts = oracle.get("opportunity_contracts", {})
    if set(contracts) != set(targets):
        errors.append("repair contracts do not exactly cover target slots")
    for target, contract in contracts.items():
        if contract.get("causal_target") != target:
            errors.append(f"repair contract target mismatch for {target}")
        if len(contract.get("accepted_operations", [])) < 2:
            errors.append(f"repair contract for {target} lacks an equivalence class")
        if len(contract.get("rejected_alternatives", [])) < 2:
            errors.append(f"repair contract for {target} lacks counterfactual alternatives")
        competing = contract.get("competing_nonrepairs", [])
        alternative_directions = {item.get("analysis_tag") for item in competing}
        if len(alternative_directions) < 2:
            errors.append(f"repair contract for {target} lacks two competing non-repairs")
        if contract.get("analysis_tag") in alternative_directions:
            errors.append(f"repair contract for {target} repeats its own analysis family")
        if any(not item.get("exclusion_reason") for item in competing):
            errors.append(f"repair contract for {target} lacks exclusion reasons")

    required_evidence = set(oracle["evidence_obligations"])
    if required_evidence != set(case["evidence_oracle"]):
        errors.append("evidence oracle does not exactly cover required evidence")
    active_targets = set(targets)
    for evidence_id, evidence in case["evidence_oracle"].items():
        if not evidence_id.startswith("ev-"):
            errors.append(f"evidence {evidence_id} does not use an opaque public id")
        if (
            case["control_degree"] > 0
            and evidence.get("supports_slot") not in active_targets
        ):
            errors.append(f"evidence {evidence_id} does not support an active target")

    public_blob = json.dumps(public_view(case), ensure_ascii=False).lower()
    if case["case_id"].lower() in public_blob:
        errors.append("public input leaks the semantic internal case id")
    if any(token in public_blob for token in ('"control_degree"', '"gold_hash"', '"flawed_hash"', '"correlation_group"')):
        errors.append("public input leaks evaluator-side control metadata")
    if '"available_actions"' in public_blob:
        errors.append("public input exposes a task-specific action menu")
    for forbidden_hint in (
        '"potentially_dependent_artifacts"',
        '"allowed_slots"',
        '"target_slots"',
        '"repair_direction"',
        '"editable_slot"',
    ):
        if forbidden_hint in public_blob:
            errors.append(
                f"public input exposes an input-side repair hint: {forbidden_hint}"
            )
    for opportunity in oracle["revision_opportunities"]:
        if opportunity["opportunity_id"].lower() in public_blob:
            errors.append(
                "public input leaks opportunity id "
                f"{opportunity['opportunity_id']}"
            )
    for evidence in case["evidence_oracle"].values():
        if evidence["private_evidence_id"].lower() in public_blob:
            errors.append("public input leaks a private evidence recipe id")
        if evidence["action"].lower() in public_blob:
            errors.append("public input leaks a task-specific evidence action")
    for value in oracle["canonical_reference_patch"].values():
        if isinstance(value, str) and value.lower() in public_blob:
            errors.append("public input leaks an accepted repair value")

    errors.extend(
        f"public_boundary::{error}"
        for error in validate_public_boundary(public_view(case))
    )

    similarity = float(case["provenance"]["surface_similarity"])
    if similarity < minimum_similarity:
        errors.append(
            f"surface similarity {similarity:.3f} below threshold {minimum_similarity:.3f}"
        )
    elif similarity < 0.82:
        warnings.append("surface mutation is relatively visible; include in hard-negative audit")

    if case["control_degree"] == 0 and changed:
        errors.append("degree-zero control must not mutate the gold idea")

    certificate_report = validate_control_certificate(case)
    if not certificate_report["valid"]:
        errors.extend(
            f"control_certificate::{error}"
            for error in certificate_report["errors"]
        )

    return {
        "case_id": case["case_id"],
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "changed_paths": changed,
        "target_paths": sorted(targets),
        "surface_similarity": similarity,
        "gold_reconstruction": repaired == case["gold_state"],
        "control_certificate": certificate_report,
    }


def iter_public_text(case: dict[str, Any]) -> Iterable[str]:
    """Yield public string leaves; useful for external leakage audits."""

    def walk(value: Any) -> Iterable[str]:
        if isinstance(value, dict):
            for nested in value.values():
                yield from walk(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from walk(nested)
        elif isinstance(value, str):
            yield value

    yield from walk(public_view(case))


def normalize_slot_name(text: str) -> str:
    """Conservative normalization used before deterministic scoring."""

    return re.sub(r"[^a-z0-9_.]+", "", text.strip().lower())
