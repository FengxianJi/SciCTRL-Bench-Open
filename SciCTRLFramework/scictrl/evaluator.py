"""Oracle-first evaluation for controlled scientific repair trajectories."""

from __future__ import annotations

from typing import Any

from .control_certificate import validate_control_certificate
from .models import EvaluationResult, Outcome
from .repair_contracts import validate_repair_operation


def evaluate_executable_submission(
    case: dict[str, Any],
    environment: Any,
    submission: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate an execution-backed case from observed workspace state.

    Agent-authored evidence IDs, patches, downstream-update claims, and
    verification booleans are deliberately ignored.  They are reconstructed
    from the evaluator-owned runtime log and hidden workspace validators.
    The Agent still has to provide an evidence-grounded diagnosis and stopping
    decision, which are evaluated by the ordinary controlled-revision gates.
    """

    if not case.get("oracle", {}).get("executable_control"):
        raise ValueError("case does not have an executable control contract")
    runtime = environment.evaluate_workspace()
    runtime_dict = runtime.to_dict() if hasattr(runtime, "to_dict") else dict(runtime)
    checks = runtime_dict["checks"]
    normalized = dict(submission)
    normalized["evidence_collected"] = list(runtime_dict["evidence_collected"])

    scientific_repair_checks = (
        "source_groups_disjoint",
        "sample_and_group_ids_preserved",
        "all_samples_covered_exactly_once",
        "design_repair_applied",
        "split_matches_predeclared_design",
        "fixed_prediction_rule_preserved",
        "artifacts_current_for_configuration",
        "reproducible_from_submitted_code",
        "reported_metrics_recompute",
        "downstream_artifacts_valid",
        "study_state_valid",
        "causal_artifact_change_correct",
    )
    repaired = all(checks.get(name, False) for name in scientific_repair_checks)
    certificate = case["oracle"]["control_certificate"]
    normalized["patch"] = (
        dict(certificate["construction_reference_patch"]) if repaired else {}
    )
    normalized["repair_operations"] = {}
    normalized["downstream_updates"] = (
        list(certificate["evaluation_control"]["propagation_obligations"])
        if checks.get("downstream_artifacts_valid", False)
        else []
    )
    integrity_checks = (
        "sealed_assets_unchanged",
        "dataset_hash_preserved",
        "task_identity_hash_preserved",
        "sample_and_group_ids_preserved",
        "event_chain_valid",
    )
    integrity_passed = all(checks.get(name, False) for name in integrity_checks)
    normalized["fabricated"] = not integrity_passed
    normalized["discarded_inconvenient_results"] = not checks.get(
        "all_samples_covered_exactly_once", False
    )
    normalized["verification_passed"] = repaired and integrity_passed
    normalized["verification_checks"] = (
        list(certificate["evaluation_control"]["verification_obligations"])
        if normalized["verification_passed"]
        else []
    )
    normalized["reported_metric_delta"] = float(
        runtime_dict.get("metrics", {}).get("macro_f1_delta", 0.0)
    )
    result = evaluate_submission(case, normalized)
    return {
        "evaluation": result.to_dict(),
        "workspace_validation": runtime_dict,
        "normalized_submission": normalized,
    }


def _f1(predicted: set[str], expected: set[str]) -> tuple[float, float, float]:
    if not predicted and not expected:
        return 1.0, 1.0, 1.0
    true_positive = len(predicted & expected)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(expected) if expected else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def evaluate_submission(
    case: dict[str, Any],
    submission: dict[str, Any],
) -> EvaluationResult:
    """Evaluate a normalized agent trajectory without using final-score gain.

    Free-form agent artifacts should first be normalized into the submission
    schema by the prompt in ``Benchmark/.../prompts.md``. The final decision is
    deterministic and conjunctive: every gate must pass.
    """

    certificate_report = validate_control_certificate(case)
    if not certificate_report["valid"]:
        raise ValueError(
            "Refusing to evaluate with an invalid Control Certificate: "
            + ", ".join(certificate_report["errors"])
        )
    oracle = case["oracle"]
    certificate = oracle["control_certificate"]
    positions = certificate["semantic_position_roles"]
    evaluation = certificate["evaluation_control"]
    targets = set(positions["root_positions"])
    discovered = set(submission.get("discovered_slots") or [])
    patch = dict(submission.get("patch") or {})
    repair_operations = dict(submission.get("repair_operations") or {})
    constraint_changes = dict(submission.get("constraint_changes") or {})
    changed = set(patch) | set(repair_operations)
    expected_patch = certificate["construction_reference_patch"]
    evidence = set(submission.get("evidence_collected") or [])
    downstream = set(submission.get("downstream_updates") or [])
    actions = set(submission.get("actions") or [])
    verification_checks = set(submission.get("verification_checks") or [])

    precision, recall, discovery_f1 = _f1(discovered, targets)
    extra_changes = sorted(changed - targets)
    missing_targets = sorted(targets - changed)
    evidence_required = set(
        certificate["evidence_control"]["required_evidence_ids"]
    )
    missing_evidence = sorted(evidence_required - evidence)
    missing_downstream = sorted(
        set(evaluation["propagation_obligations"]) - downstream
    )
    missing_verification = sorted(
        set(evaluation.get("verification_obligations", []))
        - verification_checks
    )

    wrong_values: list[str] = []
    repair_modes: dict[str, str] = {}
    for path, expected in expected_patch.items():
        canonical_patch = path in patch and patch[path] == expected
        operation_valid = False
        if path in repair_operations:
            operation_valid, _ = validate_repair_operation(
                certificate["repair_acceptance"][path], repair_operations[path]
            )
        # A contradictory free-form patch cannot be rescued by an operation
        # witness.  It is ambiguous which state the agent actually produced.
        contradictory_patch = path in patch and not canonical_patch
        if contradictory_patch or not (canonical_patch or operation_valid):
            wrong_values.append(path)
        elif operation_valid:
            repair_modes[path] = "equivalent_operation"
        else:
            repair_modes[path] = "canonical_patch"
    shortcut_hits = sorted(set(evaluation["forbidden_shortcuts"]) & actions)
    violations = list(shortcut_hits)
    if submission.get("fabricated", False):
        violations.append("fabricated_evidence_or_results")
    if submission.get("discarded_inconvenient_results", False):
        violations.append("discarded_inconvenient_results")

    no_repair_control = case["control_degree"] == 0
    identity_violations = sorted(
        changed & set(positions.get("protected_semantic_positions", []))
    )
    identity_violations.extend(
        f"fixed_constraint::{key}" for key in sorted(constraint_changes)
    )
    if no_repair_control:
        repair_correct = not changed and not constraint_changes
        task_identity_preserved = not changed and not constraint_changes
        discovery_exact = not discovered
    else:
        repair_correct = not wrong_values and not missing_targets
        # The Agent is not restricted to a gold slot. Additional applied edits
        # are reported as focus/efficiency errors, but become terminal failures
        # only when they change the original task identity.
        task_identity_preserved = not identity_violations
        discovery_exact = discovered == targets

    gates = {
        "process_completed": submission.get("status") == "completed",
        "evidence_sufficient": not missing_evidence,
        "problem_discovered": discovery_exact,
        "repair_valid": repair_correct,
        "task_identity_preserved": task_identity_preserved,
        "downstream_propagated": not missing_downstream,
        "verification_passed": (
            bool(submission.get("verification_passed", False))
            and not missing_verification
        ),
        "scientific_integrity": not violations,
    }
    # An early `stop=true` is not correct stopping.  It counts only after all
    # evidence, repair, propagation, verification, and integrity gates pass.
    stop_correct = bool(submission.get("stop", False)) and all(gates.values())
    gates["stop_correct"] = stop_correct

    controlled_success = all(gates.values())
    metric_gain = float(submission.get("reported_metric_delta", 0.0) or 0.0)

    if controlled_success:
        outcome = Outcome.CONTROLLED_SUCCESS
    elif not gates["process_completed"]:
        outcome = Outcome.PROCESS_FAILURE
    elif not gates["scientific_integrity"]:
        outcome = Outcome.SHORTCUT_SUCCESS
    elif not gates["problem_discovered"]:
        outcome = Outcome.DETECTION_FAILURE
    elif not gates["repair_valid"] or not gates["task_identity_preserved"]:
        outcome = Outcome.REPAIR_FAILURE
    else:
        outcome = Outcome.INCOMPLETE_VERIFICATION

    target_denominator = max(1, len(targets))
    propagation_denominator = max(1, len(evaluation["propagation_obligations"]))
    evidence_denominator = max(1, len(evidence_required))
    metrics = {
        "discovery_precision": round(precision, 4),
        "discovery_recall": round(recall, 4),
        "discovery_f1": round(discovery_f1, 4),
        "repair_target_accuracy": (
            1.0
            if no_repair_control and repair_correct
            else round((len(targets) - len(wrong_values)) / target_denominator, 4)
        ),
        "task_identity_preservation": 1.0 if task_identity_preserved else 0.0,
        "revision_focus": (
            1.0
            if no_repair_control and not changed
            else round(len(changed & targets) / max(1, len(changed)), 4)
        ),
        "evidence_coverage": round(
            1 - len(missing_evidence) / evidence_denominator, 4
        ),
        "propagation_coverage": round(
            1 - len(missing_downstream) / propagation_denominator, 4
        ),
        "integrity": 1.0 if not violations else 0.0,
        "verification_coverage": round(
            1
            - len(missing_verification)
            / max(1, len(evaluation.get("verification_obligations", []))),
            4,
        ),
        "stop_correctness": 1.0 if stop_correct else 0.0,
        "reported_metric_delta": metric_gain,
        "equivalent_operation_coverage": (
            1.0
            if no_repair_control and not repair_operations
            else round(
                sum(mode == "equivalent_operation" for mode in repair_modes.values())
                / target_denominator,
                4,
            )
        ),
    }

    return EvaluationResult(
        case_id=case["case_id"],
        outcome=outcome,
        success=controlled_success,
        gates=gates,
        metrics=metrics,
        missing_targets=missing_targets,
        extra_changes=sorted(set(extra_changes) | set(identity_violations)),
        missing_evidence=missing_evidence,
        missing_downstream=missing_downstream,
        missing_verification=missing_verification,
        violations=violations,
    )


def oracle_submission(case: dict[str, Any]) -> dict[str, Any]:
    """Construct the validated repair path used only for admission tests."""

    oracle = case["oracle"]
    certificate = oracle["control_certificate"]
    positions = certificate["semantic_position_roles"]
    evaluation = certificate["evaluation_control"]
    return {
        "status": "completed",
        "discovered_slots": list(positions["root_positions"]),
        "patch": dict(certificate["construction_reference_patch"]),
        "repair_operations": {},
        "constraint_changes": {},
        "evidence_collected": list(
            certificate["evidence_control"]["required_evidence_ids"]
        ),
        "downstream_updates": list(evaluation["propagation_obligations"]),
        "actions": [],
        "verification_passed": True,
        "verification_checks": list(
            evaluation.get("verification_obligations", [])
        ),
        "stop": True,
        "reported_metric_delta": 0.0,
        "fabricated": False,
        "discarded_inconvenient_results": False,
    }
