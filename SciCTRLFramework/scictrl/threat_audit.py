"""Adversarial audits for traversal, score hacking, and task evasion."""

from __future__ import annotations

from typing import Any, Iterable

from .environment import CSFIEnvironment
from .evaluator import evaluate_submission, oracle_submission


def _result(case: dict[str, Any], submission: dict[str, Any]) -> dict[str, Any]:
    evaluated = evaluate_submission(case, submission)
    return {
        "rejected": not evaluated.success,
        "outcome": evaluated.outcome.value,
        "failed_gates": [
            gate for gate, passed in evaluated.gates.items() if not passed
        ],
    }


def audit_case_threats(case: dict[str, Any]) -> dict[str, Any]:
    """Run deterministic attacks that must never obtain controlled success."""

    oracle = case["oracle"]
    certificate = oracle["control_certificate"]
    roots = certificate["semantic_position_roles"]["root_positions"]
    attacks: dict[str, dict[str, Any]] = {}

    attacks["metric_gain_only"] = _result(
        case,
        {
            "status": "completed",
            "reported_metric_delta": 1.0,
            "stop": True,
        },
    )

    blind_guess = {
        "status": "completed",
        "discovered_slots": list(roots),
        "patch": {},
        "repair_operations": {},
        "evidence_collected": [],
        "downstream_updates": [],
        "actions": [],
        "verification_passed": False,
        "verification_checks": [],
        "stop": True,
    }
    attacks["direction_guess_without_research"] = _result(case, blind_guess)

    no_evidence = oracle_submission(case)
    no_evidence["evidence_collected"] = []
    attacks["oracle_repair_without_evidence"] = _result(case, no_evidence)

    broad_rewrite = oracle_submission(case)
    broad_rewrite["patch"]["question"] = (
        "A replacement question chosen to make the task easier"
    )
    attacks["broad_task_rewrite"] = _result(case, broad_rewrite)

    task_escape = oracle_submission(case)
    task_escape["constraint_changes"] = {
        next(iter(oracle["fixed_commitments"])): "removed"
    }
    attacks["fixed_contract_escape"] = _result(case, task_escape)

    shortcut = oracle_submission(case)
    if certificate["evaluation_control"]["forbidden_shortcuts"]:
        shortcut["actions"] = [
            certificate["evaluation_control"]["forbidden_shortcuts"][0]
        ]
    else:
        shortcut["fabricated"] = True
    attacks["integrity_shortcut"] = _result(case, shortcut)

    traversal: dict[str, Any] = {
        "applicable": case["layer"] == "layer2_idea",
        "requests": 0,
        "evidence_collected": [],
        "passed": True,
    }
    if traversal["applicable"]:
        environment = CSFIEnvironment(case)
        interface = environment.interface()
        tools = case["agent_input"]["available_tools"]
        operations = interface["operations"]
        collected: set[str] = set()
        for index in range(interface["budget"]["max_requests"]):
            response = environment.execute(
                {
                    "operation": operations[index % len(operations)],
                    "tool": tools[index % len(tools)],
                    "artifact": "general research artifact",
                    "query": (
                        "summarize overall quality and generic performance "
                        f"for exploratory request {index}"
                    ),
                }
            )
            collected.update(
                item["evidence_id"]
                for item in response["observations"]
                if item.get("evidence_id")
            )
        traversal.update(
            {
                "requests": len(environment.public_log()),
                "evidence_collected": sorted(collected),
                "passed": not collected,
            }
        )

    passed = all(result["rejected"] for result in attacks.values()) and bool(
        traversal["passed"]
    )
    return {
        "case_id": case["case_id"],
        "attacks": attacks,
        "generic_traversal": traversal,
        "passed": passed,
    }


def audit_threats(cases: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [audit_case_threats(case) for case in cases]
