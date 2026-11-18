"""Blinded assignment utilities that prevent correlated-case traversal."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


def build_blinded_assignments(cases: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Distribute cases so one assignment has at most one case per gold project."""

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        groups[case["provenance"]["gold_project_id"]].append(case)
    for rows in groups.values():
        rows.sort(key=lambda row: (row["layer"], row["public_case_id"]))

    assignment_count = max((len(rows) for rows in groups.values()), default=0)
    assignments: list[dict[str, Any]] = []
    for index in range(assignment_count):
        selected = [
            rows[index]
            for _, rows in sorted(groups.items())
            if index < len(rows)
        ]
        assignments.append(
            {
                "assignment_id": f"blind-{index:03d}",
                "public_case_ids": [row["public_case_id"] for row in selected],
                "case_count": len(selected),
            }
        )
    return assignments


def validate_blinded_assignment(
    cases: Iterable[dict[str, Any]], public_case_ids: Iterable[str]
) -> list[str]:
    """Reject unknown, duplicate, or same-gold-project cases in one assignment."""

    by_public = {case["public_case_id"]: case for case in cases}
    ids = list(public_case_ids)
    errors: list[str] = []
    if len(ids) != len(set(ids)):
        errors.append("duplicate_public_case")
    if set(ids) - set(by_public):
        errors.append("unknown_public_case")
    projects = [
        by_public[case_id]["provenance"]["gold_project_id"]
        for case_id in ids
        if case_id in by_public
    ]
    if len(projects) != len(set(projects)):
        errors.append("correlated_gold_project_repeated")
    return errors
