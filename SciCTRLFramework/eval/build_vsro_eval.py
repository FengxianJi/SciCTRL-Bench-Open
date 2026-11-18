"""Materialize private evaluator labels from the pinned VSRO candidate set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build(cases_path: Path, output: Path) -> int:
    labels: list[dict[str, Any]] = []
    for line in cases_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        oracle = case["oracle"]
        project = case.get("provenance", {}).get("gold_project_id", case["domain"])
        targets = oracle.get("causal_targets", [])
        contracts = oracle.get("opportunity_contracts", {})
        categories_by_id = {
            target: contracts.get(target, {}).get("analysis_tag")
            for target in targets
            if contracts.get(target, {}).get("analysis_tag")
        }
        labels.append(
            {
                "case_id": case["case_id"],
                "pair_id": f"{project}::{case['layer']}",
                "source_family_cluster": project,
                "layer": case["layer"],
                "control_degree": case["control_degree"],
                "condition": "control" if case["control_degree"] == 0 else "intervention",
                "expected_revision": case["control_degree"] > 0,
                "intervention_ids": targets,
                "intervention_categories": sorted(set(categories_by_id.values())),
                "factor_categories_by_id": categories_by_id,
                "hidden_test_count": len(oracle.get("hidden_tests", [])),
                "repair_contract_count": len(oracle.get("opportunity_contracts", {})),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in labels),
        encoding="utf-8",
    )
    return len(labels)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("Benchmark/RevisionOpportunityFramework/internal_cases.jsonl"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"labels": build(args.cases, args.output), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
