"""Score deterministic construction gates for the checked-in VSRO candidate set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("Benchmark/RevisionOpportunityFramework")


def _load(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return value


def audit(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    cases = [
        json.loads(line)
        for line in (root / "internal_cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    reports = {name: _load(root / filename) for name, filename in {
        "validation": "validation_report.json",
        "certificates": "control_certificate_report.json",
        "threats": "control_threat_report.json",
        "oracle_paths": "oracle_path_results.json",
    }.items()}
    maps = {
        name: {row["case_id"]: row for row in rows}
        for name, rows in reports.items()
    }
    expected_ids = {case["case_id"] for case in cases}
    join_complete = all(set(rows) == expected_ids for rows in maps.values())
    changed_exactly_targets = all(
        set(maps["validation"][case_id].get("changed_paths", []))
        == set(maps["validation"][case_id].get("target_paths", []))
        for case_id in expected_ids
    )
    certificates_valid = all(
        row.get("valid") and all(row.get("checks", {}).values()) and not row.get("errors")
        for row in maps["certificates"].values()
    )
    threats_rejected = all(row.get("passed") for row in maps["threats"].values())
    oracle_repairs_pass = all(row.get("success") for row in maps["oracle_paths"].values())
    gold_reconstruction = all(
        maps["validation"][case_id].get("gold_reconstruction") is True
        for case_id in expected_ids
    )
    alternatives = [
        len(contract.get("accepted_operations", []))
        for case in cases
        for contract in case.get("oracle", {}).get("opportunity_contracts", {}).values()
    ]
    equivalent_repairs_present = bool(alternatives) and min(alternatives) >= 2

    # Scan only released public rows; private contracts are never copied to this report.
    benchmark_root = Path(__file__).resolve().parents[2] / "Benchmark"
    public_files = [benchmark_root / "Layer1/public.jsonl", benchmark_root / "Layer2/public.jsonl"]
    public_rows = []
    for path in public_files:
        public_rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    public_by_id = {row["case_id"]: row for row in public_rows}
    label_leak_count = 0
    for case in cases:
        public = public_by_id.get(case["public_case_id"])
        if public is None:
            label_leak_count += 1
            continue
        serialized = json.dumps(public, ensure_ascii=False)
        private = case.get("oracle", {})
        for target in private.get("causal_targets", []):
            # One-token slots such as `hypothesis` are part of the public
            # task contract; their generic word is not an intervention label.
            if target and "." in target and target in serialized:
                label_leak_count += 1
                break
        forbidden_keys = {"oracle", "gold_state", "flawed_state", "evidence_oracle"}
        if forbidden_keys.intersection(public):
            label_leak_count += 1

    gates = {
        "case_report_join_complete": join_complete,
        "semantic_diff_equals_registered_intervention": changed_exactly_targets,
        "certificate_checks_all_pass": certificates_valid,
        "threat_shortcuts_rejected": threats_rejected,
        "oracle_repair_paths_pass": oracle_repairs_pass,
        "gold_states_reconstruct": gold_reconstruction,
        "at_least_two_equivalent_repairs": equivalent_repairs_present,
        "public_case_label_leaks_zero": label_leak_count == 0,
    }
    return {
        "schema_version": "scictrl-construction-audit-0.1",
        "cases": len(cases),
        "projects": len({case.get("provenance", {}).get("gold_project_id") for case in cases}),
        "layers": sorted({case.get("layer") for case in cases}),
        "control_degrees": sorted({case.get("control_degree") for case in cases}),
        "gates": gates,
        "passed_gates": sum(gates.values()),
        "total_gates": len(gates),
        "status": "structural_gates_pass" if all(gates.values()) else "not_admitted",
        "limitations": [
            "This is deterministic constructor evidence, not subject-matter-expert approval.",
            "The 68 cases are correlated variants of a small number of gold projects.",
            "Agent outcomes require paired Qwen and AutoResearchClaw runs under a permitted backend API plan.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.root)
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["status"] == "structural_gates_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
