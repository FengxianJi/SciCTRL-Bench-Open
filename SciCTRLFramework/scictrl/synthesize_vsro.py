"""CLI for generating the deterministic CSFI benchmark matrix."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
import tempfile
from typing import Any, Iterable

from .catalog import DOMAIN_PACKS
from .directions import direction_manifest, validate_flaw_direction
from .evaluator import evaluate_submission, oracle_submission
from .feasibility import assess_direction_feasibility
from .injector import build_case, public_view, validate_case
from .models import Layer, SCHEMA_VERSION
from .task_output import materialize_task_outputs
from .threat_audit import audit_case_threats
from .serving import build_blinded_assignments, validate_blinded_assignment
from .executable_runtime import (
    AUDIT_SOURCE,
    ExecutableCSFIEnvironment,
    build_executable_pilot_case,
    materialize_ml_group_leakage_workspace,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    path.write_text(text, encoding="utf-8")


def generate(benchmark_root: Path) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    oracle_results: list[dict[str, Any]] = []
    feasibility_results: list[dict[str, Any]] = []
    certificate_results: list[dict[str, Any]] = []
    threat_results: list[dict[str, Any]] = []

    direction_errors = [
        error
        for pack in DOMAIN_PACKS
        for flaw in pack.flaws
        for error in validate_flaw_direction(flaw)
    ]
    if direction_errors:
        raise RuntimeError("Direction taxonomy validation failed: " + "; ".join(direction_errors))

    for pack in DOMAIN_PACKS:
        for layer in (Layer.STATE, Layer.IDEA):
            selections = [()] + [
                tuple(flaw.flaw_id for flaw in selected)
                for degree in (1, 2, 3)
                for selected in combinations(pack.flaws, degree)
            ]
            for selected_ids in selections:
                degree = len(selected_ids)
                case = build_case(
                    pack,
                    layer,
                    degree,
                    selected_flaw_ids=selected_ids,
                )
                report = validate_case(case)
                cases.append(case)
                reports.append(report)
                certificate_results.append(report["control_certificate"])
                oracle_results.append(
                    evaluate_submission(case, oracle_submission(case)).to_dict()
                )
                feasibility_results.append(assess_direction_feasibility(case))
                threat_results.append(audit_case_threats(case))

    invalid = [report for report in reports if not report["valid"]]
    failed_oracles = [result for result in oracle_results if not result["success"]]
    failed_feasibility = [result for result in feasibility_results if not result["passed"]]
    failed_threats = [result for result in threat_results if not result["passed"]]
    if invalid or failed_oracles or failed_feasibility or failed_threats:
        details = {
            "invalid_cases": invalid,
            "failed_oracles": failed_oracles,
            "failed_direction_feasibility": failed_feasibility,
            "failed_control_threat_audits": failed_threats,
        }
        raise RuntimeError(
            "CSFI admission failed:\n" + json.dumps(details, ensure_ascii=False, indent=2)
        )

    for layer, dirname in ((Layer.STATE, "Layer1"), (Layer.IDEA, "Layer2")):
        layer_cases = [case for case in cases if case["layer"] == layer.value]
        public_rows = [public_view(case) for case in layer_cases]
        oracle_rows = [
            {
                "case_id": case["case_id"],
                "public_case_id": case["public_case_id"],
                "gold_state": case["gold_state"],
                "flawed_state": case["flawed_state"],
                "evidence_oracle": case["evidence_oracle"],
                "oracle": case["oracle"],
                "provenance": case["provenance"],
            }
            for case in layer_cases
        ]
        layer_root = benchmark_root / dirname
        _write_jsonl(layer_root / "public.jsonl", public_rows)
        _write_jsonl(layer_root / "oracle.jsonl", oracle_rows)
        _write_json(
            layer_root / "manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "layer": layer.value,
                "cases": len(layer_cases),
                "domains": sorted({case["domain"] for case in layer_cases}),
                "control_degrees": [0, 1, 2, 3],
                "public_file": "public.jsonl",
                "private_file": "oracle.jsonl",
            },
        )

    csfi_root = benchmark_root / "RevisionOpportunityFramework"
    _write_jsonl(csfi_root / "internal_cases.jsonl", cases)
    _write_json(csfi_root / "validation_report.json", reports)
    _write_json(csfi_root / "oracle_path_results.json", oracle_results)
    _write_json(csfi_root / "direction_feasibility_report.json", feasibility_results)
    _write_json(
        csfi_root / "control_certificate_report.json", certificate_results
    )
    _write_json(csfi_root / "control_threat_report.json", threat_results)

    directions = Counter(
        opportunity["analysis_tag"]
        for case in cases
        for opportunity in case["oracle"]["revision_opportunities"]
    )
    similarities = [case["provenance"]["surface_similarity"] for case in cases]
    layer2_direction_sets = {
        tuple(
            sorted(
                opportunity["analysis_tag"]
                for opportunity in case["oracle"]["revision_opportunities"]
            )
        )
        for case in cases
        if case["layer"] == Layer.IDEA.value
    }
    unique_contracts = {
        contract["contract_id"]: contract
        for case in cases
        for contract in case["oracle"]["opportunity_contracts"].values()
    }
    contract_summary = {
        "unique_flaw_contracts": len(unique_contracts),
        "accepted_repair_operations": sum(
            len(contract["accepted_operations"])
            for contract in unique_contracts.values()
        ),
        "competing_nonrepair_exclusions": sum(
            len(contract["competing_nonrepairs"])
            for contract in unique_contracts.values()
        ),
        "domain_control_certificate_checks": sum(
            len(pack.environment.get("control_observations", {}))
            for pack in DOMAIN_PACKS
        ),
        "verification_source": "evaluator_owned_check_ids",
        "control_certificates": len(certificate_results),
        "control_certificate_checks": len(certificate_results[0]["checks"]),
        "control_threat_attacks_per_case": len(threat_results[0]["attacks"]),
    }
    task_classes = {
        "layer2_instances": sum(case["layer"] == Layer.IDEA.value for case in cases),
        "control_degree_classes": 4,
        "implemented_atomic_directions": len(directions),
        "layer2_direction_set_classes": len(layer2_direction_sets),
        "independent_gold_projects": len(DOMAIN_PACKS),
        "note": "Combinatorial variants sharing a gold_project_id are correlated benchmark instances, not independent research projects.",
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "SciCTRL-Bench / VSRO",
        "total_cases": len(cases),
        "release_status": "canonical_structural_candidates",
        "serving_policy": {
            "development_files": "contain correlated variants and are not a blinded release split",
            "maximum_variants_per_gold_project_per_agent": 1,
            "public_control_degree": "hidden",
            "public_case_identifiers": "opaque",
            "agent_revision_space": "open within the original task contract",
            "task_specific_action_menu": "not exposed",
        },
        "task_classes": task_classes,
        "repair_contract_summary": contract_summary,
        "layers": Counter(case["layer"] for case in cases),
        "domains": Counter(case["domain"] for case in cases),
        "control_degrees": Counter(str(case["control_degree"]) for case in cases),
        "active_repair_directions": dict(sorted(directions.items())),
        "surface_similarity": {
            "minimum": min(similarities),
            "mean": round(sum(similarities) / len(similarities), 4),
            "maximum": max(similarities),
        },
        "admission": {
            "valid_cases": len(cases),
            "invalid_cases": 0,
            "oracle_paths_passed": len(oracle_results),
            "oracle_paths_failed": 0,
            "direction_feasibility_passed": len(feasibility_results),
            "direction_feasibility_failed": 0,
            "control_certificates_passed": sum(
                result["valid"] for result in certificate_results
            ),
            "control_certificates_failed": sum(
                not result["valid"] for result in certificate_results
            ),
            "control_threat_audits_passed": sum(
                result["passed"] for result in threat_results
            ),
            "control_threat_audits_failed": sum(
                not result["passed"] for result in threat_results
            ),
        },
    }
    _write_json(csfi_root / "manifest.json", summary)
    _write_json(csfi_root / "task_class_report.json", task_classes)
    _write_json(csfi_root / "direction_taxonomy.json", direction_manifest())
    blinded_assignments = build_blinded_assignments(cases)
    assignment_errors = [
        {
            "assignment_id": assignment["assignment_id"],
            "errors": validate_blinded_assignment(
                cases, assignment["public_case_ids"]
            ),
        }
        for assignment in blinded_assignments
    ]
    assignment_errors = [row for row in assignment_errors if row["errors"]]
    if assignment_errors:
        raise RuntimeError(
            "Blinded serving-plan validation failed:\n"
            + json.dumps(assignment_errors, ensure_ascii=False, indent=2)
        )
    _write_json(
        csfi_root / "blinded_serving_plan.json",
        {
            "schema_version": SCHEMA_VERSION,
            "policy": "at_most_one_case_per_gold_project_per_assignment",
            "assignments": blinded_assignments,
        },
    )
    pilot = build_executable_pilot_case()
    clean_control = build_executable_pilot_case(clean_control=True)
    pilot_root = benchmark_root / "ExecutablePacks" / "rare_group_classification_v1"
    materialize_ml_group_leakage_workspace(pilot_root / "workspace_template")
    materialize_ml_group_leakage_workspace(
        pilot_root / "clean_control" / "workspace_template",
        clean_control=True,
    )
    _write_json(pilot_root / "public_case.json", public_view(pilot))
    _write_json(
        pilot_root / "clean_control" / "public_case.json",
        public_view(clean_control),
    )
    _write_json(
        pilot_root / "private_control.json",
        {
            "case_id": pilot["case_id"],
            "public_case_id": pilot["public_case_id"],
            "oracle": pilot["oracle"],
            "provenance": pilot["provenance"],
            "internal_case": pilot,
        },
    )
    _write_json(
        pilot_root / "clean_control" / "private_control.json",
        {
            "case_id": clean_control["case_id"],
            "public_case_id": clean_control["public_case_id"],
            "oracle": clean_control["oracle"],
            "provenance": clean_control["provenance"],
            "internal_case": clean_control,
        },
    )

    def admit_executable(case: dict[str, Any], *, repair: bool) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="scictrl-admission-") as temporary:
            environment = ExecutableCSFIEnvironment(case, Path(temporary))
            initial_run = environment.execute(
                {"operation": "execute_experiment", "script": "experiment.py"}
            )
            environment.execute(
                {
                    "operation": "write_file",
                    "path": "private_admission_audit.py",
                    "content": AUDIT_SOURCE,
                }
            )
            environment.execute(
                {"operation": "run_analysis", "script": "private_admission_audit.py"}
            )
            repaired_run = initial_run
            if repair:
                config_path = Path(temporary) / "split_config.json"
                config = json.loads(config_path.read_text(encoding="utf-8"))
                config["partition_unit"] = "source_group"
                environment.execute(
                    {
                        "operation": "write_file",
                        "path": "split_config.json",
                        "content": json.dumps(config, indent=2, sort_keys=True) + "\n",
                    }
                )
                repaired_run = environment.execute(
                    {"operation": "execute_experiment", "script": "experiment.py"}
                )
            validation = environment.evaluate_workspace().to_dict()
            return {
                "execution_backend": "local_subprocess_reproducibility_harness",
                "security_boundary": "container_or_microvm_required_for_untrusted_agents",
                "initial_metrics": json.loads(initial_run["stdout"]),
                "terminal_metrics": json.loads(repaired_run["stdout"]),
                "terminal_validation": validation,
            }

    pilot_admission = admit_executable(pilot, repair=True)
    control_admission = admit_executable(clean_control, repair=False)
    _write_json(pilot_root / "admission_report.json", pilot_admission)
    _write_json(
        pilot_root / "clean_control" / "admission_report.json",
        control_admission,
    )
    summary["executable_pilot"] = {
        "pack": "rare_group_classification_v1",
        "cases": 2,
        "repair_case_admission_passed": pilot_admission["terminal_validation"]["success"],
        "clean_control_admission_passed": control_admission["terminal_validation"]["success"],
        "public_case": "ExecutablePacks/rare_group_classification_v1/public_case.json",
        "workspace_template": "ExecutablePacks/rare_group_classification_v1/workspace_template",
        "admission_report": "ExecutablePacks/rare_group_classification_v1/admission_report.json",
    }
    summary["pipeline_output"] = materialize_task_outputs(
        cases, benchmark_root / "PipelineOutput"
    )
    summary["blinded_serving"] = {
        "assignments": len(blinded_assignments),
        "validation_errors": 0,
        "maximum_cases_per_assignment": max(
            (row["case_count"] for row in blinded_assignments),
            default=0,
        ),
    }
    _write_json(csfi_root / "manifest.json", summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        required=True,
        help="Benchmark output root containing Layer1, Layer2, and RevisionOpportunityFramework directories",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = generate(args.benchmark_root.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=dict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
