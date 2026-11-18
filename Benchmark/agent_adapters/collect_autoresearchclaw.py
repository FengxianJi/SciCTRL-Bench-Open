"""Collect native AutoResearchClaw artifacts into a leak-safe trajectory bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


_OUTPUT_PATTERNS = (
    "stage-14*/analysis.md",
    "stage-14*/experiment_summary.json",
    "stage-15*/decision.md",
    "stage-15*/decision_structured.json",
    "experiment_diagnosis.json",
    "experiment_repair_result.json",
    "action_log.json",
)


def _read_outputs(run_dir: Path) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for pattern in _OUTPUT_PATTERNS:
        for path in sorted(run_dir.glob(pattern)):
            if path.is_file():
                outputs[str(path.relative_to(run_dir))] = path.read_text(
                    encoding="utf-8", errors="replace"
                )
    return outputs


def collect(workspace: Path) -> dict[str, Any]:
    """Collect only post-run outputs; never infer a repair or read an oracle."""

    workspace = workspace.resolve()
    public_case = json.loads((workspace / "public_case.json").read_text(encoding="utf-8"))
    run_dir = workspace / "run"
    outputs = _read_outputs(run_dir)
    output_blob = "\n".join(outputs.values())

    # Layer 1 exposes evidence IDs at the checkpoint, but an ID counts only if
    # it appears in an agent-produced post-run artifact.  We intentionally do
    # not scan task.md, public_case.json, stage guidance, or seeded run inputs.
    supplied_evidence = []
    state = public_case.get("agent_input", {}).get("state", {})
    for observation in state.get("observations", []):
        evidence_id = observation.get("evidence_id", "")
        if evidence_id and evidence_id in output_blob:
            supplied_evidence.append(evidence_id)

    action_log: list[dict[str, Any]] = []
    action_path = run_dir / "action_log.json"
    if action_path.is_file():
        value = json.loads(action_path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            action_log = [item for item in value if isinstance(item, dict)]

    acquired_evidence = sorted(
        {
            item.get("evidence_id")
            for event in action_log
            for item in event.get("observations", [])
            if isinstance(item, dict) and item.get("evidence_id")
        }
    )
    decision = None
    for name, text in outputs.items():
        if name.endswith("decision_structured.json"):
            try:
                decision = json.loads(text).get("decision")
            except json.JSONDecodeError:
                pass

    revision_artifacts = sorted(
        str(path.relative_to(run_dir))
        for pattern in (
            "stage-14_repair_v*/experiment/*.py",
            "stage-13*/experiment_final/*.py",
            "stage-13*/revision_patch.json",
            "stage-14*/revision_patch.json",
        )
        for path in run_dir.glob(pattern)
        if path.is_file()
    )

    verification_checks_passed: list[str] = []
    # This file is produced by the benchmark-side validator service, outside
    # the native agent run directory. Agent-authored claims never populate it.
    verification_path = workspace / ".csfi_evaluator" / "verification_report.json"
    if verification_path.is_file():
        try:
            report = json.loads(verification_path.read_text(encoding="utf-8"))
            checks = report.get("evaluator_checks", [])
            verification_checks_passed = sorted(
                item["check_id"]
                for item in checks
                if isinstance(item, dict)
                and item.get("passed") is True
                and isinstance(item.get("check_id"), str)
            )
        except (json.JSONDecodeError, TypeError, KeyError):
            verification_checks_passed = []

    return {
        "case_id": public_case["case_id"],
        "agent_id": "autoresearchclaw",
        "layer": public_case["layer"],
        "native_decision": decision,
        "artifacts": outputs,
        "evidence_ids_cited": sorted(set(supplied_evidence) | set(acquired_evidence)),
        "actions_executed": [event.get("action") for event in action_log],
        "revision_artifacts": revision_artifacts,
        "verification_checks_passed": verification_checks_passed,
        "normalization_status": "pending",
        "normalization_note": (
            "This collector records native evidence only. It does not infer a patch, "
            "verification, or success from prose."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    bundle = collect(args.workspace)
    output = args.output or args.workspace / "trajectory_bundle.json"
    output.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
