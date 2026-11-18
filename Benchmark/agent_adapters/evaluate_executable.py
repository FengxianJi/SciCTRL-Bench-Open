"""Terminal evaluation bridge for execution-backed AutoResearchClaw runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = PROJECT_ROOT / "SciCTRLFramework"
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from scictrl.evaluator import evaluate_executable_submission  # noqa: E402
from scictrl.executable_runtime import (  # noqa: E402
    validate_attached_executable_workspace,
)


def _numeric_reports(value: Any) -> list[dict[str, float]]:
    reports: list[dict[str, float]] = []
    if isinstance(value, dict):
        if "source_group_overlap_count" in value:
            reports.append(
                {
                    key: float(item)
                    for key, item in value.items()
                    if isinstance(item, (int, float))
                }
            )
        for item in value.values():
            reports.extend(_numeric_reports(item))
    elif isinstance(value, list):
        for item in value:
            reports.extend(_numeric_reports(item))
    elif isinstance(value, str):
        for line in value.splitlines():
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            reports.extend(_numeric_reports(parsed))
    return reports


def _native_execution_trace(
    case: dict[str, Any], run_dir: Path
) -> tuple[bool, list[str], dict[str, float] | None]:
    reports: list[dict[str, float]] = []
    completed_records = 0
    for pattern in (
        "stage-12*/runs/*.json",
        "stage-14*/experiment_summary.json",
        "action_log.json",
    ):
        for path in sorted(run_dir.glob(pattern)):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(record, dict) and record.get("status") == "completed":
                completed_records += 1
            reports.extend(_numeric_reports(record))
    expect_overlap = case["control_degree"] > 0
    evidence_report = next(
        (
            report
            for report in reports
            if (report.get("source_group_overlap_count", 0.0) > 0)
            == expect_overlap
        ),
        None,
    )
    evidence = (
        list(case["oracle"]["evidence_obligations"])
        if evidence_report is not None
        else []
    )
    return completed_records > 0 and evidence_report is not None, evidence, evidence_report


def _experiment_workspace(run_dir: Path) -> Path:
    candidates = [
        *sorted(run_dir.glob("stage-13*/experiment_final"), reverse=True),
        *sorted(run_dir.glob("stage-10*/experiment"), reverse=True),
    ]
    for candidate in candidates:
        if (candidate / "experiment.py").is_file() and (
            candidate / "split_config.json"
        ).is_file():
            return candidate
    raise FileNotFoundError("no execution-backed experiment workspace found")


class _AttachedEnvironment:
    def __init__(self, result: Any):
        self.result = result

    def evaluate_workspace(self) -> Any:
        return self.result


def evaluate_autoresearchclaw_workspace(
    private_control_path: Path,
    workspace: Path,
    submission: dict[str, Any],
) -> dict[str, Any]:
    private = json.loads(private_control_path.read_text(encoding="utf-8"))
    case = private["internal_case"]
    run_dir = workspace / "run"
    trace_valid, evidence, initial_metrics = _native_execution_trace(case, run_dir)
    validation = validate_attached_executable_workspace(
        case,
        _experiment_workspace(run_dir),
        evidence_collected=evidence,
        trusted_trace_valid=trace_valid,
        initial_metrics=initial_metrics,
    )
    return evaluate_executable_submission(
        case,
        _AttachedEnvironment(validation),
        submission,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-control", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    submission = json.loads(args.submission.read_text(encoding="utf-8"))
    report = evaluate_autoresearchclaw_workspace(
        args.private_control.resolve(), args.workspace.resolve(), submission
    )
    output = args.output or args.workspace / ".csfi_evaluator" / "executable_evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
