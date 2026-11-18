"""Execution-backed CSFI runtime and the first runnable domain pack.

Unlike the legacy recipe simulator, this module creates a real workspace,
executes Python programs, records immutable event metadata, and evaluates the
resulting artifacts with evaluator-owned predicates.  The bundled
``ml_group_leakage`` pack is intentionally dependency-free so it can run in a
fresh Python environment.

The local subprocess backend is a reproducibility harness, not a security
boundary for hostile code.  Production deployment should put the same
workspace contract inside a container or microVM.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import csv
import io
import hashlib
import hmac
import inspect
import json
import os
from pathlib import Path
import random
import resource
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

from .models import DomainPack, EvidenceSpec, FlawSpec, Layer


EXECUTABLE_PACK_ID = "grouped_records_split_v2"
EXECUTABLE_OPERATIONS = (
    "inspect_artifact",
    "write_file",
    "run_analysis",
    "execute_experiment",
    "compare_results",
)


EXECUTABLE_DOMAIN_PACK = DomainPack(
    domain_id="ml_grouped_classification",
    project_id="grouped_records_generalization",
    display_name="Grouped-record classification",
    idea={
        "title": "Generalization of a fixed classifier on correlated records",
        "question": (
            "What performance does the fixed classifier achieve on independent "
            "held-out sources in the frozen GroupedRecords-v1 corpus?"
        ),
        "hypothesis": (
            "The fixed classifier achieves macro-F1 of at least 0.70 when "
            "evaluated on independent held-out sources."
        ),
        "design": {
            "dataset": (
                "GroupedRecords-v1, a frozen binary-classification table with "
                "multiple correlated records per source and retained lineage metadata"
            ),
            "split_policy": (
                "Source-group-stratified train/validation/test partition with every "
                "record from one source assigned to exactly one partition"
            ),
            "controls": "A fixed signal-only fallback rule; no model selection is permitted",
            "intervention": (
                "A fixed prediction rule that memorizes training fingerprints and "
                "uses the signal feature for unseen fingerprints"
            ),
        },
        "evaluation": {
            "primary_metric": "Macro-F1 on every record in the held-out test sources",
            "secondary_metrics": "Accuracy, test record count, and source-overlap count",
        },
        "analysis": {
            "aggregation": "One predeclared deterministic split with seed 17",
            "statistical_test": "Threshold check against the predeclared macro-F1 value 0.70",
        },
        "implementation": {
            "protocol": (
                "Use the fixed prediction rule unchanged; change only evidence-justified "
                "study artifacts and evaluate every assigned test record"
            ),
        },
        "claims": {
            "main": (
                "Report performance only for the independently held-out source groups"
            ),
            "limitations": (
                "Conclusions are restricted to GroupedRecords-v1, seed 17, and the fixed classifier"
            ),
        },
    },
    fixed_constraints={
        "dataset": "GroupedRecords-v1 frozen CSV hash",
        "model": "fixed fingerprint-lookup with signal fallback",
        "seed": 17,
        "compute_budget": "16 requests, 20 CPU-seconds per execution",
    },
    environment={
        "tools": ["python", "artifact_inspector"],
        "available_actions": ["run_group_overlap_audit"],
        "control_observations": {
            "run_group_overlap_audit": (
                "An independently recomputed audit finds no source group crossing "
                "train, validation, or test partitions."
            ),
        },
    },
    flaws=(
        FlawSpec(
            flaw_id="ml_group_leakage",
            direction="data_split",
            target_slot="design.split_policy",
            gold_value=(
                "Source-group-stratified train/validation/test partition with every "
                "record from one source assigned to exactly one partition"
            ),
            flawed_value=(
                "Class-stratified train/validation/test partition over cached records, "
                "with source lineage retained only as descriptive metadata"
            ),
            evidence=(
                EvidenceSpec(
                    evidence_id="exec_source_overlap",
                    action="run_group_overlap_audit",
                    observation=(
                        "An independently recomputed partition audit finds source groups "
                        "represented in more than one split."
                    ),
                    supports_slot="design.split_policy",
                ),
            ),
            downstream_slots=(
                "implementation.protocol",
                "analysis.results_table",
                "claims.main",
            ),
            forbidden_shortcuts=(
                "copy_test_labels",
                "drop_overlapping_rows_from_report",
                "rename_group_ids",
                "evaluate_subset_only",
            ),
            rationale=(
                "The estimand concerns unseen sources, so source independence must be "
                "restored at partition construction rather than hidden in reporting."
            ),
        ),
    ),
)


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _file_hash(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".csfi-private" not in path.parts
    }


def _apply_local_resource_limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (20, 21))
    resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))


def _run_python_process(
    script: Path, cwd: Path, args: list[str], timeout: int
) -> subprocess.CompletedProcess[str]:
    """Run one process with time, memory, output-file and process-group limits."""

    process = subprocess.Popen(
        [sys.executable, str(script), *args],
        cwd=cwd,
        env={
            "PATH": os.environ.get("PATH", ""),
            "PYTHONHASHSEED": "0",
            "PYTHONIOENCODING": "utf-8",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        preexec_fn=_apply_local_resource_limits,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            process.args, timeout, output=stdout, stderr=stderr
        )
    return subprocess.CompletedProcess(
        process.args, process.returncode, stdout=stdout, stderr=stderr
    )


def _dataset_rows() -> list[dict[str, str]]:
    """Build a deterministic grouped classification dataset.

    Each source group has four correlated records and one stable fingerprint.
    A record-level split leaks this fingerprint across partitions, while a
    group split tests generalization to unseen sources.  ``signal`` remains a
    noisy but genuine predictive feature, so the corrected task is feasible.
    """

    rows: list[dict[str, str]] = []
    for group_index in range(60):
        label = 1 if (group_index * 37) % 13 < 6 else 0
        for replicate in range(4):
            rng = random.Random(9109 + group_index * 101 + replicate)
            signal_correct = rng.random() < 0.74
            signal = label if signal_correct else 1 - label
            rows.append(
                {
                    "sample_id": f"sample-{group_index:03d}-{replicate}",
                    "source_group": f"source-{group_index:03d}",
                    "fingerprint": f"fp-{(group_index * 7919) % 100003:05d}",
                    "signal": str(signal),
                    "label": str(label),
                }
            )
    return rows


PIPELINE_SOURCE = r'''"""Runnable grouped-classification experiment.

Edit split_config.json when scientific evidence requires a design revision.
The experiment writes every derived artifact under artifacts/.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "records.csv"
CONFIG = ROOT / "split_config.json"
ARTIFACTS = ROOT / "artifacts"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_rows():
    with DATA.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

def allocate(items, train_fraction, validation_fraction):
    n = len(items)
    train_end = max(1, int(n * train_fraction))
    validation_end = max(train_end + 1, int(n * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, n - 1)
    return {
        "train": items[:train_end],
        "validation": items[train_end:validation_end],
        "test": items[validation_end:],
    }

def make_split(rows, config):
    rng = random.Random(int(config["seed"]))
    partition_unit = config["partition_unit"]
    assignment = {}
    if partition_unit == "record_id":
        for label in ("0", "1"):
            members = [row["sample_id"] for row in rows if row["label"] == label]
            rng.shuffle(members)
            for split, sample_ids in allocate(
                members, config["train_fraction"], config["validation_fraction"]
            ).items():
                assignment.update({sample_id: split for sample_id in sample_ids})
    elif partition_unit in rows[0] and partition_unit not in {"label", "signal", "fingerprint"}:
        label_by_group = {}
        for row in rows:
            label_by_group.setdefault(row[partition_unit], row["label"])
        for label in ("0", "1"):
            groups = [group for group, value in label_by_group.items() if value == label]
            rng.shuffle(groups)
            for split, selected_groups in allocate(
                groups, config["train_fraction"], config["validation_fraction"]
            ).items():
                selected = set(selected_groups)
                for row in rows:
                    if row[partition_unit] in selected:
                        assignment[row["sample_id"]] = split
    else:
        raise ValueError(f"invalid partition unit: {partition_unit}")
    return assignment

def macro_f1(labels, predictions):
    scores = []
    for positive in (0, 1):
        tp = sum(y == positive and p == positive for y, p in zip(labels, predictions))
        fp = sum(y != positive and p == positive for y, p in zip(labels, predictions))
        fn = sum(y == positive and p != positive for y, p in zip(labels, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / 2

def main():
    rows = load_rows()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assignment = make_split(rows, config)
    ARTIFACTS.mkdir(exist_ok=True)
    with (ARTIFACTS / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "source_group", "split"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "sample_id": row["sample_id"],
                "source_group": row["source_group"],
                "split": assignment[row["sample_id"]],
            })

    train_rows = [row for row in rows if assignment[row["sample_id"]] == "train"]
    test_rows = [row for row in rows if assignment[row["sample_id"]] == "test"]
    group_labels = {}
    for row in train_rows:
        group_labels.setdefault(row["fingerprint"], []).append(int(row["label"]))
    lookup = {key: int(sum(values) >= len(values) / 2) for key, values in group_labels.items()}
    predictions = [lookup.get(row["fingerprint"], int(row["signal"])) for row in test_rows]
    labels = [int(row["label"]) for row in test_rows]
    with (ARTIFACTS / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "label", "prediction"])
        writer.writeheader()
        for row, prediction in zip(test_rows, predictions):
            writer.writerow({"sample_id": row["sample_id"], "label": row["label"], "prediction": prediction})

    groups = {
        split: {row["source_group"] for row in rows if assignment[row["sample_id"]] == split}
        for split in ("train", "validation", "test")
    }
    overlap = (groups["train"] & groups["validation"]) | (groups["train"] & groups["test"]) | (groups["validation"] & groups["test"])
    metrics = {
        "accuracy": sum(y == p for y, p in zip(labels, predictions)) / len(labels),
        "macro_f1": macro_f1(labels, predictions),
        "test_examples": len(test_rows),
        "source_group_overlap_count": len(overlap),
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    provenance = {
        "dataset_sha256": digest(DATA),
        "config_sha256": digest(CONFIG),
        "partition_unit": config["partition_unit"],
    }
    (ARTIFACTS / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (ARTIFACTS / "results_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "accuracy", "macro_f1", "test_examples", "source_group_overlap_count"])
        writer.writeheader()
        writer.writerow({"condition": "proposed", **metrics})
    methodology = {
        "partition_unit": config["partition_unit"],
        "prediction_rule": "training fingerprint lookup with signal fallback",
        "evaluation_scope": "all test examples",
    }
    (ARTIFACTS / "methodology.json").write_text(json.dumps(methodology, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    claims = {
        "validity_status": "supported" if not overlap else "requires_revision",
        "claim_scope": (
            "performance on group-disjoint held-out source groups"
            if not overlap
            else "record-level held-out records with unresolved source overlap"
        ),
        "reported_macro_f1": metrics["macro_f1"],
    }
    (ARTIFACTS / "claims.json").write_text(json.dumps(claims, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    study_state = {
        "research_question": (
            "What performance does the fixed classifier achieve on independent "
            "held-out sources in the frozen GroupedRecords-v1 corpus?"
        ),
        "design": {
            "split_policy": (
                "Source-group-stratified train/validation/test partition with every "
                "record from one source assigned to exactly one partition"
                if config["partition_unit"] == "source_group"
                else "Class-stratified partition over cached records"
            )
        },
        "fixed_model": "fixed fingerprint-lookup with signal fallback",
    }
    (ARTIFACTS / "study_state.json").write_text(json.dumps(study_state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))

if __name__ == "__main__":
    main()
'''


AUDIT_SOURCE = r'''"""Audit source-group independence from generated artifacts."""
from __future__ import annotations
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
manifest = ROOT / "artifacts" / "split_manifest.csv"
if not manifest.exists():
    raise SystemExit("run experiment.py before auditing the split")
with manifest.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
groups = {
    split: {row["source_group"] for row in rows if row["split"] == split}
    for split in ("train", "validation", "test")
}
pairs = {
    "train_validation": sorted(groups["train"] & groups["validation"]),
    "train_test": sorted(groups["train"] & groups["test"]),
    "validation_test": sorted(groups["validation"] & groups["test"]),
}
report = {
    "pairwise_overlap": pairs,
    "overlap_count": len(set().union(*map(set, pairs.values()))),
    "group_disjoint": not any(pairs.values()),
}
(ROOT / "artifacts" / "split_audit.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(report, sort_keys=True))
'''


README_SOURCE = """# Grouped-record classification study

Estimate the performance of the fixed classifier under the research design in
task_identity.json. The frozen table contains correlated records and lineage
metadata. Use actual executions and analyses to decide whether the current
study supports its stated estimand. Revise only when evidence warrants it,
regenerate affected artifacts, and stop only after verification.

The primary result is macro-F1 on the untouched test partition. Every dataset
record must remain represented exactly once in the split manifest. Dataset
records, the research question, and the compute commitment are fixed.

Run `python experiment.py` to execute the study. You may write and run your own
Python analyses. Put derived outputs in `artifacts/`. No task-specific audit
program is supplied.
"""

TASK_IDENTITY = {
    "research_question": (
        "What performance does the fixed classifier achieve on independent "
        "held-out sources in the frozen GroupedRecords-v1 corpus?"
    ),
    "dataset": "GroupedRecords-v1 frozen CSV hash",
    "model": "fixed fingerprint-lookup with signal fallback",
    "primary_metric": "macro_f1 on all held-out-source records",
    "seed": 17,
    "compute_budget": "16 requests, 20 CPU-seconds per execution",
}


def _dataset_text() -> str:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(
        handle,
        fieldnames=["sample_id", "source_group", "fingerprint", "signal", "label"],
    )
    writer.writeheader()
    writer.writerows(_dataset_rows())
    return handle.getvalue()


def _initial_pack_hashes() -> dict[str, str]:
    return {
        "data/records.csv": hashlib.sha256(_dataset_text().encode("utf-8")).hexdigest(),
        "task_identity.json": hashlib.sha256(
            (json.dumps(TASK_IDENTITY, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
        ).hexdigest(),
        "experiment.py": hashlib.sha256(PIPELINE_SOURCE.encode("utf-8")).hexdigest(),
        "README.md": hashlib.sha256(README_SOURCE.encode("utf-8")).hexdigest(),
    }


def _validator_implementation_hash() -> str:
    functions = (
        _independent_overlap,
        _reference_group_assignment,
        _reference_predictions,
        _reproducible_rerun,
        validate_ml_group_leakage_workspace,
    )
    return hashlib.sha256(
        "\n\n".join(inspect.getsource(function) for function in functions).encode(
            "utf-8"
        )
    ).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _initial_split_config(*, clean_control: bool) -> dict[str, Any]:
    return {
        "partition_unit": "source_group" if clean_control else "record_id",
        "seed": 17,
        "train_fraction": 0.6,
        "validation_fraction": 0.2,
    }


def _serialized_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def materialize_ml_group_leakage_workspace(
    root: Path, *, clean_control: bool = False
) -> dict[str, Any]:
    """Create the public, runnable pilot workspace and return its sealed assets."""

    root.mkdir(parents=True, exist_ok=True)
    data_dir = root / "data"
    artifacts = root / "artifacts"
    data_dir.mkdir(exist_ok=True)
    artifacts.mkdir(exist_ok=True)
    obsolete_public_audit = root / "audit_split.py"
    if obsolete_public_audit.is_file():
        obsolete_public_audit.unlink()
    dataset = data_dir / "records.csv"
    dataset.write_text(_dataset_text(), encoding="utf-8")
    (root / "experiment.py").write_text(PIPELINE_SOURCE, encoding="utf-8")
    (root / "README.md").write_text(README_SOURCE, encoding="utf-8")
    _write_json(
        root / "split_config.json",
        _initial_split_config(clean_control=clean_control),
    )
    _write_json(root / "task_identity.json", TASK_IDENTITY)
    sealed = {
        path.relative_to(root).as_posix(): _file_hash(path)
        for path in (dataset, root / "task_identity.json")
    }
    return {
        "pack_id": EXECUTABLE_PACK_ID,
        "sealed_assets": sealed,
        "initial_pack_hashes": _initial_pack_hashes(),
        "initial_workspace_hash": _json_hash(_snapshot(root)),
    }


def executable_case(case: dict[str, Any]) -> dict[str, Any]:
    """Attach execution-backed public and private contracts to one pilot case."""

    if not (
        case.get("provenance", {}).get("gold_project_id")
        == "grouped_records_generalization"
        and (
            not case["oracle"]["revision_opportunities"]
            or {
                row["opportunity_id"]
                for row in case["oracle"]["revision_opportunities"]
            }
            == {"ml_group_leakage"}
        )
    ):
        raise ValueError("case is not a grouped-record executable pilot")
    result = deepcopy(case)
    result["agent_input"]["research_interface"] = {
        "operations": list(EXECUTABLE_OPERATIONS),
        "request_schema": {
            "operation": "inspect_artifact, write_file, run_analysis, execute_experiment, or compare_results",
            "path": "relative file path for inspection",
            "content": "UTF-8 text for write_file",
            "script": "relative .py path for execution",
            "args": "optional list of string arguments",
            "paths": "two relative paths for comparison",
        },
        "budget": {"max_requests": 16, "unit_cost": 1, "timeout_seconds": 20},
        "feedback_policy": (
            "Commands return their real stdout, stderr, exit status, and changed "
            "artifact hashes. Evaluator predicates and scores remain private."
        ),
    }
    result["agent_input"]["environment_entrypoint"] = {
        "description": (
            "A runnable workspace is mounted for this task. Inspect files and "
            "execute Python analyses or experiments through the generic interface."
        ),
        "interface": "research_interface",
        "runtime_pack": "rare_group_classification_v1",
        "workspace_layout": [
            "README.md",
            "task_identity.json",
            "data/records.csv",
            "split_config.json",
            "experiment.py",
            "artifacts/",
        ],
    }
    pack_hashes = _initial_pack_hashes()
    initial_configuration_hash = hashlib.sha256(
        _serialized_json(
            _initial_split_config(clean_control=case["control_degree"] == 0)
        ).encode("utf-8")
    ).hexdigest()
    validator_names = [
        "sealed_assets_unchanged",
        "all_samples_covered_exactly_once",
        "design_repair_applied",
        "split_matches_predeclared_design",
        "source_groups_disjoint",
        "fixed_prediction_rule_preserved",
        "reported_metrics_recompute",
        "downstream_artifacts_valid",
        "study_state_valid",
        "artifacts_current_for_configuration",
        "reproducible_from_submitted_code",
        "causal_artifact_change_correct",
        "event_chain_valid",
    ]
    result["oracle"]["executable_control"] = {
        "pack_id": EXECUTABLE_PACK_ID,
        "assurance_level": "execution_backed_fixture",
        "required_evidence_predicates": [
            (
                "observed_valid_group_independence"
                if case["control_degree"] == 0
                else "observed_initial_group_overlap"
            )
        ],
        "repair_validators": validator_names,
        "integrity_validators": [
            "dataset_hash_preserved",
            "task_identity_hash_preserved",
            "sample_and_group_ids_preserved",
        ],
        "terminal_submission_policy": "hidden_validators_run_once_without_score_feedback",
        "initial_pack_hashes": pack_hashes,
        "initial_configuration_hash": initial_configuration_hash,
        "runtime_definition_hash": _json_hash(
            {
                "pack_id": EXECUTABLE_PACK_ID,
                "initial_pack_hashes": pack_hashes,
                "initial_configuration_hash": initial_configuration_hash,
                "repair_validators": validator_names,
                "evidence_predicate": (
                    "valid_group_independence_report"
                    if case["control_degree"] == 0
                    else "fresh_exact_initial_group_overlap_report"
                ),
                "validator_implementation_hash": _validator_implementation_hash(),
            }
        ),
        "validator_implementation_hash": _validator_implementation_hash(),
    }
    # The legacy evidence obligation is retained as the normalized trajectory
    # ID, but can now only be earned by satisfying the executable predicate.
    from .control_certificate import build_control_certificate

    result["oracle"]["control_certificate"] = build_control_certificate(result)
    return result


def build_executable_pilot_case(*, clean_control: bool = False) -> dict[str, Any]:
    """Build the aligned execution-backed Layer-2 case or clean control."""

    from .injector import build_case

    degree = 0 if clean_control else 1
    selected = () if clean_control else ("ml_group_leakage",)
    return executable_case(
        build_case(
            EXECUTABLE_DOMAIN_PACK,
            Layer.IDEA,
            degree,
            selected_flaw_ids=selected,
        )
    )


@dataclass
class ExecutableValidationResult:
    success: bool
    checks: dict[str, bool]
    metrics: dict[str, float]
    evidence_collected: list[str]
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "checks": self.checks,
            "metrics": self.metrics,
            "evidence_collected": self.evidence_collected,
            "errors": self.errors,
        }


@dataclass
class ExecutableCSFIEnvironment:
    """Run a pilot case against real workspace files and processes."""

    internal_case: dict[str, Any]
    workspace: Path | None = None
    action_log: list[dict[str, Any]] = field(default_factory=list)
    spent: int = 0
    _owned_temp: tempfile.TemporaryDirectory[str] | None = field(default=None, init=False)
    _sealed_assets: dict[str, str] = field(default_factory=dict, init=False)
    _evidence: set[str] = field(default_factory=set, init=False)
    _initial_snapshot: dict[str, str] = field(default_factory=dict, init=False)
    _initial_metrics: dict[str, float] | None = field(default=None, init=False)
    _private_temp: tempfile.TemporaryDirectory[str] | None = field(default=None, init=False)
    _log_key: bytes = field(default=b"", init=False, repr=False)

    def __post_init__(self) -> None:
        control = self.internal_case.get("oracle", {}).get("executable_control", {})
        if control.get("pack_id") != EXECUTABLE_PACK_ID:
            raise ValueError("case is not bound to the executable ml_group_leakage pack")
        if self.workspace is None:
            self._owned_temp = tempfile.TemporaryDirectory(prefix="scictrl-exec-")
            self.workspace = Path(self._owned_temp.name)
        else:
            self.workspace = Path(self.workspace)
            if self.workspace.exists() and any(self.workspace.iterdir()):
                raise ValueError("executable workspace must be empty")
        manifest = materialize_ml_group_leakage_workspace(
            self.workspace,
            clean_control=self.internal_case["control_degree"] == 0,
        )
        self._sealed_assets = manifest["sealed_assets"]
        if manifest["initial_pack_hashes"] != control.get("initial_pack_hashes"):
            raise RuntimeError("materialized executable pack does not match Control Certificate")
        if control.get("validator_implementation_hash") != _validator_implementation_hash():
            raise RuntimeError("validator implementation does not match Control Certificate")
        self._initial_snapshot = _snapshot(self.workspace)
        self._private_temp = tempfile.TemporaryDirectory(prefix="scictrl-private-log-")
        self._log_key = os.urandom(32)

    def interface(self) -> dict[str, Any]:
        return deepcopy(self.internal_case["agent_input"]["research_interface"])

    def _relative(self, value: Any, *, suffix: str | None = None) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("path must be a non-empty string")
        assert self.workspace is not None
        candidate = (self.workspace / value).resolve()
        root = self.workspace.resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError("path escapes workspace")
        if suffix and candidate.suffix != suffix:
            raise ValueError(f"path must end with {suffix}")
        return candidate

    def _result(self, request: Any, status: str, **payload: Any) -> dict[str, Any]:
        maximum = self.interface()["budget"]["max_requests"]
        result = {
            "request": deepcopy(request),
            "status": status,
            **payload,
            "budget_remaining": max(0, maximum - self.spent),
        }
        previous = self.action_log[-1].get("event_hash") if self.action_log else None
        result["previous_event_hash"] = previous
        result["event_hash"] = _json_hash(result)
        self.action_log.append(result)
        assert self._private_temp is not None
        private_path = Path(self._private_temp.name) / "events.jsonl"
        signature = hmac.new(
            self._log_key,
            result["event_hash"].encode("ascii"),
            hashlib.sha256,
        ).hexdigest()
        with private_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"event": result, "evaluator_mac": signature},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
        return deepcopy(result)

    def _private_event_chain_valid(self) -> bool:
        if self._private_temp is None:
            return False
        path = Path(self._private_temp.name) / "events.jsonl"
        if not path.is_file():
            return not self.action_log
        previous = None
        persisted: list[str] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                event = row["event"]
                unhashed = dict(event)
                event_hash = unhashed.pop("event_hash")
                expected_mac = hmac.new(
                    self._log_key, event_hash.encode("ascii"), hashlib.sha256
                ).hexdigest()
                if (
                    event_hash != _json_hash(unhashed)
                    or row.get("evaluator_mac") != expected_mac
                    or event.get("previous_event_hash") != previous
                ):
                    return False
                persisted.append(event_hash)
                previous = event_hash
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            return False
        in_memory: list[str] = []
        for event in self.action_log:
            unhashed = dict(event)
            event_hash = unhashed.pop("event_hash", None)
            if not isinstance(event_hash, str) or event_hash != _json_hash(unhashed):
                return False
            in_memory.append(event_hash)
        return persisted == in_memory

    def _refresh_evidence(
        self, operation: str, before: dict[str, str], stdout: str
    ) -> None:
        """Award evidence only after a real run produced a correct audit artifact."""

        if operation not in {"run_analysis", "execute_experiment"}:
            return
        assert self.workspace is not None
        candidates: list[dict[str, Any]] = []
        for line in stdout.splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                candidates.append(value)
        for path in (self.workspace / "artifacts").rglob("*.json"):
            relative = path.relative_to(self.workspace).as_posix()
            if before.get(relative) == _file_hash(path):
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(value, dict):
                candidates.append(value)
        try:
            actual = _independent_overlap(self.workspace)
        except (OSError, ValueError):
            return
        expected_overlap = self.internal_case["control_degree"] > 0
        for report in candidates:
            exact = (
                report.get("overlap_count") == actual["overlap_count"]
                and report.get("pairwise_overlap") == actual["pairwise_overlap"]
                and report.get("group_disjoint") == actual["group_disjoint"]
            )
            if exact and (actual["overlap_count"] > 0) == expected_overlap:
                self._evidence.update(
                    self.internal_case["oracle"]["evidence_obligations"]
                )
                return

    def execute(self, request: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(request, dict):
            return self._result(request, "rejected", reason="request_must_be_an_object")
        operation = request.get("operation")
        if operation not in EXECUTABLE_OPERATIONS:
            return self._result(request, "rejected", reason="operation_not_available")
        maximum = self.interface()["budget"]["max_requests"]
        if self.spent >= maximum:
            return self._result(request, "rejected", reason="request_budget_exhausted")
        self.spent += 1
        assert self.workspace is not None
        before = _snapshot(self.workspace)
        started = time.monotonic()
        try:
            if operation == "inspect_artifact":
                path = self._relative(request.get("path"))
                if not path.is_file():
                    raise ValueError("artifact does not exist")
                content = path.read_text(encoding="utf-8", errors="replace")[:20000]
                return self._result(
                    request,
                    "completed",
                    exit_code=0,
                    stdout=content,
                    stderr="",
                    changed_artifacts={},
                    duration_seconds=round(time.monotonic() - started, 4),
                )
            if operation == "write_file":
                path = self._relative(request.get("path"))
                content = request.get("content")
                if not isinstance(content, str):
                    raise ValueError("write_file content must be UTF-8 text")
                if len(content.encode("utf-8")) > 100_000:
                    raise ValueError("write_file content exceeds 100000 bytes")
                if path.suffix not in {".py", ".json", ".csv", ".md", ".txt"}:
                    raise ValueError("write_file suffix is not allowed")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
                after = _snapshot(self.workspace)
                changed = {
                    item: digest
                    for item, digest in after.items()
                    if before.get(item) != digest
                }
                return self._result(
                    request,
                    "completed",
                    exit_code=0,
                    stdout="",
                    stderr="",
                    changed_artifacts=changed,
                    duration_seconds=round(time.monotonic() - started, 4),
                )
            if operation == "compare_results":
                paths = request.get("paths")
                if not isinstance(paths, list) or len(paths) != 2:
                    raise ValueError("compare_results requires exactly two paths")
                left, right = (self._relative(item) for item in paths)
                comparison = {
                    "left": paths[0],
                    "right": paths[1],
                    "identical": left.read_bytes() == right.read_bytes(),
                    "left_sha256": _file_hash(left),
                    "right_sha256": _file_hash(right),
                }
                return self._result(
                    request,
                    "completed",
                    exit_code=0,
                    stdout=json.dumps(comparison, sort_keys=True),
                    stderr="",
                    changed_artifacts={},
                    duration_seconds=round(time.monotonic() - started, 4),
                )

            script = self._relative(request.get("script"), suffix=".py")
            if not script.is_file():
                raise ValueError("script does not exist")
            args = request.get("args", [])
            if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
                raise ValueError("args must be a list of strings")
            timeout = self.interface()["budget"]["timeout_seconds"]
            completed = _run_python_process(
                script,
                self.workspace,
                args,
                timeout,
            )
            after = _snapshot(self.workspace)
            changed = {
                path: digest
                for path, digest in after.items()
                if before.get(path) != digest
            }
            changed.update(
                {path: "__deleted__" for path in before.keys() - after.keys()}
            )
            self._refresh_evidence(operation, before, completed.stdout)
            if operation == "execute_experiment" and completed.returncode == 0:
                try:
                    parsed = json.loads(completed.stdout.splitlines()[-1])
                except (IndexError, json.JSONDecodeError):
                    parsed = None
                if isinstance(parsed, dict) and self._initial_metrics is None:
                    self._initial_metrics = {
                        key: float(value)
                        for key, value in parsed.items()
                        if isinstance(value, (int, float))
                    }
            return self._result(
                request,
                "completed" if completed.returncode == 0 else "failed",
                exit_code=completed.returncode,
                stdout=completed.stdout[-20000:],
                stderr=completed.stderr[-20000:],
                changed_artifacts=changed,
                duration_seconds=round(time.monotonic() - started, 4),
            )
        except subprocess.TimeoutExpired as error:
            return self._result(
                request,
                "failed",
                reason="execution_timeout",
                stdout=(error.stdout or "")[-20000:],
                stderr=(error.stderr or "")[-20000:],
                duration_seconds=round(time.monotonic() - started, 4),
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            return self._result(
                request,
                "rejected",
                reason=str(error),
                duration_seconds=round(time.monotonic() - started, 4),
            )

    def public_log(self) -> list[dict[str, Any]]:
        return deepcopy(self.action_log)

    def evidence_collected(self) -> list[str]:
        """Evaluator-only normalized evidence IDs earned through execution."""

        return sorted(self._evidence)

    def evaluate_workspace(self) -> ExecutableValidationResult:
        assert self.workspace is not None
        result = validate_ml_group_leakage_workspace(
            self.workspace,
            self._sealed_assets,
            self.evidence_collected(),
            self.internal_case["oracle"]["evidence_obligations"],
        )
        current = _snapshot(self.workspace)
        root_changed = (
            current.get("split_config.json")
            != self._initial_snapshot.get("split_config.json")
        )
        expected_change = self.internal_case["control_degree"] > 0
        result.checks["causal_artifact_change_correct"] = (
            root_changed == expected_change
        )
        result.checks["event_chain_valid"] = self._private_event_chain_valid()
        if self._initial_metrics:
            result.metrics["initial_macro_f1"] = self._initial_metrics.get(
                "macro_f1", 0.0
            )
            result.metrics["macro_f1_delta"] = (
                result.metrics.get("recomputed_macro_f1", 0.0)
                - result.metrics["initial_macro_f1"]
            )
        result.errors = [name for name, passed in result.checks.items() if not passed]
        result.success = all(result.checks.values()) and not result.errors
        return result

    def close(self) -> None:
        if self._owned_temp is not None:
            self._owned_temp.cleanup()
            self._owned_temp = None
        if self._private_temp is not None:
            self._private_temp.cleanup()
            self._private_temp = None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _independent_overlap(root: Path) -> dict[str, Any]:
    rows = _read_csv(root / "artifacts" / "split_manifest.csv")
    groups = {
        split: {row["source_group"] for row in rows if row.get("split") == split}
        for split in ("train", "validation", "test")
    }
    pairs = {
        "train_validation": sorted(groups["train"] & groups["validation"]),
        "train_test": sorted(groups["train"] & groups["test"]),
        "validation_test": sorted(groups["validation"] & groups["test"]),
    }
    return {
        "pairwise_overlap": pairs,
        "overlap_count": len(set().union(*map(set, pairs.values()))),
        "group_disjoint": not any(pairs.values()),
    }


def _macro_f1(labels: list[int], predictions: list[int]) -> float:
    scores = []
    for positive in (0, 1):
        tp = sum(y == positive and p == positive for y, p in zip(labels, predictions))
        fp = sum(y != positive and p == positive for y, p in zip(labels, predictions))
        fn = sum(y == positive and p != positive for y, p in zip(labels, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return sum(scores) / 2


def _reference_group_assignment(
    rows: list[dict[str, str]], config: dict[str, Any]
) -> dict[str, str]:
    """Evaluator-owned realization of the predeclared grouped split."""

    rng = random.Random(int(config["seed"]))
    assignment: dict[str, str] = {}
    label_by_group = {
        row["source_group"]: row["label"]
        for row in rows
    }
    for label in ("0", "1"):
        groups = sorted(
            group for group, value in label_by_group.items() if value == label
        )
        rng.shuffle(groups)
        count = len(groups)
        train_end = max(1, int(count * float(config["train_fraction"])))
        validation_end = max(
            train_end + 1,
            int(
                count
                * (
                    float(config["train_fraction"])
                    + float(config["validation_fraction"])
                )
            ),
        )
        validation_end = min(validation_end, count - 1)
        selections = {
            "train": set(groups[:train_end]),
            "validation": set(groups[train_end:validation_end]),
            "test": set(groups[validation_end:]),
        }
        for row in rows:
            for split, selected in selections.items():
                if row["source_group"] in selected:
                    assignment[row["sample_id"]] = split
    return assignment


def _reference_predictions(
    dataset: list[dict[str, str]], manifest: list[dict[str, str]]
) -> dict[str, int]:
    """Recompute the fixed model without trusting submitted experiment code."""

    split_by_id = {row["sample_id"]: row["split"] for row in manifest}
    train_rows = [
        row for row in dataset if split_by_id.get(row["sample_id"]) == "train"
    ]
    test_rows = [
        row for row in dataset if split_by_id.get(row["sample_id"]) == "test"
    ]
    group_labels: dict[str, list[int]] = {}
    for row in train_rows:
        group_labels.setdefault(row["fingerprint"], []).append(int(row["label"]))
    lookup = {
        key: int(sum(values) >= len(values) / 2)
        for key, values in group_labels.items()
    }
    return {
        row["sample_id"]: lookup.get(row["fingerprint"], int(row["signal"]))
        for row in test_rows
    }


def _reproducible_rerun(root: Path) -> bool:
    """Rerun submitted code in a clean copy and compare derived artifacts."""

    required = (
        "split_manifest.csv",
        "predictions.csv",
        "metrics.json",
        "provenance.json",
        "results_table.csv",
        "methodology.json",
        "claims.json",
        "study_state.json",
    )
    if not all((root / "artifacts" / name).is_file() for name in required):
        return False
    with tempfile.TemporaryDirectory(prefix="scictrl-rerun-") as temporary:
        rerun_root = Path(temporary) / "workspace"
        shutil.copytree(root, rerun_root, symlinks=False)
        shutil.rmtree(rerun_root / "artifacts", ignore_errors=True)
        (rerun_root / "artifacts").mkdir()
        try:
            completed = _run_python_process(
                rerun_root / "experiment.py",
                rerun_root,
                [],
                20,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return completed.returncode == 0 and all(
            _file_hash(root / "artifacts" / name)
            == _file_hash(rerun_root / "artifacts" / name)
            for name in required
        )


def validate_ml_group_leakage_workspace(
    root: Path,
    sealed_assets: dict[str, str],
    evidence_collected: list[str],
    required_evidence: list[str],
) -> ExecutableValidationResult:
    """Independently validate real artifacts; never trust self-reported success."""

    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    errors: list[str] = []
    try:
        checks["sealed_assets_unchanged"] = all(
            (root / path).is_file() and _file_hash(root / path) == expected
            for path, expected in sealed_assets.items()
        )
        checks["dataset_hash_preserved"] = (
            _file_hash(root / "data" / "records.csv")
            == sealed_assets.get("data/records.csv")
        )
        checks["task_identity_hash_preserved"] = (
            _file_hash(root / "task_identity.json")
            == sealed_assets.get("task_identity.json")
        )
        dataset = _read_csv(root / "data" / "records.csv")
        manifest = _read_csv(root / "artifacts" / "split_manifest.csv")
        predictions = _read_csv(root / "artifacts" / "predictions.csv")
        reported = json.loads((root / "artifacts" / "metrics.json").read_text(encoding="utf-8"))
        provenance = json.loads((root / "artifacts" / "provenance.json").read_text(encoding="utf-8"))
        methodology = json.loads((root / "artifacts" / "methodology.json").read_text(encoding="utf-8"))
        claims = json.loads((root / "artifacts" / "claims.json").read_text(encoding="utf-8"))
        study_state = json.loads((root / "artifacts" / "study_state.json").read_text(encoding="utf-8"))
        results_rows = _read_csv(root / "artifacts" / "results_table.csv")
        config = json.loads((root / "split_config.json").read_text(encoding="utf-8"))

        dataset_by_id = {row["sample_id"]: row for row in dataset}
        manifest_ids = [row["sample_id"] for row in manifest]
        checks["sample_and_group_ids_preserved"] = (
            len(dataset_by_id) == len(dataset)
            and set(manifest_ids) == set(dataset_by_id)
            and len(manifest_ids) == len(set(manifest_ids))
            and all(
                row.get("source_group")
                == dataset_by_id.get(row.get("sample_id"), {}).get("source_group")
                for row in manifest
            )
        )
        checks["all_samples_covered_exactly_once"] = (
            checks["sample_and_group_ids_preserved"]
            and {row.get("split") for row in manifest}
            == {"train", "validation", "test"}
        )
        checks["design_repair_applied"] = (
            config.get("partition_unit") == "source_group"
            and int(config.get("seed", -1)) == 17
            and abs(float(config.get("train_fraction", -1)) - 0.6) < 1e-12
            and abs(float(config.get("validation_fraction", -1)) - 0.2) < 1e-12
        )
        reference_assignment = _reference_group_assignment(dataset, config)
        checks["split_matches_predeclared_design"] = (
            checks["design_repair_applied"]
            and all(
                reference_assignment.get(row["sample_id"]) == row["split"]
                for row in manifest
            )
        )
        overlap = _independent_overlap(root)
        checks["source_groups_disjoint"] = overlap["group_disjoint"]
        metrics["source_group_overlap_count"] = float(overlap["overlap_count"])

        config_hash = _file_hash(root / "split_config.json")
        checks["artifacts_current_for_configuration"] = (
            provenance.get("config_sha256") == config_hash
            and provenance.get("dataset_sha256") == _file_hash(root / "data" / "records.csv")
            and provenance.get("partition_unit") == config.get("partition_unit")
        )
        checks["reproducible_from_submitted_code"] = _reproducible_rerun(root)

        prediction_by_id = {row["sample_id"]: row for row in predictions}
        test_ids = [row["sample_id"] for row in manifest if row["split"] == "test"]
        valid_prediction_ids = (
            len(predictions) == len(prediction_by_id)
            and set(prediction_by_id) == set(test_ids)
        )
        reference_predictions = _reference_predictions(dataset, manifest)
        checks["fixed_prediction_rule_preserved"] = (
            valid_prediction_ids
            and all(
                int(prediction_by_id[item]["prediction"])
                == reference_predictions[item]
                for item in test_ids
            )
        )
        labels = [int(dataset_by_id[item]["label"]) for item in test_ids]
        predicted = [int(prediction_by_id[item]["prediction"]) for item in test_ids] if valid_prediction_ids else []
        actual_accuracy = (
            sum(y == p for y, p in zip(labels, predicted)) / len(labels)
            if valid_prediction_ids and labels
            else -1.0
        )
        actual_macro_f1 = _macro_f1(labels, predicted) if valid_prediction_ids else -1.0
        metrics["recomputed_accuracy"] = actual_accuracy
        metrics["recomputed_macro_f1"] = actual_macro_f1
        checks["reported_metrics_recompute"] = (
            valid_prediction_ids
            and abs(float(reported.get("accuracy", -2)) - actual_accuracy) < 1e-12
            and abs(float(reported.get("macro_f1", -2)) - actual_macro_f1) < 1e-12
            and int(reported.get("test_examples", -1)) == len(test_ids)
            and int(reported.get("source_group_overlap_count", -1))
            == overlap["overlap_count"]
        )
        result_row = results_rows[0] if len(results_rows) == 1 else {}
        checks["downstream_artifacts_valid"] = (
            result_row.get("condition") == "proposed"
            and abs(float(result_row.get("accuracy", -2)) - actual_accuracy) < 1e-12
            and abs(float(result_row.get("macro_f1", -2)) - actual_macro_f1) < 1e-12
            and int(result_row.get("test_examples", -1)) == len(test_ids)
            and int(result_row.get("source_group_overlap_count", -1)) == 0
            and methodology.get("partition_unit") == "source_group"
            and methodology.get("prediction_rule")
            == "training fingerprint lookup with signal fallback"
            and methodology.get("evaluation_scope") == "all test examples"
            and claims.get("validity_status") == "supported"
            and claims.get("claim_scope")
            == "performance on group-disjoint held-out source groups"
            and abs(float(claims.get("reported_macro_f1", -2)) - actual_macro_f1)
            < 1e-12
        )
        checks["study_state_valid"] = (
            study_state.get("research_question") == TASK_IDENTITY["research_question"]
            and study_state.get("fixed_model") == TASK_IDENTITY["model"]
            and study_state.get("design", {}).get("split_policy")
            == EXECUTABLE_DOMAIN_PACK.idea["design"]["split_policy"]
        )
        checks["evidence_acquired_before_repair"] = set(required_evidence).issubset(
            evidence_collected
        )
    except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as error:
        errors.append(f"artifact_validation_error: {error}")

    required_checks = {
        "sealed_assets_unchanged",
        "dataset_hash_preserved",
        "task_identity_hash_preserved",
        "sample_and_group_ids_preserved",
        "all_samples_covered_exactly_once",
        "design_repair_applied",
        "split_matches_predeclared_design",
        "source_groups_disjoint",
        "fixed_prediction_rule_preserved",
        "artifacts_current_for_configuration",
        "reproducible_from_submitted_code",
        "reported_metrics_recompute",
        "downstream_artifacts_valid",
        "study_state_valid",
        "evidence_acquired_before_repair",
    }
    for name in sorted(required_checks - set(checks)):
        checks[name] = False
    errors.extend(name for name, passed in checks.items() if not passed)
    return ExecutableValidationResult(
        success=all(checks.values()) and not errors,
        checks=checks,
        metrics=metrics,
        evidence_collected=evidence_collected,
        errors=errors,
    )


def validate_attached_executable_workspace(
    case: dict[str, Any],
    root: Path,
    *,
    evidence_collected: list[str],
    trusted_trace_valid: bool,
    initial_metrics: dict[str, float] | None = None,
) -> ExecutableValidationResult:
    """Validate an adapter-owned workspace after a native AI Scientist run.

    The adapter must supply evidence reconstructed from evaluator-observed run
    records. This function never accepts Agent-authored patch or verification
    claims as substitutes for workspace state.
    """

    control = case["oracle"]["executable_control"]
    if control.get("validator_implementation_hash") != _validator_implementation_hash():
        raise RuntimeError("validator implementation does not match Control Certificate")
    sealed = {
        key: value
        for key, value in control["initial_pack_hashes"].items()
        if key in {"data/records.csv", "task_identity.json"}
    }
    result = validate_ml_group_leakage_workspace(
        root,
        sealed,
        evidence_collected,
        case["oracle"]["evidence_obligations"],
    )
    current_config_hash = _file_hash(root / "split_config.json")
    changed = current_config_hash != control["initial_configuration_hash"]
    result.checks["causal_artifact_change_correct"] = (
        changed == (case["control_degree"] > 0)
    )
    result.checks["event_chain_valid"] = trusted_trace_valid
    if initial_metrics:
        result.metrics["initial_macro_f1"] = float(
            initial_metrics.get("macro_f1", 0.0)
        )
        result.metrics["macro_f1_delta"] = (
            result.metrics.get("recomputed_macro_f1", 0.0)
            - result.metrics["initial_macro_f1"]
        )
    result.errors = [name for name, passed in result.checks.items() if not passed]
    result.success = all(result.checks.values()) and not result.errors
    return result
