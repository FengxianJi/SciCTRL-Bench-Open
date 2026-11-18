from __future__ import annotations

import csv
import json
from pathlib import Path

from scictrl.control_certificate import validate_control_certificate
from scictrl.environment import CSFIEnvironment
from scictrl.evaluator import evaluate_executable_submission
from scictrl.executable_runtime import (
    ExecutableCSFIEnvironment,
    build_executable_pilot_case,
)
from scictrl.injector import public_view


def _case(*, clean_control: bool = False) -> dict:
    return build_executable_pilot_case(clean_control=clean_control)


AUDIT_CODE = '''import csv, json
from pathlib import Path
root = Path(__file__).resolve().parent
with (root / "artifacts" / "split_manifest.csv").open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
groups = {split: {r["source_group"] for r in rows if r["split"] == split} for split in ("train", "validation", "test")}
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
print(json.dumps(report, sort_keys=True))
'''


def _run(env: ExecutableCSFIEnvironment, script: str, operation: str) -> dict:
    return env.execute({"operation": operation, "script": script, "args": []})


def _repair(env: ExecutableCSFIEnvironment) -> None:
    assert env.workspace is not None
    path = env.workspace / "split_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["partition_unit"] = "source_group"
    result = env.execute(
        {
            "operation": "write_file",
            "path": "split_config.json",
            "content": json.dumps(config, indent=2, sort_keys=True) + "\n",
        }
    )
    assert result["status"] == "completed"


def _audit(env: ExecutableCSFIEnvironment) -> dict:
    write = env.execute(
        {"operation": "write_file", "path": "diagnose.py", "content": AUDIT_CODE}
    )
    assert write["status"] == "completed"
    return _run(env, "diagnose.py", "run_analysis")


def _execute_full_repair(env: ExecutableCSFIEnvironment) -> tuple[float, float]:
    initial = _run(env, "experiment.py", "execute_experiment")
    initial_metrics = json.loads(initial["stdout"])
    audit = _audit(env)
    assert json.loads(audit["stdout"])["overlap_count"] > 0
    _repair(env)
    repaired = _run(env, "experiment.py", "execute_experiment")
    repaired_metrics = json.loads(repaired["stdout"])
    _audit(env)
    return initial_metrics["macro_f1"], repaired_metrics["macro_f1"]


def test_executable_case_has_real_runtime_contract_and_valid_certificate() -> None:
    case = _case()
    public = public_view(case)

    assert case["oracle"]["control_certificate"]["assurance_level"] == "execution_backed_fixture"
    assert validate_control_certificate(case)["valid"]
    assert case["oracle"]["executable_control"]["runtime_definition_hash"]
    assert len(case["oracle"]["executable_control"]["initial_pack_hashes"]) == 4
    assert public["agent_input"]["environment_entrypoint"]["workspace_layout"]
    assert "executable_control" not in json.dumps(public)
    public_blob = json.dumps(public).lower()
    assert "focal" not in public_blob
    assert "temperature calibration" not in public_blob
    assert public["agent_input"]["fixed_constraints"]["dataset"] == "GroupedRecords-v1 frozen CSV hash"


def test_real_experiment_exposes_leakage_then_real_repair_passes(tmp_path: Path) -> None:
    env = ExecutableCSFIEnvironment(_case(), tmp_path / "workspace")

    initial_f1, repaired_f1 = _execute_full_repair(env)
    result = env.evaluate_workspace()

    assert initial_f1 > 0.95
    assert repaired_f1 < initial_f1
    assert result.success, result.to_dict()
    assert result.metrics["source_group_overlap_count"] == 0
    assert result.checks["reproducible_from_submitted_code"]
    assert result.checks["study_state_valid"]
    assert result.checks["causal_artifact_change_correct"]
    assert result.checks["event_chain_valid"]
    assert result.metrics["macro_f1_delta"] < 0
    assert env.evidence_collected()


def test_evidence_is_earned_by_executed_correct_audit_not_experiment_alone(tmp_path: Path) -> None:
    env = ExecutableCSFIEnvironment(_case(), tmp_path / "workspace")
    _run(env, "experiment.py", "execute_experiment")
    assert env.evidence_collected() == []

    _audit(env)
    assert env.evidence_collected() == env.internal_case["oracle"]["evidence_obligations"]


def test_main_environment_delegates_to_execution_backend(tmp_path: Path) -> None:
    env = CSFIEnvironment(_case(), workspace=tmp_path / "workspace")
    result = env.execute(
        {"operation": "execute_experiment", "script": "experiment.py", "args": []}
    )

    assert result["status"] == "completed"
    assert "macro_f1" in result["stdout"]
    assert result["event_hash"]
    assert result["previous_event_hash"] is None


def test_terminal_evaluator_ignores_forged_agent_patch_and_uses_workspace(tmp_path: Path) -> None:
    case = _case()
    env = ExecutableCSFIEnvironment(case, tmp_path / "workspace")
    _run(env, "experiment.py", "execute_experiment")
    _audit(env)
    forged = {
        "status": "completed",
        "discovered_slots": ["design.split_policy"],
        "patch": dict(case["oracle"]["canonical_reference_patch"]),
        "evidence_collected": list(case["oracle"]["evidence_obligations"]),
        "downstream_updates": list(case["oracle"]["propagation_obligations"]),
        "verification_passed": True,
        "verification_checks": list(case["oracle"]["verification_obligations"]),
        "stop": True,
    }

    report = evaluate_executable_submission(case, env, forged)

    assert not report["evaluation"]["success"]
    assert not report["workspace_validation"]["checks"]["source_groups_disjoint"]
    assert report["normalized_submission"]["patch"] == {}


def test_execution_backed_evaluation_succeeds_after_applied_repair(tmp_path: Path) -> None:
    case = _case()
    env = ExecutableCSFIEnvironment(case, tmp_path / "workspace")
    _execute_full_repair(env)
    submission = {
        "status": "completed",
        "discovered_slots": ["design.split_policy"],
        "stop": True,
        "actions": [],
        "constraint_changes": {},
    }

    report = evaluate_executable_submission(case, env, submission)

    assert report["evaluation"]["success"], report
    assert report["evaluation"]["outcome"] == "controlled_success"


def test_clean_control_rewards_evidence_grounded_no_revision(tmp_path: Path) -> None:
    case = _case(clean_control=True)
    env = ExecutableCSFIEnvironment(case, tmp_path / "workspace")
    _run(env, "experiment.py", "execute_experiment")
    audit = _audit(env)
    assert json.loads(audit["stdout"])["group_disjoint"]
    submission = {
        "status": "completed",
        "discovered_slots": [],
        "stop": True,
        "actions": [],
        "constraint_changes": {},
    }

    report = evaluate_executable_submission(case, env, submission)

    assert report["evaluation"]["success"], report


def test_copying_public_labels_as_predictions_is_rejected(tmp_path: Path) -> None:
    env = ExecutableCSFIEnvironment(_case(), tmp_path / "workspace")
    _run(env, "experiment.py", "execute_experiment")
    _audit(env)
    _repair(env)
    assert env.workspace is not None
    source = (env.workspace / "experiment.py").read_text(encoding="utf-8")
    original = 'predictions = [lookup.get(row["fingerprint"], int(row["signal"])) for row in test_rows]'
    cheating = 'predictions = [int(row["label"]) for row in test_rows]'
    assert original in source
    env.execute(
        {
            "operation": "write_file",
            "path": "experiment.py",
            "content": source.replace(original, cheating),
        }
    )
    _run(env, "experiment.py", "execute_experiment")

    result = env.evaluate_workspace()

    assert not result.success
    assert not result.checks["fixed_prediction_rule_preserved"]


def test_deleting_rows_cannot_fake_a_clean_split(tmp_path: Path) -> None:
    env = ExecutableCSFIEnvironment(_case(), tmp_path / "workspace")
    _execute_full_repair(env)
    assert env.workspace is not None
    manifest = env.workspace / "artifacts" / "split_manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows[:-1])

    result = env.evaluate_workspace()

    assert not result.success
    assert not result.checks["all_samples_covered_exactly_once"]
    assert not result.checks["reproducible_from_submitted_code"]


def test_renaming_groups_and_forging_metrics_are_rejected(tmp_path: Path) -> None:
    env = ExecutableCSFIEnvironment(_case(), tmp_path / "workspace")
    _execute_full_repair(env)
    assert env.workspace is not None
    manifest = env.workspace / "artifacts" / "split_manifest.csv"
    text = manifest.read_text(encoding="utf-8").replace("source-000", "source-renamed")
    manifest.write_text(text, encoding="utf-8")
    metrics_path = env.workspace / "artifacts" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["macro_f1"] = 1.0
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    result = env.evaluate_workspace()

    assert not result.success
    assert not result.checks["sample_and_group_ids_preserved"]
    assert not result.checks["reported_metrics_recompute"]


def test_changing_dataset_or_task_identity_is_rejected(tmp_path: Path) -> None:
    env = ExecutableCSFIEnvironment(_case(), tmp_path / "workspace")
    _execute_full_repair(env)
    assert env.workspace is not None
    (env.workspace / "task_identity.json").write_text("{}\n", encoding="utf-8")
    dataset = env.workspace / "data" / "records.csv"
    dataset.write_text(dataset.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    result = env.evaluate_workspace()

    assert not result.success
    assert not result.checks["task_identity_hash_preserved"]
    assert not result.checks["dataset_hash_preserved"]


def test_workspace_path_escape_is_rejected_and_budgeted(tmp_path: Path) -> None:
    env = ExecutableCSFIEnvironment(_case(), tmp_path / "workspace")
    result = env.execute({"operation": "inspect_artifact", "path": "../secret.txt"})

    assert result["status"] == "rejected"
    assert result["reason"] == "path escapes workspace"
    assert result["budget_remaining"] == 15


def test_tampered_runtime_event_log_is_rejected(tmp_path: Path) -> None:
    env = ExecutableCSFIEnvironment(_case(), tmp_path / "workspace")
    _execute_full_repair(env)
    env.action_log[-1]["status"] = "forged"

    result = env.evaluate_workspace()

    assert not result.success
    assert not result.checks["event_chain_valid"]
