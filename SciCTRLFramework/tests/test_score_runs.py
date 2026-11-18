import json

from eval.build_vsro_eval import build
from eval.score_runs import score


def _run(case_id: str, predicted: bool, **extra) -> dict:
    return {
        "case_id": case_id,
        "framework": "candidate",
        "model_snapshot": "fixture",
        "seed": 1,
        "predicted_revision": predicted,
        "predicted_factor_ids": [],
        "valid_repair": False,
        "evidence_verified": False,
        "integrity_verified": False,
        **extra,
    }


def test_score_reports_balanced_revision_diagnosis_and_strata():
    labels = [
        {"case_id": "c0", "condition": "control", "expected_revision": False, "layer": "layer1", "control_degree": 0},
        {"case_id": "c1", "condition": "control", "expected_revision": False, "layer": "layer2", "control_degree": 0},
        {"case_id": "i0", "condition": "intervention", "expected_revision": True, "layer": "layer1", "control_degree": 1},
        {"case_id": "i1", "condition": "intervention", "expected_revision": True, "layer": "layer2", "control_degree": 2},
    ]
    report = score(labels, [_run("c0", True), _run("c1", False), _run("i0", True), _run("i1", False)])

    diagnosis = report["framework_profiles"]["candidate"]["revision_diagnosis"]
    assert {key: diagnosis[key] for key in ("tp", "tn", "fp", "fn")} == {
        "tp": 1, "tn": 1, "fp": 1, "fn": 1
    }
    assert diagnosis["sensitivity"] == 0.5
    assert diagnosis["specificity"] == 0.5
    assert diagnosis["balanced_accuracy"] == 0.5
    assert report["framework_profiles"]["candidate"]["diagnosis_by_stratum"]["control_degree:0"]["specificity"] == 0.5
    assert report["framework_profiles"]["candidate"]["clustered_95_ci"]["balanced_accuracy"] is not None


def test_score_reports_factor_localization_and_control_factor_false_positives():
    labels = [
        {"case_id": "control", "condition": "control", "expected_revision": False, "intervention_ids": [], "intervention_categories": []},
        {"case_id": "i1", "condition": "intervention", "expected_revision": True, "intervention_ids": ["a", "b"], "intervention_categories": ["data_split", "metric"]},
        {"case_id": "i2", "condition": "intervention", "expected_revision": True, "intervention_ids": ["c"], "intervention_categories": ["metric"]},
    ]
    runs = [
        _run("control", False, predicted_factor_ids=["noise"], predicted_factor_categories=["metric"]),
        _run("i1", True, predicted_factor_ids=["a"], predicted_factor_categories=["metric"]),
        _run("i2", True, predicted_factor_ids=["c", "extra"], predicted_factor_categories=["metric", "statistical_test"]),
    ]

    profile = score(labels, runs)["framework_profiles"]["candidate"]
    factors = profile["factor_localization"]
    assert factors["micro_precision"] == 0.6667
    assert factors["micro_recall"] == 0.6667
    assert factors["macro_f1_per_case"] == 0.6667
    assert factors["exact_set_match_rate"] == 0.0
    assert profile["control_spurious_factor_rate"] == 1.0
    assert profile["mean_spurious_factor_count_on_controls"] == 1.0
    assert profile["category_localization"]["micro_f1"] == 0.6667
    assert profile["control_spurious_category_rate"] == 1.0


def test_score_adds_conditional_repair_evidence_status_and_calibration_metrics():
    labels = [
        {"case_id": "control", "condition": "control", "expected_revision": False, "source_family_cluster": "p1"},
        {"case_id": "i1", "condition": "intervention", "expected_revision": True, "source_family_cluster": "p1", "control_degree": 1, "intervention_ids": ["a"]},
        {"case_id": "i2", "condition": "intervention", "expected_revision": True, "source_family_cluster": "p2", "control_degree": 2, "intervention_ids": ["b"]},
    ]
    runs = [
        _run("control", False, run_status="success", revision_confidence=0.1, evidence_verified=True, integrity_verified=True),
        _run("i1", True, valid_repair=True, run_status="success", revision_confidence=0.8, evidence_verified=True, integrity_verified=True, total_tokens=100, cost_usd=0.2),
        _run("i2", False, valid_repair=False, run_status="timeout", revision_confidence=0.7, evidence_verified=False, integrity_verified=True, total_tokens=300, cost_usd=0.3),
    ]

    profile = score(labels, runs)["framework_profiles"]["candidate"]
    assert profile["repair_attempt_rate_on_interventions"] == 0.5
    assert profile["valid_repair_rate"] == 0.5
    assert profile["valid_repair_rate_given_attempt"] == 1.0
    assert profile["evidence_verification_by_condition"]["intervention"] == 0.5
    assert profile["joint_evidence_and_integrity_rate"] == 0.6667
    assert profile["end_to_end_intervention_success_rate"] == 0.5
    assert profile["evidence_supported_control_stop_rate"] == 1.0
    assert profile["run_status_counts"] == {"success": 2, "timeout": 1}
    assert profile["resource_telemetry_coverage"]["total_tokens"]["rate"] == 0.6667
    assert profile["revision_confidence_metrics"]["n"] == 2
    assert profile["revision_confidence_metrics"]["brier_score"] == 0.025
    assert profile["revision_confidence_metrics"]["auc"] == 1.0
    assert profile["revision_confidence_coverage"] == 1.0
    assert profile["clustered_95_ci"]["balanced_accuracy"] is None
    assert profile["repair_by_control_degree"]["1"]["valid_repair_rate"] == 1.0
    assert profile["repair_by_control_degree"]["2"]["valid_repair_rate"] == 0.0
    assert profile["quality_per_resource"]["valid_repairs_per_1000_tokens"] == 2.5


def test_score_rejects_inconsistent_revision_label():
    label = {"case_id": "x", "condition": "control", "expected_revision": True}
    try:
        score([label], [_run("x", True)])
    except ValueError as error:
        assert "disagree" in str(error)
    else:
        raise AssertionError("inconsistent condition and expected_revision must fail")


def test_score_rejects_invalid_optional_confidence():
    label = {"case_id": "x", "condition": "control", "expected_revision": False}
    try:
        score([label], [_run("x", False, revision_confidence=1.5)])
    except ValueError as error:
        assert "revision_confidence" in str(error)
    else:
        raise AssertionError("out-of-range confidence must fail")


def test_failed_run_counts_as_failure_but_not_as_a_revision_prediction():
    label = {
        "case_id": "i1",
        "condition": "intervention",
        "expected_revision": True,
        "intervention_ids": ["factor-a"],
    }
    run = {
        "case_id": "i1",
        "framework": "candidate",
        "model_snapshot": "fixture",
        "seed": 1,
        "run_status": "timeout",
        "total_tokens": 50,
    }

    profile = score([label], [run])["framework_profiles"]["candidate"]
    assert profile["run_status_counts"] == {"timeout": 1}
    assert profile["run_success_rate"] == 0.0
    assert profile["attempted_case_coverage"] == 1.0
    assert profile["case_coverage"] == 0.0
    assert profile["revision_diagnosis"]["n"] == 0
    assert profile["factor_localization"]["micro_recall"] == 0.0
    assert profile["valid_repair_rate"] == 0.0
    assert profile["evidence_verification_rate"] == 0.0
    assert profile["end_to_end_intervention_success_rate"] == 0.0


def test_skipped_case_is_reported_as_missing_coverage_not_failure():
    label = {"case_id": "c1", "condition": "control", "expected_revision": False}
    run = {
        "case_id": "c1",
        "framework": "candidate",
        "model_snapshot": "fixture",
        "seed": 1,
        "run_status": "skipped",
    }

    profile = score([label], [run])["framework_profiles"]["candidate"]
    assert profile["run_status_counts"] == {"skipped": 1}
    assert profile["recorded_case_coverage"] == 1.0
    assert profile["attempted_case_coverage"] == 0.0
    assert profile["run_success_rate"] is None
    assert profile["revision_diagnosis"]["n"] == 0
    assert profile["evidence_verification_rate"] is None


def test_label_builder_exports_layer_and_control_degree(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "labels.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "case_id": "case-a",
                "domain": "math",
                "layer": "layer1",
                "control_degree": 2,
                "provenance": {"gold_project_id": "project-a"},
                "oracle": {
                    "causal_targets": ["analysis.metric"],
                    "hidden_tests": ["x"],
                    "opportunity_contracts": {"analysis.metric": {"analysis_tag": "metric"}},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert build(cases_path, output_path) == 1
    label = json.loads(output_path.read_text(encoding="utf-8"))
    assert label["layer"] == "layer1"
    assert label["control_degree"] == 2
    assert label["condition"] == "intervention"
    assert label["expected_revision"] is True
    assert label["intervention_categories"] == ["metric"]
    assert label["factor_categories_by_id"] == {"analysis.metric": "metric"}
