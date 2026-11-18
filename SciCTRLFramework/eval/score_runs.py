"""Score paired framework runs using evaluator-owned labels and hidden checks."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _diagnosis_metrics(items: list[tuple[bool, bool]]) -> dict[str, Any]:
    """Score revision decisions against labels as (expected, predicted)."""
    tp = sum(expected and predicted for expected, predicted in items)
    tn = sum(not expected and not predicted for expected, predicted in items)
    fp = sum(not expected and predicted for expected, predicted in items)
    fn = sum(expected and not predicted for expected, predicted in items)
    sensitivity = _rate([predicted for expected, predicted in items if expected])
    specificity = _rate([not predicted for expected, predicted in items if not expected])
    precision = round(tp / (tp + fp), 4) if tp + fp else None
    f1 = (
        round(2 * precision * sensitivity / (precision + sensitivity), 4)
        if precision is not None
        and sensitivity is not None
        and precision + sensitivity
        else None
    )
    balanced = (
        round((sensitivity + specificity) / 2, 4)
        if sensitivity is not None and specificity is not None
        else None
    )
    return {
        "n": len(items),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": _rate([expected == predicted for expected, predicted in items]),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "balanced_accuracy": balanced,
    }


def _factor_metrics(rows: list[tuple[set[str], set[str]]]) -> dict[str, Any]:
    tp = sum(len(predicted & expected) for predicted, expected in rows)
    fp = sum(len(predicted - expected) for predicted, expected in rows)
    fn = sum(len(expected - predicted) for predicted, expected in rows)
    precisions = []
    recalls = []
    f1s = []
    exact = []
    for predicted, expected in rows:
        row_tp = len(predicted & expected)
        row_fp = len(predicted - expected)
        row_fn = len(expected - predicted)
        precision = row_tp / (row_tp + row_fp) if row_tp + row_fp else float(not expected)
        recall = row_tp / (row_tp + row_fn) if row_tp + row_fn else float(not predicted)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        exact.append(predicted == expected)
    return {
        "n_intervention_cases": len(rows),
        "micro_precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "micro_recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "micro_f1": round(2 * tp / (2 * tp + fp + fn), 4) if 2 * tp + fp + fn else None,
        "macro_precision_per_case": round(sum(precisions) / len(precisions), 4) if rows else None,
        "macro_recall_per_case": round(sum(recalls) / len(recalls), 4) if rows else None,
        "macro_f1_per_case": round(sum(f1s) / len(f1s), 4) if rows else None,
        "exact_set_match_rate": _rate(exact),
    }


def _calibration_metrics(rows: list[tuple[float, bool]], bins: int = 10) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "brier_score": None, "ece": None, "auc": None, "bins": bins}
    brier = sum((confidence - float(expected)) ** 2 for confidence, expected in rows) / len(rows)
    ece = 0.0
    for index in range(bins):
        bucket = [
            item for item in rows
            if min(int(item[0] * bins), bins - 1) == index
        ]
        if bucket:
            avg_confidence = sum(value for value, _ in bucket) / len(bucket)
            avg_accuracy = sum(expected for _, expected in bucket) / len(bucket)
            ece += len(bucket) / len(rows) * abs(avg_confidence - avg_accuracy)
    positives = [confidence for confidence, expected in rows if expected]
    negatives = [confidence for confidence, expected in rows if not expected]
    auc = (
        sum(1.0 if pos > neg else 0.5 if pos == neg else 0.0 for pos in positives for neg in negatives)
        / (len(positives) * len(negatives))
        if positives and negatives else None
    )
    return {
        "n": len(rows),
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
        "auc": round(auc, 4) if auc is not None else None,
        "bins": bins,
    }


def _clustered_bootstrap_metric_ci(
    rows: list[tuple[str, bool, bool]], seed: int = 17, draws: int = 2000
) -> dict[str, list[float] | None]:
    clusters: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for cluster, expected, predicted in rows:
        clusters[cluster].append((expected, predicted))
    groups = list(clusters.values())
    if len(groups) < 2:
        return {"balanced_accuracy": None, "control_false_revision_rate": None, "intervention_detection_rate": None}
    rng = random.Random(seed)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(draws):
        sample = [item for _ in groups for item in rng.choice(groups)]
        metrics = _diagnosis_metrics(sample)
        if metrics["balanced_accuracy"] is not None:
            values["balanced_accuracy"].append(metrics["balanced_accuracy"])
        controls = [predicted for expected, predicted in sample if not expected]
        interventions = [predicted for expected, predicted in sample if expected]
        if controls:
            values["control_false_revision_rate"].append(sum(controls) / len(controls))
        if interventions:
            values["intervention_detection_rate"].append(sum(interventions) / len(interventions))
    intervals: dict[str, list[float] | None] = {}
    for metric in ("balanced_accuracy", "control_false_revision_rate", "intervention_detection_rate"):
        draws_for_metric = sorted(values[metric])
        intervals[metric] = (
            [round(draws_for_metric[int(len(draws_for_metric) * 0.025)], 4),
             round(draws_for_metric[min(len(draws_for_metric) - 1, int(len(draws_for_metric) * 0.975))], 4)]
            if draws_for_metric else None
        )
    return intervals


def _bootstrap_clustered_mean_ci(
    values: list[tuple[str, float]], seed: int = 17, draws: int = 2000
) -> list[float] | None:
    if not values:
        return None
    clusters: dict[str, list[float]] = defaultdict(list)
    for cluster, value in values:
        clusters[cluster].append(value)
    groups = list(clusters.values())
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        sampled = [rng.choice(groups) for _ in groups]
        means.append(sum(value for group in sampled for value in group) / sum(map(len, sampled)))
    means.sort()
    return [round(means[int(draws * 0.025)], 4), round(means[min(draws - 1, int(draws * 0.975))], 4)]


def score(labels: list[dict[str, Any]], runs: list[dict[str, Any]]) -> dict[str, Any]:
    label_map = {row["case_id"]: row for row in labels}
    if len(label_map) != len(labels):
        raise ValueError("Duplicate case_id in evaluator labels")
    unique_runs = set()
    for row in runs:
        if row.get("case_id") not in label_map:
            raise ValueError(f"Run references unknown case_id: {row.get('case_id')}")
        required = {"framework", "model_snapshot", "seed"}
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Run row missing required fields: {sorted(missing)}")
        run_status = row.get("run_status", "success")
        allowed_statuses = {"success", "failed", "timeout", "skipped", "parse_error"}
        if not isinstance(run_status, str) or run_status not in allowed_statuses:
            raise ValueError(f"Invalid run_status: {run_status!r}")
        outcome_fields = {
            "predicted_revision", "predicted_factor_ids", "valid_repair",
            "evidence_verified", "integrity_verified",
        }
        missing_outcomes = outcome_fields - row.keys()
        if run_status == "success" and missing_outcomes:
            raise ValueError(f"Successful run missing outcome fields: {sorted(missing_outcomes)}")
        for field in ("predicted_revision", "valid_repair", "evidence_verified", "integrity_verified"):
            if field in row and not isinstance(row[field], bool):
                raise ValueError(f"Run field {field} must be boolean")
        if "predicted_factor_ids" in row and (
            not isinstance(row["predicted_factor_ids"], list)
            or not all(isinstance(value, str) for value in row["predicted_factor_ids"])
        ):
            raise ValueError("predicted_factor_ids must be a list of strings")
        if "predicted_factor_categories" in row and (
            not isinstance(row["predicted_factor_categories"], list)
            or not all(isinstance(value, str) for value in row["predicted_factor_categories"])
        ):
            raise ValueError("predicted_factor_categories must be a list of strings")
        confidence = row.get("revision_confidence")
        if confidence is not None and (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise ValueError("revision_confidence must be a finite number in [0, 1]")
        for field in ("total_tokens", "cost_usd", "elapsed_seconds"):
            value = row.get(field)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{field} must be a finite nonnegative number")
        key = (row["case_id"], row["framework"], row["model_snapshot"], str(row["seed"]))
        if key in unique_runs:
            raise ValueError(f"Duplicate run for case/model/seed: {key}")
        unique_runs.add(key)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        grouped[row["framework"]].append(row)

    profiles: dict[str, Any] = {}
    per_case: dict[tuple[str, str, str], dict[str, Any]] = {}
    for framework, items in sorted(grouped.items()):
        control_fp: list[bool] = []
        intervention_detect: list[bool] = []
        intervention_repair: list[bool] = []
        evidence_checks: list[bool] = []
        integrity_checks: list[bool] = []
        token_values: list[float] = []
        costs: list[float] = []
        durations: list[float] = []
        intervention_token_values: list[float] = []
        intervention_costs: list[float] = []
        intervention_token_success: list[bool] = []
        intervention_cost_success: list[bool] = []
        factor_rows: list[tuple[set[str], set[str]]] = []
        category_rows: list[tuple[set[str], set[str]]] = []
        factor_strata: dict[str, list[tuple[set[str], set[str]]]] = defaultdict(list)
        control_factor_predictions: list[bool] = []
        control_category_predictions: list[bool] = []
        control_extra_factor_counts: list[int] = []
        diagnosis_items: list[tuple[bool, bool]] = []
        clustered_diagnosis: list[tuple[str, bool, bool]] = []
        strata: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
        evidence_by_condition: dict[str, list[bool]] = defaultdict(list)
        integrity_by_condition: dict[str, list[bool]] = defaultdict(list)
        joint_verification: list[bool] = []
        end_to_end_intervention_success: list[bool] = []
        evidence_supported_control_stop: list[bool] = []
        repair_attempts: list[bool] = []
        successful_attempts: list[bool] = []
        repair_by_degree: dict[str, list[bool]] = defaultdict(list)
        attempts_by_degree: dict[str, list[bool]] = defaultdict(list)
        confidence_rows: list[tuple[float, bool]] = []
        run_statuses: dict[str, int] = defaultdict(int)
        attempted_run_count = 0
        resource_counts = {"total_tokens": 0, "cost_usd": 0, "elapsed_seconds": 0}
        for row in items:
            label = label_map[row["case_id"]]
            condition = label.get("condition")
            status = row.get("run_status", "success")
            run_statuses[status] += 1
            if condition not in {"control", "intervention"}:
                raise ValueError(f"Invalid condition for {row['case_id']}: {condition!r}")
            expected = bool(label.get("expected_revision", condition == "intervention"))
            if expected != (condition == "intervention"):
                raise ValueError(f"Label condition and expected_revision disagree for {row['case_id']}")
            if status == "skipped":
                continue
            attempted_run_count += 1
            for key, target in (("total_tokens", token_values), ("cost_usd", costs), ("elapsed_seconds", durations)):
                value = row.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                    target.append(float(value))
                    resource_counts[key] += 1
                    if condition == "intervention" and key == "total_tokens":
                        intervention_token_values.append(float(value))
                        intervention_token_success.append(status == "success" and row.get("valid_repair") is True)
                    if condition == "intervention" and key == "cost_usd":
                        intervention_costs.append(float(value))
                        intervention_cost_success.append(status == "success" and row.get("valid_repair") is True)
            if status != "success":
                evidence_checks.append(False)
                integrity_checks.append(False)
                evidence_by_condition[condition].append(False)
                integrity_by_condition[condition].append(False)
                joint_verification.append(False)
                if condition == "intervention":
                    end_to_end_intervention_success.append(False)
                    repair_attempts.append(False)
                    intervention_repair.append(False)
                    if label.get("control_degree") is not None:
                        degree_key = str(label["control_degree"])
                        repair_by_degree[degree_key].append(False)
                        attempts_by_degree[degree_key].append(False)
                    expected_factors = set(label.get("intervention_ids", []))
                    factor_rows.append((set(), expected_factors))
                    expected_categories = set(label.get("intervention_categories", []))
                    if expected_categories:
                        category_rows.append((set(), expected_categories))
                    factor_strata[f"layer:{label.get('layer', 'unknown')}"].append((set(), expected_factors))
                    degree = label.get("control_degree")
                    if degree is not None:
                        factor_strata[f"control_degree:{degree}"].append((set(), expected_factors))
                else:
                    evidence_supported_control_stop.append(False)
                continue
            predicted = row["predicted_revision"]
            diagnosis_items.append((expected, predicted))
            clustered_diagnosis.append((label.get("source_family_cluster", row["case_id"]), expected, predicted))
            layer = label.get("layer", "unknown")
            strata[f"layer:{layer}"].append((expected, predicted))
            degree = label.get("control_degree")
            if degree is not None:
                strata[f"control_degree:{degree}"].append((expected, predicted))
            predicted_factors = set(row.get("predicted_factor_ids", []))
            expected_factors = set(label.get("intervention_ids", []))
            category_map = label.get("factor_categories_by_id", {})
            predicted_categories = set(row.get("predicted_factor_categories", []))
            if not predicted_categories:
                predicted_categories = {category_map[item] for item in predicted_factors if item in category_map}
            expected_categories = set(label.get("intervention_categories", []))
            evidence_ok = bool(row.get("evidence_verified", False))
            integrity_ok = bool(row.get("integrity_verified", False))
            evidence_by_condition[condition].append(evidence_ok)
            integrity_by_condition[condition].append(integrity_ok)
            joint_verification.append(evidence_ok and integrity_ok)
            if condition == "intervention":
                end_to_end_intervention_success.append(
                    predicted and bool(row.get("valid_repair", False)) and evidence_ok and integrity_ok
                )
            else:
                evidence_supported_control_stop.append(not predicted and evidence_ok and integrity_ok)
            confidence = row.get("revision_confidence")
            if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) and math.isfinite(confidence) and 0 <= confidence <= 1:
                confidence_rows.append((float(confidence), expected))
            if condition == "control":
                control_fp.append(predicted)
                control_factor_predictions.append(bool(predicted_factors))
                control_extra_factor_counts.append(len(predicted_factors))
                control_category_predictions.append(bool(predicted_categories))
            elif condition == "intervention":
                intervention_detect.append(predicted)
                factor_rows.append((predicted_factors, expected_factors))
                if expected_categories:
                    category_rows.append((predicted_categories, expected_categories))
                factor_strata[f"layer:{layer}"].append((predicted_factors, expected_factors))
                if degree is not None:
                    factor_strata[f"control_degree:{degree}"].append((predicted_factors, expected_factors))
                intervention_repair.append(bool(row.get("valid_repair", False)))
                repair_attempts.append(predicted)
                if degree is not None:
                    degree_key = str(degree)
                    repair_by_degree[degree_key].append(bool(row.get("valid_repair", False)))
                    attempts_by_degree[degree_key].append(predicted)
                if predicted:
                    successful_attempts.append(bool(row.get("valid_repair", False)))
            else:
                raise ValueError(f"Invalid condition for {row['case_id']}: {condition!r}")
            evidence_checks.append(bool(row.get("evidence_verified", False)))
            integrity_checks.append(bool(row.get("integrity_verified", False)))
            per_case[(row["case_id"], framework, row["model_snapshot"], str(row["seed"]))] = {
                "revision": predicted,
                "valid_repair": bool(row.get("valid_repair", False)),
                "condition": condition,
            }
        factor_summary = _factor_metrics(factor_rows)
        profiles[framework] = {
            "runs": len(items),
            "case_coverage": round(len({row["case_id"] for row in items if row.get("run_status", "success") == "success"}) / len(labels), 4) if labels else 0.0,
            "attempted_case_coverage": round(len({row["case_id"] for row in items if row.get("run_status", "success") != "skipped"}) / len(labels), 4) if labels else 0.0,
            "recorded_case_coverage": round(len({row["case_id"] for row in items}) / len(labels), 4) if labels else 0.0,
            "run_success_rate": _rate([row.get("run_status", "success") == "success" for row in items if row.get("run_status", "success") != "skipped"]),
            "revision_diagnosis": _diagnosis_metrics(diagnosis_items),
            "diagnosis_by_stratum": {
                name: _diagnosis_metrics(values) for name, values in sorted(strata.items())
            },
            "false_revision_rate_on_controls": _rate(control_fp),
            "revision_detection_rate": _rate(intervention_detect),
            "factor_localization": factor_summary,
            "category_localization": _factor_metrics(category_rows),
            "factor_localization_by_stratum": {
                name: _factor_metrics(values) for name, values in sorted(factor_strata.items())
            },
            "control_spurious_factor_rate": _rate(control_factor_predictions),
            "control_spurious_category_rate": _rate(control_category_predictions),
            "mean_spurious_factor_count_on_controls": round(sum(control_extra_factor_counts) / len(control_extra_factor_counts), 4) if control_extra_factor_counts else None,
            "factor_identification_precision": factor_summary["micro_precision"],
            "factor_identification_recall": factor_summary["micro_recall"],
            "valid_repair_rate": _rate(intervention_repair),
            "repair_attempt_rate_on_interventions": _rate(repair_attempts),
            "valid_repair_rate_given_attempt": _rate(successful_attempts) if successful_attempts else None,
            "repair_by_control_degree": {
                degree: {
                    "attempt_rate": _rate(attempts_by_degree[degree]),
                    "valid_repair_rate": _rate(repair_by_degree[degree]),
                    "n_interventions": len(repair_by_degree[degree]),
                }
                for degree in sorted(repair_by_degree, key=int)
            },
            "evidence_verification_rate": _rate(evidence_checks),
            "task_integrity_verification_rate": _rate(integrity_checks),
            "evidence_verification_by_condition": {key: _rate(value) for key, value in sorted(evidence_by_condition.items())},
            "integrity_verification_by_condition": {key: _rate(value) for key, value in sorted(integrity_by_condition.items())},
            "joint_evidence_and_integrity_rate": _rate(joint_verification),
            "end_to_end_intervention_success_rate": _rate(end_to_end_intervention_success),
            "evidence_supported_control_stop_rate": _rate(evidence_supported_control_stop),
            "revision_confidence_metrics": _calibration_metrics(confidence_rows),
            "revision_confidence_coverage": round(len(confidence_rows) / len(diagnosis_items), 4) if diagnosis_items else 0.0,
            "run_status_counts": dict(sorted(run_statuses.items())),
            "source_clusters_with_completed_runs": len({cluster for cluster, _, _ in clustered_diagnosis}),
            "resource_telemetry_coverage": {key: {"n": count, "rate": round(count / attempted_run_count, 4) if attempted_run_count else None} for key, count in resource_counts.items()},
            "quality_per_resource": {
                "valid_repairs_per_1000_tokens": round(sum(intervention_token_success) * 1000 / sum(intervention_token_values), 4) if intervention_token_values and sum(intervention_token_values) else None,
                "valid_repairs_per_usd": round(sum(intervention_cost_success) / sum(intervention_costs), 4) if intervention_costs and sum(intervention_costs) else None,
            },
            "clustered_95_ci": _clustered_bootstrap_metric_ci(clustered_diagnosis),
            "mean_total_tokens": round(sum(token_values) / len(token_values), 2) if token_values else None,
            "mean_cost_usd": round(sum(costs) / len(costs), 6) if costs else None,
            "mean_elapsed_seconds": round(sum(durations) / len(durations), 3) if durations else None,
        }

    paired: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for (case_id, framework, model, seed), result in per_case.items():
        label = label_map[case_id]
        paired[(case_id, model, seed)][framework] = {
            **result,
            "condition": label["condition"],
            "source_family_cluster": label.get("source_family_cluster", case_id),
        }

    comparisons: dict[str, Any] = {}
    frameworks = sorted(grouped)
    for i, left in enumerate(frameworks):
        for right in frameworks[i + 1 :]:
            deltas: dict[str, list[tuple[str, float]]] = defaultdict(list)
            complete_pairs = 0
            for values in paired.values():
                if left not in values or right not in values:
                    continue
                if values[left]["condition"] != values[right]["condition"]:
                    continue
                complete_pairs += 1
                cluster = values[left]["source_family_cluster"]
                if values[left]["condition"] == "intervention":
                    for metric in ("revision", "valid_repair"):
                        deltas[metric].append((cluster, float(values[left][metric]) - float(values[right][metric])))
                else:
                    deltas["revision_on_controls"].append(
                        (cluster, float(values[left]["revision"]) - float(values[right]["revision"]))
                    )
            comparisons[f"{left}_minus_{right}"] = {
                "paired_cells": complete_pairs,
                "revision_rate_difference_ci95": _bootstrap_clustered_mean_ci(deltas["revision"]),
                "valid_repair_rate_difference_ci95": _bootstrap_clustered_mean_ci(deltas["valid_repair"]),
                "control_false_revision_difference_ci95": _bootstrap_clustered_mean_ci(deltas["revision_on_controls"]),
                "difference_direction": "left_minus_right",
            }

    return {
        "schema_version": "scictrl-eval-0.2",
        "label_cases": len(labels),
        "run_rows": len(runs),
        "framework_profiles": profiles,
        "paired_comparisons": comparisons,
        "limitations": [
            "This report measures agent outcomes conditional on supplied labels and hidden evaluator outputs; it is not a domain-expert validation.",
            "Bootstrap intervals cluster by source family, but the small number of families makes them provisional.",
            "A missing run is missing coverage, not a scientific failure; report coverage separately before comparing success rates.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = score(_read_jsonl(args.labels), _read_jsonl(args.runs))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_rows": report["run_rows"], "frameworks": list(report["framework_profiles"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
