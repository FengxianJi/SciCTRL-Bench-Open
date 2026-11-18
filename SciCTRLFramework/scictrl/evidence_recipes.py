"""Private evidence-acquisition recipes.

Recipe IDs remain evaluator-side. Public tasks expose only tools and generic
operations; a request must select an appropriate tool and articulate the
scientific concepts needed for the corresponding observation.
"""

from __future__ import annotations

import copy
from typing import Any


_RECIPES: dict[str, dict[str, Any]] = {
    "inspect_split_manifest": {
        "tool": "dataset_audit", "operations": ["inspect_artifact", "run_analysis"],
        "artifact_concepts": ["split", "manifest"], "query_concepts": ["partition", "provenance"],
    },
    "run_group_overlap_audit": {
        "tool": "dataset_audit", "operations": ["run_analysis", "compare_results"],
        "artifact_concepts": ["group", "provenance"], "query_concepts": ["group", "overlap", "lineage"],
    },
    "inspect_confusion_matrix": {
        "tool": "metric_report", "operations": ["inspect_artifact", "run_analysis"],
        "artifact_concepts": ["confusion", "prediction"], "query_concepts": ["class", "recall", "imbalance"],
    },
    "rerun_all_declared_seeds": {
        "tool": "metric_report", "operations": ["execute_experiment", "compare_results"],
        "artifact_concepts": ["seed", "run"], "query_concepts": ["seed", "variance", "aggregate"],
    },
    "inspect_shift_strata": {
        "tool": "shift_audit", "operations": ["inspect_artifact", "run_analysis"],
        "artifact_concepts": ["shift", "strata"], "query_concepts": ["severity", "threshold", "site"],
    },
    "run_shift_interaction_probe": {
        "tool": "interaction_report", "operations": ["run_analysis", "compare_results"],
        "artifact_concepts": ["shift", "result"], "query_concepts": ["interaction", "severity", "effect"],
    },
    "inspect_metric_and_split_checks": {
        "tool": "shift_audit", "operations": ["inspect_artifact", "run_analysis"],
        "artifact_concepts": ["metric", "split"], "query_concepts": ["metric", "split", "site"],
    },
    "run_untouched_confirmation": {
        "tool": "confirmation_runner", "operations": ["execute_experiment", "compare_results"],
        "artifact_concepts": ["confirmation", "site"], "query_concepts": ["untouched", "severity", "effect"],
    },
    "inspect_patient_split": {
        "tool": "cohort_audit", "operations": ["inspect_artifact", "run_analysis"],
        "artifact_concepts": ["patient", "split"], "query_concepts": ["patient", "replicate", "overlap"],
    },
    "run_batch_predictability_probe": {
        "tool": "batch_report", "operations": ["run_analysis", "compare_results"],
        "artifact_concepts": ["batch", "site"], "query_concepts": ["batch", "site", "predictability"],
    },
    "inspect_pathway_pvalues": {
        "tool": "stats_report", "operations": ["inspect_artifact", "run_analysis"],
        "artifact_concepts": ["pathway", "pvalue"], "query_concepts": ["pathway", "multiplicity", "family"],
    },
    "rerun_patient_bootstrap": {
        "tool": "stats_report", "operations": ["execute_experiment", "run_analysis"],
        "artifact_concepts": ["patient", "result"], "query_concepts": ["patient", "bootstrap", "interval"],
    },
    "compare_control_settings": {
        "tool": "simulation_runner", "operations": ["compare_results", "run_analysis"],
        "artifact_concepts": ["control", "structure"], "query_concepts": ["cell", "composition", "temperature"],
    },
    "run_convergence_audit": {
        "tool": "convergence_audit", "operations": ["execute_experiment", "run_analysis"],
        "artifact_concepts": ["convergence", "barrier"], "query_concepts": ["cutoff", "kpoint", "barrier"],
    },
    "inspect_unit_manifest": {
        "tool": "unit_check", "operations": ["inspect_artifact", "run_analysis"],
        "artifact_concepts": ["unit", "manifest"], "query_concepts": ["unit", "energy", "consistency"],
    },
    "inspect_all_paths": {
        "tool": "path_report", "operations": ["inspect_artifact", "compare_results"],
        "artifact_concepts": ["path", "barrier"], "query_concepts": ["path", "barrier", "scope"],
    },
    "compare_stopping_rules": {
        "tool": "tolerance_audit", "operations": ["compare_results", "run_analysis"],
        "artifact_concepts": ["stopping", "residual"], "query_concepts": ["stopping", "residual", "tolerance"],
    },
    "inspect_failed_matrices": {
        "tool": "failure_report", "operations": ["inspect_artifact", "run_analysis"],
        "artifact_concepts": ["matrix", "failure"], "query_concepts": ["matrix", "timeout", "excluded"],
    },
    "recompute_shared_residual": {
        "tool": "tolerance_audit", "operations": ["execute_experiment", "run_analysis"],
        "artifact_concepts": ["solver", "residual"], "query_concepts": ["shared", "residual", "accuracy"],
    },
    "run_full_heldout_suite": {
        "tool": "solver_runner", "operations": ["execute_experiment", "compare_results"],
        "artifact_concepts": ["heldout", "suite"], "query_concepts": ["matrix", "family", "heterogeneity"],
    },
}


def private_probe_recipe(probe_id: str) -> dict[str, Any]:
    try:
        return copy.deepcopy(_RECIPES[probe_id])
    except KeyError as error:
        raise KeyError(f"No private evidence recipe for {probe_id!r}") from error


def known_probe_ids() -> set[str]:
    return set(_RECIPES)
