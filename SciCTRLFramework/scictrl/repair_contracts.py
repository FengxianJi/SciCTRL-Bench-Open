"""Hidden equivalence classes of scientifically acceptable repair operations.

The canonical gold string is useful for construction tests, but it must not be
the only way an agent can succeed.  A repair witness names an operation and
structured parameters.  Multiple operations may satisfy the same invariants.
"""

from __future__ import annotations

import copy
from typing import Any

from .models import FlawSpec


_CONTRACTS: dict[str, dict[str, Any]] = {
    "ml_group_leakage": {
        "invariants": ["evaluation_groups_disjoint", "preprocessing_fit_on_train_only"],
        "accepted_operations": [
            {"operation": "rebuild_group_disjoint_split", "parameters": {"group_key": "source_group", "augmentation_stage": "after_split", "preprocessing_fit": "train_only"}},
            {"operation": "partition_provenance_components", "parameters": {"component_key": "augmentation_lineage", "component_disjoint": True, "preprocessing_fit": "train_only"}},
        ],
        "rejected_alternatives": [
            {"operation": "drop_overlaps_from_report", "reason": "Hides contaminated samples without repairing the evaluation boundary."},
            {"operation": "change_model_or_metric", "reason": "Cannot remove shared provenance across train and evaluation."},
        ],
    },
    "ml_accuracy_imbalance": {
        "invariants": ["primary_metric_weights_classes", "rare_class_failure_visible"],
        "accepted_operations": [
            {"operation": "use_class_balanced_primary_metric", "parameters": {"metric": "macro_f1", "class_weighting": "equal"}},
            {"operation": "use_guarded_balanced_metric", "parameters": {"metric": "balanced_accuracy", "guard": "per_class_recall_non_degradation"}},
        ],
        "rejected_alternatives": [
            {"operation": "remove_rare_classes", "reason": "Changes the estimand rather than measuring it."},
            {"operation": "retain_accuracy_add_cosmetic_table", "reason": "Leaves the primary decision rule misaligned."},
        ],
    },
    "ml_best_seed_reporting": {
        "invariants": ["all_predeclared_seeds_included", "uncertainty_reported"],
        "accepted_operations": [
            {"operation": "aggregate_all_predeclared_seeds", "parameters": {"seeds": [0, 1, 2, 3, 4], "summary": "mean_95ci"}},
            {"operation": "bootstrap_predeclared_seed_mean", "parameters": {"seeds": [0, 1, 2, 3, 4], "interval": "bootstrap_95ci"}},
        ],
        "rejected_alternatives": [
            {"operation": "report_best_seed", "reason": "Selection remains conditioned on validation performance."},
            {"operation": "discard_failed_seeds", "reason": "Violates the predeclared experimental population."},
        ],
    },
    "ml_universal_shift_hypothesis": {
        "invariants": [
            "universal_prediction_marked_falsified",
            "negative_severe_shift_result_retained",
            "posthoc_hypothesis_not_mislabeled_confirmatory",
        ],
        "accepted_operations": [
            {
                "operation": "reject_universal_hypothesis",
                "parameters": {
                    "original_status": "falsified",
                    "negative_regime": "retained",
                    "replacement_hypothesis": "exploratory_only",
                },
            },
            {
                "operation": "revise_conditional_hypothesis_and_reconfirm",
                "parameters": {
                    "derivation_data": "development_sites_only",
                    "condition": "predeclared_shift_severity",
                    "confirmation_data": "untouched_sites",
                    "retain_severe_negative_effect": True,
                },
            },
        ],
        "rejected_alternatives": [
            {
                "operation": "change_metric_until_universal_gain",
                "reason": "Metric substitution cannot overturn the predeclared sign-reversing interaction without changing the estimand.",
            },
            {
                "operation": "retain_hypothesis_narrow_claim_only",
                "reason": "A narrower claim does not update the explicitly falsified universal prediction or its confirmatory status.",
            },
        ],
    },
    "bio_replicate_leakage": {
        "invariants": ["patient_disjoint_evaluation", "replicates_colocated"],
        "accepted_operations": [
            {"operation": "rebuild_patient_disjoint_site_holdout", "parameters": {"group_key": "patient_id", "replicates": "same_partition", "heldout_unit": "site"}},
            {"operation": "partition_connected_patient_components", "parameters": {"component_key": "patient_id", "site_heldout": True, "replicates": "same_partition"}},
        ],
        "rejected_alternatives": [
            {"operation": "deduplicate_after_scoring", "reason": "The fitted model has already seen patient-linked information."},
            {"operation": "change_to_sample_level_claim", "reason": "Changes the fixed patient-level research question."},
        ],
    },
    "bio_batch_confound": {
        "invariants": ["matched_nuisance_controls_present", "same_patient_site_splits", "pathway_compared_to_joint_nuisance"],
        "accepted_operations": [
            {"operation": "add_matched_nuisance_baselines", "parameters": {"baselines": ["batch_only", "site_only", "batch_plus_site"], "splits": "identical"}},
            {"operation": "add_joint_nuisance_falsification_test", "parameters": {"nuisance_features": ["batch", "site"], "comparison": "pathway_minus_joint_nuisance", "splits": "identical"}},
        ],
        "rejected_alternatives": [
            {"operation": "erase_batch_metadata", "reason": "Prevents testing whether the signal is explained by nuisance variables."},
            {"operation": "remove_difficult_site", "reason": "Narrows the fixed site-transfer evaluation."},
        ],
    },
    "bio_uncorrected_pathways": {
        "invariants": ["predeclared_family_preserved", "family_level_error_control"],
        "accepted_operations": [
            {"operation": "apply_bh_family_correction", "parameters": {"family": "predeclared_pathways", "target": "fdr"}},
            {"operation": "apply_holm_family_correction", "parameters": {"family": "predeclared_pathways", "target": "fwer"}},
        ],
        "rejected_alternatives": [
            {"operation": "report_nominal_subset", "reason": "Post-hoc filtering does not control family-level error."},
            {"operation": "redefine_test_family", "reason": "Changes multiplicity scope after observing results."},
        ],
    },
    "mat_unmatched_control": {
        "invariants": ["control_composition_matched", "control_cell_effect_removed", "temperature_matched"],
        "accepted_operations": [
            {"operation": "rerun_matched_supercell_control", "parameters": {"composition": "matched", "cell_size": "matched", "temperature": "matched"}},
            {"operation": "finite_size_extrapolate_both_conditions", "parameters": {"cell_sizes": "shared_ladder", "extrapolation": "common_limit", "temperature": "matched"}},
        ],
        "rejected_alternatives": [
            {"operation": "manual_energy_offset", "reason": "An arbitrary offset does not estimate finite-size bias."},
            {"operation": "drop_pristine_control", "reason": "Removes the counterfactual required by the question."},
        ],
    },
    "mat_unconverged_mesh": {
        "invariants": ["numerical_settings_matched", "barrier_converged_below_0_01_ev"],
        "accepted_operations": [
            {"operation": "run_shared_convergence_ladder", "parameters": {"settings": ["cutoff", "kpoint_density", "neb", "thermostat"], "barrier_tolerance_ev": 0.01}},
            {"operation": "extrapolate_common_numerical_limit", "parameters": {"settings": ["cutoff", "kpoint_density"], "shared_sequence": True, "barrier_tolerance_ev": 0.01}},
        ],
        "rejected_alternatives": [
            {"operation": "increase_convergence_tolerance", "reason": "Weakens the validity condition after observing instability."},
            {"operation": "mix_precision_levels", "reason": "Leaves physical and numerical effects confounded."},
        ],
    },
    "mat_conductivity_overclaim": {
        "invariants": ["claim_limited_to_measured_barrier", "no_unmeasured_conductivity_claim"],
        "accepted_operations": [
            {"operation": "restrict_claim_to_simulated_barrier", "parameters": {"scope": "tested_structures_and_protocol", "quantity": "migration_barrier"}},
            {"operation": "state_conditional_barrier_mechanism", "parameters": {"scope": "tested_paths_only", "exclude": ["bulk_conductivity", "experimental_transport"]}},
        ],
        "rejected_alternatives": [
            {"operation": "invent_conductivity_estimate", "reason": "Introduces evidence that was never generated."},
            {"operation": "retain_transport_claim_add_disclaimer", "reason": "A limitation cannot rescue an unsupported headline claim."},
        ],
    },
    "opt_stopping_mismatch": {
        "invariants": ["solver_accuracy_matched", "runtime_measured_to_common_target"],
        "accepted_operations": [
            {"operation": "measure_time_to_shared_residual", "parameters": {"target": "shared_residual_tolerance", "residual": "externally_recomputed"}},
            {"operation": "measure_time_to_shared_objective_gap", "parameters": {"target": "common_objective_gap", "quality_guard": "shared_residual_tolerance"}},
        ],
        "rejected_alternatives": [
            {"operation": "loosen_proposed_tolerance", "reason": "Makes runtime incomparable by changing achieved accuracy."},
            {"operation": "trust_native_stop_flags", "reason": "Native rules encode different accuracy targets."},
        ],
    },
    "opt_survivor_average": {
        "invariants": ["all_heldout_matrices_included", "timeouts_counted", "paired_population_preserved"],
        "accepted_operations": [
            {"operation": "aggregate_timeout_penalized_runtimes", "parameters": {"population": "all_heldout_matrices", "timeout_seconds": 300, "summary": "paired_geometric_mean"}},
            {"operation": "report_all_instance_performance_profile", "parameters": {"population": "all_heldout_matrices", "timeouts": "right_censored_at_300", "pairing": "by_matrix"}},
        ],
        "rejected_alternatives": [
            {"operation": "average_completed_instances", "reason": "Conditions the population on solver success."},
            {"operation": "raise_proposed_timeout_only", "reason": "Breaks the fixed paired budget."},
        ],
    },
    "opt_generalization_overclaim": {
        "invariants": ["claim_bounded_by_tested_suite", "negative_family_retained"],
        "accepted_operations": [
            {"operation": "restrict_claim_to_fixed_suite", "parameters": {"scope": "fixed_heldout_suite", "hardware": "tested_cpu_node"}},
            {"operation": "report_family_stratified_claim", "parameters": {"positive_families": "only_supported_families", "negative_family": "reported", "universal": False}},
        ],
        "rejected_alternatives": [
            {"operation": "remove_negative_family", "reason": "Deletes a predeclared part of the evaluation population."},
            {"operation": "retain_universal_claim", "reason": "Contradicts observed family heterogeneity."},
        ],
    },
}


_COMPETING_REPAIR_FAMILIES: dict[str, tuple[str, str]] = {
    "ml_group_leakage": ("metric", "claim_scope"),
    "ml_accuracy_imbalance": ("data_split", "claim_scope"),
    "ml_best_seed_reporting": ("metric", "claim_scope"),
    "ml_universal_shift_hypothesis": ("metric", "claim_scope"),
    "bio_replicate_leakage": ("experimental_control", "claim_scope"),
    "bio_batch_confound": ("data_split", "claim_scope"),
    "bio_uncorrected_pathways": ("statistical_aggregation", "claim_scope"),
    "mat_unmatched_control": ("implementation_protocol", "claim_scope"),
    "mat_unconverged_mesh": ("experimental_control", "claim_scope"),
    "mat_conductivity_overclaim": ("implementation_protocol", "experimental_control"),
    "opt_stopping_mismatch": ("statistical_aggregation", "claim_scope"),
    "opt_survivor_average": ("metric", "claim_scope"),
    "opt_generalization_overclaim": ("metric", "statistical_aggregation"),
}


def repair_contract(flaw: FlawSpec) -> dict[str, Any]:
    """Return a JSON-safe hidden contract for one catalog flaw."""

    contract = copy.deepcopy(_CONTRACTS[flaw.flaw_id])
    contract.update(
        {
            "contract_id": f"repair::{flaw.flaw_id}",
            "analysis_tag": flaw.direction,
            "causal_target": flaw.target_slot,
        }
    )
    contract["competing_nonrepairs"] = [
        {
            "analysis_tag": direction,
            "exclusion_reason": contract["rejected_alternatives"][index]["reason"],
        }
        for index, direction in enumerate(
            _COMPETING_REPAIR_FAMILIES[flaw.flaw_id]
        )
    ]
    return contract


def validate_repair_operation(
    contract: dict[str, Any], witness: Any
) -> tuple[bool, str | None]:
    """Validate a structured operation against an accepted equivalence class."""

    if not isinstance(witness, dict):
        return False, "repair witness is not an object"
    operation = witness.get("operation")
    parameters = witness.get("parameters")
    if not isinstance(operation, str) or not isinstance(parameters, dict):
        return False, "repair witness requires operation and parameters"
    def contains_required(actual: Any, required: Any) -> bool:
        if isinstance(required, dict):
            return isinstance(actual, dict) and all(
                key in actual and contains_required(actual[key], value)
                for key, value in required.items()
            )
        if isinstance(required, list):
            return isinstance(actual, list) and all(item in actual for item in required)
        return actual == required

    for option in contract["accepted_operations"]:
        if operation == option["operation"] and contains_required(
            parameters, option["parameters"]
        ):
            return True, None
    return False, f"operation {operation!r} does not satisfy the hidden repair invariants"


def canonical_repair_operation(contract: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(contract["accepted_operations"][0])
