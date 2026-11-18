"""Curated domain packs and surface-realistic flaw realizations."""

from __future__ import annotations

from .models import DomainPack, EvidenceSpec, FlawSpec


def _e(
    evidence_id: str,
    action: str,
    observation: str,
    slot: str,
) -> EvidenceSpec:
    return EvidenceSpec(evidence_id, action, observation, slot)


DOMAIN_PACKS: tuple[DomainPack, ...] = (
    DomainPack(
        domain_id="ml_classification",
        project_id="ml_rare_class_calibration",
        display_name="Machine-learning classification",
        idea={
            "title": "Robust rare-class recognition with calibrated focal training",
            "question": "Can calibrated focal training improve rare-class recognition without reducing majority-class reliability?",
            "hypothesis": "Calibrated focal training improves rare-class recognition over cross-entropy while preserving calibration.",
            "design": {
                "dataset": "RareGroup-Image v1, a frozen imbalanced image corpus with source-group identifiers",
                "split_policy": "Group-stratified train/validation/test split created before augmentation or normalization fitting",
                "controls": "Cross-entropy baseline and focal-loss ablation under the identical architecture and budget",
                "intervention": "Focal loss followed by held-out temperature calibration",
            },
            "evaluation": {
                "primary_metric": "Macro-F1 on the untouched grouped test split",
                "secondary_metrics": "Per-class recall, expected calibration error, and compute",
            },
            "analysis": {
                "aggregation": "Mean and 95% interval across five predeclared seeds",
                "statistical_test": "Paired seed-level comparison against the baseline",
            },
            "implementation": {
                "protocol": "Fit preprocessing on training groups only; evaluate every test example",
            },
            "claims": {
                "main": "Improvement is claimed only if rare-class performance rises and calibration does not materially degrade",
                "limitations": "Conclusions are restricted to the fixed corpus and compute budget",
            },
        },
        fixed_constraints={
            "dataset": "RareGroup-Image v1 (frozen release hash)",
            "compute_budget": "2 accelerator-hours per seed",
            "architecture": "frozen backbone family",
        },
        environment={
            "tools": ["python", "pytest", "dataset_audit", "metric_report"],
            "available_actions": [
                "inspect_split_manifest",
                "run_group_overlap_audit",
                "inspect_confusion_matrix",
                "rerun_all_declared_seeds",
            ],
            "control_observations": {
                "run_group_overlap_audit": "No source group or augmentation lineage crosses train, validation, or test boundaries.",
                "inspect_confusion_matrix": "The class-balanced primary report includes every class and exposes each rare-class recall.",
                "rerun_all_declared_seeds": "All five predeclared seeds are included in the aggregate and its 95% interval.",
            },
        },
        flaws=(
            FlawSpec(
                flaw_id="ml_group_leakage",
                direction="data_split",
                target_slot="design.split_policy",
                gold_value="Group-stratified train/validation/test split created before augmentation or normalization fitting",
                flawed_value="Class-stratified split over cached sample records with deterministic augmentation lineage retained",
                evidence=(
                    _e(
                        "ml_overlap_hashes",
                        "run_group_overlap_audit",
                        "The metric implementation and declared seed runs pass their checks, but several validation items share source-group hashes with training items although filenames differ.",
                        "design.split_policy",
                    ),
                ),
                downstream_slots=("implementation.protocol", "analysis.results_table", "claims.main"),
                forbidden_shortcuts=("drop_overlapping_rows_from_report", "evaluate_subset_only", "rename_group_ids"),
                rationale="The split must be repaired at the group boundary; metric or model changes cannot remove shared provenance.",
            ),
            FlawSpec(
                flaw_id="ml_accuracy_imbalance",
                direction="metric",
                target_slot="evaluation.primary_metric",
                gold_value="Macro-F1 on the untouched grouped test split",
                flawed_value="Overall accuracy on the untouched grouped test split",
                evidence=(
                    _e(
                        "ml_minority_collapse",
                        "inspect_confusion_matrix",
                        "On the same clean grouped split and unchanged predictions, overall accuracy declares success while equal-class aggregation reverses the decision because two rare classes have near-zero recall.",
                        "evaluation.primary_metric",
                    ),
                ),
                downstream_slots=("analysis.results_table", "claims.main"),
                forbidden_shortcuts=("remove_rare_classes", "merge_labels", "report_accuracy_only"),
                rationale="The estimand concerns rare classes, so only a class-balanced primary metric matches the question.",
            ),
            FlawSpec(
                flaw_id="ml_best_seed_reporting",
                direction="statistical_aggregation",
                target_slot="analysis.aggregation",
                gold_value="Mean and 95% interval across five predeclared seeds",
                flawed_value="Best validation-selected seed reported as the representative test result",
                evidence=(
                    _e(
                        "ml_seed_variance",
                        "rerun_all_declared_seeds",
                        "The selected seed is an outlier; the remaining seeds overlap or trail the baseline.",
                        "analysis.aggregation",
                    ),
                ),
                downstream_slots=("analysis.results_table", "claims.main", "claims.limitations"),
                forbidden_shortcuts=("discard_failed_seeds", "change_seed_after_test", "hide_variance"),
                rationale="A fixed multi-seed aggregation is the only repair compatible with the declared stochastic experiment.",
            ),
        ),
    ),
    DomainPack(
        domain_id="ml_classification",
        project_id="ml_shift_hypothesis",
        display_name="Machine-learning classification",
        idea={
            "title": "Representation alignment across acquisition-shift regimes",
            "question": "Does frozen representation alignment improve held-out-site classification under moderate and severe acquisition shifts?",
            "hypothesis": "Alignment improves held-out-site Macro-F1 under moderate shift, while severe shift may reverse the effect.",
            "design": {
                "dataset": "ShiftBench-Clinical v1 with predeclared moderate/severe acquisition strata and untouched confirmation sites",
                "split_policy": "Source-site-disjoint development and confirmation partitions fixed before modeling",
                "controls": "No-alignment baseline with identical backbone, training budget, and site partitions",
                "intervention": "Frozen representation alignment applied with one predeclared strength",
            },
            "evaluation": {
                "primary_metric": "Macro-F1 stratified by predeclared shift severity on untouched confirmation sites",
                "secondary_metrics": "Calibration error, per-class recall, pooled Macro-F1, and compute",
            },
            "analysis": {
                "aggregation": "Mean and 95% interval across five fixed seeds within each severity stratum",
                "statistical_test": "Predeclared alignment-by-shift interaction followed by stratified contrasts",
                "results_table": "Development and confirmation effects reported separately for both shift strata",
            },
            "implementation": {
                "protocol": "Shift severity thresholds and alignment strength are frozen before confirmation evaluation",
            },
            "claims": {
                "main": "Conclusions are reported separately for moderate and severe shift before any pooled summary",
                "limitations": "Any hypothesis derived from development results is exploratory until tested on untouched confirmation sites",
            },
        },
        fixed_constraints={
            "dataset": "ShiftBench-Clinical v1 frozen release",
            "compute_budget": "3 accelerator-hours per seed",
            "confirmation_set": "untouched held-out acquisition sites",
        },
        environment={
            "tools": ["python", "shift_audit", "interaction_report", "confirmation_runner"],
            "available_actions": [
                "inspect_shift_strata",
                "run_shift_interaction_probe",
                "inspect_metric_and_split_checks",
                "run_untouched_confirmation",
            ],
            "control_observations": {
                "inspect_shift_strata": "Moderate and severe shift strata use frozen thresholds and contain disjoint acquisition sites.",
                "inspect_metric_and_split_checks": "The source-site split, stratified Macro-F1 implementation, controls, and all five seeds pass their declared checks.",
                "run_untouched_confirmation": "The predeclared conditional prediction is evaluated on untouched sites, with both severity strata and negative effects retained.",
            },
        },
        flaws=(
            FlawSpec(
                flaw_id="ml_universal_shift_hypothesis",
                direction="hypothesis",
                target_slot="hypothesis",
                gold_value="Alignment improves held-out-site Macro-F1 under moderate shift, while severe shift may reverse the effect.",
                flawed_value="Alignment improves held-out-site classification across all acquisition-shift regimes.",
                evidence=(
                    _e(
                        "ml_shift_sign_reversal",
                        "run_shift_interaction_probe",
                        "On development sites, alignment improves Macro-F1 under moderate shift but decreases it under severe shift; the predeclared interaction interval excludes zero, while split, metric, control, and seed checks pass.",
                        "hypothesis",
                    ),
                ),
                downstream_slots=("analysis.results_table", "claims.main", "claims.limitations"),
                forbidden_shortcuts=(
                    "drop_severe_shift",
                    "retune_shift_boundary_posthoc",
                    "label_exploratory_as_confirmatory",
                    "rerun_until_universal_gain",
                ),
                rationale="The universal prediction is directly falsified; valid responses must retain the negative regime and either reject it or condition it before independent confirmation.",
            ),
        ),
    ),
    DomainPack(
        domain_id="computational_biology",
        project_id="bio_multisite_response",
        display_name="Computational biology",
        idea={
            "title": "Transcriptomic response prediction across clinical sites",
            "question": "Can a sparse pathway model predict treatment response across clinical sites?",
            "hypothesis": "Pathway aggregation improves patient-level response prediction under site transfer.",
            "design": {
                "dataset": "Fixed multi-site cohort with repeated samples per patient and recorded processing batches",
                "split_policy": "Patient-disjoint, site-held-out evaluation with all replicates assigned together",
                "controls": "Expression-only reference plus batch-only, site-only, and joint batch+site nuisance baselines under matched preprocessing and identical patient/site splits",
                "intervention": "Sparse pathway aggregation model",
            },
            "evaluation": {
                "primary_metric": "Patient-level AUROC on the held-out site",
                "secondary_metrics": "AUPRC, calibration slope, and subgroup sensitivity",
            },
            "analysis": {
                "aggregation": "Bootstrap interval over patients, not samples",
                "statistical_test": "Benjamini-Hochberg correction over the predeclared pathway family",
            },
            "implementation": {
                "protocol": "Batch correction parameters are learned within each training fold only",
            },
            "claims": {
                "main": "The pathway model retains patient-level response signal on held-out clinical sites",
                "limitations": "No causal treatment-effect claim is made",
            },
        },
        fixed_constraints={
            "dataset": "multisite_cohort_v2",
            "compute_budget": "8 CPU-hours",
            "outcome": "fixed binary response label",
        },
        environment={
            "tools": ["python", "cohort_audit", "batch_report", "stats_report"],
            "available_actions": [
                "inspect_patient_split",
                "run_batch_predictability_probe",
                "inspect_pathway_pvalues",
                "rerun_patient_bootstrap",
            ],
            "control_observations": {
                "inspect_patient_split": "No patient or replicate group crosses an evaluation boundary, including the held-out site.",
                "run_batch_predictability_probe": "Expression, batch-only, site-only, and joint nuisance baselines use identical splits; the pathway comparison is reported against the joint nuisance baseline.",
                "inspect_pathway_pvalues": "Every predeclared pathway is included in the declared family-level error-control procedure.",
            },
        },
        flaws=(
            FlawSpec(
                flaw_id="bio_replicate_leakage",
                direction="data_split",
                target_slot="design.split_policy",
                gold_value="Patient-disjoint, site-held-out evaluation with all replicates assigned together",
                flawed_value="Sample-stratified split balancing response labels across train and test",
                evidence=(
                    _e(
                        "bio_patient_overlap",
                        "inspect_patient_split",
                        "Repeated samples from the same patient appear on both sides of the evaluation split.",
                        "design.split_policy",
                    ),
                ),
                downstream_slots=("implementation.protocol", "analysis.results_table", "claims.main"),
                forbidden_shortcuts=("deduplicate_after_scoring", "hide_patient_ids", "sample_level_claim"),
                rationale="Patient-disjoint assignment is forced by the patient-level estimand.",
            ),
            FlawSpec(
                flaw_id="bio_batch_confound",
                direction="experimental_control",
                target_slot="design.controls",
                gold_value="Expression-only reference plus batch-only, site-only, and joint batch+site nuisance baselines under matched preprocessing and identical patient/site splits",
                flawed_value="Expression-only reference under matched preprocessing and identical patient/site splits",
                evidence=(
                    _e(
                        "bio_batch_probe",
                        "run_batch_predictability_probe",
                        "Under the identical patient/site split, a joint batch+site nuisance predictor reproduces most of the pathway signal; split integrity and patient-level aggregation checks both pass, but the current comparison contains only the expression reference.",
                        "design.controls",
                    ),
                ),
                downstream_slots=("analysis.results_table", "claims.main", "claims.limitations"),
                forbidden_shortcuts=("remove_difficult_site", "erase_batch_metadata", "claim_causality"),
                rationale="The held-out-site signal cannot be interpreted as pathway-specific until matched nuisance baselines are included.",
            ),
            FlawSpec(
                flaw_id="bio_uncorrected_pathways",
                direction="statistical_test",
                target_slot="analysis.statistical_test",
                gold_value="Benjamini-Hochberg correction over the predeclared pathway family",
                flawed_value="Benjamini-Hochberg correction applied separately within each pathway annotation source",
                evidence=(
                    _e(
                        "bio_multiplicity",
                        "inspect_pathway_pvalues",
                        "The annotation sources overlap heavily and repeat related hypotheses; hits surviving source-wise correction vanish when the full predeclared pathway family is corrected once.",
                        "analysis.statistical_test",
                    ),
                ),
                downstream_slots=("analysis.results_table", "claims.main"),
                forbidden_shortcuts=("report_subset_of_tests", "redefine_family_posthoc", "omit_null_pathways"),
                rationale="The predeclared pathway family fixes the multiplicity correction scope.",
            ),
        ),
    ),
    DomainPack(
        domain_id="materials_simulation",
        project_id="materials_lithium_diffusion",
        display_name="Computational materials science",
        idea={
            "title": "Defect-assisted lithium diffusion in a solid electrolyte",
            "question": "Does a vacancy motif lower the lithium migration barrier under fixed composition and temperature?",
            "hypothesis": "The vacancy motif lowers the converged migration barrier relative to the pristine control.",
            "design": {
                "dataset": "Fixed crystal structures and pseudopotential family",
                "split_policy": "Candidate structures selected before observing barrier estimates",
                "controls": "Pristine and vacancy structures at matched composition, cell size, and temperature",
                "intervention": "Introduce the predeclared vacancy motif",
            },
            "evaluation": {
                "primary_metric": "Converged migration-barrier difference in eV",
                "secondary_metrics": "Path stability, force convergence, and compute cost",
            },
            "analysis": {
                "aggregation": "Report all predeclared paths with convergence uncertainty",
                "statistical_test": "Convergence check over k-point and cutoff refinements",
            },
            "implementation": {
                "protocol": "Identical cutoff, k-point density, cell size, NEB settings, and thermostat settings, each refined until the barrier changes by less than 0.01 eV",
            },
            "claims": {
                "main": "The vacancy motif changes the simulated barrier under the tested conditions",
                "limitations": "Conclusions are restricted to the tested structures and simulation protocol",
            },
        },
        fixed_constraints={
            "structures": "fixed_structure_pack_v1",
            "compute_budget": "500 CPU-hours",
            "pseudopotentials": "fixed family",
        },
        environment={
            "tools": ["simulation_runner", "convergence_audit", "unit_check", "path_report"],
            "available_actions": [
                "compare_control_settings",
                "run_convergence_audit",
                "inspect_unit_manifest",
                "inspect_all_paths",
            ],
            "control_observations": {
                "compare_control_settings": "Pristine and vacancy controls match composition, cell size, and temperature.",
                "run_convergence_audit": "Matched cutoff and k-point refinements change every reported barrier by less than 0.01 eV.",
                "inspect_all_paths": "All predeclared paths are reported, and the conclusion is limited to simulated barriers under the tested protocol.",
            },
        },
        flaws=(
            FlawSpec(
                flaw_id="mat_unmatched_control",
                direction="experimental_control",
                target_slot="design.controls",
                gold_value="Pristine and vacancy structures at matched composition, cell size, and temperature",
                flawed_value="Pristine reference from the smallest available cell compared with the vacancy production cell",
                evidence=(
                    _e(
                        "mat_cell_mismatch",
                        "compare_control_settings",
                        "The pristine and vacancy energies were produced with different cell sizes and finite-size errors.",
                        "design.controls",
                    ),
                ),
                downstream_slots=("analysis.results_table", "claims.main"),
                forbidden_shortcuts=("offset_energy_manually", "drop_pristine_control", "compare_per_atom_only"),
                rationale="A matched cell is required to isolate the vacancy motif from finite-size effects.",
            ),
            FlawSpec(
                flaw_id="mat_unconverged_mesh",
                direction="implementation_protocol",
                target_slot="implementation.protocol",
                gold_value="Identical cutoff, k-point density, cell size, NEB settings, and thermostat settings, each refined until the barrier changes by less than 0.01 eV",
                flawed_value="Use the fastest cutoff and k-point mesh that completes for each structure",
                evidence=(
                    _e(
                        "mat_convergence_drift",
                        "run_convergence_audit",
                        "All predeclared paths and matched structures are present, yet rerunning the same paths with shared cutoff and k-point refinements reverses the barrier ordering until changes fall below 0.01 eV.",
                        "implementation.protocol",
                    ),
                ),
                downstream_slots=("analysis.results_table", "claims.main", "claims.limitations"),
                forbidden_shortcuts=("increase_tolerance", "omit_failed_refinement", "mix_precision_levels"),
                rationale="Only a matched converged protocol can separate the physical effect from numerical error.",
            ),
            FlawSpec(
                flaw_id="mat_conductivity_overclaim",
                direction="claim_scope",
                target_slot="claims.main",
                gold_value="The vacancy motif changes the simulated barrier under the tested conditions",
                flawed_value="The lower barrier is consistent with improved room-temperature ionic transport across related members of the tested material family",
                evidence=(
                    _e(
                        "mat_claim_gap",
                        "inspect_all_paths",
                        "The same barrier values produce conflicting conductivity rankings under plausible transport-network models; no conductivity model or experimental sample was evaluated, while the barrier result itself is stable.",
                        "claims.main",
                    ),
                ),
                downstream_slots=("claims.limitations",),
                forbidden_shortcuts=("invent_conductivity", "cite_unrun_experiment", "suppress_scope_limit"),
                rationale="The available evidence can support only the tested simulation-level barrier claim.",
            ),
        ),
    ),
    DomainPack(
        domain_id="numerical_optimization",
        project_id="optimization_adaptive_preconditioning",
        display_name="Numerical optimization",
        idea={
            "title": "Adaptive preconditioning for sparse inverse problems",
            "question": "Does adaptive preconditioning reduce time-to-tolerance on fixed sparse inverse problems?",
            "hypothesis": "Adaptive preconditioning reduces wall-clock time to a common residual tolerance without sacrificing solution quality.",
            "design": {
                "dataset": "Fixed matrix suite spanning condition number and sparsity",
                "split_policy": "Tune on development matrices and evaluate once on a held-out matrix suite",
                "controls": "Unpreconditioned and fixed-preconditioner solvers with identical stopping rules",
                "intervention": "Adaptive preconditioner update schedule",
            },
            "evaluation": {
                "primary_metric": "Wall-clock time to the shared residual tolerance",
                "secondary_metrics": "Iterations, peak memory, residual, and objective gap",
            },
            "analysis": {
                "aggregation": "Geometric mean of paired reference/proposed runtimes over every held-out matrix, assigning the fixed 300-second budget to each timeout",
                "statistical_test": "Paired bootstrap over held-out matrices",
            },
            "implementation": {
                "protocol": "Same hardware, precision, timeout, and residual computation for every solver",
            },
            "claims": {
                "main": "The method improves time-to-tolerance on the fixed held-out suite",
                "limitations": "No claim beyond the tested matrix families and hardware",
            },
        },
        fixed_constraints={
            "matrix_suite": "sparse_suite_v3",
            "hardware": "fixed CPU node",
            "timeout": "300 seconds per matrix",
        },
        environment={
            "tools": ["solver_runner", "tolerance_audit", "runtime_report", "failure_report"],
            "available_actions": [
                "compare_stopping_rules",
                "inspect_failed_matrices",
                "recompute_shared_residual",
                "run_full_heldout_suite",
            ],
            "control_observations": {
                "compare_stopping_rules": "Every solver runtime is measured to the same externally recomputed residual tolerance.",
                "inspect_failed_matrices": "Every held-out matrix is retained, with each timeout assigned the fixed 300-second budget.",
                "run_full_heldout_suite": "All matrix families are reported and the conclusion is bounded to the fixed suite and tested hardware.",
            },
        },
        flaws=(
            FlawSpec(
                flaw_id="opt_stopping_mismatch",
                direction="metric",
                target_slot="evaluation.primary_metric",
                gold_value="Wall-clock time to the shared residual tolerance",
                flawed_value="Wall-clock time until each solver's native stopping rule fires",
                evidence=(
                    _e(
                        "opt_residual_mismatch",
                        "compare_stopping_rules",
                        "The adaptive solver stops earlier at a substantially looser recomputed residual.",
                        "evaluation.primary_metric",
                    ),
                ),
                downstream_slots=("analysis.results_table", "claims.main"),
                forbidden_shortcuts=("loosen_tolerance", "compare_native_status_only", "omit_residual"),
                rationale="A shared accuracy target is required for runtime comparisons to measure the same task.",
            ),
            FlawSpec(
                flaw_id="opt_survivor_average",
                direction="statistical_aggregation",
                target_slot="analysis.aggregation",
                gold_value="Geometric mean of paired reference/proposed runtimes over every held-out matrix, assigning the fixed 300-second budget to each timeout",
                flawed_value="Arithmetic mean speedup over matrices completed by every solver",
                evidence=(
                    _e(
                        "opt_failure_selection",
                        "inspect_failed_matrices",
                        "Most hard matrices disappear from the aggregate because the proposed solver times out on them.",
                        "analysis.aggregation",
                    ),
                ),
                downstream_slots=("analysis.results_table", "claims.main", "claims.limitations"),
                forbidden_shortcuts=("drop_timeouts", "raise_timeout_for_proposed_only", "average_successes_only"),
                rationale="Failure-aware aggregation is forced by the fixed held-out suite and timeout.",
            ),
            FlawSpec(
                flaw_id="opt_generalization_overclaim",
                direction="claim_scope",
                target_slot="claims.main",
                gold_value="The method improves time-to-tolerance on the fixed held-out suite",
                flawed_value="The method is universally faster for sparse inverse problems",
                evidence=(
                    _e(
                        "opt_family_heterogeneity",
                        "run_full_heldout_suite",
                        "Speedups are positive for two matrix families and negative for the remaining family.",
                        "claims.main",
                    ),
                ),
                downstream_slots=("claims.limitations",),
                forbidden_shortcuts=("remove_negative_family", "relabel_heldout_suite", "claim_universal"),
                rationale="The fixed heterogeneous suite permits only a bounded empirical claim.",
            ),
        ),
    ),
)


def get_domain_pack(project_or_domain_id: str) -> DomainPack:
    for pack in DOMAIN_PACKS:
        if pack.project_id == project_or_domain_id:
            return pack
    domain_matches = [
        pack for pack in DOMAIN_PACKS if pack.domain_id == project_or_domain_id
    ]
    if len(domain_matches) == 1:
        return domain_matches[0]
    if len(domain_matches) > 1:
        projects = ", ".join(pack.project_id for pack in domain_matches)
        raise KeyError(
            f"Domain {project_or_domain_id!r} has multiple projects; "
            f"use one of: {projects}"
        )
    known = ", ".join(pack.project_id for pack in DOMAIN_PACKS)
    raise KeyError(
        f"Unknown domain project {project_or_domain_id!r}; expected one of: {known}"
    )
