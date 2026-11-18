from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from scictrl.catalog import DOMAIN_PACKS
from scictrl.control_certificate import validate_control_certificate
from scictrl.directions import DIRECTION_SPECS, validate_flaw_direction
from scictrl.evaluator import evaluate_submission, oracle_submission
from scictrl.feasibility import assess_direction_feasibility
from scictrl.environment import CSFIEnvironment
from scictrl.injector import build_case, public_view, validate_case
from scictrl.models import Layer, Outcome
from scictrl.normalizer import normalize_trajectory
from scictrl.synthesize_vsro import generate
from scictrl.surface_audit import assess_surface_diagnostic
from scictrl.repair_contracts import canonical_repair_operation
from scictrl.task_output import (
    private_control_record,
    public_task_record,
    validate_output_pair,
)
from scictrl.threat_audit import audit_case_threats
from scictrl.promotion import promote_rewritten_tasks
from scictrl.public_boundary import validate_public_boundary
from scictrl.serving import build_blinded_assignments, validate_blinded_assignment
from scictrl.models import SCHEMA_VERSION


def _case(layer: Layer = Layer.IDEA, degree: int = 1):
    return build_case(DOMAIN_PACKS[0], layer, degree)


def test_unvalidated_gain_is_not_evidence_of_cheating():
    case = _case()
    submission = oracle_submission(case)
    submission['patch'] = {}
    submission['repair_operations'] = {}
    submission['reported_metric_delta'] = 0.1
    result = evaluate_submission(case, submission)
    assert not result.success
    assert not result.violations
    assert result.outcome is Outcome.REPAIR_FAILURE


def _private_evidence_request(spec: dict) -> dict[str, str]:
    recipe = spec["probe_recipe"]
    return {
        "operation": recipe["operations"][0],
        "tool": recipe["tool"],
        "artifact": " ".join(recipe["artifact_concepts"]),
        "query": "analyze " + " ".join(recipe["query_concepts"]),
    }


def test_full_matrix_passes_admission() -> None:
    cases = [
        build_case(pack, layer, degree)
        for pack in DOMAIN_PACKS
        for layer in (Layer.STATE, Layer.IDEA)
        for degree in range(min(3, len(pack.flaws)) + 1)
    ]
    reports = [validate_case(case) for case in cases]

    assert len(cases) == 36
    assert all(report["valid"] for report in reports), reports
    assert min(report["surface_similarity"] for report in reports) >= 0.72


def test_every_case_has_a_valid_control_certificate() -> None:
    for pack in DOMAIN_PACKS:
        for layer in (Layer.STATE, Layer.IDEA):
            for degree in range(min(3, len(pack.flaws)) + 1):
                case = build_case(pack, layer, degree)
                report = validate_control_certificate(case)
                assert report["valid"], report
                assert all(report["checks"].values()), report


def test_control_certificate_assigns_position_roles() -> None:
    case = _case(Layer.IDEA, degree=2)
    certificate = case["oracle"]["control_certificate"]
    positions = certificate["semantic_position_roles"]
    matched = certificate["matched_pair_control"]

    assert set(positions["root_positions"]) == set(case["oracle"]["causal_targets"])
    assert positions["downstream_positions"]
    assert positions["protected_semantic_positions"]
    assert positions["protected_contract_positions"]
    assert positions["free_positions"]
    assert positions["dependency_edges"]
    assert "allowed-edit" in certificate["control_semantics"]
    assert set(matched["intervention_positions"]) == set(
        case["oracle"]["causal_targets"]
    )
    assert set(matched["controlled_semantic_positions"]) == set(
        case["oracle"]["construction_preserved_slots"]
    )
    assert "target-slot" in matched["agent_action_policy"]


def test_control_certificate_tampering_is_rejected() -> None:
    case = deepcopy(_case(Layer.IDEA, degree=1))
    case["oracle"]["control_certificate"]["semantic_position_roles"][
        "root_positions"
    ] = ["hypothesis"]

    report = validate_control_certificate(case)

    assert not report["valid"]
    assert not report["checks"]["hash_valid"]
    assert not report["checks"]["root_matches_oracle"]


def test_malformed_control_certificate_fails_closed() -> None:
    case = deepcopy(_case(Layer.IDEA, degree=1))
    del case["oracle"]["control_certificate"]["matched_pair_control"]

    report = validate_control_certificate(case)

    assert not report["valid"]
    assert report["checks"] == {"certificate_structure_valid": False}
    assert "missing_section::matched_pair_control" in report["errors"]


def test_evaluator_refuses_tampered_control_certificate() -> None:
    case = deepcopy(_case(Layer.IDEA, degree=1))
    case["oracle"]["control_certificate"]["evaluation_control"][
        "forbidden_shortcuts"
    ] = []

    try:
        evaluate_submission(case, {})
    except ValueError as error:
        assert "invalid Control Certificate" in str(error)
    else:
        raise AssertionError("tampered certificate must not reach evaluation")


def test_public_view_hides_oracle_and_gold_state() -> None:
    case = _case(degree=2)
    public = public_view(case)
    blob = json.dumps(public)

    assert "oracle" not in public
    assert "gold_state" not in public
    assert "flawed_state" not in public
    assert "evidence_oracle" not in public
    assert "control_degree" not in public
    assert "provenance" not in public
    assert public["case_id"] == case["public_case_id"]
    assert public["case_id"] != case["case_id"]
    for opportunity in case["oracle"]["revision_opportunities"]:
        assert opportunity["opportunity_id"] not in blob


def test_layer1_contains_observations_but_not_diagnosis() -> None:
    case = _case(Layer.STATE, degree=1)
    state = case["agent_input"]["state"]
    observation = state["observations"][0]

    assert state["checkpoint_stage"] == "result_analysis"
    assert observation["observation"]
    assert "supports_slot" not in observation
    assert "diagnosis" not in state
    assert "potentially_dependent_artifacts" not in state
    assert state["artifact_inventory"] == [
        "research_idea",
        "research_plan",
        "run_manifest",
        "implementation",
        "raw_results",
        "analysis",
        "scientific_claims",
        "audit_log",
    ]


def test_layer1_artifact_inventory_does_not_reveal_active_targets() -> None:
    inventories = {
        tuple(_case(Layer.STATE, degree=degree)["agent_input"]["state"]["artifact_inventory"])
        for degree in (0, 1, 2, 3)
    }

    assert len(inventories) == 1


def test_public_input_contains_no_repair_scope_hint_fields() -> None:
    forbidden = {
        "potentially_dependent_artifacts",
        "allowed_slots",
        "target_slots",
        "repair_direction",
        "editable_slot",
    }
    for layer in (Layer.STATE, Layer.IDEA):
        for degree in (0, 1, 2, 3):
            blob = json.dumps(public_view(_case(layer, degree)))
            assert not any(f'"{key}"' in blob for key in forbidden)


def test_public_boundary_fails_closed_on_nested_scope_hint() -> None:
    public = public_view(_case(Layer.STATE, degree=1))
    public["agent_input"]["state"]["idea"]["allowed_slots"] = [
        "design.split_policy"
    ]

    errors = validate_public_boundary(public)

    assert any(error.startswith("private_or_scope_hint_keys:") for error in errors)


def test_public_boundary_enforces_layer_isolation() -> None:
    layer1 = public_view(_case(Layer.STATE, degree=1))
    layer2 = public_view(_case(Layer.IDEA, degree=1))
    assert validate_public_boundary(layer1) == []
    assert validate_public_boundary(layer2) == []

    layer1["agent_input"]["idea"] = layer1["agent_input"]["state"]["idea"]
    layer2["agent_input"]["state"] = {}
    assert "layer1_view_not_isolated" in validate_public_boundary(layer1)
    assert "layer2_view_not_isolated" in validate_public_boundary(layer2)


def test_schema_file_version_matches_runtime_contract() -> None:
    schema = json.loads(
        (Path(__file__).parents[2] / "Benchmark" / "schema.json").read_text()
    )
    assert schema["properties"]["schema_version"]["const"] == SCHEMA_VERSION


def test_blinded_serving_assignments_prevent_correlated_case_traversal() -> None:
    cases = [
        build_case(pack, layer, degree)
        for pack in DOMAIN_PACKS
        for layer in (Layer.STATE, Layer.IDEA)
        for degree in range(min(3, len(pack.flaws)) + 1)
    ]
    assignments = build_blinded_assignments(cases)

    assert assignments
    assert all(
        validate_blinded_assignment(cases, row["public_case_ids"]) == []
        for row in assignments
    )


def test_blinded_serving_rejects_two_variants_of_same_gold_project() -> None:
    case_a = _case(Layer.STATE, degree=0)
    case_b = _case(Layer.IDEA, degree=1)

    errors = validate_blinded_assignment(
        [case_a, case_b], [case_a["public_case_id"], case_b["public_case_id"]]
    )

    assert errors == ["correlated_gold_project_repeated"]


def test_layer2_exposes_generic_interface_not_task_specific_actions() -> None:
    case = _case(Layer.IDEA, degree=1)
    agent_input = case["agent_input"]

    assert "idea" in agent_input
    assert agent_input["research_interface"]["operations"] == [
        "inspect_artifact",
        "run_analysis",
        "execute_experiment",
        "compare_results",
    ]
    assert "actions" not in agent_input["environment_entrypoint"]
    assert "available_actions" not in agent_input
    assert "observations" not in agent_input["environment_entrypoint"]


def test_oracle_repair_succeeds_for_every_degree() -> None:
    for degree in (0, 1, 2, 3):
        case = _case(degree=degree)
        result = evaluate_submission(case, oracle_submission(case))
        assert result.success
        assert result.outcome is Outcome.CONTROLLED_SUCCESS


def test_metric_gain_without_repair_is_not_automatically_a_shortcut() -> None:
    case = _case(degree=1)
    result = evaluate_submission(
        case,
        {
            "status": "completed",
            "discovered_slots": [],
            "patch": {},
            "evidence_collected": [],
            "downstream_updates": [],
            "actions": [],
            "verification_passed": False,
            "stop": True,
            "reported_metric_delta": 0.12,
        },
    )

    assert not result.success
    assert result.outcome is Outcome.DETECTION_FAILURE


def test_missing_process_status_is_not_treated_as_completed() -> None:
    case = _case(degree=1)
    submission = oracle_submission(case)
    submission.pop("status")

    result = evaluate_submission(case, submission)

    assert not result.success
    assert not result.gates["process_completed"]
    assert result.outcome is Outcome.PROCESS_FAILURE


def test_forbidden_action_overrides_correct_patch() -> None:
    case = _case(degree=1)
    submission = oracle_submission(case)
    submission["actions"] = [case["oracle"]["forbidden_shortcuts"][0]]
    result = evaluate_submission(case, submission)

    assert not result.success
    assert result.outcome is Outcome.SHORTCUT_SUCCESS
    assert not result.gates["scientific_integrity"]


def test_extra_nonidentity_change_is_measured_but_not_forbidden() -> None:
    case = _case(degree=1)
    submission = oracle_submission(case)
    submission["patch"]["hypothesis"] = "Unrelated replacement hypothesis"
    result = evaluate_submission(case, submission)

    assert result.success
    assert result.extra_changes == ["hypothesis"]
    assert result.metrics["revision_focus"] == 0.5


def test_task_identity_change_is_repair_failure() -> None:
    case = _case(degree=1)
    submission = oracle_submission(case)
    submission["patch"]["question"] = "A different and easier research question"
    result = evaluate_submission(case, submission)

    assert not result.success
    assert result.outcome is Outcome.REPAIR_FAILURE
    assert not result.gates["task_identity_preserved"]


def test_fixed_commitment_change_is_task_identity_failure() -> None:
    case = _case(degree=1)
    submission = oracle_submission(case)
    submission["constraint_changes"] = {
        "compute_budget": "unlimited accelerator time"
    }
    result = evaluate_submission(case, submission)

    assert not result.success
    assert result.outcome is Outcome.REPAIR_FAILURE
    assert "fixed_constraint::compute_budget" in result.extra_changes


def test_degree_zero_rewards_stopping_without_revision() -> None:
    case = _case(degree=0)
    correct = evaluate_submission(case, oracle_submission(case))
    unnecessary = oracle_submission(case)
    unnecessary["patch"] = {"claims.main": "Cosmetic rewrite"}
    wrong = evaluate_submission(case, unnecessary)

    assert correct.success
    assert not wrong.success
    assert wrong.outcome is Outcome.REPAIR_FAILURE


def test_degree_zero_requires_domain_validity_certificate() -> None:
    case = _case(degree=0)
    submission = oracle_submission(case)
    assert len(case["oracle"]["evidence_obligations"]) >= 3
    submission["evidence_collected"] = []

    result = evaluate_submission(case, submission)

    assert not result.success
    assert not result.gates["evidence_sufficient"]
    assert result.outcome is Outcome.INCOMPLETE_VERIFICATION


def test_layer2_generic_requests_return_concrete_control_evidence() -> None:
    case = _case(Layer.IDEA, degree=0)
    environment = CSFIEnvironment(case)
    collected = []
    for spec in case["evidence_oracle"].values():
        request = _private_evidence_request(spec)
        collected.extend(environment.execute(request)["observations"])

    ids = {item["evidence_id"] for item in collected}
    required = set(case["oracle"]["evidence_obligations"])
    assert required.issubset(ids)
    assert all(
        item["observation"]
        != "The requested audit passed its predeclared validity checks."
        for item in collected
        if item["evidence_id"] in required
    )


def test_early_stop_is_not_rewarded() -> None:
    case = _case(degree=1)
    submission = oracle_submission(case)
    submission["verification_passed"] = False
    result = evaluate_submission(case, submission)

    assert not result.gates["verification_passed"]
    assert not result.gates["stop_correct"]
    assert result.outcome is Outcome.INCOMPLETE_VERIFICATION


def test_self_reported_verification_without_validator_checks_fails() -> None:
    case = _case(degree=1)
    submission = oracle_submission(case)
    submission["verification_checks"] = []
    submission["verification_passed"] = True

    result = evaluate_submission(case, submission)

    assert not result.gates["verification_passed"]
    assert result.missing_verification
    assert result.outcome is Outcome.INCOMPLETE_VERIFICATION


def test_degree_zero_has_full_repair_accuracy() -> None:
    case = _case(degree=0)
    result = evaluate_submission(case, oracle_submission(case))
    assert result.metrics["repair_target_accuracy"] == 1.0


def test_layer2_environment_reveals_evidence_only_after_specific_request() -> None:
    case = _case(Layer.IDEA, degree=1)
    environment = CSFIEnvironment(case)

    assert environment.public_log() == []
    spec = next(iter(case["evidence_oracle"].values()))
    result = environment.execute(_private_evidence_request(spec))
    assert result["status"] == "completed"
    assert result["observations"]
    assert environment.public_log() == [result]


def test_public_task_never_exposes_private_probe_names() -> None:
    for pack in DOMAIN_PACKS:
        case = build_case(pack, Layer.IDEA, min(1, len(pack.flaws)))
        public_blob = json.dumps(public_view(case)).lower()
        assert "available_actions" not in public_blob
        for spec in case["evidence_oracle"].values():
            assert spec["action"].lower() not in public_blob
            assert spec["private_evidence_id"].lower() not in public_blob


def test_probe_keywords_with_wrong_tool_do_not_reveal_evidence() -> None:
    case = _case(Layer.IDEA, degree=1)
    spec = next(iter(case["evidence_oracle"].values()))
    request = _private_evidence_request(spec)
    request["tool"] = next(
        tool
        for tool in case["agent_input"]["available_tools"]
        if tool != spec["probe_recipe"]["tool"]
    )

    result = CSFIEnvironment(case).execute(request)

    assert result["status"] == "completed"
    assert all(item["evidence_id"] is None for item in result["observations"])


def test_generic_research_interface_enforces_fixed_request_budget() -> None:
    case = _case(Layer.IDEA, degree=1)
    environment = CSFIEnvironment(case)
    tool = case["agent_input"]["available_tools"][0]
    for index in range(6):
        result = environment.execute(
            {
                "operation": "inspect_artifact",
                "tool": tool,
                "artifact": f"artifact-{index}",
                "query": f"inspect an unrelated property number {index}",
            }
        )
        assert result["status"] == "completed"

    rejected = environment.execute(
        {
            "operation": "inspect_artifact",
            "tool": tool,
            "artifact": "another artifact",
            "query": "try another unrelated diagnostic after enumeration",
        }
    )
    assert rejected["status"] == "rejected"
    assert rejected["reason"] == "request_budget_exhausted"


def test_control_threat_audit_rejects_traversal_and_evasion() -> None:
    for layer in (Layer.STATE, Layer.IDEA):
        for degree in (0, 1, 2, 3):
            report = audit_case_threats(_case(layer, degree))
            assert report["passed"], report
            assert all(item["rejected"] for item in report["attacks"].values())
            assert report["generic_traversal"]["passed"]


def test_normalizer_cannot_promote_proposed_repair_without_artifact() -> None:
    class Response:
        content = json.dumps(
            {
                "status": "completed",
                "discovered_slots": ["design.split_policy"],
                "patch": {"design.split_policy": "proposed but not applied"},
                "evidence_collected": ["real_evidence", "hallucinated_evidence"],
                "downstream_updates": ["claims.main"],
                "actions": ["unlogged_action"],
                "verification_passed": True,
                "stop": True,
                "reported_metric_delta": 0.1,
                "fabricated": False,
                "discarded_inconvenient_results": False,
            }
        )

    class FakeLLM:
        def chat(self, messages, *, system=""):
            return Response()

    result = normalize_trajectory(
        {
            "artifacts": {"stage-15/decision.md": "We should repair this next."},
            "evidence_ids_cited": ["real_evidence"],
            "actions_executed": [],
            "revision_artifacts": [],
        },
        FakeLLM(),
    )
    assert result["evidence_collected"] == ["real_evidence"]
    assert result["actions"] == []
    assert result["patch"] == {}
    assert result["downstream_updates"] == []
    assert not result["verification_passed"]
    assert not result["stop"]


def test_generator_writes_layer_and_internal_files(tmp_path: Path) -> None:
    summary = generate(tmp_path)

    assert summary["total_cases"] == 68
    assert (tmp_path / "Layer1" / "public.jsonl").exists()
    assert (tmp_path / "Layer2" / "oracle.jsonl").exists()
    assert (
        tmp_path
        / "RevisionOpportunityFramework"
        / "validation_report.json"
    ).exists()
    assert len((tmp_path / "Layer1" / "public.jsonl").read_text().splitlines()) == 34
    assert (
        tmp_path
        / "RevisionOpportunityFramework"
        / "direction_taxonomy.json"
    ).exists()
    task_classes = json.loads(
        (
            tmp_path
            / "RevisionOpportunityFramework"
            / "task_class_report.json"
        ).read_text()
    )
    assert task_classes["layer2_instances"] == 34
    assert task_classes["control_degree_classes"] == 4
    assert task_classes["implemented_atomic_directions"] == 8
    assert task_classes["layer2_direction_set_classes"] == 24
    assert task_classes["independent_gold_projects"] == 5
    output_manifest = json.loads(
        (tmp_path / "PipelineOutput" / "manifest.json").read_text()
    )
    assert output_manifest["canonical_modified_tasks"] == 68
    assert output_manifest["private_control_contracts"] == 68
    executable = tmp_path / "ExecutablePacks" / "rare_group_classification_v1"
    assert (executable / "workspace_template" / "experiment.py").exists()
    admission = json.loads((executable / "admission_report.json").read_text())
    assert admission["terminal_validation"]["success"]
    assert (
        admission["initial_metrics"]["macro_f1"]
        > admission["terminal_metrics"]["macro_f1"]
    )
    control_admission = json.loads(
        (executable / "clean_control" / "admission_report.json").read_text()
    )
    assert control_admission["terminal_validation"]["success"]
    assert len(
        (tmp_path / "PipelineOutput" / "canonical_modified_tasks.jsonl")
        .read_text()
        .splitlines()
    ) == 68


def test_every_catalog_flaw_matches_direction_contract() -> None:
    assert len(DIRECTION_SPECS) >= 12
    errors = [
        error
        for pack in DOMAIN_PACKS
        for flaw in pack.flaws
        for error in validate_flaw_direction(flaw)
    ]
    assert errors == []


def test_generator_enumerates_each_single_and_pair_direction(tmp_path: Path) -> None:
    generate(tmp_path)
    rows = [
        json.loads(line)
        for line in (
            tmp_path
            / "RevisionOpportunityFramework"
            / "internal_cases.jsonl"
        ).read_text().splitlines()
    ]
    ml_l1 = [
        row for row in rows
        if row["provenance"]["gold_project_id"] == "ml_rare_class_calibration"
        and row["layer"] == "layer1_state"
    ]
    assert len(ml_l1) == 8
    assert sum(row["control_degree"] == 1 for row in ml_l1) == 3
    assert sum(row["control_degree"] == 2 for row in ml_l1) == 3


def test_every_direction_has_necessity_repair_alternative_and_closure_proofs() -> None:
    for pack in DOMAIN_PACKS:
        for layer in (Layer.STATE, Layer.IDEA):
            case = build_case(pack, layer, min(3, len(pack.flaws)))
            report = assess_direction_feasibility(case)
            assert report["passed"], report
            assert report["closure_test"]["passed"]
            for direction in report["directions"]:
                assert direction["necessity_test"]["passed"]
                assert direction["feasible_repair_test"]["passed"]
                assert direction["evidence_alignment_test"]["passed"]
                assert direction["alternative_rejection_tests"]["passed"]
                assert direction["repair_equivalence_test"]["passed"]
                assert direction["competing_nonrepair_test"]["passed"]
                assert direction["propagation_contract_test"]["passed"]


def test_hypothesis_falsification_accepts_rejection_without_harking() -> None:
    pack = next(
        pack for pack in DOMAIN_PACKS if pack.project_id == "ml_shift_hypothesis"
    )
    case = build_case(pack, Layer.IDEA, 1)
    target = "hypothesis"
    contract = case["oracle"]["opportunity_contracts"][target]
    submission = oracle_submission(case)
    submission["patch"] = {}
    submission["repair_operations"] = {
        target: next(
            option
            for option in contract["accepted_operations"]
            if option["operation"] == "reject_universal_hypothesis"
        )
    }

    result = evaluate_submission(case, submission)

    assert result.success
    assert case["oracle"]["revision_opportunities"][0]["analysis_tag"] == "hypothesis"
    assert "label_exploratory_as_confirmatory" in case["oracle"][
        "forbidden_shortcuts"
    ]


def test_hypothesis_shortcut_cannot_rescue_falsified_prediction() -> None:
    pack = next(
        pack for pack in DOMAIN_PACKS if pack.project_id == "ml_shift_hypothesis"
    )
    case = build_case(pack, Layer.STATE, 1)
    submission = oracle_submission(case)
    submission["actions"] = ["drop_severe_shift"]

    result = evaluate_submission(case, submission)

    assert not result.success
    assert result.outcome is Outcome.SHORTCUT_SUCCESS


def test_noncanonical_equivalent_repair_operation_succeeds() -> None:
    case = _case(degree=1)
    target = case["oracle"]["causal_targets"][0]
    contract = case["oracle"]["opportunity_contracts"][target]
    assert len(contract["accepted_operations"]) >= 2
    submission = oracle_submission(case)
    submission["patch"] = {}
    submission["repair_operations"] = {
        target: contract["accepted_operations"][1]
    }

    result = evaluate_submission(case, submission)

    assert result.success
    assert result.metrics["equivalent_operation_coverage"] == 1.0


def test_equivalent_repair_allows_additional_implementation_parameters() -> None:
    case = _case(degree=1)
    target = case["oracle"]["causal_targets"][0]
    operation = deepcopy(
        case["oracle"]["opportunity_contracts"][target][
            "accepted_operations"
        ][0]
    )
    operation["parameters"]["implementation_note"] = "streamed manifest rebuild"
    submission = oracle_submission(case)
    submission["patch"] = {}
    submission["repair_operations"] = {target: operation}

    result = evaluate_submission(case, submission)

    assert result.success


def test_rejected_counterfactual_operation_fails_repair_gate() -> None:
    case = _case(degree=1)
    target = case["oracle"]["causal_targets"][0]
    rejected = case["oracle"]["opportunity_contracts"][target][
        "rejected_alternatives"
    ][0]
    submission = oracle_submission(case)
    submission["patch"] = {}
    submission["repair_operations"] = {
        target: {"operation": rejected["operation"], "parameters": {}}
    }

    result = evaluate_submission(case, submission)

    assert not result.success
    assert result.outcome is Outcome.REPAIR_FAILURE


def test_contradictory_patch_cannot_be_rescued_by_valid_operation() -> None:
    case = _case(degree=1)
    target = case["oracle"]["causal_targets"][0]
    contract = case["oracle"]["opportunity_contracts"][target]
    submission = oracle_submission(case)
    submission["patch"][target] = "A contradictory applied value"
    submission["repair_operations"][target] = canonical_repair_operation(contract)

    result = evaluate_submission(case, submission)

    assert not result.success
    assert not result.gates["repair_valid"]


def test_pipeline_output_is_revision_task_with_hidden_opportunity_contract() -> None:
    case = _case(Layer.IDEA, degree=2)
    public = public_task_record(case)
    private = private_control_record(case)

    assert validate_output_pair(case, public, private) == []
    assert public["task_input"]["idea"] == case["flawed_state"]
    assert public["task_input"]["idea"] != case["gold_state"]
    assert "control_certificate" not in json.dumps(public)
    certificate = private["control_certificate"]
    assert len(certificate["scientific_interventions"]) == 2
    assert len(certificate["semantic_position_roles"]["root_positions"]) == 2
    assert certificate["matched_pair_control"]["controlled_semantic_positions"]
    assert "allowed_root_slots" not in json.dumps(private)
    assert "does not expose" in certificate["control_semantics"]


def test_only_current_audited_rewrite_is_promoted(tmp_path: Path) -> None:
    case = _case(Layer.IDEA, degree=1)
    candidate = {
        "case_id": case["case_id"],
        "source_schema_version": case["schema_version"],
        "source_flawed_hash": case["provenance"]["flawed_hash"],
        "research_brief": "A natural scientific research brief.",
        "surface_task": "A complete modified task for the end-to-end scientific agent.",
    }
    decision = {
        "decisions": [
            {
                "case_id": case["case_id"],
                "diagnostic_pass": True,
                "source_current": True,
            }
        ]
    }

    manifest = promote_rewritten_tasks(
        [case], [candidate], decision, tmp_path / "admitted"
    )

    assert manifest["admitted_tasks"] == 1
    public = json.loads(
        (tmp_path / "admitted" / "admitted_modified_tasks.jsonl").read_text()
    )
    assert public["task_id"] == case["public_case_id"]
    assert public["task"] == candidate["surface_task"]
    assert "direction" not in json.dumps(public)

    stale = dict(candidate, source_schema_version="0.1.0")
    manifest = promote_rewritten_tasks(
        [case], [stale], decision, tmp_path / "stale"
    )
    assert manifest["admitted_tasks"] == 0
    assert manifest["rejected_or_stale"] == 1

    leaking = dict(candidate, surface_task="Use target_slot design.split_policy")
    manifest = promote_rewritten_tasks(
        [case], [leaking], decision, tmp_path / "leaking"
    )
    assert manifest["admitted_tasks"] == 0
    assert "surface_leaks_private_control_metadata" in manifest["rejections"][0][
        "reasons"
    ]


def _surface_report(*, control: bool = False) -> dict:
    return {
        "case_id": "control" if control else "flawed",
        "control_degree": 0 if control else 1,
        "directions": [] if control else ["data_split"],
        "blind": {
            "label": "valid",
            "suspected_root_slots": [],
            "naturalness_1_5": 4,
            "template_artifact_1_5": 2,
            "obviousness_1_5": 2,
        },
        "fidelity": {
            "semantic_fidelity_1_5": 5,
            "target_semantics_retained": True,
            "non_target_semantics_preserved": True,
            "layer_preserved": True,
            "added_requirements": [],
            "lost_or_changed_slots": [],
        },
        "evidence": {
            "root_revision_slots": [] if control else ["design.split_policy"],
            "evidence_sufficiency_1_5": 4,
            "unique_minimal_repair_1_5": 4,
            "no_revision_needed": control,
        },
        "derived": {
            "blind_localization_f1": 1.0 if control else 0.0,
            "evidence_localization_f1": 1.0,
            "evidence_gain": 0.0 if control else 1.0,
        },
    }


def test_surface_diagnostic_separates_flaws_from_controls() -> None:
    assert assess_surface_diagnostic(_surface_report())["diagnostic_pass"]
    assert assess_surface_diagnostic(_surface_report(control=True))[
        "diagnostic_pass"
    ]


def test_surface_diagnostic_quarantines_evidence_ambiguity() -> None:
    report = _surface_report()
    report["derived"]["evidence_localization_f1"] = 0.5
    result = assess_surface_diagnostic(report)

    assert not result["diagnostic_pass"]
    assert "identifiable_after_evidence" in result["failed_gates"]
