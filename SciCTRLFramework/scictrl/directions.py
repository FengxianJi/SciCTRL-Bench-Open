"""Executable taxonomy of controlled scientific revision directions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .models import FlawSpec


@dataclass(frozen=True)
class DirectionSpec:
    direction_id: str
    stage: str
    repair_object: str
    allowed_slot_prefixes: tuple[str, ...]
    evidence_requirement: str
    typical_propagation: tuple[str, ...]
    support: str = "implemented"


DIRECTION_SPECS: tuple[DirectionSpec, ...] = (
    DirectionSpec(
        "problem_formulation", "problem", "research question and estimand",
        ("question",), "Evidence that the posed question is unanswerable or mismatched to the available setting.",
        ("hypothesis", "design", "evaluation", "claims"), "planned",
    ),
    DirectionSpec(
        "literature_grounding", "literature", "premise or novelty grounding",
        ("literature",), "A retrieved source that contradicts a premise or invalidates a novelty claim.",
        ("hypothesis", "claims"), "planned",
    ),
    DirectionSpec(
        "hypothesis", "hypothesis", "testable hypothesis",
        ("hypothesis",), "Predeclared evidence that falsifies or conditions the prediction while competing design and analysis failures pass their checks.",
        ("analysis.results_table", "claims.main", "claims.limitations"),
    ),
    DirectionSpec(
        "data_split", "experiment_design", "sampling, grouping, and split policy",
        ("design.split_policy", "design.dataset"), "A provenance, overlap, or representativeness audit tied to the estimand.",
        ("implementation.protocol", "analysis.results_table", "claims.main"),
    ),
    DirectionSpec(
        "preprocessing", "experiment_design", "preprocessing and leakage boundary",
        ("implementation.protocol", "design.dataset"), "A fold-isolation, transformation, or provenance audit.",
        ("analysis.results_table", "claims.main"), "planned",
    ),
    DirectionSpec(
        "experimental_control", "experiment_design", "control or baseline design",
        ("design.controls",), "A nuisance probe or matched-control comparison showing confounding.",
        ("analysis.results_table", "claims.main", "claims.limitations"),
    ),
    DirectionSpec(
        "implementation_protocol", "execution", "scientific execution protocol",
        ("implementation.protocol",), "A convergence, unit, environment, or parity audit.",
        ("analysis.results_table", "claims.main", "claims.limitations"),
    ),
    DirectionSpec(
        "metric", "evaluation", "estimand-aligned primary metric",
        ("evaluation.primary_metric", "evaluation.secondary_metrics"), "A diagnostic showing the current metric rewards the wrong behavior.",
        ("analysis.results_table", "claims.main"),
    ),
    DirectionSpec(
        "statistical_aggregation", "analysis", "aggregation population and failure policy",
        ("analysis.aggregation",), "Per-unit results showing selection, survivor, or best-run bias.",
        ("analysis.results_table", "claims.main", "claims.limitations"),
    ),
    DirectionSpec(
        "statistical_test", "analysis", "inferential test and error control",
        ("analysis.statistical_test",), "Multiplicity, dependence, or uncertainty evidence invalidating the current test.",
        ("analysis.results_table", "claims.main"),
    ),
    DirectionSpec(
        "claim_scope", "interpretation", "claim strength and scope",
        ("claims.main", "claims.limitations"), "A measured-vs-claimed capability audit exposing an inference gap.",
        ("claims.limitations",),
    ),
    DirectionSpec(
        "reporting_reproducibility", "writing", "reproducibility and evidence reporting",
        ("reporting",), "An artifact audit showing a result cannot be independently reconstructed.",
        ("claims.limitations",), "planned",
    ),
)


_BY_ID = {item.direction_id: item for item in DIRECTION_SPECS}


def validate_flaw_direction(flaw: FlawSpec) -> list[str]:
    errors: list[str] = []
    spec = _BY_ID.get(flaw.direction)
    if spec is None:
        return [f"unknown direction: {flaw.direction}"]
    if not any(
        flaw.target_slot == prefix or flaw.target_slot.startswith(prefix + ".")
        for prefix in spec.allowed_slot_prefixes
    ):
        errors.append(
            f"{flaw.flaw_id}: target {flaw.target_slot!r} is outside direction "
            f"{flaw.direction!r} prefixes {spec.allowed_slot_prefixes}"
        )
    if not flaw.evidence:
        errors.append(f"{flaw.flaw_id}: direction has no observable evidence")
    if not flaw.downstream_slots:
        errors.append(f"{flaw.flaw_id}: direction has no propagation obligation")
    return errors


def direction_manifest() -> list[dict[str, Any]]:
    return [asdict(item) for item in DIRECTION_SPECS]
