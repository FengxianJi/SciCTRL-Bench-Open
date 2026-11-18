"""Diagnostic gates for LLM-rewritten CSFI task surfaces.

These gates decide whether a rewrite candidate is ready for independent review.
They never replace deterministic construction checks or final human admission.
"""

from __future__ import annotations

from typing import Any


DEFAULT_THRESHOLDS = {
    "naturalness_min": 4.0,
    "template_artifact_max": 2.0,
    "semantic_fidelity_min": 4.0,
    "obviousness_max": 3.0,
    "blind_localization_f1_max": 0.5,
    "evidence_localization_f1_min": 0.8,
    "evidence_gain_min": 0.3,
    "evidence_sufficiency_min": 4.0,
    "unique_minimal_repair_min": 4.0,
}


def assess_surface_diagnostic(
    report: dict[str, Any],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Assess one blind/fidelity/evidence report produced by an LLM judge.

    ``diagnostic_pass`` means the surface can advance to independent-model and
    human review. It deliberately does not mean final benchmark admission.
    """

    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    blind = report["blind"]
    fidelity = report["fidelity"]
    evidence = report["evidence"]
    derived = report["derived"]
    is_control = report["control_degree"] == 0

    gates = {
        "natural_surface": (
            float(blind["naturalness_1_5"]) >= limits["naturalness_min"]
            and float(blind["template_artifact_1_5"])
            <= limits["template_artifact_max"]
        ),
        "rewrite_fidelity": (
            float(fidelity["semantic_fidelity_1_5"])
            >= limits["semantic_fidelity_min"]
            and bool(fidelity["target_semantics_retained"])
            and bool(fidelity["non_target_semantics_preserved"])
            and bool(fidelity["layer_preserved"])
            and not fidelity.get("added_requirements", [])
            and not fidelity.get("lost_or_changed_slots", [])
        ),
    }

    if is_control:
        gates.update(
            {
                "blind_control_validity": (
                    blind["label"] == "valid"
                    and not blind.get("suspected_root_slots", [])
                ),
                "evidence_control_validity": (
                    bool(evidence["no_revision_needed"])
                    and not evidence.get("root_revision_slots", [])
                ),
            }
        )
    else:
        gates.update(
            {
                "hidden_before_evidence": (
                    float(blind["obviousness_1_5"])
                    <= limits["obviousness_max"]
                    and float(derived["blind_localization_f1"])
                    <= limits["blind_localization_f1_max"]
                ),
                "identifiable_after_evidence": (
                    not bool(evidence["no_revision_needed"])
                    and float(derived["evidence_localization_f1"])
                    >= limits["evidence_localization_f1_min"]
                    and float(derived["evidence_gain"])
                    >= limits["evidence_gain_min"]
                    and float(evidence["evidence_sufficiency_1_5"])
                    >= limits["evidence_sufficiency_min"]
                    and float(evidence["unique_minimal_repair_1_5"])
                    >= limits["unique_minimal_repair_min"]
                ),
            }
        )

    failed = [name for name, passed in gates.items() if not passed]
    return {
        "case_id": report["case_id"],
        "control_degree": report["control_degree"],
        "directions": report.get("directions", []),
        "gates": gates,
        "diagnostic_pass": not failed,
        "next_status": "independent_review" if not failed else "revise",
        "failed_gates": failed,
        "thresholds": limits,
        "scope_note": (
            "This is an M3 surface diagnostic, not final benchmark admission."
        ),
    }
