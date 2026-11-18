"""Typed data contracts for CSFI cases and evaluations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


SCHEMA_VERSION = "0.8.0"


class Layer(str, Enum):
    """Benchmark start-state granularity."""

    STATE = "layer1_state"
    IDEA = "layer2_idea"


class Outcome(str, Enum):
    """Mutually exclusive terminal attribution labels."""

    PROCESS_FAILURE = "process_failure"
    DETECTION_FAILURE = "detection_failure"
    REPAIR_FAILURE = "repair_failure"
    INCOMPLETE_VERIFICATION = "incomplete_verification"
    SHORTCUT_SUCCESS = "shortcut_success"
    CONTROLLED_SUCCESS = "controlled_success"


@dataclass(frozen=True)
class EvidenceSpec:
    evidence_id: str
    action: str
    observation: str
    supports_slot: str


@dataclass(frozen=True)
class FlawSpec:
    flaw_id: str
    direction: str
    target_slot: str
    gold_value: Any
    flawed_value: Any
    evidence: tuple[EvidenceSpec, ...]
    downstream_slots: tuple[str, ...]
    forbidden_shortcuts: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class DomainPack:
    domain_id: str
    project_id: str
    display_name: str
    idea: dict[str, Any]
    fixed_constraints: dict[str, Any]
    environment: dict[str, Any]
    flaws: tuple[FlawSpec, ...]


@dataclass
class EvaluationResult:
    case_id: str
    outcome: Outcome
    success: bool
    gates: dict[str, bool]
    metrics: dict[str, float]
    missing_targets: list[str] = field(default_factory=list)
    extra_changes: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    missing_downstream: list[str] = field(default_factory=list)
    missing_verification: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["outcome"] = self.outcome.value
        return result


def case_id(
    domain_id: str,
    layer: Layer,
    control_degree: int,
    variant: str = "",
) -> str:
    layer_token = "l1" if layer is Layer.STATE else "l2"
    base = f"csfi-{domain_id}-{layer_token}-k{control_degree}"
    return f"{base}-{variant}" if variant else base
