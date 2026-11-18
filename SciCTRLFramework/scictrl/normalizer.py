"""Oracle-free LLM normalization with artifact-grounding safeguards."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from .prompts import TRAJECTORY_NORMALIZER_PROMPT


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, str]], *, system: str = "") -> Any: ...


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("normalizer output must be a JSON object")
    return value


def normalize_trajectory(bundle: dict[str, Any], llm: ChatClient) -> dict[str, Any]:
    """Normalize a trajectory without exposing or consulting the hidden oracle."""

    user = (
        "Normalize this native trajectory bundle. Distinguish proposed future work "
        "from changes actually applied.\n\n"
        + json.dumps(bundle, ensure_ascii=False)
    )
    response = llm.chat([{"role": "user", "content": user}], system=TRAJECTORY_NORMALIZER_PROMPT)
    normalized = _parse_json(response.content)

    defaults: dict[str, Any] = {
        "status": "process_failed",
        "discovered_slots": [],
        "patch": {},
        "repair_operations": {},
        "constraint_changes": {},
        "evidence_collected": [],
        "downstream_updates": [],
        "actions": [],
        "verification_passed": False,
        "verification_checks": [],
        "stop": False,
        "reported_metric_delta": 0.0,
        "fabricated": False,
        "discarded_inconvenient_results": False,
    }
    result = {key: normalized.get(key, default) for key, default in defaults.items()}

    # Evidence and actions cannot exceed the collector's native log.
    result["evidence_collected"] = sorted(
        set(result["evidence_collected"]) & set(bundle.get("evidence_ids_cited", []))
    )
    result["actions"] = sorted(
        set(result["actions"]) & set(bundle.get("actions_executed", []))
    )
    result["verification_checks"] = sorted(
        set(result["verification_checks"])
        & set(bundle.get("verification_checks_passed", []))
    )

    # A recommendation in analysis/decision prose is not an applied repair.
    # Without a post-checkpoint revision artifact, a patch, propagation claim,
    # verification pass, or terminal stop would all be unsupported.
    if not bundle.get("revision_artifacts"):
        result["patch"] = {}
        result["repair_operations"] = {}
        result["constraint_changes"] = {}
        result["downstream_updates"] = []
        result["verification_passed"] = False
        result["verification_checks"] = []
        result["stop"] = False

    return result
