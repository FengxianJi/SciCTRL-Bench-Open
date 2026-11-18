"""Controlled whole-task rewriting for surface-realistic CSFI cases."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol


class ChatClient(Protocol):
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        system: str = "",
        max_tokens: int | None = None,
    ) -> Any: ...


REWRITE_SYSTEM = """You realize benchmark tasks as natural scientific briefs.
Rewrite the ENTIRE task, not one visibly edited sentence. Preserve every
scientific choice, constraint, method, evaluation decision, and claim in the
input—even if a choice is subtly questionable. Do not repair, criticize,
highlight, or hint at any issue. Do not add a requirement absent from the
source. Avoid field names such as target_slot, flaw, injected, gold, oracle, or
repair direction. The result should read like a coherent task written by a
research lead. Return JSON only with two fields:
1. research_brief: the scientific idea/design/claim in natural prose, excluding
   benchmark mechanics, tool lists, required-output schema, checkpoint
   observations, and instructions to diagnose or revise;
2. surface_task: the complete Agent-facing input, including the research brief,
   layer-appropriate checkpoint or action context, fixed resources, available
   tools/actions, and output obligations. Integrate these naturally rather than
   reproducing JSON headings or schema language. Use concrete dataset/resource
   identifiers from the source where available. Avoid repeating phrases such as
   fixed, untouched, predeclared, and claimed-only-if more than needed."""


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def rewrite_public_case(public_case: dict[str, Any], llm: ChatClient) -> dict[str, str]:
    """Create a global surface rewrite while keeping canonical semantics private."""

    prompt = (
        "Rewrite this complete benchmark input as one self-contained scientific "
        "brief. Keep the same starting state and required horizon for its layer.\n\n"
        + json.dumps(public_case, ensure_ascii=False, sort_keys=True)
    )
    response = llm.chat(
        [{"role": "user", "content": prompt}],
        system=REWRITE_SYSTEM,
        max_tokens=2400,
    )
    result = parse_json_object(response.content)
    research_brief = result.get("research_brief")
    surface = result.get("surface_task")
    if not isinstance(research_brief, str) or len(research_brief.split()) < 60:
        raise ValueError("rewrite did not return a sufficiently complete research_brief")
    if not isinstance(surface, str) or len(surface.split()) < 100:
        raise ValueError("rewrite did not return a sufficiently complete surface_task")
    return {"research_brief": research_brief.strip(), "surface_task": surface.strip()}


def semantic_slots(case: dict[str, Any]) -> list[str]:
    idea = (
        case["agent_input"]["state"]["idea"]
        if case["layer"] == "layer1_state"
        else case["agent_input"]["idea"]
    )
    slots: list[str] = ["question", "hypothesis"]
    for section in ("design", "evaluation", "analysis", "implementation", "claims"):
        slots.extend(f"{section}.{key}" for key in idea[section])
    return slots
