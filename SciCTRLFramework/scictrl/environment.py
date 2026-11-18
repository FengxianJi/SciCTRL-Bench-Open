"""Budgeted, open-ended research interface for evidence acquisition.

The public interface exposes generic research operations rather than a menu of
flaw-specific buttons. Evidence recipes remain private to the evaluator.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import re
from typing import Any

from .executable_runtime import ExecutableCSFIEnvironment


GENERIC_OPERATIONS = (
    "inspect_artifact",
    "run_analysis",
    "execute_experiment",
    "compare_results",
)

def _tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if len(token) > 2
    }


@dataclass
class CSFIEnvironment:
    """Execute generic requests while keeping evidence recipes private.

    A production worker must host this object outside the agent-readable
    workspace. The Agent sees generic operations, tools, a fixed request budget,
    and observations resulting from its own queries. It never sees private
    probe names, the evidence-to-target map, or the acceptance contract.
    """

    internal_case: dict[str, Any]
    action_log: list[dict[str, Any]] = field(default_factory=list)
    spent: int = 0
    workspace: Any = None
    _executable: ExecutableCSFIEnvironment | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.internal_case.get("oracle", {}).get("executable_control"):
            self._executable = ExecutableCSFIEnvironment(
                self.internal_case, workspace=self.workspace
            )

    def interface(self) -> dict[str, Any]:
        if self._executable is not None:
            return self._executable.interface()
        return deepcopy(self.internal_case["agent_input"]["research_interface"])

    def _reject(self, request: Any, reason: str) -> dict[str, Any]:
        result = {
            "request": deepcopy(request),
            "status": "rejected",
            "reason": reason,
            "observations": [],
            "budget_remaining": max(
                0, self.interface()["budget"]["max_requests"] - self.spent
            ),
        }
        self.action_log.append(result)
        return deepcopy(result)

    def execute(self, request: dict[str, str]) -> dict[str, Any]:
        """Execute one open-ended research request under a fixed budget."""

        if self._executable is not None:
            return self._executable.execute(request)

        if not isinstance(request, dict):
            return self._reject(request, "request_must_be_an_object")
        operation = request.get("operation")
        tool = request.get("tool")
        query = request.get("query")
        if operation not in GENERIC_OPERATIONS:
            return self._reject(request, "operation_not_available")
        if tool not in self.internal_case["agent_input"]["available_tools"]:
            return self._reject(request, "tool_not_available")
        if not isinstance(query, str) or len(query.strip()) < 8:
            return self._reject(request, "query_not_specific")

        maximum = self.interface()["budget"]["max_requests"]
        if self.spent >= maximum:
            return self._reject(request, "request_budget_exhausted")
        self.spent += 1

        artifact_tokens = _tokens(str(request.get("artifact", "")))
        query_tokens = _tokens(query)
        matched = []
        for evidence_id, spec in self.internal_case["evidence_oracle"].items():
            recipe = spec["probe_recipe"]
            artifact_required = set(recipe["artifact_concepts"])
            query_required = set(recipe["query_concepts"])
            if (
                tool == recipe["tool"]
                and operation in recipe["operations"]
                and artifact_required.issubset(artifact_tokens | query_tokens)
                and query_required.issubset(query_tokens)
            ):
                matched.append(
                    {
                        "evidence_id": evidence_id,
                        "observation": spec["observation"],
                    }
                )

        if not matched:
            matched = [
                {
                    "evidence_id": None,
                    "observation": (
                        "The requested operation completed, but this query did not "
                        "produce a task-specific diagnostic observation."
                    ),
                }
            ]
        result = {
            "request": deepcopy(request),
            "status": "completed",
            "observations": matched,
            "budget_remaining": maximum - self.spent,
        }
        self.action_log.append(result)
        return deepcopy(result)

    def public_log(self) -> list[dict[str, Any]]:
        if self._executable is not None:
            return self._executable.public_log()
        return deepcopy(self.action_log)

    def evidence_collected(self) -> list[str]:
        if self._executable is None:
            return []
        return self._executable.evidence_collected()

    def evaluate_workspace(self) -> dict[str, Any]:
        if self._executable is None:
            raise RuntimeError("this case uses the legacy deterministic environment")
        return self._executable.evaluate_workspace().to_dict()

    def close(self) -> None:
        if self._executable is not None:
            self._executable.close()
