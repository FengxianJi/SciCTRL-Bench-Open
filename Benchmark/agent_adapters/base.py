"""Common result contract for end-to-end AI Scientist adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class AdapterBundle:
    case_id: str
    agent_id: str
    workspace: Path
    config_path: Path
    command: tuple[str, ...]
    start_stage: str
    end_stage: str


class AgentAdapter(Protocol):
    agent_id: str

    def render(self, case: dict, workspace: Path) -> AdapterBundle:
        """Render a canonical public case into a native agent workspace."""
