"""Map Layer 1 and Layer 2 CSFI cases into AutoResearchClaw inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import yaml

from .base import AdapterBundle


DEFAULT_AGENT_ROOT = Path(__file__).resolve().parents[2] / "Method" / "AutoResearchClaw"
DEFAULT_BENCHMARK_ROOT = Path(__file__).resolve().parents[2] / "Benchmark"
SUPPORTED_SCHEMA_VERSION = "0.8.0"


def _require_current_public_schema(case: dict[str, Any]) -> None:
    version = case.get("schema_version")
    if version != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            "Refusing stale SciCTRL public input: expected schema "
            f"{SUPPORTED_SCHEMA_VERSION}, received {version!r}. Regenerate the "
            "benchmark and render a new adapter workspace."
        )


def _load_case(path: Path, case_id: str) -> dict[str, Any]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        if case["case_id"] == case_id:
            forbidden = {"oracle", "gold_state", "flawed_state", "evidence_oracle"}
            leaked = forbidden & set(case)
            if leaked:
                raise ValueError(f"Refusing to adapt private fields: {sorted(leaked)}")
            _require_current_public_schema(case)
            return case
    raise KeyError(f"Case {case_id!r} not found in {path}")


def _idea_markdown(idea: dict[str, Any]) -> str:
    lines = [
        f"# {idea['title']}",
        f"\nResearch question: {idea['question']}",
        f"\nHypothesis: {idea['hypothesis']}",
    ]
    for section in ("design", "evaluation", "analysis", "implementation", "claims"):
        lines.append(f"\n## {section.title()}")
        for key, value in idea[section].items():
            lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def _base_config(case: dict[str, Any], topic: str) -> dict[str, Any]:
    return {
        "project": {
            "name": f"scictrl-{case['case_id']}",
            "mode": "full-auto",
        },
        "research": {
            "topic": topic,
            "domains": [case["domain"]],
            "daily_paper_count": 3,
            "quality_threshold": 3.0,
        },
        "runtime": {
            "timezone": "Etc/UTC",
            "max_parallel_tasks": 1,
            "approval_timeout_hours": 1,
            "retry_limit": 1,
        },
        "notifications": {
            "channel": "console",
            "target": "",
            "on_stage_start": True,
            "on_stage_fail": True,
            "on_gate_required": True,
        },
        "knowledge_base": {"backend": "markdown", "root": "docs/kb"},
        "llm": {
            "provider": "minimax",
            "base_url": "https://api.minimaxi.com/v1",
            "wire_api": "chat_completions",
            "api_key_env": "MINIMAX_API_KEY",
            "api_key": "",
            "primary_model": "MiniMax-M3",
            "fallback_models": [],
            "thinking_mode": "disabled",
        },
        "literature_search": {
            "sources": ["arxiv"],
            "max_results_per_query": 8,
            "inter_query_delay_sec": 3.0,
            "openalex_email": "researchclaw@users.noreply.github.com",
        },
        "experiment": {
            "mode": "sandbox",
            "time_budget_sec": 120,
            "max_iterations": 2,
            "metric_key": "primary_metric",
            "metric_direction": "maximize",
            "sandbox": {
                "python_path": str(DEFAULT_AGENT_ROOT / ".venv/bin/python"),
                "gpu_required": False,
                "max_memory_mb": 2048,
            },
            "opencode": {"enabled": False, "auto": False},
            "benchmark_agent": {"enabled": False},
            "figure_agent": {"enabled": False},
        },
        "web_search": {"enabled": False},
        "security": {
            "hitl_required_stages": [5, 9, 20],
            "allow_publish_without_approval": False,
            "redact_sensitive_logs": True,
        },
        "hitl": {"enabled": False, "mode": "full-auto", "cost_budget_usd": 0.0},
        "export": {"target_conference": "generic", "authors": "Anonymous", "bib_file": "references"},
        "prompts": {"extra_prompts": {}},
    }


def _state_guidance(case: dict[str, Any]) -> str:
    state = case["agent_input"]["state"]
    observations = "\n".join(
        f"- [{item['evidence_id']}] {item['observation']}"
        for item in state["observations"]
    )
    return (
        f"This is a continuation from a fixed research checkpoint.\n\n"
        f"{case['agent_input']['task']}\n\n"
        f"Recorded observations (not diagnoses):\n{observations}\n\n"
        "The checkpoint completion manifest is authoritative: every declared "
        "condition, seed, secondary metric, statistical report, and compute-parity "
        "obligation not contradicted by a recorded observation was completed. "
        "A compressed metric view is not evidence that an artifact is missing. "
        "Do not invent undeclared requirements.\n\n"
        "Determine whether the current study is valid from this evidence. State which "
        "research components, if any, require revision; preserve unaffected premises; "
        "identify downstream artifacts that must be regenerated; and give explicit "
        "post-revision verification and stopping criteria."
    )


def _idea_guidance(case: dict[str, Any]) -> str:
    interface = case["agent_input"]["research_interface"]
    operations = ", ".join(interface["operations"])
    budget = interface["budget"]["max_requests"]
    runtime_pack = case["agent_input"].get("environment_entrypoint", {}).get(
        "runtime_pack"
    )
    runtime_note = ""
    if runtime_pack:
        runtime_note = (
            "\n\nA fixed executable study workspace has been mounted in "
            "run/stage-10/experiment. Its data and task_identity.json are "
            "authoritative. Use the existing runnable files, obtain evidence "
            "from actual executions, preserve fixed assets, and revise the "
            "applied code/configuration plus every affected artifact."
        )
    return (
        f"Canonical benchmark task:\n{case['agent_input']['task']}\n\n"
        "The idea is an ordinary research proposal, not an instruction about where to "
        "edit. Use experiments and audits to determine whether revision is necessary. "
        "Do not change fixed constraints or improve a metric by narrowing evaluation.\n\n"
        f"The benchmark research interface permits at most {budget} open-ended requests "
        f"using generic operations ({operations}). Formulate each request around an "
        "artifact and a scientific question; no task-specific diagnostic menu or "
        "evaluator score is available."
        + runtime_note
    )


def _seed_executable_pack(case: dict[str, Any], run_dir: Path) -> Path | None:
    runtime_pack = case["agent_input"].get("environment_entrypoint", {}).get(
        "runtime_pack"
    )
    if not runtime_pack:
        return None
    template = DEFAULT_BENCHMARK_ROOT / "ExecutablePacks" / runtime_pack / "workspace_template"
    if not template.is_dir():
        raise FileNotFoundError(
            f"Executable CSFI pack has not been generated: {template}"
        )
    destination = run_dir / "stage-10" / "experiment"
    destination.mkdir(parents=True, exist_ok=True)
    for source in template.iterdir():
        target = destination / source.name
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)
    # AutoResearchClaw's native sandbox expects main.py. Keep experiment.py as
    # the editable implementation and use a minimal native entry point.
    (destination / "main.py").write_text(
        "from experiment import main\n\nif __name__ == '__main__':\n    main()\n",
        encoding="utf-8",
    )
    return destination


def _secondary_metric_values(domain: str, primary: float) -> dict[str, float]:
    if domain == "ml_classification":
        return {
            "rare_class_recall": round(primary - 0.035, 3),
            "majority_class_recall": round(primary + 0.041, 3),
            "expected_calibration_error": round(0.082 - primary / 10, 4),
            "compute_hours": 1.8,
        }
    if domain == "computational_biology":
        return {
            "auprc": round(primary - 0.061, 3),
            "calibration_slope": round(0.98 + (primary - 0.7) / 10, 3),
            "subgroup_sensitivity": round(primary - 0.028, 3),
            "compute_hours": 6.4,
        }
    if domain == "materials_simulation":
        return {
            "path_stability": round(primary - 0.018, 3),
            "force_convergence": 0.012,
            "compute_cost": 412.0,
        }
    return {
        "constraint_violation": 0.0,
        "wall_clock_seconds": 92.0,
        "robustness_score": round(primary - 0.022, 3),
    }


def _checkpoint_metrics(domain: str) -> dict[str, float]:
    """Return a complete, non-degenerate result matrix for a local case.

    Layer 1 is meant to isolate a validated scientific revision opportunity. A sparse
    placeholder run makes a native agent quite reasonably diagnose missing
    baselines, seeds, and secondary metrics as additional flaws.  The adapter
    therefore supplies a closed checkpoint: three declared conditions, five
    seeds, and all generic reporting obligations.  The case observations are
    the only intentionally failing validity checks.
    """

    values = {
        "baseline": (0.701, 0.704, 0.699, 0.706, 0.702),
        "proposed": (0.731, 0.734, 0.728, 0.736, 0.732),
        "ablation": (0.716, 0.719, 0.713, 0.721, 0.717),
    }
    metrics: dict[str, float] = {}
    for condition, seeds in values.items():
        for seed, primary in enumerate(seeds):
            prefix = f"{condition}/main/{seed}"
            metrics[f"{prefix}/primary_metric"] = primary
            for name, value in _secondary_metric_values(domain, primary).items():
                metrics[f"{prefix}/{name}"] = value
    return metrics


def _checkpoint_code(
    idea: dict[str, Any], observations: list[dict[str, Any]], domain: str
) -> str:
    """Create an editable representation of the checkpoint research state.

    This is deliberately ordinary Python rather than an oracle patch.  It
    contains exactly the public idea and observations, allowing an adapted
    agent to revise the scientific state without exposing the hidden gold.
    """

    payload = json.dumps(
        {"idea": idea, "observations": observations},
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    metrics = json.dumps(_checkpoint_metrics(domain), indent=2, sort_keys=True)
    return (
        '"""Editable CSFI checkpoint supplied to the native agent."""\n\n'
        "import json\n\n"
        f"RESEARCH_STATE = {payload}\n\n"
        f"METRICS = {metrics}\n\n"
        "def main() -> None:\n"
        "    for key, value in METRICS.items():\n"
        "        condition = key.split('/', 1)[0]\n"
        "        if key.endswith('/primary_metric'):\n"
        "            print(f'condition={condition} primary_metric={value}')\n"
        "    print(json.dumps(RESEARCH_STATE, ensure_ascii=False))\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )


def _seed_layer1_run(case: dict[str, Any], run_dir: Path) -> None:
    state = case["agent_input"]["state"]
    idea = state["idea"]
    (run_dir / "stage-08").mkdir(parents=True, exist_ok=True)
    (run_dir / "stage-09").mkdir(parents=True, exist_ok=True)
    experiment_dir = run_dir / "stage-10" / "experiment"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = run_dir / "stage-12" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "stage-08" / "hypotheses.md").write_text(
        f"# Checkpoint hypothesis\n\n{idea['hypothesis']}\n",
        encoding="utf-8",
    )
    (run_dir / "stage-09" / "exp_plan.yaml").write_text(
        yaml.safe_dump(
            {
                "conditions": [
                    {"name": "baseline", "role": "matched baseline"},
                    {"name": "proposed", "role": "proposed method"},
                    {"name": "ablation", "role": "predeclared ablation"},
                ],
                "seeds": [0, 1, 2, 3, 4],
                "design": idea["design"],
                "evaluation": idea["evaluation"],
                "analysis": idea["analysis"],
                "implementation": idea["implementation"],
                "fixed_constraints": case["agent_input"]["fixed_constraints"],
                "completion_manifest": {
                    "all_conditions_completed": True,
                    "all_seeds_completed": True,
                    "secondary_metrics_reported": True,
                    "compute_parity_verified": True,
                    "non_observation_checks_passed": True,
                },
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    observation_text = "\n".join(
        f"[{item['evidence_id']}] {item['observation']}"
        for item in state["observations"]
    )
    metrics = _checkpoint_metrics(case["domain"])
    run_record = {
        "run_id": "csfi-checkpoint-run",
        "task_id": "controlled-checkpoint",
        "status": "completed",
        "metrics": metrics,
        "primary_metric": metrics["proposed/main/3/primary_metric"],
        "stdout": (
            "CHECKPOINT_COMPLETE: 3/3 conditions, 5/5 seeds each, all declared "
            "secondary metrics and matched budgets present.\n"
            + observation_text
        ),
        "stderr": "",
        "timed_out": False,
        "structured_results": {
            "idea": idea,
            "observations": state["observations"],
            "completion_manifest": {
                "conditions": 3,
                "seeds_per_condition": 5,
                "all_declared_metrics_present": True,
                "all_non_observation_checks_passed": True,
            },
            "statistical_report": {
                "seed_level_values_available": True,
                "paired_test_completed": True,
                "paired_test_p_value": 0.0079,
                "multiple_comparison_policy": "Only the predeclared primary comparison is confirmatory",
                "confidence_intervals_reported": True,
                "effect_size_reported": True,
            },
            "reporting_note": (
                "All domain-specific secondary metrics named in the plan are present "
                "in the per-seed metric keys; no declared output is omitted."
            ),
        },
    }
    (runs_dir / "run-001.json").write_text(
        json.dumps(run_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (experiment_dir / "main.py").write_text(
        _checkpoint_code(idea, state["observations"], case["domain"]),
        encoding="utf-8",
    )
    (run_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "last_completed_stage": 13,
                "last_completed_name": "ITERATIVE_REFINE",
                "run_id": f"seed-{case['case_id']}",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


class AutoResearchClawAdapter:
    agent_id = "autoresearchclaw"

    def __init__(
        self,
        agent_root: Path = DEFAULT_AGENT_ROOT,
        *,
        backend: str = "minimax",
        codex_model: str = "gpt-5.6-sol",
        qwen_model: str = "qwen3.7-plus",
        qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ):
        if backend not in {"minimax", "codex", "qwen"}:
            raise ValueError(f"Unsupported AutoResearchClaw backend: {backend}")
        self.agent_root = agent_root.resolve()
        self.backend = backend
        self.codex_model = codex_model
        self.qwen_model = qwen_model
        self.qwen_base_url = qwen_base_url.rstrip("/")

    @property
    def model(self) -> str:
        if self.backend == "codex":
            return self.codex_model
        if self.backend == "qwen":
            return self.qwen_model
        return "MiniMax-M3"

    def render(self, case: dict[str, Any], workspace: Path) -> AdapterBundle:
        _require_current_public_schema(case)
        workspace = workspace.resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        layer = case["layer"]
        executable_workspace = None
        if layer == "layer1_state":
            idea = case["agent_input"]["state"]["idea"]
            start_stage = "RESULT_ANALYSIS"
            end_stage = "RESEARCH_DECISION"
            run_dir = workspace / "run"
            _seed_layer1_run(case, run_dir)
            guidance = _state_guidance(case)
            guidance_stages = ("result_analysis", "research_decision")
        elif layer == "layer2_idea":
            idea = case["agent_input"]["idea"]
            start_stage = "TOPIC_INIT"
            end_stage = "RESEARCH_DECISION"
            run_dir = workspace / "run"
            executable_workspace = _seed_executable_pack(case, run_dir)
            guidance = _idea_guidance(case)
            guidance_stages = (
                "topic_init",
                "problem_decompose",
                "experiment_design",
                "result_analysis",
            )
        else:
            raise ValueError(f"Unsupported layer: {layer}")

        topic = f"{idea['title']}: {idea['question']} Hypothesis: {idea['hypothesis']}"
        config = _base_config(case, topic)
        if self.backend == "codex":
            config["llm"] = {
                "provider": "acp",
                "primary_model": self.codex_model,
                "fallback_models": [],
                "thinking_mode": "",
                "acp": {
                    "agent": "codex",
                    "model": self.codex_model,
                    "cwd": str(workspace),
                    "session_name": f"scictrl-{case['case_id']}",
                    "timeout_sec": 1800,
                    "max_turns": 1,
                },
            }
        elif self.backend == "qwen":
            config["llm"] = {
                "provider": "openai-compatible",
                "base_url": self.qwen_base_url,
                "wire_api": "chat_completions",
                "api_key_env": "DASHSCOPE_API_KEY",
                "api_key": "",
                "primary_model": self.qwen_model,
                "fallback_models": [],
                "thinking_mode": "",
            }
        if executable_workspace is not None:
            interface_budget = case["agent_input"]["research_interface"]["budget"]
            config["experiment"]["time_budget_sec"] = interface_budget[
                "timeout_seconds"
            ]
            config["experiment"]["max_iterations"] = interface_budget[
                "max_requests"
            ]
        guidance_path = workspace / "stage_guidance.md"
        guidance_path.write_text(guidance + "\n", encoding="utf-8")
        config["prompts"]["extra_prompts"] = {
            stage: str(guidance_path) for stage in guidance_stages
        }
        config_path = workspace / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        (workspace / "task.md").write_text(
            _idea_markdown(idea) + "\n## Task\n\n" + case["agent_input"]["task"] + "\n",
            encoding="utf-8",
        )
        (workspace / "public_case.json").write_text(
            json.dumps(case, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        command = [
            str(self.agent_root / ".venv/bin/researchclaw"),
            "run",
            "--config",
            str(config_path),
            "--output",
            str(run_dir),
            "--from-stage",
            start_stage,
            "--to-stage",
            end_stage,
            "--auto-approve",
        ]
        if layer == "layer1_state":
            command.append("--resume")

        bundle = AdapterBundle(
            case_id=case["case_id"],
            agent_id=self.agent_id,
            workspace=workspace,
            config_path=config_path,
            command=tuple(command),
            start_stage=start_stage,
            end_stage=end_stage,
        )
        (workspace / "adapter_manifest.json").write_text(
            json.dumps(
                {
                    "case_id": bundle.case_id,
                    "agent_id": bundle.agent_id,
                    "config_path": str(bundle.config_path),
                    "command": list(bundle.command),
                    "start_stage": bundle.start_stage,
                    "end_stage": bundle.end_stage,
                    "backend": self.backend,
                    "model": self.model,
                    "thinking_mode": "codex-configured" if self.backend == "codex" else config["llm"].get("thinking_mode", ""),
                    "private_fields_present": False,
                    "executable_workspace": (
                        str(executable_workspace)
                        if executable_workspace
                        else None
                    ),
                    "terminal_evaluator": (
                        "agent_adapters.evaluate_executable"
                        if executable_workspace
                        else None
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return bundle


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-file", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--agent-root", type=Path, default=DEFAULT_AGENT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    case = _load_case(args.case_file, args.case_id)
    bundle = AutoResearchClawAdapter(args.agent_root).render(case, args.workspace)
    print(json.dumps({"workspace": str(bundle.workspace), "command": list(bundle.command)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
