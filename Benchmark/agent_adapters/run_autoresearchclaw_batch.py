"""Run CSFI Layer 1/2 cases through AutoResearchClaw with safe resume semantics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Iterable

from .autoresearchclaw import AutoResearchClawAdapter, DEFAULT_AGENT_ROOT
from .collect_autoresearchclaw import collect


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    DEFAULT_ROOT / "benchmark_runs" / "scictrl_autoresearchclaw_schema_0_8"
)
LAYER_FILES = {
    "Layer1": DEFAULT_ROOT / "Benchmark" / "Layer1" / "public.jsonl",
    "Layer2": DEFAULT_ROOT / "Benchmark" / "Layer2" / "public.jsonl",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _read_cases(path: Path) -> list[dict[str, Any]]:
    cases = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [case["case_id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate case_id in {path}")
    return cases


def _jobs(layers: Iterable[str], case_limit: int | None) -> list[tuple[str, dict[str, Any]]]:
    jobs: list[tuple[str, dict[str, Any]]] = []
    for layer in layers:
        cases = _read_cases(LAYER_FILES[layer])
        if case_limit is not None:
            cases = cases[:case_limit]
        jobs.extend((layer, case) for case in cases)
    return jobs


def _load_status(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def run_case(
    *,
    layer: str,
    case: dict[str, Any],
    output_root: Path,
    adapter: AutoResearchClawAdapter,
    timeout_seconds: int,
    resume: bool,
    dry_run: bool,
) -> dict[str, Any]:
    workspace = output_root / layer / case["case_id"]
    status_path = workspace / "batch_status.json"
    previous = _load_status(status_path)
    if resume and previous and previous.get("status") == "completed":
        return {**previous, "batch_action": "skipped_completed"}

    workspace.mkdir(parents=True, exist_ok=True)
    bundle = adapter.render(case, workspace)
    status: dict[str, Any] = {
        "case_id": case["case_id"],
        "layer": layer,
        "schema_version": case.get("schema_version"),
        "agent_id": bundle.agent_id,
        "backend": adapter.backend,
        "model": (
            adapter.model
        ),
        "thinking_mode": (
            "codex-configured" if adapter.backend == "codex" else "configured-default"
        ),
        "workspace": str(workspace),
        "status": "dry_run" if dry_run else "running",
        "started_at": _now(),
        "attempt": int((previous or {}).get("attempt", 0)) + 1,
        "timeout_seconds": timeout_seconds,
    }
    _write_json(status_path, status)
    if dry_run:
        status["finished_at"] = _now()
        status["batch_action"] = "rendered"
        _write_json(status_path, status)
        return status

    stdout_path = workspace / "native_stdout.log"
    stderr_path = workspace / "native_stderr.log"
    started = time.monotonic()
    try:
        with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open(
            "a", encoding="utf-8"
        ) as stderr:
            process = subprocess.Popen(
                bundle.command,
                cwd=adapter.agent_root,
                env=os.environ.copy(),
                text=True,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            try:
                exit_code = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
                raise
        status["exit_code"] = exit_code
        status["status"] = "completed" if exit_code == 0 else "failed"
    except subprocess.TimeoutExpired:
        status["exit_code"] = None
        status["status"] = "timed_out"
    except Exception as error:  # Keep the remaining benchmark runnable.
        status["exit_code"] = None
        status["status"] = "runner_error"
        status["error_type"] = type(error).__name__
        status["error"] = str(error)

    try:
        trajectory = collect(workspace)
        _write_json(workspace / "trajectory_bundle.json", trajectory)
        status["trajectory_collected"] = True
    except Exception as error:
        status["trajectory_collected"] = False
        status["collector_error"] = f"{type(error).__name__}: {error}"
    status["duration_seconds"] = round(time.monotonic() - started, 3)
    status["finished_at"] = _now()
    status["batch_action"] = "executed"
    _write_json(status_path, status)
    return status


def _summary(results: list[dict[str, Any]], total_jobs: int) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for result in results:
        name = str(result.get("status", "unknown"))
        counts[name] = counts.get(name, 0) + 1
    return {
        "updated_at": _now(),
        "total_jobs": total_jobs,
        "processed_this_invocation": len(results),
        "status_counts": counts,
        "cases": results,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layers", nargs="+", choices=sorted(LAYER_FILES), default=sorted(LAYER_FILES)
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--agent-root", type=Path, default=DEFAULT_AGENT_ROOT)
    parser.add_argument("--backend", choices=("minimax", "codex", "qwen"), default="minimax")
    parser.add_argument("--codex-model", default="gpt-5.6-sol")
    parser.add_argument("--qwen-model", default="qwen3.7-plus")
    parser.add_argument(
        "--qwen-base-url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="Region-specific OpenAI-compatible Model Studio endpoint for a backend API key",
    )
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument(
        "--case-limit", type=int, help="Per-layer prefix size, intended for smoke tests"
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Return zero even if a native run fails",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    if args.case_limit is not None and args.case_limit <= 0:
        raise SystemExit("--case-limit must be positive")
    if (
        args.backend == "minimax"
        and not args.dry_run
        and not os.environ.get("MINIMAX_API_KEY")
    ):
        raise SystemExit(
            "MINIMAX_API_KEY is not set; refusing to start an online benchmark."
        )
    if args.backend == "qwen" and not args.dry_run and not os.environ.get("DASHSCOPE_API_KEY"):
        raise SystemExit(
            "DASHSCOPE_API_KEY is not set; refusing to start an online backend-agent benchmark."
        )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    jobs = _jobs(args.layers, args.case_limit)
    adapter = AutoResearchClawAdapter(
        args.agent_root,
        backend=args.backend,
        codex_model=args.codex_model,
        qwen_model=args.qwen_model,
        qwen_base_url=args.qwen_base_url,
    )
    results: list[dict[str, Any]] = []
    for index, (layer, case) in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {layer} {case['case_id']}", flush=True)
        result = run_case(
            layer=layer,
            case=case,
            output_root=output_root,
            adapter=adapter,
            timeout_seconds=args.timeout_seconds,
            resume=not args.no_resume,
            dry_run=args.dry_run,
        )
        results.append(result)
        _write_json(output_root / "batch_summary.json", _summary(results, len(jobs)))
        print(f"  -> {result['status']}", flush=True)

    failures = [
        result
        for result in results
        if result.get("status") not in {"completed", "dry_run"}
    ]
    return 0 if args.allow_failures or not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
