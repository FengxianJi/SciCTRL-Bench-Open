# Agent adapters

Adapters accept only the current public benchmark schema. Historical rendered
workspaces remain available as experiment records, but must be regenerated
before reuse so that stale input-side repair hints cannot re-enter a run.

Adapters translate a canonical public CSFI case into an end-to-end AI
Scientist's native configuration without exposing private oracle fields.

## AutoResearchClaw

```bash
cd /path/to/SciCTRL-Bench
PYTHONPATH=SciCTRLFramework:Benchmark python -m agent_adapters.autoresearchclaw \
  --case-file Benchmark/Layer2/public.jsonl \
  --case-id "$TASK_ID" \
  --workspace runs/example
```

- Layer 1 is rendered as a seeded checkpoint and resumes at
  `RESULT_ANALYSIS`, ending at `RESEARCH_DECISION`. The checkpoint contains
  three completed conditions, five seeds, secondary metrics, matched budgets,
  and editable state code so unrelated missing-work defects are not introduced
  by the adapter.
- Layer 2 starts at `TOPIC_INIT` and runs through `RESEARCH_DECISION`; it is the
  end-to-end evidence/discovery/revision-decision track, not a hypothesis-only
  smoke test.
- Execution-backed Layer-2 cases mount their public data, code, configuration,
  and artifact directory directly at `run/stage-10/experiment`. A native
  `main.py` delegates to the supplied experiment while supporting files remain
  available for evidence-driven revision.
- The terminal bridge `agent_adapters.evaluate_executable` reads the private
  case only after the native run, reconstructs evidence from completed native
  execution records, validates the final experiment workspace, and then calls
  the common controlled-revision evaluator. Agent-authored patch and
  verification claims cannot replace these checks.
- MiniMax runs set `thinking_mode: disabled`; record provider-default reasoning
  settings for other backends explicitly in the run manifest.

The adapter writes only the public case into the agent workspace. Oracle files
remain under the evaluator-controlled benchmark directory.

After a native run, collect its output without inferring success:

```bash
PYTHONPATH=SciCTRLFramework:Benchmark python -m agent_adapters.collect_autoresearchclaw \
  --workspace runs/example
```

The resulting `trajectory_bundle.json` contains native artifacts, cited
evidence IDs, and executed actions. It deliberately leaves patch and
verification normalization pending; the private deterministic evaluator makes
the final decision.

## Resumable Layer 1 + Layer 2 batch

The batch runner creates one isolated native workspace per case, writes logs and
status atomically, collects a trajectory after both successful and failed runs,
and skips completed cases when restarted:

```bash
cd /path/to/SciCTRL-Bench
export MINIMAX_API_KEY='...'
PYTHONPATH=SciCTRLFramework:Benchmark Method/AutoResearchClaw/.venv/bin/python \
  -m agent_adapters.run_autoresearchclaw_batch
```

Use `--case-limit 1` for an online smoke test or `--dry-run` to render and
validate all 68 native workspaces without making API calls. Outputs default to
`benchmark_runs/`.

For Qwen through a backend API key that permits server-side agent use, set
`DASHSCOPE_API_KEY` and run `--backend qwen --qwen-model qwen3.7-plus` (change
`--qwen-base-url` for the key's region). Do not use a Token Plan credential for
AutoResearchClaw backend calls: Model Studio documents that plan for
interactive coding tools, not backend services. The adapter deliberately reads
only the environment variable and never reads `qwenkey.txt`.
