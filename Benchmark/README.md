# SciCTRL-Bench

**SciCTRL-Bench: A Controlled Feedback–Revision Benchmark for End-to-End AI
Scientists** evaluates whether an agent can acquire evidence, discover a
scientific validity problem, perform a scientifically valid repair, propagate
the repair, verify it, and stop.

It contains three executable parts:

1. **VSRO task construction** creates matched gold/flawed task pairs with genuine
   validated revision opportunities and private acceptance contracts.
2. **Agent adapters** map the same canonical case into each end-to-end AI
   Scientist's native input and collect its trajectory.
3. **Reasoning Valuator** normalizes trajectories and applies deterministic
   evidence, causal repair, task identity, propagation, integrity, verification, and stopping
   gates.

## Data layout

```text
Benchmark/
├── Layer1/       # state-conditioned local revision
├── Layer2/       # idea-conditioned end-to-end revision
├── PipelineOutput/ # public modified tasks paired with private contracts
├── ExecutablePacks/ # runnable data/code fixtures and pilot admission reports
├── agent_adapters/
└── RevisionOpportunityFramework/ # private construction, controls and audit records
```

`public.jsonl` is safe to expose to evaluated agents. `oracle.jsonl` and the
RevisionOpportunityFramework internal files are evaluator-only and must never be placed in the agent
workspace.

The public input controls only neutral task conditions such as the research
question, data, tools and budget. It never exposes a target slot, repair
direction, selected downstream-artifact list, or accepted repair. The private
matched-control certificate proves the one-factor intervention and evaluates
the open Agent trajectory.

The current twenty-round control refinement and its remaining validity boundary
are recorded in `RevisionOpportunityFramework/CONTROL_ITERATIONS_20.md`.

The current MVP contains 68 structurally executable cases and two
execution-backed ML instances: one repair case and one clean control. The
proposed full release scales
the same schema to approximately 400 cases after human and execution-based
admission audits.

`PipelineOutput/canonical_modified_tasks.jsonl` contains 68 semantically
modified candidates. `PipelineOutput/admitted_modified_tasks.jsonl` is the
actual final Agent-facing output and currently contains zero tasks: schema-0.1
M3 rewrites are stale after the schema-0.7 revisions and cannot be promoted.
