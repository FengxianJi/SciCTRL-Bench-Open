# Pipeline output contract

## Definition

The final public output of the Pipeline is a **modified scientific task** for
an end-to-end AI Scientist. The task contains a naturally embedded scientific
problem, but it does not tell the Agent what causal component is responsible,
how many scientific obligations fail, or which repairs are accepted.

The validated scientific opportunity exists only in the paired evaluator-side
contract:

```text
private valid task
  → choose a restorable scientific obligation
  → construct a state that genuinely violates that obligation
  → rewrite the complete task
  → audit hiddenness, fidelity and evidence identifiability
  → public modified task
       ↕ opaque task_id
    private revision-opportunity contract
```

## Public task

An admitted public task contains only:

- opaque task ID;
- layer and domain;
- natural `research_brief`;
- complete Agent-facing `task`.

The task is the modified task, not the original gold task and not an
instruction such as “please modify the metric.” The causal problem must be
discovered through scientific actions and observations.

## Private contract

The evaluator-only side contains:

- a hash-bound Control Certificate containing private causal targets and
  analysis tags (not allowed-edit slots);
- accepted repair-operation equivalence classes;
- invariants shared by all accepted repairs;
- task-identity commitments that prevent replacing the research problem;
- downstream propagation obligations;
- required evidence and validator-owned verification checks;
- forbidden shortcuts and correct stopping behavior.

Thus “controlled” means the scientific weakness, obtainable evidence and
acceptance obligations are validated. It does not mean the Agent is ordered or
technically restricted to edit one slot. Additional non-identity edits are
reported through revision focus; replacing the original task fails.

## Output stages

`Benchmark/PipelineOutput/` separates two stages:

1. `canonical_modified_tasks.jsonl`: semantic modification has been applied,
   but the surface is still awaiting whole-task rewrite/admission;
2. `admitted_modified_tasks.jsonl`: current-schema M3 rewrite, source-hash
   check and surface diagnostics all passed; these are the final tasks that may
   be sent to an end-to-end Agent.

The corresponding private files never enter the Agent workspace. At present
there are 68 canonical modified tasks and zero admitted schema-0.5 tasks,
because the available M3 reports were generated from older schemas and correctly
fail freshness checks.
