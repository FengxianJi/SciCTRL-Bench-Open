# Architecture: validated revision-opportunity control

## Why direction labels are insufficient

Saying that a case is about “the hypothesis” or “the experiment” only controls
a broad category. It leaves many scientifically inequivalent edits and invites
the objection that an evaluator merely rewards one arbitrary author-preferred
solution.

VSRO therefore controls a **scientific obligation and its equivalence class of
valid repairs**, while leaving the Agent's action space open. A canonical case
is structurally admitted only when
all five objects exist:

```text
Gold state
  -> minimal semantic mutation
  -> observable evidence signature
  -> repair contract
  -> hidden validation suite
```

The hash-bound `Matched Control Certificate` defines a genuine causal repair
opportunity. It first proves that the gold and flawed states hold every
non-intervened semantic position and every public running condition constant.
It then records:

- `causal_targets`: private components whose scientific invariants are violated;
- `task_identity_invariants`: constitutive premises that prevent task evasion;
- `propagation_obligations`: downstream artifacts made stale by the repair;
- `evidence_obligations`: observations the agent must obtain or cite;
- `forbidden_shortcuts`: actions that can improve a score without restoring
  scientific validity.

The Agent action trajectory is deliberately not a controlled variable. The
Agent may inspect and revise any non-protected research artifact. Control is
implemented by the matched state pair and private acceptance semantics, not by
an input-side allowed-edit list.

The evaluator accepts semantically equivalent repairs after normalization. It
does not require exact wording. Schema 0.8 provides at least two structured
repair operations per target; each operation must satisfy shared hidden
invariants. Canonical patch values remain construction regressions, while an
LLM normalizer may map applied artifacts to an operation witness but cannot
override hidden tests or validator-owned verification records.

For the execution-backed pilot, normalization is subordinate to workspace
state. Agent-authored claims that it collected evidence, applied a patch,
updated downstream artifacts, or passed verification are ignored. Those fields
are reconstructed from the command event chain and private validators over the
actual files. The evaluator also reruns the submitted experiment in a clean
copy and requires byte-identical derived artifacts.

## Layer views

```text
                         shared hidden repair contract
                                      |
          +---------------------------+---------------------------+
          |                                                       |
Layer 1: failed checkpoint                              Layer 2: flawed idea
artifacts + observations                                normal E2E task input
          |                                                       |
inspect/acquire evidence                                 design/run experiments
          +------------------> diagnose --------------------------+
                                      |
                         evidence-justified scientific repair
                                      |
                         propagate -> verify -> stop
```

## Control degree

`control_degree = k` means exactly `k` independent repair atoms are active.
Degree zero is the no-repair control and explicitly measures unnecessary
editing and stopping behavior. Degrees one to three test increasing repair-set
compositionality without changing the capability definition.

## Admission tests

A synthesized canonical case passes structural admission only if:

1. mutation diff equals the hidden target-slot set;
2. applying the oracle patch exactly reconstructs the gold state;
3. no target labels or accepted values leak into public task metadata;
4. evidence signatures exist for every active repair atom;
5. target slots are non-overlapping;
6. gold/flawed surface similarity remains above the configured threshold;
7. a deterministic evaluator rejects shortcuts and task-identity replacement;
8. public Layer-2 inputs expose only generic budgeted research operations, not
   an enumerable task-specific diagnostic menu.

This replaces the impossible requirement to enumerate a modification
direction's theoretical maximum potential with a finite, testable controlled
repair opportunity.

Public release additionally requires whole-task rewrite fidelity, blind
hiddenness, evidence-aware identifiability, a held-out model-family audit and
human domain review. Structural admission alone does not imply publication.
