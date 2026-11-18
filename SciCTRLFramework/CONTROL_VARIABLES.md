# SciCTRL control variables and implementation boundary

## Experimental definition

SciCTRL treats one benchmark unit as a matched pair:

```text
gold state    = (held-constant conditions C, valid factor X0)
flawed state  = (held-constant conditions C, intervened factor X1)
```

The benchmark measures the Agent trajectory `Y` under `X1`, conditional on
`C`. The Agent's edits are observations in `Y`; they are not control variables.

The terms have distinct meanings:

- **controlled variables:** task identity, source data, tools, budgets,
  interface, and output obligations held fixed;
- **scientific intervention:** the one-to-three independently registered
  validity factors changed by the constructor;
- **violated invariant:** the scientific obligation made false by an
  intervention;
- **revision site:** the semantic location at which the intervention is
  realized;
- **repair equivalence class:** multiple implementations that restore the same
  invariant;
- **action constraints:** task-identity and integrity protections that prevent
  evasion. These are supporting safeguards, not the controlled intervention.

## Public/private boundary

Agent-visible input may state neutral experimental conditions:

```text
research question, dataset, workspace, tools, budget, generic operations,
raw observations available at the selected layer, and output obligations
```

It must not state or structurally reveal:

```text
whether revision is required, target slots, allowed slots, repair direction,
selected downstream artifacts, accepted values, private probes, or tests
```

An edit whitelist is not causal control. It leaks localization and turns the
task into constrained execution. SciCTRL instead keeps the action space open
inside protected task identity and applies private semantic acceptance tests.

## Matched Control Certificate 2.0

Every internal case contains an evaluator-only certificate with four distinct
parts:

1. `matched_pair_control` lists every held-constant semantic position, binds
   their values by hash, identifies only the registered intervention positions,
   and freezes public running conditions.
2. `scientific_interventions` links each violated obligation to evidence,
   causal root, propagation obligations, and a repair contract.
3. `semantic_position_roles` separates causal roots, downstream artifacts,
   protected task identity, and otherwise free positions.
4. `evidence_control`, `task_identity_control`, and `evaluation_control` bind
   private probes, non-evasion commitments, hidden tests, shortcuts,
   verification, and stopping.

The certificate validator fails closed when a held-constant value differs,
an unregistered semantic position changes, public conditions are altered, the
intervention set does not match the causal roots, or the certificate hash is
invalid.

## Layer interpretation

Layer is a stratification variable, not a repair direction:

- Layer 1 exposes a checkpoint and raw observations, but no diagnosis or
  selected artifact list.
- Layer 2 exposes the initial idea and a generic executable interface, so the
  Agent must produce diagnostic evidence itself.

Both layers use the same private repair and validation semantics.

## Admission requirements

A canonical case is admitted only when:

1. the gold/flawed semantic diff exactly equals the intervention positions;
2. all other semantic values and public running conditions are fixed;
3. the flawed state violates each registered invariant;
4. at least two equivalent repair operations restore each invariant;
5. competing non-repairs and integrity shortcuts fail;
6. required downstream artifacts are regenerated and privately verified;
7. a degree-zero control accepts evidence-grounded stopping without edits;
8. public input contains no repair-scope hint fields.

This supports the precise claim that SciCTRL controls a scientific validity
intervention under matched conditions. It does not claim to enumerate the
theoretical potential of every open-ended research direction.
