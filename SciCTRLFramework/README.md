# SciCTRL VSRO Pipeline

This directory implements the data-generation core of the SciCTRL-Bench VSRO
(Validated Scientific Revision Opportunity) framework. It turns a
validated scientific idea into a normal scientific task containing a genuine,
experimentally discoverable revision opportunity.

Its final interface is defined in `OUTPUT_CONTRACT.md`: the public output is a
whole task, while the causal opportunity annotation and acceptance suite remain
in a paired private contract. The private causal target is an evaluation label,
not a public restriction on what the Agent may revise.

The central object is a **validated revision opportunity**, not an injected
instruction. Each case contains:

1. a gold research state;
2. a surface-realistic flawed state produced by changing one to three semantic
   slots;
3. a budgeted generic research interface whose observations can expose the
   resulting scientific failure;
4. hidden obligations specifying which causal problem must be resolved and
   which task-identity commitments remain invariant;
5. deterministic structural validation for the full matrix, plus an
   execution-backed pilot that runs real data analysis and experiment code.

Every internal case materializes these elements as a hash-bound **Matched
Control Certificate**. The certificate proves which semantic positions and
public running conditions are held constant between the gold and flawed state,
registers the scientific intervention, and assigns root, propagation,
protected and free position roles. It is the evaluator's source of truth.

The public task never includes an issue label, repair target, gold state, or
reference patch. A repair succeeds only when it discharges every violated
scientific obligation,
propagates required downstream changes, avoids forbidden shortcuts, verifies
the repaired state, and stops. A metric increase alone is never sufficient.

## Canonical layer definitions

- **Layer 1 — state-conditioned revision:** start from a failed intermediate
  research state containing artifacts and observations. The agent must acquire
  or inspect evidence, diagnose the problem, repair it, re-verify,
  and stop.
- **Layer 2 — idea-conditioned end-to-end revision:** start from a plausible
  but flawed idea. The agent must design and execute the research process that
  reveals the fault before repairing it.

Both layers measure the same capability. They differ only in starting state,
available horizon, and terminal artifact scope.

## Generate the benchmark

Install from the repository root with `python -m pip install -r requirements-dev.txt`.
Run the following command from that root; generated outputs are separate from the
versioned benchmark snapshot.

```bash
python -m scictrl.synthesize_vsro \
  --benchmark-root generated/canonical
```

The current matrix contains five independent gold projects across four
computational domains. Four projects enumerate every subset of three repair
atoms; the hypothesis-falsification pilot contributes only `k=0` and `k=1`
rather than manufacturing meaningless multi-flaw combinations. The result is
68 deterministic canonical cases, 34 per layer.

## Execution-backed pilot

Schema 0.8 retains `rare_group_classification_v1`, the first runnable VSRO domain
Pack. It materializes a grouped CSV dataset, editable split configuration,
a fixed-model Python experiment, and derived artifacts. The environment
launches actual subprocesses and returns their stdout, stderr, exit status and
changed-file hashes. A private evaluator independently checks source-group
disjointness, the predeclared split realization, complete sample coverage,
fixed-model predictions, metric recomputation, downstream artifacts, sealed
inputs, and reproducibility from the submitted code. No task-specific audit
program is exposed; agents may create analyses through the generic write/run
interface.

The pack includes both a flawed repair case and a clean `k=0` control. The
control requires evidence-grounded stopping without an unnecessary edit.

Layer-1 public inputs expose a complete, case-invariant artifact inventory.
They never expose a selected list of potentially dependent artifacts, target
slots, allowed slots, or repair directions. Those objects exist only in the
private matched-control certificate.

The pilot deliberately demonstrates that a scientifically correct repair may
lower the headline score: the leaked record split scores near 0.98 macro-F1,
whereas the valid group-held-out estimate is near 0.73. Controlled success is
therefore tied to restored validity rather than score improvement.

The local runner enforces request, timeout, address-space, file-size and
process-group limits and signs a private event chain. It is still not a
filesystem or network security boundary. Evaluated third-party agents should
run inside a container or microVM while preserving the same workspace contract.
