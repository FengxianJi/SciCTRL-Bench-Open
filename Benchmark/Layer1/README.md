# Layer 1 — state-conditioned revision

Input is a completed intermediate research checkpoint containing the
idea, experiment summary, artifacts, and observations, but no diagnosis and no
repair instruction.

The public artifact inventory is complete and identical across cases. It is
not filtered to the artifacts downstream of the hidden intervention, because a
filtered list would reveal the likely revision scope.

The evaluated behavior is:

```text
inspect/acquire evidence -> discover -> revise -> propagate -> rerun -> verify -> stop
```

Layer 1 isolates revision behavior with a shorter horizon. It contains a
validated scientific revision opportunity but does not tell the agent which
causal component is responsible. In the AutoResearchClaw adapter, the state
is rendered as a checkpoint bundle and mapped to the closest supported resume
stage.
