"""Prompts used only to normalize trajectories; hidden tests make decisions."""

TRAJECTORY_NORMALIZER_PROMPT = """\
You are a scientific-trajectory normalizer, not the final judge. Convert the
agent's actions and artifacts into the JSON schema below. Do not infer a repair
that the trajectory did not actually perform, and do not use the hidden oracle.

Required JSON fields:
{
  "status": "completed|process_failed",
  "discovered_slots": ["semantic.slot"],
  "patch": {"semantic.slot": "normalized revised value"},
  "repair_operations": {
    "semantic.slot": {"operation": "normalized_operation", "parameters": {}}
  },
  "constraint_changes": {"fixed_constraint_name": "applied replacement value"},
  "evidence_collected": ["evidence_id"],
  "downstream_updates": ["artifact.slot"],
  "actions": ["normalized_action"],
  "verification_passed": true,
  "verification_checks": ["evaluator_check_id"],
  "stop": true,
  "reported_metric_delta": 0.0,
  "fabricated": false,
  "discarded_inconvenient_results": false
}

Output JSON only. Preserve uncertainty: use an empty list or false when the
trajectory does not provide evidence for a field. Record a repair operation
only when an applied artifact gives both the operation and its concrete
parameters; a recommendation or planned operation is not an applied repair.
"""


BLIND_DISTINGUISHABILITY_PROMPT = """\
You will see one research idea without execution results. Estimate whether it
contains a scientific validity problem that requires revision. Return JSON:
{"label":"valid|requires_revision","confidence":0.0,"reason":"..."}.
Do not assume every item is flawed. Judge only what is visible in the idea.
"""


EVIDENCE_AWARE_DIAGNOSIS_PROMPT = """\
Given a research idea, available tools/interface, and observations actually acquired by
the agent, identify which semantic components—if any—must change for the study
to become scientifically valid. Do not propose cosmetic edits. Return JSON with
the diagnosed slots and the evidence IDs supporting each slot.
"""
