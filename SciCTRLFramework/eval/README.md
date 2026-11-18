# SciCTRL framework evaluation protocol

## Why this dataset first

The first external development source is the DiscoveryBench synthetic
development split under `Benchmark/ExternalDatasets/DiscoveryBench/synth/dev`.
Its task folders vary the height of observed nodes and depth of the target in a
semantic tree. The dev split has 69 folders and 153 query records; each folder
has its own CSV, so these are **task-family variants, not matched causal pairs**.
Use them to calibrate task ingestion, axis coverage, and difficulty. Use the
VSRO gold/intervention states for paired causal comparisons. The dev split is
not a source of final claims. The official test split must remain untouched
until the protocol and thresholds are frozen. DiscoveryBench
contains 264 paper-derived real tasks and 903 synthetic tasks overall; the
synthetic set is the appropriate controlled first-stage calibration source.

ScienceAgentBench is a strong later external-validity check: its 102 tasks were
derived from 44 peer-reviewed papers, validated by nine subject-matter experts,
and scored through generated programs, execution, costs, and rubrics. Its
official repository explicitly says not to redistribute the full unzipped
benchmark. We therefore do not copy its private/full artifacts into this
repository. Use only the official access route if a later local-only study is
approved.

## What makes a control framework good

Do not collapse framework validity into one agent score. A framework is
admissible only if every scored case passes these deterministic gates:

1. The baseline and intervened case have identical registered context,
   dataset, tools, budget, and output contract hashes.
2. The private semantic diff contains exactly the preregistered intervention;
   no unregistered position changes.
3. Baseline/gold states pass, each flawed state violates its intended hidden
   obligation, and at least two semantically equivalent repairs pass the same
   hidden tests.
4. Public prompts and file paths reveal no intervention labels, allowed-edit
   lists, accepted values, or hidden checks.
5. Degree-zero controls permit justified stopping and detect unnecessary
   edits. Execution evidence and final artifacts are evaluator-verified.
6. Independent task families and blinded domain review support the claims.
   A deterministic constructor pass alone is not domain validation.

Report the gate profile and separate agent outcomes: revision detection,
factor-identification precision/recall, valid-repair rate, false-revision rate
on controls, evidence/verification completeness, task coverage, tokens, cost,
and elapsed time. Compare systems on identical task IDs, Qwen model snapshot,
prompts, tools, budgets, and seeds. Randomize run order; use paired differences
and bootstrap confidence intervals clustered by source project/task family.
Never count correlated semantic-tree variants as independent projects.

For the current VSRO release, existing constructor reports in
`Benchmark/RevisionOpportunityFramework/` provide deterministic construction
checks. They do not establish an AutoResearchClaw or Qwen success rate. The
AutoResearchClaw adapter is the native-agent baseline; its workspace must be
evaluated by the private SciCTRL evaluator after the run, not by the agent's
self-reported success.

## Evaluation data contract

The input is two JSONL files:

- `public.jsonl`: opaque case ID, public task, public data paths, and agent
  output. Never include private labels or the evaluator key in a model prompt.
- `vsro_labels.jsonl`: evaluator-only matched-pair ID, source-family cluster,
  condition (`control` or `intervention`), expected revision decision, and
  hidden intervention IDs. Actual integrity/repair results are attached only
  after the private evaluator runs.

Each run row in `runs.jsonl` contains `case_id`, `framework`, `model_snapshot`,
`seed`, `predicted_revision`, `predicted_factor_ids`, `valid_repair`,
`evidence_verified`, `integrity_verified`, and optional token/cost/time fields.
It may also include predicted factor categories, revision confidence, and a
run status. A failed or timed-out attempt may contain only run identity, status,
and any observed resource usage; successful attempts must contain all evaluator
outcome fields. Confidence is a probability of needing revision.
The `valid_repair`, evidence, and integrity fields must be emitted by the
private evaluator, never copied from the model response. The complete
machine-readable contract is `run_record.schema.json`.

Run the dataset adapter with:

```bash
python SciCTRLFramework/eval/build_discoverybench.py \
  --output Benchmark/ExternalDatasets/DiscoveryBench/processed
```

Audit the downloaded split and perform a safe data-computation smoke test with:

```bash
python SciCTRLFramework/eval/audit_discoverybench.py \
  --output runs/eval/discoverybench_data_audit.json
python SciCTRLFramework/eval/smoke_discoverybench.py \
  --output runs/eval/discoverybench_smoke.json
```

The audit separates file/privacy integrity gates from upstream annotation
warnings. Do not use tasks with malformed gold expressions or missing query
columns for exact gold scoring. The smoke test computes a Pearson correlation
from numeric columns and deliberately never executes dataset-provided code.

This creates opaque public task IDs and a separate private labels file. It
copies the development CSVs to opaque filenames so directory names cannot
leak semantic-tree levels. It writes `discoverybench_public.jsonl` and
`discoverybench_source_labels.jsonl`. These source labels describe the upstream
DiscoveryBench tasks; they are **not** SciCTRL revision-control labels and are
not a direct input to the VSRO scorecard. The processed directory is local
evaluation data and is git-ignored. For VSRO agent scoring, create a separate
private `vsro_labels.jsonl` from the pinned canonical cases using the private
case certificate; only the evaluator can write the verified result fields to
`runs.jsonl`:

```bash
python SciCTRLFramework/eval/build_vsro_eval.py \
  --output runs/eval/vsro_labels.jsonl
```

`score_runs.py` computes per-system profiles and paired
comparisons from those evaluator outputs. It reports a revision confusion
matrix, balanced accuracy, and layer/control-degree slices so a high flaw
detection rate cannot hide false revisions on degree-zero controls:

- **Diagnosis:** confusion counts, accuracy, sensitivity, specificity,
  precision/F1, and balanced accuracy.
- **Localization:** micro and per-case macro factor precision/recall/F1,
  exact-set match, taxonomy-category localization, plus spurious factor and
  category rates on clean controls.
- **Revision quality:** repair attempt rate, evaluator-verified repair rate
  over all interventions, conditional on an attempted revision, and by
  intervention degree.
- **Evidence and integrity:** separate rates by control/intervention and their
  joint pass rate, end-to-end verified intervention success, and evidence-
  supported stopping on clean controls.
- **Calibration and uncertainty:** optional Brier/ECE/AUC from revision
  confidence and source-project-clustered intervals for diagnosis rates.
- **Efficiency and operations:** run-status counts, resource telemetry
  coverage, and valid repairs per 1,000 intervention tokens / per USD.

Metrics requiring fields absent from a run remain null or report zero
confidence coverage; missing telemetry is never imputed. Run status and
revision confidence are optional evaluator-run fields in the JSON schema.
Confidence is the system's probability that revision is required, not an
evaluator score. Clustered intervals are provisional with only a few source
projects, and token/cost-normalized outcomes are descriptive rather than
quality-adjusted causal effects.
Failed, timed-out, and parse-error attempts count against end-to-end
repair/evidence outcomes and execution coverage, but do not enter the decision
confusion matrix without a valid prediction. Skipped rows are reported as
missing coverage rather than failures.

```bash
python SciCTRLFramework/eval/score_runs.py \
  --labels runs/eval/vsro_labels.jsonl \
  --runs runs/eval/runs.jsonl \
  --output runs/eval/report.json
```

No API credentials are read by these offline utilities. Live API runs should
use an endpoint and credential whose terms permit backend-agent use.
