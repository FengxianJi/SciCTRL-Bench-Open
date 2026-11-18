# SciCTRL-Bench

SciCTRL-Bench studies whether scientific agents can identify and repair a controlled validity problem while preserving the research task, gathering evidence, updating dependent artifacts, verifying the result, and stopping appropriately. The core contribution is a matched-pair task and evaluation framework: baseline and intervened tasks share registered conditions, while private evaluator contracts measure whether the scientific invariant was actually restored.

> **Release boundary:** This public repository contains framework code, tests, and public task inputs. It does not contain private evaluator labels, oracle repairs, hidden control contracts, unpublished paper drafts, API credentials, or third-party agent environments. The private SciCTRL-Bench repository remains the source for restricted evaluator assets.

## Repository map

- `SciCTRLFramework/scictrl/` — task models, control certificates, task construction, runtime, and evaluator.
- `SciCTRLFramework/eval/` — data audits, scoring, and evaluation schemas.
- `Benchmark/agent_adapters/` — adapters and trajectory collection for compatible agent runs.
- `Benchmark/{Layer1,Layer2,ExecutablePacks}/eight_domains_v4/public.jsonl` — bundled agent-visible task inputs. No evaluator labels are included.
- `Method/` — shared transport and reporting utilities; third-party research-agent environments are not included.
- `results/pilot_summary.json` — aggregate diagnostic-pilot counts and limitations.

## Install and test

Python 3.10+ is recommended. Tests run offline and need no API key, GPU, or external agent checkout.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

To install the framework package directly, run `python -m pip install -e './SciCTRLFramework[dev]'`.

## Generate a fresh public task set

```bash
scictrl-synthesize-vsro --benchmark-root generated/canonical
python -m scictrl.synthesize_cpu_diagnostic --benchmark-root generated/cpu
```

Outputs are written outside the repository snapshot. Agent-visible task files are `public.jsonl`; do not put evaluator-only material in an agent workspace. The full evaluator requires a matching private label/contract file, which is intentionally not distributed here.

## What current results establish

The bundled aggregate pilot reports 32/32 programmatic diagnostic tasks passing numerical and final-source checks, 16/16 original proposals classified correctly, and 14/16 rewritten proposals classified correctly (87.5% balanced accuracy; AUC 0.883). A CPU control stress test contains 96 executions. These are task-construction and diagnostic-pilot results; they do **not** demonstrate superiority of an RSI framework or end-to-end scientific research ability. A matched native-agent comparison remains future work.

## Data and external dependencies

Bundled task inputs are under CC BY 4.0; framework code is under MIT. The two licenses are provided separately. DiscoveryBench is not redistributed here; obtain it from [AllenAI's official repository](https://github.com/allenai/discoverybench) and follow its ODC-By attribution terms. Native agent frameworks are external dependencies and retain their own licenses.

See `SciCTRLFramework/README.md`, `SciCTRLFramework/CONTROL_VARIABLES.md`, and `SciCTRLFramework/eval/README.md` for architecture, controls, and the scoring protocol.
