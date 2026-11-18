# Native methods

This directory tracks SciCTRL adapters and runners. Downloaded upstream
repositories and their environments are local dependencies, excluded from Git.
The offline benchmark and its tests do not require these repositories.

[`sources.json`](sources.json) records the exact upstream commits present during
repository preparation. [`reproduction_manifest.json`](reproduction_manifest.json)
and [`cpu_api_dependency_lock.json`](cpu_api_dependency_lock.json) are historical
environment inventories; consult upstream installation instructions for each
method's dependencies.

## Restore AutoResearchClaw

From the repository root, into a fresh `Method/AutoResearchClaw` directory:

```bash
git clone https://github.com/aiming-lab/AutoResearchClaw.git Method/AutoResearchClaw
git -C Method/AutoResearchClaw checkout be4ba4755bf1b52220f25e13b2293b5956590070
git -C Method/AutoResearchClaw apply ../patches/AutoResearchClaw.patch
cp -R Method/overlays/AutoResearchClaw/. Method/AutoResearchClaw/
```

The patch preserves tracked local provider, transport and pipeline compatibility
changes and their upstream tests. The overlay preserves the additional local
configuration examples, reproduction notes and launch script. Follow the
restored upstream README to install its dependencies in a separate environment.
The upstream code retains its original license.

For other methods, clone the repository URL from `sources.json` into
`Method/<name>` and check out its recorded commit. No local tracked changes were
present in those five checkouts when this snapshot was prepared.

## Entrypoints

| Script | Purpose |
| --- | --- |
| `reproduce_cpu.py` | Inventory native imports, prompt hashes and dependencies |
| `check_native_cpu_entrypoints.py` | Check native entrypoints and prerequisites |
| `run_native_cpu.py` | Run one native method on a public CPU task |
| `run_native_cpu_batch.py` | Schedule native CPU tasks |
| `evaluate_native_cpu.py` | Evaluate generated artifacts separately |
| `summarize_native_runs.py` | Summarize saved runs and update manuscript blocks |

Use `PYTHONPATH=SciCTRLFramework:Benchmark:Method` when invoking scripts from the root.
Runtime output under `Benchmark/Results/` and `benchmark_runs/` is ignored.
Reporting scripts require those separately retained experiment outputs.
See [CPU_REPRODUCTION.md](CPU_REPRODUCTION.md) and
[NATIVE_RUN_PROTOCOL.md](NATIVE_RUN_PROTOCOL.md) for the experiment protocol.
