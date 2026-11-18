# Public release scope

This repository is a sanitized, source-only release snapshot. The upstream research repository remains private because it contains evaluator-only annotations and contracts required for blind scoring. No upstream git history is copied into this repository.

Included: framework source and tests, adapters and shared utilities, documentation needed to use the code, three agent-visible v4 JSONL input files, and aggregate pilot metrics.

Excluded: all private/oracle JSONL and control contracts, private label folders, raw model traces and audit reports, API keys, virtual environments, caches, paper PDFs/drafts, downloaded DiscoveryBench data, and third-party agent repositories.

Adapter integration tests that load internal evaluator cases are intentionally withheld; public-source unit tests remain in the repository. The public `public.jsonl` inputs must be treated as model-visible material. A valid blind benchmark deployment must keep its evaluator labels and repair contracts on a separate evaluator-controlled system.
