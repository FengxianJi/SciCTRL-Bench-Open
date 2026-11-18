"""Create opaque public inputs and a separate evaluator-only key for DiscoveryBench dev."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "Benchmark/ExternalDatasets/DiscoveryBench/synth/dev"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def build(output: Path) -> dict[str, int]:
    output = output.resolve()
    public_data = output / "public_data"
    public_data.mkdir(parents=True, exist_ok=True)
    public: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []

    for folder in sorted(path for path in SOURCE.iterdir() if path.is_dir()):
        csv_files = sorted(folder.glob("*.csv"))
        if not csv_files:
            continue
        # Keep each upstream variant's data isolated under a non-semantic name.
        family = folder.name.rsplit("_", 2)[0]
        for metadata_path in sorted(folder.glob("metadata_*.json")):
            raw_metadata = metadata_path.read_bytes()
            try:
                metadata = json.loads(raw_metadata.decode("utf-8"))
            except UnicodeDecodeError:
                # A small number of upstream metadata files contain Windows-1252
                # punctuation despite the repository otherwise being UTF-8.
                metadata = json.loads(raw_metadata.decode("cp1252"))
            for query in metadata.get("queries", []):
                source_key = f"{folder.name}/{metadata_path.name}/{query['qid']}"
                case_id = "dbdev-" + _digest(source_key)
                data_name = case_id + ".csv"
                shutil.copy2(csv_files[0], public_data / data_name)
                datasets = metadata.get("datasets", [])
                public.append(
                    {
                        "case_id": case_id,
                        "dataset": {
                            "description": datasets[0].get("description", "")
                            if datasets
                            else "",
                            "columns": datasets[0].get("columns", [])
                            if datasets
                            else [],
                            "csv": f"public_data/{data_name}",
                            "sha256": hashlib.sha256(csv_files[0].read_bytes()).hexdigest(),
                        },
                        "domain": metadata.get("domain", ""),
                        "workflow_tags": metadata.get("workflow_tags", []),
                        "domain_knowledge": metadata.get("domain_knowledge", ""),
                        "task": query.get("question", ""),
                    }
                )
                labels.append(
                    {
                        "case_id": case_id,
                        "source_family_cluster": family,
                        "upstream_split": "synth/dev",
                        "upstream_variant": folder.name,
                        "start_height": int(folder.name.rsplit("_", 2)[1]),
                        "target_depth": int(folder.name.rsplit("_", 1)[1]),
                        "qid": query.get("qid"),
                        "difficulty": query.get("difficulty"),
                        "target_col": query.get("target_col"),
                        "relevant_cols": query.get("relevant_cols", []),
                        "true_hypothesis": query.get("true_hypothesis", ""),
                        "true_hypothesis_expr": query.get("true_hypothesis_expr", ""),
                    }
                )

    if not public:
        raise RuntimeError(f"No DiscoveryBench development cases found under {SOURCE}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "discoverybench_public.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in public),
        encoding="utf-8",
    )
    (output / "discoverybench_source_labels.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in labels),
        encoding="utf-8",
    )
    return {"cases": len(public), "source_families": len({x["source_family_cluster"] for x in labels})}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
