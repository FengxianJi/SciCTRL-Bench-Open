"""Run a safe numeric-analysis smoke test against one DiscoveryBench task."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "Benchmark/ExternalDatasets/DiscoveryBench/synth/dev"


def _read_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    try:
        return json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        return json.loads(raw.decode("cp1252"))


def smoke(source: Path = DEFAULT_SOURCE) -> dict[str, Any]:
    for metadata_path in sorted(source.rglob("metadata_*.json")):
        metadata = _read_json(metadata_path)
        csv_path = next(iter(sorted(metadata_path.parent.glob("*.csv"))), None)
        if csv_path is None:
            continue
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            continue
        columns = set(rows[0])
        for query in metadata.get("queries", []):
            target = query.get("target_col")
            if not query.get("question") or target not in columns:
                continue
            for feature in query.get("relevant_cols", []):
                if feature == target or feature not in columns:
                    continue
                pairs = []
                for row in rows:
                    try:
                        pairs.append((float(row[feature]), float(row[target])))
                    except (TypeError, ValueError):
                        continue
                if len(pairs) < 3:
                    continue
                xs, ys = zip(*pairs)
                if len(set(xs)) < 2 or len(set(ys)) < 2:
                    continue
                return {
                    "status": "pass",
                    "method": "Pearson correlation over numeric task columns",
                    "gold_expression_executed": False,
                    "source_metadata": str(metadata_path.relative_to(source)),
                    "qid": query.get("qid"),
                    "feature": feature,
                    "target": target,
                    "valid_rows": len(pairs),
                    "pearson_r": statistics.correlation(xs, ys),
                }
    raise RuntimeError(f"No usable numeric task found under {source}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = smoke(args.source)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
