"""Validate downloaded DiscoveryBench dev files and their public/private conversion."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "Benchmark/ExternalDatasets/DiscoveryBench/synth/dev"
DEFAULT_PROCESSED = ROOT / "Benchmark/ExternalDatasets/DiscoveryBench/processed"
FORBIDDEN_PUBLIC_FIELDS = {
    "target_col",
    "relevant_cols",
    "true_hypothesis",
    "true_hypothesis_expr",
    "qid",
    "difficulty",
    "start_height",
    "target_depth",
}


def _metadata(path: Path) -> dict[str, Any]:
    content = path.read_bytes()
    try:
        return json.loads(content.decode("utf-8"))
    except UnicodeDecodeError:
        return json.loads(content.decode("cp1252"))


def audit(source: Path = DEFAULT_SOURCE, processed: Path = DEFAULT_PROCESSED) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    task_count = 0
    metadata_count = 0
    folder_count = 0
    row_count = 0
    column_count = 0
    query_ids: list[str] = []
    family_counts: Counter[str] = Counter()
    observed_levels: set[tuple[int, int]] = set()

    for folder in sorted(path for path in source.iterdir() if path.is_dir()):
        folder_count += 1
        name_parts = folder.name.rsplit("_", 2)
        if len(name_parts) != 3 or not name_parts[1].isdigit() or not name_parts[2].isdigit():
            errors.append({"path": str(folder), "error": "invalid_synthetic_variant_name"})
            continue
        family, start_height, target_depth = name_parts[0], int(name_parts[1]), int(name_parts[2])
        family_counts[family] += 1
        observed_levels.add((start_height, target_depth))
        csv_files = sorted(folder.glob("*.csv"))
        if len(csv_files) != 1:
            errors.append({"path": str(folder), "error": f"expected_one_csv_got_{len(csv_files)}"})
            continue
        try:
            with csv_files[0].open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                columns = reader.fieldnames or []
                rows = list(reader)
            if not columns or not rows:
                errors.append({"path": str(csv_files[0]), "error": "empty_table_or_header"})
                continue
            row_count += len(rows)
            column_count += len(columns)
        except (OSError, csv.Error, UnicodeDecodeError) as error:
            errors.append({"path": str(csv_files[0]), "error": f"csv_read_error:{type(error).__name__}"})
            continue

        metadata_files = sorted(folder.glob("metadata_*.json"))
        for path in metadata_files:
            metadata_count += 1
            try:
                metadata = _metadata(path)
            except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
                errors.append({"path": str(path), "error": f"metadata_read_error:{type(error).__name__}"})
                continue
            source_columns = {
                item.get("name")
                for dataset in metadata.get("datasets", [])
                for item in dataset.get("columns", [])
            }
            missing_descriptions = source_columns - set(columns)
            if missing_descriptions:
                errors.append({"path": str(path), "error": "described_column_absent_from_csv"})
            for query in metadata.get("queries", []):
                task_count += 1
                qid = str(query.get("qid"))
                query_ids.append(qid)
                needed = set(query.get("relevant_cols", [])) | {query.get("target_col")}
                if not query.get("question"):
                    errors.append({"path": str(path), "error": f"query_{qid}_missing_question"})
                if None in needed or not needed <= set(columns):
                    errors.append({"path": str(path), "error": f"query_{qid}_columns_missing_from_csv"})
                expression = query.get("true_hypothesis_expr", "")
                try:
                    ast.parse(expression, mode="eval")
                except (SyntaxError, ValueError):
                    errors.append({"path": str(path), "error": f"query_{qid}_invalid_gold_expression_syntax"})

    if len(query_ids) != len(set(query_ids)):
        errors.append({"path": str(source), "error": "duplicate_qid_in_dev_split"})

    public_path = processed / "discoverybench_public.jsonl"
    labels_path = processed / "discoverybench_source_labels.jsonl"
    public_rows = [json.loads(line) for line in public_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    label_rows = [json.loads(line) for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    public_ids = [row.get("case_id") for row in public_rows]
    label_ids = [row.get("case_id") for row in label_rows]
    if len(public_ids) != len(set(public_ids)) or public_ids != label_ids:
        errors.append({"path": str(processed), "error": "public_private_id_mismatch_or_duplicate"})
    leaks = 0
    hash_errors = 0
    for row in public_rows:
        leaked_keys = FORBIDDEN_PUBLIC_FIELDS & row.keys()
        if leaked_keys:
            leaks += len(leaked_keys)
        data_path = processed / row["dataset"]["csv"]
        if not data_path.is_file():
            hash_errors += 1
        elif hashlib.sha256(data_path.read_bytes()).hexdigest() != row["dataset"].get("sha256"):
            hash_errors += 1
    if leaks:
        errors.append({"path": str(public_path), "error": "private_field_leak"})
    if hash_errors:
        errors.append({"path": str(public_path), "error": "missing_or_hash_mismatched_public_csv"})

    source_errors = [item for item in errors if "query_" in item["error"]]
    annotation_errors = [
        item for item in source_errors
        if "columns_missing" in item["error"] or "expression_syntax" in item["error"]
    ]
    gates = {
        "source_tables_and_metadata_read": folder_count > 0 and metadata_count > 0 and not any("read_error" in item["error"] for item in errors),
        "metadata_described_columns_exist": not any("column_absent" in item["error"] for item in errors),
        "unique_query_ids": "duplicate_qid_in_dev_split" not in {item["error"] for item in errors},
        "public_labels_do_not_leak": leaks == 0,
        "opaque_data_files_match_hashes": hash_errors == 0,
        "public_private_cases_align": "public_private_id_mismatch_or_duplicate" not in {item["error"] for item in errors},
    }
    annotation_summary = Counter(
        "missing_query_columns" if "columns_missing" in item["error"] else "invalid_gold_expression_syntax"
        for item in annotation_errors
    )
    return {
        "schema_version": "discoverybench-data-audit-0.1",
        "source_split": "DiscoveryBench synthetic/dev",
        "source_folders": folder_count,
        "metadata_files": metadata_count,
        "tasks": task_count,
        "source_families": len(family_counts),
        "family_folder_counts": dict(sorted(family_counts.items())),
        "semantic_depth_pairs": len(observed_levels),
        "csv_rows_total": row_count,
        "csv_columns_total_across_variants": column_count,
        "converted_public_rows": len(public_rows),
        "converted_private_label_rows": len(label_rows),
        "public_leaked_private_fields": leaks,
        "public_data_hash_errors": hash_errors,
        "gates": gates,
        "passed_gates": sum(gates.values()),
        "total_gates": len(gates),
        "status": "pass" if all(gates.values()) else "fail",
        "annotation_status": "clean" if not annotation_errors else "warnings_found",
        "annotation_warning_counts": dict(sorted(annotation_summary.items())),
        "annotation_warnings": annotation_errors[:100],
        "errors": errors[:100],
        "limitations": [
            "DiscoveryBench source-level semantic variants use distinct task CSV files and are not matched causal pairs.",
            "Upstream query-column and gold-expression defects are reported as annotation warnings; source files are left unchanged.",
            "Gold expressions are syntax-checked but never executed by this audit.",
            "This verifies dataset integrity and public/private separation, not Qwen or AutoResearchClaw scientific performance.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.source, args.processed)
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
