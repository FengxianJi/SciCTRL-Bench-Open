"""Deterministically score one normalized CSFI submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .evaluator import evaluate_submission


def _load_case(path: Path, case_id: str) -> dict[str, Any]:
    for line in path.read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        if case["case_id"] == case_id:
            return case
    raise KeyError(case_id)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal-cases", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    case = _load_case(args.internal_cases, args.case_id)
    submission = json.loads(args.submission.read_text(encoding="utf-8"))
    result = evaluate_submission(case, submission).to_dict()
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
