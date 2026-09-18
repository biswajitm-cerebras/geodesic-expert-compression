#!/usr/bin/env python3
"""Collect local Gemma MATH-500 judge scores for dense, REAP, and GCGM runs."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path("/lustre/scratch/users/biswajit.mishra/model_merge/msmoe/reap")
SCORE_FILE = "judge_scores_gemma-4-E2B-it.json"
OUTPUT_DIR = ROOT / "artifacts/eval"

RUNS = [
    ("GCGM", "seed_42", ROOT / "artifacts/eval/fgcluster_reap_localfr_gl_0p5"),
    ("GCGM", "seed_11", ROOT / "artifacts/eval/gcgm_three_seed/gcgm_seed_11/math500"),
    ("GCGM", "seed_99", ROOT / "artifacts/eval/gcgm_three_seed/gcgm_seed_99/math500"),
    ("REAP controlled Instruct-2507", "seed_42", ROOT / "artifacts/eval/instruct2507_controlled/reap_seed_42/math500"),
    ("REAP controlled Instruct-2507", "seed_11", ROOT / "artifacts/eval/instruct2507_controlled/reap_seed_11/math500"),
    ("REAP controlled Instruct-2507", "seed_99", ROOT / "artifacts/eval/instruct2507_controlled/reap_seed_99/math500"),
    ("REAP original paper parent", "seed_42", ROOT / "artifacts/eval/paper_reproduction/reap_seed_42/math500"),
    ("REAP original paper parent", "seed_11", ROOT / "artifacts/eval/paper_reproduction/reap_seed_11/math500"),
    ("REAP original paper parent", "seed_99", ROOT / "artifacts/eval/paper_reproduction/reap_seed_99/math500"),
    ("Dense Instruct-2507", "single", ROOT / "artifacts/eval/instruct2507_controlled/dense/math500"),
    ("Dense original paper parent", "single", ROOT / "artifacts/eval/paper_reproduction/dense/math500"),
]


def main() -> int:
    rows = []
    grouped: dict[str, list[float]] = {}
    for method, run, directory in RUNS:
        path = directory / SCORE_FILE
        if not path.exists():
            raise FileNotFoundError(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        score = 100 * float(data["overall_accuracy"])
        rows.append({"method": method, "run": run, "gemma_math500_percent": score})
        grouped.setdefault(method, []).append(score)

    for method, scores in grouped.items():
        if len(scores) == 3:
            rows.append(
                {
                    "method": method,
                    "run": "three_seed_mean",
                    "gemma_math500_percent": statistics.mean(scores),
                }
            )
            rows.append(
                {
                    "method": method,
                    "run": "three_seed_sample_std",
                    "gemma_math500_percent": statistics.stdev(scores),
                }
            )

    csv_path = OUTPUT_DIR / "gemma_math500_judge_summary.csv"
    json_path = OUTPUT_DIR / "gemma_math500_judge_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())