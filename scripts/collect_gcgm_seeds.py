#!/usr/bin/env python3
"""Collect seed-matched GCGM evaluations and report mean and sample std."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, stdev

from collect_paper_reproduction import FIELDS, collect, write_xlsx


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "eval" / "gcgm_three_seed"
SEEDS = (42, 11, 99)


def aggregate(
    rows: list[dict[str, str | float | None]], statistic: str
) -> dict[str, str | float | None]:
    result: dict[str, str | float | None] = {
        "Model": f"GCGM 50% (3-seed {statistic})"
    }
    for field in FIELDS[1:]:
        values = [row[field] for row in rows]
        if not all(isinstance(value, float) for value in values):
            result[field] = None
        elif statistic == "mean":
            result[field] = round(mean(values), 2)
        else:
            result[field] = round(stdev(values), 2)
    return result


def main() -> None:
    rows = [
        collect(f"GCGM 50% seed {seed}", RESULTS / f"gcgm_seed_{seed}")
        for seed in SEEDS
    ]
    missing = [
        f"seed {seed}: {field}"
        for seed, row in zip(SEEDS, rows)
        for field in FIELDS[1:]
        if row[field] is None
    ]
    if missing:
        raise RuntimeError("Missing completed metrics: " + ", ".join(missing))

    output_rows = [*rows, aggregate(rows, "mean"), aggregate(rows, "std")]
    RESULTS.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS / "gcgm_three_seed_scores.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    write_xlsx(output_rows, RESULTS / "gcgm_three_seed_scores.xlsx")
    print(csv_path.read_text())


if __name__ == "__main__":
    main()