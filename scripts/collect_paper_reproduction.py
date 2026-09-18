#!/usr/bin/env python3
"""Collect dense and three-seed REAP paper-reproduction scores."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from statistics import mean

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(
    os.environ.get(
        "REPRO_RESULTS_DIR",
        ROOT / "artifacts" / "eval" / "paper_reproduction",
    )
)
DENSE_LABEL = os.environ.get("REPRO_DENSE_LABEL", "Dense paper checkpoint")
REAP_LABEL = os.environ.get("REPRO_REAP_LABEL", "REAP 50%")
FIELDS = [
    "Model", "HumanEval", "HumanEval+", "MBPP", "MBPP+", "Eval+ Avg",
    "MC Avg", "LiveCodeBench", "Paper Code Avg", "GSM8K", "MATH-500", "Math Avg",
]
MC_TASKS = [
    "arc_challenge", "arc_easy", "boolq", "hellaswag", "mmlu",
    "openbookqa", "rte", "winogrande",
]


def percent(value: float) -> float:
    return round(100.0 * value, 2)


def evalplus(path: Path, suite: str, plus: bool) -> float | None:
    file = path / f"{suite}.json"
    if not file.exists():
        return None
    data = json.loads(file.read_text())
    key = "plus" if plus else "base"
    return percent(float(data["pass_at_k"][key]["pass@1"]))


def livecode(path: Path) -> float | None:
    file = path / "Scenario.codegeneration_1_0.2_eval.json"
    if not file.exists():
        return None
    return percent(float(json.loads(file.read_text())[0]["pass@1"]))


def evalscope(path: Path, dataset: str) -> float | None:
    reports = list(path.glob(f"evalscope_results/*/reports/*/{dataset}.json"))
    if not reports:
        return None
    report = max(reports, key=lambda item: item.stat().st_mtime)
    return percent(float(json.loads(report.read_text())["score"]))


def mc_average(path: Path) -> float | None:
    file = path / "lm_eval_results.json"
    if not file.exists():
        return None
    results = json.loads(file.read_text()).get("results", {})
    scores: list[float] = []
    for task in MC_TASKS:
        metrics = results.get(task)
        if not metrics:
            continue
        value = metrics.get("acc_norm,none", metrics.get("acc,none"))
        if value is not None:
            scores.append(percent(float(value)))
    return round(mean(scores), 2) if len(scores) == len(MC_TASKS) else None


def avg(*values: float | None) -> float | None:
    return round(mean(values), 2) if values and all(value is not None for value in values) else None


def collect(label: str, root: Path) -> dict[str, str | float | None]:
    core = root / "core"
    gsm = root / "gsm8k"
    math = root / "math500"
    he = evalplus(core, "humaneval", False)
    hep = evalplus(core, "humaneval", True)
    mbpp = evalplus(core, "mbpp", False)
    mbppp = evalplus(core, "mbpp", True)
    # Paper definition: Eval+ Avg is the mean of the two EvalPlus scores,
    # HumanEval+ and MBPP+, not the mean of all four code columns.
    eval_avg = avg(hep, mbppp)
    lcb = livecode(core)
    gsm_score = evalscope(gsm, "gsm8k")
    math_score = evalscope(math, "math_500")
    return {
        "Model": label,
        "HumanEval": he,
        "HumanEval+": hep,
        "MBPP": mbpp,
        "MBPP+": mbppp,
        "Eval+ Avg": eval_avg,
        "MC Avg": mc_average(core),
        "LiveCodeBench": lcb,
        "Paper Code Avg": avg(eval_avg, lcb),
        "GSM8K": gsm_score,
        "MATH-500": math_score,
        "Math Avg": avg(gsm_score, math_score),
    }


def reap_mean(rows: list[dict[str, str | float | None]]) -> dict[str, str | float | None]:
    result: dict[str, str | float | None] = {"Model": f"{REAP_LABEL} (3-seed mean)"}
    for field in FIELDS[1:]:
        values = [row[field] for row in rows]
        result[field] = round(mean(values), 2) if all(isinstance(v, float) for v in values) else None
    return result


def write_xlsx(rows: list[dict[str, str | float | None]], path: Path) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Paper reproduction"
    sheet.append(FIELDS)
    for row in rows:
        sheet.append([row[field] for field in FIELDS])
    for cell in sheet[1]:
        cell.fill = PatternFill("solid", fgColor="1E3A8A")
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
    for row in sheet.iter_rows(min_row=2):
        for cell in row[1:]:
            cell.number_format = "0.00"
            cell.alignment = Alignment(horizontal="center")
    sheet.freeze_panes = "B2"
    sheet.auto_filter.ref = sheet.dimensions
    sheet.sheet_view.showGridLines = False
    sheet.column_dimensions["A"].width = 27
    for index in range(2, len(FIELDS) + 1):
        sheet.column_dimensions[get_column_letter(index)].width = 16
    workbook.save(path)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    seed_rows = [collect(f"{REAP_LABEL} seed {seed}", RESULTS / f"reap_seed_{seed}") for seed in (42, 11, 99)]
    rows = [collect(DENSE_LABEL, RESULTS / "dense"), *seed_rows, reap_mean(seed_rows)]
    csv_path = RESULTS / "paper_reproduction_scores.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    write_xlsx(rows, RESULTS / "paper_reproduction_scores.xlsx")
    print(csv_path.read_text())


if __name__ == "__main__":
    main()
