#!/usr/bin/env python3
"""Judge saved EvalScope MATH-500 generations with the established local Gemma judge."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


ROOT = Path("/lustre/scratch/users/biswajit.mishra/model_merge")
JAIS_EVAL = ROOT / "jais-3-evals"
JUDGE_MODEL = "google/gemma-4-E2B-it"
OUTPUT_NAME = "judge_scores_gemma-4-E2B-it.json"
DETAILS_NAME = "judge_scores_gemma-4-E2B-it.jsonl"


def latest_prediction_dir(results_dir: Path) -> Path:
    candidates = [
        path
        for path in results_dir.glob("evalscope_results/*/predictions/*")
        if path.is_dir() and list(path.glob("math_500_*.jsonl"))
    ]
    if not candidates:
        raise FileNotFoundError(f"No EvalScope MATH-500 predictions under {results_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_evalscope_samples(results_dir: Path) -> list[dict]:
    prediction_dir = latest_prediction_dir(results_dir)
    samples: list[dict] = []
    seen: set[str] = set()
    for path in sorted(prediction_dir.glob("math_500_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                raw_input = row.get("raw_input", {})
                sample_id = str(raw_input.get("unique_id", row.get("id", "")))
                if sample_id in seen:
                    continue
                seen.add(sample_id)
                choices = row.get("choices") or []
                prediction = ""
                if choices:
                    prediction = str(choices[0].get("message", {}).get("content") or "")
                samples.append(
                    {
                        "sample_id": sample_id,
                        "problem": str(raw_input.get("problem", "")),
                        "gold_answer": str(raw_input.get("answer", "")),
                        "prediction": prediction,
                    }
                )
    if len(samples) != 500:
        raise RuntimeError(
            f"Expected exactly 500 unique MATH-500 samples in {prediction_dir}; "
            f"found {len(samples)}"
        )
    return samples


def prepare_lm_eval_dir(directory: Path, samples: list[dict]) -> None:
    sample_path = directory / "samples_minerva_math500_2026-09-15T00-00-00.jsonl"
    with sample_path.open("w", encoding="utf-8") as handle:
        for doc_id, sample in enumerate(samples):
            row = {
                "doc_id": doc_id,
                "doc": {
                    "problem": sample["problem"],
                    "answer_not_normalized": sample["gold_answer"],
                },
                "target": sample["gold_answer"],
                "resps": [[sample["prediction"]]],
                "generation_metadata": [[{"reasoning": ""}]],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (directory / "results_evalscope_adapter.json").write_text(
        json.dumps({"results": {"minerva_math500": {}}}), encoding="utf-8"
    )


def save_outputs(results_dir: Path, temp_dir: Path, samples: list[dict]) -> None:
    judged_files = sorted(
        (temp_dir / "judge__gemma-4-E2B-it").glob("samples_minerva_math500_*.jsonl")
    )
    if len(judged_files) != 1:
        raise RuntimeError(f"Expected one Gemma output in {temp_dir}; found {judged_files}")

    by_doc_id: dict[int, dict] = {}
    with judged_files[0].open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            by_doc_id[int(row["doc_id"])] = row
    if len(by_doc_id) != 500:
        raise RuntimeError(f"Expected 500 Gemma verdicts in {judged_files[0]}; found {len(by_doc_id)}")

    details = []
    for doc_id, sample in enumerate(samples):
        verdict = by_doc_id[doc_id]
        details.append(
            {
                **sample,
                "judge_model": JUDGE_MODEL,
                "judge_reasoning": verdict.get("reasoning", ""),
                "judge_extracted_answer": verdict.get("extracted_answer"),
                "is_correct": bool(verdict.get("is_correct", False)),
            }
        )

    correct = sum(item["is_correct"] for item in details)
    summary = {
        "model": JUDGE_MODEL,
        "judge_type": "local_vllm_math_equivalence",
        "total_samples": 500,
        "correct": correct,
        "overall_accuracy": correct / 500,
        "per_task": {
            "math500": {"accuracy": correct / 500, "total": 500, "correct": correct}
        },
    }
    (results_dir / OUTPUT_NAME).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (results_dir / DETAILS_NAME).open("w", encoding="utf-8") as handle:
        for item in details:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"{results_dir}: {correct}/500 = {100 * correct / 500:.2f}%", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dirs", nargs="+", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--gpu-mem-utilization", type=float, default=0.85)
    args = parser.parse_args()

    targets = []
    for results_dir in args.results_dirs:
        results_dir = results_dir.resolve()
        output = results_dir / OUTPUT_NAME
        if output.exists() and not args.overwrite:
            print(f"[SKIP] {output} already exists", flush=True)
            continue
        targets.append((results_dir, load_evalscope_samples(results_dir)))
    if not targets:
        print("Nothing to judge.")
        return 0

    sys.path.insert(0, str(JAIS_EVAL))
    from judge_vllm import main as run_gemma_judge

    with tempfile.TemporaryDirectory(prefix="evalscope_gemma_judge_") as tmp:
        temp_root = Path(tmp)
        temp_dirs = []
        for index, (_, samples) in enumerate(targets):
            temp_dir = temp_root / f"target_{index:02d}"
            temp_dir.mkdir()
            prepare_lm_eval_dir(temp_dir, samples)
            temp_dirs.append(temp_dir)

        run_gemma_judge(
            SimpleNamespace(
                results_path=temp_dirs,
                judge_model_name=JUDGE_MODEL,
                batch_size=16384,
                judge_params={
                    "seed": 42,
                    "max_tokens": 8192,
                    "temperature": 0,
                    "enable_thinking": False,
                },
                override=False,
                show_subbatch_progress=False,
                limit=None,
                gpu_mem_utilization=args.gpu_mem_utilization,
            )
        )
        for (results_dir, samples), temp_dir in zip(targets, temp_dirs):
            save_outputs(results_dir, temp_dir, samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())