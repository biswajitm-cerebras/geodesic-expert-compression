#!/usr/bin/env python3
"""Judge saved EvalScope MATH-500 generations with the project GPT judge."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI


JUDGE_DIR = Path("/lustre/scratch/users/biswajit.mishra/model_merge/judge")
sys.path.insert(0, str(JUDGE_DIR))
from run_math500_judge import (  # noqa: E402
    best_model_answer,
    judge_all,
    judge_one,
    save_details,
    save_scores,
)


def latest_prediction_dir(results_dir: Path) -> Path:
    candidates = [
        path
        for path in results_dir.glob("evalscope_results/*/predictions/*")
        if path.is_dir() and list(path.glob("math_500_*.jsonl"))
    ]
    if not candidates:
        raise FileNotFoundError(f"No EvalScope MATH-500 predictions under {results_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_evalscope_samples(prediction_dir: Path) -> list[dict]:
    samples: list[dict] = []
    seen: set[str] = set()
    for path in sorted(prediction_dir.glob("math_500_*.jsonl")):
        with path.open() as handle:
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
                        "problem": str(raw_input.get("problem", "")),
                        "gold_answer": str(raw_input.get("answer", "")),
                        "prediction": prediction,
                        "model_answer": best_model_answer(prediction),
                        "exact_match": None,
                        "math_verify": None,
                    }
                )
    if len(samples) != 500:
        raise RuntimeError(
            f"Expected exactly 500 unique MATH-500 samples in {prediction_dir}; "
            f"found {len(samples)}"
        )
    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--judge-model", default="gpt-5.2-2025-12-11")
    parser.add_argument("--max-concurrent", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = args.results_dir / "judge_scores.json"
    details = args.results_dir / "judge_scores.jsonl"
    if output.exists() and not args.overwrite:
        print(f"[SKIP] {output} already exists")
        return 0
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    prediction_dir = latest_prediction_dir(args.results_dir)
    samples = load_evalscope_samples(prediction_dir)
    base_url = os.environ.get("OPENAI_BASE_URL")
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        **({"base_url": base_url} if base_url else {}),
    )
    preflight = judge_one(client, args.judge_model, samples[0], max_retries=1)
    if str(preflight["judge_reply"]).startswith("ERROR:"):
        raise RuntimeError(
            "Judge API preflight failed; no batch requests or scores were saved: "
            + str(preflight["judge_reply"])
        )
    label = args.results_dir.parent.name
    started = time.time()
    results = judge_all(client, args.judge_model, samples, args.max_concurrent, label)
    errors = [result for result in results if str(result["judge_reply"]).startswith("ERROR:")]
    if errors:
        raise RuntimeError(f"Judge failed for {len(errors)}/500 samples; no score saved")
    save_scores(output, args.judge_model, results, task_name="math500")
    save_details(details, results, task_name="math500")
    correct = sum(bool(result["is_correct"]) for result in results)
    print(f"\n{correct}/500 = {100 * correct / 500:.2f}% in {time.time() - started:.1f}s")
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())