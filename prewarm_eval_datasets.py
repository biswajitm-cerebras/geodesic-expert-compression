"""Pre-download all datasets needed by the REAP eval harness into the HF cache,
so the offline compute nodes can run evals. Run on the (online) login node.
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("HF_DATASETS_OFFLINE", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

LM_EVAL_TASKS = [
    "winogrande", "arc_challenge", "arc_easy", "boolq",
    "hellaswag", "mmlu", "openbookqa", "rte",
]

print("=== [1/3] lm-eval task datasets ===", flush=True)
try:
    from lm_eval.tasks import TaskManager, get_task_dict
    tm = TaskManager()
    # Building the task dict triggers dataset download for each task.
    td = get_task_dict(LM_EVAL_TASKS, task_manager=tm)
    print(f"lm-eval task dict built for {len(td)} top-level tasks.", flush=True)
except Exception as e:
    print(f"lm-eval prewarm error: {e}", flush=True)

print("=== [2/3] evalscope math datasets (gsm8k, math_500) ===", flush=True)
for name, kwargs in [
    ("openai/gsm8k", {"name": "main"}),
    ("HuggingFaceH4/MATH-500", {}),
]:
    try:
        from datasets import load_dataset
        ds = load_dataset(name, **kwargs)
        print(f"downloaded {name}: {list(ds.keys())}", flush=True)
    except Exception as e:
        print(f"math prewarm error for {name}: {e}", flush=True)

print("=== [3/3] evalplus (humaneval, mbpp) ===", flush=True)
try:
    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    he = get_human_eval_plus()
    mb = get_mbpp_plus()
    print(f"evalplus cached: humaneval={len(he)} tasks, mbpp={len(mb)} tasks", flush=True)
except Exception as e:
    print(f"evalplus prewarm error: {e}", flush=True)

print("=== prewarm complete ===", flush=True)
