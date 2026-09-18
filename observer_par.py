#!/usr/bin/env python3
"""Distributed observer for REAP.

Each rank (one process per node) loads the full model sharded on its local GPUs
via device_map="auto". The 128 calibration batches are split evenly across ranks
so the observer wall-time scales linearly with the number of nodes.

After all batches are processed each rank's partial statistics are gathered to
rank 0, merged using Welford's incremental update formula, and saved to the same
.pt file that main.py would produce. A subsequent run of main.py will find the
file and skip the observer phase automatically.

Launch (SLURM, 2 nodes × 4 GPUs):
    srun --nodes=2 --ntasks-per-node=1 \\
        python observer_par.py \\
        --model_name "Qwen/Qwen3-30B-A3B-Instruct-2507" \\
        --dataset_name "theblackcat102/evol-codealpaca-v1" \\
        --batches_per_category 128 --batch_size 8 \\
        --model_max_length 2048 \\
        --output_file_name "observations_1024_cosine-seed_42_v27gl_paper.pt" \\
        --seed 42

Environment variables (set automatically by SLURM + srun):
    SLURM_PROCID  → rank
    SLURM_NTASKS  → world_size
    SLURM_STEP_NODELIST or SLURM_NODELIST → used to derive MASTER_ADDR

If running outside SLURM set RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT.
"""
from __future__ import annotations

import logging
import os
import pathlib
import re
import sys

import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from accelerate.utils import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Ensure the reap package is importable from this directory.
_REAP_SRC = pathlib.Path(__file__).parent / "src"
if str(_REAP_SRC) not in sys.path:
    sys.path.insert(0, str(_REAP_SRC))

from reap.args import ModelArgs, DatasetArgs, ObserverArgs, ReapArgs
from reap.data import load_category_batches
from reap.metrics import OnlineStatsTracker
from reap.observer import OBSERVER_CONFIG_REGISTRY, MoETransformerObserver
from reap.model_util import patched_model_map


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def _init_dist() -> tuple[int, int]:
    """Initialize a Gloo process group; return ``(rank, world_size)``.

    Each rank performs model inference independently on its four local GPUs.
    The process group is only used for barriers and gathering CPU observer
    state objects, for which Gloo is more appropriate than NCCL.
    """
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    world_size = int(
        os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1"))
    )
    if world_size == 1:
        return 0, 1

    master_addr = os.environ.get("MASTER_ADDR")
    if master_addr is None:
        # Derive from SLURM_STEP_NODELIST / SLURM_NODELIST (first hostname).
        nodelist = os.environ.get(
            "SLURM_STEP_NODELIST", os.environ.get("SLURM_NODELIST", "localhost")
        )
        import subprocess
        try:
            master_addr = subprocess.check_output(
                ["scontrol", "show", "hostname", nodelist],
                text=True,
            ).split()[0]
        except Exception:
            master_addr = "localhost"
        os.environ["MASTER_ADDR"] = master_addr

    master_port = os.environ.get("MASTER_PORT", "29500")
    os.environ["MASTER_PORT"] = master_port

    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
    )
    logger.info("Distributed initialized: rank %d / %d  addr=%s:%s",
                rank, world_size, master_addr, master_port)
    return rank, world_size


def _barrier(world_size: int):
    if world_size > 1:
        dist.barrier()


# ---------------------------------------------------------------------------
# State serialisation / merge
# ---------------------------------------------------------------------------

def _serialize_state(state: dict) -> dict:
    """Convert an observer state dict (OnlineStatsTrackers + Tensors) to a
    plain Python/tensor dict suitable for dist.all_gather_object."""
    out: dict = {}
    for layer_num, layer_state in state.items():
        out[layer_num] = {}
        for key, val in layer_state.items():
            if isinstance(val, OnlineStatsTracker):
                out[layer_num][key] = {
                    "__type__": "OnlineStatsTracker",
                    "mean": val.mean.cpu(),
                    "count": val.count.cpu(),
                    "shape": val.shape,
                    "count_shape": val.count_shape,
                    "dtype": val.dtype,
                }
            elif isinstance(val, torch.Tensor):
                out[layer_num][key] = val.cpu()
            else:
                out[layer_num][key] = val
    return out


def _merge_serialized_states(states: list[dict]) -> dict:
    """Merge N partial serialised observer states into one.

    - OnlineStatsTracker fields: merged via Welford's incremental update
      (equivalent to computing the weighted mean over all tokens seen by
      all ranks).
    - Plain tensor accumulators (counts, sums): element-wise sum across ranks.
    - max_activations: element-wise max across ranks.
    """
    if not states:
        raise ValueError("Empty states list")
    merged: dict = {}
    for layer_num in states[0]:
        merged[layer_num] = {}
        for key in states[0][layer_num]:
            val0 = states[0][layer_num][key]

            if isinstance(val0, dict) and val0.get("__type__") == "OnlineStatsTracker":
                # Initialise from rank-0 state.
                tracker = OnlineStatsTracker(
                    shape=val0["shape"],
                    count_shape=val0["count_shape"],
                    device=torch.device("cpu"),
                    dtype=val0["dtype"],
                )
                tracker.mean = val0["mean"].clone()
                tracker.count = val0["count"].clone()
                # Fold in each other rank's contribution.
                for s in states[1:]:
                    other = s[layer_num][key]
                    tracker.update(other["mean"], other["count"])
                merged[layer_num][key] = tracker

            elif isinstance(val0, torch.Tensor):
                if key == "max_activations":
                    merged[layer_num][key] = torch.stack(
                        [s[layer_num][key] for s in states]
                    ).max(dim=0).values
                else:
                    # Counts / sums: just add.
                    merged[layer_num][key] = sum(
                        s[layer_num][key] for s in states
                    )
            else:
                # Scalar — take from rank 0.
                merged[layer_num][key] = val0

    return merged


def _report_and_save(merged_state: dict, obs_file: pathlib.Path):
    """Convert merged state to the same format as observer.save_state() and
    write to disk."""
    report = {
        layer_num: {
            k: v.mean if isinstance(v, OnlineStatsTracker) else v
            for k, v in layer_state.items()
        }
        for layer_num, layer_state in merged_state.items()
    }
    obs_file.parent.mkdir(parents=True, exist_ok=True)
    with open(obs_file, "wb") as f:
        torch.save(report, f)
    logger.info("Observations saved → %s", obs_file)


# ---------------------------------------------------------------------------
# Observer setup (mirrors main._setup_observer)
# ---------------------------------------------------------------------------

def _setup_observer(model, obs_args) -> MoETransformerObserver:
    model_cls = model.__class__.__name__
    if model_cls not in OBSERVER_CONFIG_REGISTRY:
        raise ValueError(
            f"No observer config registered for '{model_cls}'. "
            f"Supported: {list(OBSERVER_CONFIG_REGISTRY.keys())}"
        )
    renormalize = (
        getattr(model.config, "norm_topk_prob", False)
        and obs_args.renormalize_router_weights
    )
    observer_config = OBSERVER_CONFIG_REGISTRY[model_cls](
        distance_measure="cosine",
        renormalize_router_weights=renormalize,
        record_pruning_metrics_only=obs_args.record_pruning_metrics_only,
    )
    return MoETransformerObserver(model=model, hook_config=observer_config)


# ---------------------------------------------------------------------------
# Directory helpers (mirrors main.py)
# ---------------------------------------------------------------------------

def _str_to_dir(s: str) -> str:
    return re.sub(r"[^\w\-_.]", "_", s)


def _results_dir(model_name: str, dataset_name: str) -> pathlib.Path:
    import hashlib
    model_clean = _str_to_dir(model_name.split("/")[-1])
    if "," in dataset_name:
        h = hashlib.md5(dataset_name.encode()).hexdigest()[:8]
        dataset_clean = f"composite_{h}"
    else:
        dataset_clean = _str_to_dir(dataset_name.split("/")[-1])
    return pathlib.Path("./artifacts") / model_clean / dataset_clean


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = HfArgumentParser((ModelArgs, DatasetArgs, ObserverArgs, ReapArgs))
    model_args, ds_args, obs_args, reap_args = parser.parse_args_into_dataclasses()

    rank, world_size = _init_dist()
    set_seed(reap_args.seed)

    # Determine output path.
    results_dir = _results_dir(model_args.model_name, ds_args.dataset_name)
    cat_dir = results_dir / _str_to_dir("all")
    obs_file = cat_dir / obs_args.output_file_name

    if obs_file.exists() and not obs_args.overwrite_observations:
        if rank == 0:
            logger.info("Obs file already exists at %s — nothing to do.", obs_file)
        if world_size > 1:
            dist.destroy_process_group()
        return

    # ---- Load model (each rank uses its node's GPUs via device_map="auto") --
    model_name = patched_model_map(model_args.model_name)
    logger.info("Rank %d loading model %s …", rank, model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )

    # ---- Load batches (all ranks load the same data; each uses its slice) ----
    all_batches_dict = load_category_batches(
        dataset_name=ds_args.dataset_name,
        split=ds_args.split,
        subset=ds_args.dataset_config_name,
        tokenizer=tokenizer,
        model_max_length=obs_args.model_max_length,
        split_by_category=False,
        return_vllm_tokens_prompt=obs_args.return_vllm_tokens_prompt,
        truncate=obs_args.truncate,
        batches_per_category=obs_args.batches_per_category,
        batch_size=obs_args.batch_size,
    )
    all_batches = all_batches_dict["all"]

    # Assign this rank's subset.
    n_total = len(all_batches)
    per_rank = (n_total + world_size - 1) // world_size
    my_batches = all_batches[rank * per_rank : (rank + 1) * per_rank]
    logger.info(
        "Rank %d/%d processing batches %d–%d (%d of %d total)",
        rank, world_size,
        rank * per_rank,
        min((rank + 1) * per_rank, n_total) - 1,
        len(my_batches),
        n_total,
    )

    # ---- Warm-up pass (flushes CUDA caches, avoids OOM surprise on first batch)
    observer = _setup_observer(model, obs_args)
    with torch.no_grad():
        s = "hello " * obs_args.model_max_length
        tok = tokenizer(
            [s],
            return_tensors="pt",
            truncation=True,
            max_length=obs_args.model_max_length,
        )
        tok = {k: v.to(model.device) for k, v in tok.items()}
        _ = model(**tok)
    observer.reset()
    logger.info("Rank %d warm-up done.", rank)

    # ---- Observer forward passes -------------------------------------------
    with torch.no_grad():
        for sample in tqdm(
            my_batches,
            desc=f"[rank {rank}] observer",
            position=rank,
            leave=True,
        ):
            attention_mask = sample.get("attention_mask", None)
            sample_gpu = {
                k: v.to(model.device) if torch.is_tensor(v) else v
                for k, v in sample.items()
            }
            with observer.set_attention_mask(attention_mask):
                model(**sample_gpu)

    # Serialise partial state before freeing the model.
    partial = _serialize_state(observer.state)
    observer.close_hooks()
    del model
    torch.cuda.empty_cache()

    logger.info("Rank %d finished observer, waiting for peers …", rank)
    _barrier(world_size)

    # ---- Gather partial states to rank 0 -----------------------------------
    if world_size > 1:
        all_partials: list = [None] * world_size
        dist.all_gather_object(all_partials, partial)
    else:
        all_partials = [partial]

    if rank == 0:
        logger.info("Merging %d partial states …", world_size)
        merged = _merge_serialized_states(all_partials)
        _report_and_save(merged, obs_file)
        logger.info("Done. Obs file: %s", obs_file)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
