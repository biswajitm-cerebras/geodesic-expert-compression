"""
Per-expert diagonal-Fisher collection for the v27 Fisher-Geodesic expert merge.

This mirrors the unit-wise diagonal Fisher used in the model_merge `lws_GL_diag_v27`
checkpoint-merge recipe, but computes it *per MoE expert* so that experts inside an
M-SMoE cluster can be blended with a per-unit Fisher-weighted rule instead of a plain
frequency-weighted average.

For a single expert projection with weight ``W`` of shape ``[out, in]`` we define the
per-output-unit (per-neuron) scalar Fisher as

    F[u] = mean_batch( mean_in( (dL/dW[u, :])^2 ) )

i.e. the mean squared gradient of the causal-LM loss w.r.t. the row of ``W`` feeding
output neuron ``u``, averaged over calibration batches. This is exactly the reduction
used in v27's ``collect_unit_fisher`` (``g2.view(n_units, usz).mean(dim=1)``) applied to
each expert Linear separately.

Fisher values are floored and L1-normalised **per (layer, expert)** so that the blend
weights ``w_i(u) = F_i(u) / sum_j F_j(u)`` are comparable across the experts of a
cluster, matching v27's per-model L1 normalisation.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger(__name__)

F_FLOOR = 1e-10
# Element-wise Fisher is stored in fp16 to keep the full [out, in] tensors
# affordable on host RAM (~58 GB vs ~116 GB for the Qwen3-30B expert set).
# fp16 cannot represent F_FLOOR (1e-10 underflows to 0), so the element-wise
# path uses a larger, fp16-representable floor that still keeps every expert's
# blend weight strictly positive (guarding the merge denominator).
F_FLOOR_EW = 1e-4
_PARAM_NAMES = ("gate_proj", "up_proj", "down_proj")


def _expert_linears(expert: nn.Module, model_attrs: Dict[str, Any]) -> Dict[str, nn.Linear]:
    """Return {logical_param_name: nn.Linear} for one expert."""
    out = {}
    for pname in _PARAM_NAMES:
        attr = model_attrs[pname]
        mod = getattr(expert, attr, None)
        if mod is not None:
            out[pname] = mod
    return out


@torch.enable_grad()
def collect_expert_unit_fisher(
    model: nn.Module,
    calibration_batches: List[Dict[str, torch.Tensor]],
    model_attrs: Dict[str, Any],
    sparse_layer_indices: List[int],
    *,
    cache_path: Optional[str] = None,
    max_batches: Optional[int] = None,
    l1_normalize: bool = True,
    overwrite: bool = False,
) -> Dict[int, Dict[int, Dict[str, torch.Tensor]]]:
    """Collect per-expert, per-unit diagonal Fisher.

    Returns a nested dict ``fisher[layer_idx][expert_idx][param_name] -> Tensor[out_units]``
    on CPU (float32), floored and (optionally) L1-normalised per (layer, expert).
    """
    # Weight-wise (element-wise) variant: keep the full [out, in] diagonal Fisher
    # for every weight element instead of reducing each output row to one scalar.
    elementwise = os.environ.get("FISHER_ELEMENTWISE", "0") not in ("0", "", "false", "False")
    if elementwise and cache_path is not None:
        _root, _ext = os.path.splitext(cache_path)
        cache_path = f"{_root}_ew{_ext}"

    if cache_path is not None and os.path.exists(cache_path) and not overwrite:
        logger.info(f"[v27-Fisher] Loading cached Fisher: {cache_path}")
        return torch.load(cache_path, weights_only=False)

    model.eval()  # no dropout, deterministic (matches v27)

    # Freeze everything, then enable grad only on expert projection weights to
    # bound memory (grads are only allocated for the expert Linears we care about).
    for p in model.parameters():
        p.requires_grad_(False)

    # Map: id(weight) -> (layer_idx, expert_idx, param_name)
    param_index: Dict[int, tuple] = {}
    # Accumulators live on the same device as their weight (device_map="auto").
    accum: Dict[tuple, torch.Tensor] = {}

    layers = model.model.layers
    moe_attr = model_attrs["moe_block"]
    experts_attr = model_attrs["experts"]

    n_tracked = 0
    for layer_idx in sparse_layer_indices:
        moe = getattr(layers[layer_idx], moe_attr)
        experts = getattr(moe, experts_attr)
        for expert_idx, expert in enumerate(experts):
            for pname, lin in _expert_linears(expert, model_attrs).items():
                w = lin.weight
                w.requires_grad_(True)
                key = (layer_idx, expert_idx, pname)
                param_index[id(w)] = key
                accum[key] = torch.zeros(
                    (w.shape[0], w.shape[1]) if elementwise else (w.shape[0],),
                    device=w.device,
                    dtype=torch.float32,
                )
                n_tracked += 1

    logger.info(
        f"[v27-Fisher] Tracking {n_tracked} expert projections across "
        f"{len(sparse_layer_indices)} layers."
    )

    # Keep a stable list of (weight, key) to iterate grads after each backward.
    tracked_weights = []
    for layer_idx in sparse_layer_indices:
        moe = getattr(layers[layer_idx], moe_attr)
        experts = getattr(moe, experts_attr)
        for expert_idx, expert in enumerate(experts):
            for pname, lin in _expert_linears(expert, model_attrs).items():
                tracked_weights.append((lin.weight, (layer_idx, expert_idx, pname)))

    batches = calibration_batches
    if max_batches is not None:
        batches = batches[:max_batches]

    # Distributed data-parallel Fisher: each rank processes a disjoint shard of
    # batches, then all-reduces the squared-gradient accumulators via Gloo (CPU).
    _dist_ok = torch.distributed.is_available() and torch.distributed.is_initialized()
    _rank = torch.distributed.get_rank() if _dist_ok else 0
    _world = torch.distributed.get_world_size() if _dist_ok else 1
    if _world > 1:
        batches = batches[_rank::_world]
        logger.info(
            f"[v27-Fisher] Rank {_rank}/{_world}: shard = {len(batches)} batches."
        )

    n_batches = 0

    for sample in tqdm(batches, desc=f"[v27-Fisher] backprop rank={_rank}"):
        sample = {
            k: (v.to(model.device) if torch.is_tensor(v) else v)
            for k, v in sample.items()
        }
        input_ids = sample.get("input_ids")
        if input_ids is None:
            continue
        labels = input_ids.clone()
        attn = sample.get("attention_mask", None)
        if attn is not None:
            labels = labels.masked_fill(attn == 0, -100)

        model.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        loss = out.loss
        if loss is None or not torch.isfinite(loss):
            logger.warning("[v27-Fisher] Non-finite loss on a batch; skipping.")
            continue
        loss.backward()

        # Accumulate per-unit squared gradients (mean over the input dimension).
        for w, key in tracked_weights:
            g = w.grad
            if g is None:
                continue
            # g: [out, in]. Element-wise keeps the full diagonal Fisher [out, in];
            # per-unit reduces each output row to one scalar via mean over the inputs.
            g2 = g.detach().float().pow(2)
            accum[key].add_(g2 if elementwise else g2.mean(dim=1))
        n_batches += 1

    model.zero_grad(set_to_none=True)
    if n_batches == 0:
        raise RuntimeError("[v27-Fisher] No valid calibration batches processed.")

    # All-reduce squared-gradient accumulators and batch count across distributed ranks.
    if _dist_ok and _world > 1:
        logger.info(
            f"[v27-Fisher] Rank {_rank}: all_reduce {len(accum)} accumulators."
        )
        for key in list(accum.keys()):
            t_cpu = accum[key].cpu()
            torch.distributed.all_reduce(t_cpu, op=torch.distributed.ReduceOp.SUM)
            accum[key] = t_cpu
        n_batches_t = torch.tensor(n_batches, dtype=torch.long)
        torch.distributed.all_reduce(n_batches_t, op=torch.distributed.ReduceOp.SUM)
        n_batches = int(n_batches_t.item())

    logger.info(f"[v27-Fisher] Accumulated over {n_batches} batches. Reducing...")

    # Finalise: mean over batches, floor, move to CPU, group by (layer, expert).
    fisher: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}
    for (layer_idx, expert_idx, pname), t in accum.items():
        t = (t / n_batches).clamp_min(F_FLOOR).cpu()
        fisher.setdefault(layer_idx, {}).setdefault(expert_idx, {})[pname] = t

    if l1_normalize:
        for layer_idx, experts in fisher.items():
            for expert_idx, params in experts.items():
                # Normalise across all of this expert's units (all 3 matrices) so
                # per-unit weights are comparable across experts of a cluster.
                all_vals = torch.cat([v.flatten() for v in params.values()])
                mean_f = all_vals.mean().item()
                if mean_f > 0:
                    for pname in params:
                        params[pname] = (params[pname] / mean_f).clamp_min(F_FLOOR)

    # Downcast element-wise Fisher to fp16 *after* the fp32 normalisation: raw
    # squared gradients can exceed the fp16 range, but the L1-normalised values
    # are O(1) and safe. Re-floor at the fp16-representable F_FLOOR_EW so no
    # expert's blend weight underflows to exactly 0.
    # IMPORTANT: clamp to fp16 max BEFORE the cast. L1-normalised values are
    # typically O(1) but can spike large for very active weight elements. Without
    # the clamp, values > 65504 become Inf in fp16, and Inf/Inf in the Fw
    # normalisation produces NaN in the merged weights.
    if elementwise:
        fp16_max = torch.finfo(torch.float16).max  # 65504
        for experts in fisher.values():
            for params in experts.values():
                for pname in params:
                    params[pname] = (
                        params[pname].clamp(max=fp16_max).to(torch.float16).clamp_min(F_FLOOR_EW)
                    )

    if cache_path is not None and _rank == 0:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(fisher, cache_path)
        logger.info(f"[v27-Fisher] Saved Fisher cache -> {cache_path}")

    # Restore grad flags so downstream merge (torch.no_grad) is clean.
    for p in model.parameters():
        p.requires_grad_(False)

    return fisher
