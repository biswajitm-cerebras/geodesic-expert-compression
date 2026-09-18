"""
Gauss-Legendre *path-integrated* diagonal Fisher for the v27 expert merge.

Port of the `lws_GL_diag_v27` checkpoint-merge recipe to the MoE-expert setting.

Background
----------
The endpoint Fisher in :mod:`reap.fisher_v27` evaluates the per-unit diagonal Fisher of
each expert *at its own weights* ``W_i`` and merges once. The original v27 recipe instead
**path-integrates** the Fisher along the geodesic between each endpoint and the running
merged state, using a 3-point Gauss-Legendre quadrature on ``[0, 1]``, wrapped in a Picard
fixed-point iteration:

    for k in 1..K:                                   # GL fixed-point iterations
        for each endpoint i:
            gamma_i(t) = (1 - t) * W_i + t * merged  # geodesic node interpolant
            F_hat_i(u) = sum_n w_n * F(gamma_i(t_n)) # GL quadrature of the Fisher
        merged[u]   = sum_i F_hat_i(u) * W_i[u] / sum_i F_hat_i(u)

The 3-point GL nodes/weights on ``[0, 1]`` are ``t = 0.5 +/- 0.5*sqrt(3/5), 0.5`` with
weights ``5/18, 8/18, 5/18`` (i.e. Legendre-Gauss on ``[-1, 1]`` remapped, weights halved
so they sum to 1).

MoE mapping
-----------
For an M-SMoE cluster ``c`` with experts ``{W_i}`` the "merged" target is the cluster's
shared expert ``theta*_c``. Expert ``i``'s geodesic path is ``gamma_i(t) = (1-t) W_i +
t theta*_c``. To evaluate ``F(gamma_i(t_n))`` for *all* experts in one backward pass we set
every clustered expert to its node-``t_n`` interpolant simultaneously and reuse the
per-expert backprop in :func:`reap.fisher_v27.collect_expert_unit_fisher` (each expert's
gradient is independent given routing, so one pass yields Fisher for all experts at that
node). The merge itself reuses :func:`reap.main.merge` (dominant selection, permutation,
tying) with the GL-averaged Fisher and the *original* expert weights.

At ``n_nodes == 1`` (single GL node at ``t = 0``) with ``n_iters == 0`` this reduces
exactly to the endpoint Fisher of :mod:`reap.fisher_v27`.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from reap.fisher_v27 import F_FLOOR, _PARAM_NAMES, _expert_linears

logger = logging.getLogger(__name__)


def gauss_legendre_nodes_01(n_nodes: int) -> List[Tuple[float, float]]:
    """Return ``[(t_n, w_n), ...]`` for an ``n_nodes``-point Gauss-Legendre rule on ``[0, 1]``.

    Weights are normalised to sum to 1 (Legendre-Gauss weights on ``[-1, 1]`` sum to 2).
    For ``n_nodes == 1`` this returns the single node ``(0.0, 1.0)`` (endpoint), so that
    the GL Fisher reduces exactly to the endpoint Fisher.
    """
    if n_nodes < 1:
        raise ValueError(f"n_nodes must be >= 1, got {n_nodes}")
    if n_nodes == 1:
        # Endpoint node (t=0) rather than the GL midpoint, so 1-node GL == endpoint Fisher.
        return [(0.0, 1.0)]
    x, w = np.polynomial.legendre.leggauss(n_nodes)  # nodes/weights on [-1, 1]
    t = 0.5 * (x + 1.0)  # remap to [0, 1]
    wt = 0.5 * w  # weights now sum to 1
    return [(float(ti), float(wi)) for ti, wi in zip(t, wt)]


def clustered_expert_keys(
    cluster_labels: Dict[int, torch.Tensor],
) -> Dict[int, List[int]]:
    """Return ``{layer_idx: [expert_idx, ...]}`` for experts in clusters of size > 1.

    Singleton clusters are untouched by the merge, so we neither snapshot nor perturb them.
    """
    out: Dict[int, List[int]] = {}
    for layer_idx, labels in cluster_labels.items():
        labels_t = labels if torch.is_tensor(labels) else torch.as_tensor(labels)
        keep: List[int] = []
        for cid in labels_t.unique():
            members = torch.where(labels_t == cid)[0].tolist()
            if len(members) > 1:
                keep.extend(members)
        if keep:
            out[layer_idx] = sorted(keep)
    return out


def snapshot_expert_weights(
    model: nn.Module,
    keys: Dict[int, List[int]],
    model_attrs: Dict[str, Any],
) -> Dict[Tuple[int, int, str], torch.Tensor]:
    """Clone the clustered experts' projection weights to CPU float32.

    Returns ``{(layer_idx, expert_idx, param_name): Tensor}``.
    """
    layers = model.model.layers
    moe_attr = model_attrs["moe_block"]
    experts_attr = model_attrs["experts"]
    snap: Dict[Tuple[int, int, str], torch.Tensor] = {}
    for layer_idx, expert_indices in keys.items():
        experts = getattr(getattr(layers[layer_idx], moe_attr), experts_attr)
        for expert_idx in expert_indices:
            for pname, lin in _expert_linears(experts[expert_idx], model_attrs).items():
                snap[(layer_idx, expert_idx, pname)] = (
                    lin.weight.detach().float().cpu().clone()
                )
    return snap


@torch.no_grad()
def set_expert_interpolants(
    model: nn.Module,
    originals: Dict[Tuple[int, int, str], torch.Tensor],
    merged: Dict[Tuple[int, int, str], torch.Tensor],
    t: float,
    keys: Dict[int, List[int]],
    model_attrs: Dict[str, Any],
) -> None:
    """Set each clustered expert weight to ``gamma_i(t) = (1 - t) * original + t * merged``.

    Writes into the live model (dtype/device of the target weight preserved).
    """
    layers = model.model.layers
    moe_attr = model_attrs["moe_block"]
    experts_attr = model_attrs["experts"]
    for layer_idx, expert_indices in keys.items():
        experts = getattr(getattr(layers[layer_idx], moe_attr), experts_attr)
        for expert_idx in expert_indices:
            for pname, lin in _expert_linears(experts[expert_idx], model_attrs).items():
                key = (layer_idx, expert_idx, pname)
                w0 = originals[key]
                w1 = merged[key]
                interp = (1.0 - t) * w0 + t * w1
                lin.weight.data.copy_(
                    interp.to(dtype=lin.weight.dtype, device=lin.weight.device)
                )


@torch.no_grad()
def restore_expert_weights(
    model: nn.Module,
    weights: Dict[Tuple[int, int, str], torch.Tensor],
    model_attrs: Dict[str, Any],
) -> None:
    """Write ``weights`` (from :func:`snapshot_expert_weights`) back into the live model."""
    layers = model.model.layers
    moe_attr = model_attrs["moe_block"]
    experts_attr = model_attrs["experts"]
    # Group keys by layer for efficient module access.
    by_layer: Dict[int, List[Tuple[int, str]]] = {}
    for (layer_idx, expert_idx, pname) in weights:
        by_layer.setdefault(layer_idx, []).append((expert_idx, pname))
    for layer_idx, items in by_layer.items():
        experts = getattr(getattr(layers[layer_idx], moe_attr), experts_attr)
        for expert_idx, pname in items:
            lin = _expert_linears(experts[expert_idx], model_attrs)[pname]
            w = weights[(layer_idx, expert_idx, pname)]
            lin.weight.data.copy_(w.to(dtype=lin.weight.dtype, device=lin.weight.device))


def gl_accumulate_fisher(
    accum: Dict[int, Dict[int, Dict[str, torch.Tensor]]] | None,
    node_fisher: Dict[int, Dict[int, Dict[str, torch.Tensor]]],
    weight: float,
) -> Dict[int, Dict[int, Dict[str, torch.Tensor]]]:
    """In-place GL accumulation: ``accum += weight * node_fisher`` (nested dict form).

    ``node_fisher`` has the shape returned by ``collect_expert_unit_fisher``:
    ``{layer_idx: {expert_idx: {param_name: Tensor[out_units]}}}``. Returns ``accum``.
    """
    if accum is None:
        accum = {}
    for layer_idx, experts in node_fisher.items():
        acc_l = accum.setdefault(layer_idx, {})
        for expert_idx, params in experts.items():
            acc_e = acc_l.setdefault(expert_idx, {})
            for pname, t in params.items():
                contrib = weight * t
                if pname in acc_e:
                    acc_e[pname] = acc_e[pname] + contrib
                else:
                    acc_e[pname] = contrib.clone()
    return accum


def floor_fisher(
    fisher: Dict[int, Dict[int, Dict[str, torch.Tensor]]],
) -> Dict[int, Dict[int, Dict[str, torch.Tensor]]]:
    """Floor all Fisher values at ``F_FLOOR`` (guards against zero denominators in merge)."""
    for experts in fisher.values():
        for params in experts.values():
            for pname in params:
                params[pname] = params[pname].clamp_min(F_FLOOR)
    return fisher


@torch.no_grad()
def cluster_relative_change(
    prev: Dict[Tuple[int, int, str], torch.Tensor],
    curr: Dict[Tuple[int, int, str], torch.Tensor],
) -> float:
    """Relative Frobenius change ``||curr - prev|| / ||prev||`` over all clustered experts.

    Used as the Picard fixed-point convergence criterion (mirrors v27's ``rel_change``).
    """
    num = 0.0
    den = 0.0
    for key, p in prev.items():
        c = curr[key]
        num += float((c - p).pow(2).sum())
        den += float(p.pow(2).sum())
    if den <= 0.0:
        return 0.0
    return (num ** 0.5) / (den ** 0.5)
