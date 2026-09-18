"""Router-weighted Fisher geometry for clustering MoE experts.

The full Fisher--Rao geodesic between every pair of experts would require
``O(num_experts**2 * num_quadrature_nodes)`` full-model backward passes per
layer.  That is not practical for a 128-expert, 48-layer model.  This module
therefore implements the clustering approximation used by the runnable method:

* use the endpoint-average diagonal Fisher metric for pairwise clustering;
* condition each expert's metric by its corpus router/activation importance;
* estimate parameter-space distances with a deterministic coordinate sketch;
* robustly normalise and combine Fisher, router, and activation distances.

The subsequent cluster merge can still use the path-integrated Gauss--Legendre
fixed-point implementation in :mod:`reap.fisher_gl_v27`.  Consequently, this
module deliberately calls the pairwise quantity an *approximated local
Fisher--Rao distance*, not an exact Fisher geodesic distance.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from reap.fisher_v27 import _PARAM_NAMES, _expert_linears

logger = logging.getLogger(__name__)

_EPS = torch.finfo(torch.float32).eps
_PROJECTION_SEED_OFFSETS = {
    "gate_proj": 0,
    "up_proj": 10_007,
    "down_proj": 20_011,
}


def router_importance_from_observer(
    layer_data: dict[str, Any],
    source: str = "reap",
) -> torch.Tensor:
    """Return a positive, mean-one importance vector for one layer.

    ``source='reap'`` uses ``weighted_ean_sum / total_tokens`` when available.
    This is the observer's direct corpus estimate of
    ``E[1_routed * router_weight * ||expert_output||]``.  Older caches fall back
    to the reported conditional REAP mean multiplied by routing frequency.

    The mean-one normalisation preserves relative importance while avoiding an
    arbitrary global scale in the Fisher metric.
    """
    valid_sources = {"none", "frequency", "router", "reap"}
    if source not in valid_sources:
        raise ValueError(
            f"Unknown Fisher router-weight source '{source}'. "
            f"Expected one of {sorted(valid_sources)}."
        )

    frequency = torch.as_tensor(layer_data["expert_frequency"]).float().cpu()
    if source == "none":
        return torch.ones_like(frequency)

    total_tokens = float(torch.as_tensor(layer_data["total_tokens"]).item())
    total_tokens = max(total_tokens, 1.0)

    if source == "frequency":
        raw = frequency / total_tokens
    elif source == "router":
        if "weighted_expert_frequency_sum" in layer_data:
            raw = (
                torch.as_tensor(layer_data["weighted_expert_frequency_sum"])
                .float()
                .cpu()
                / total_tokens
            )
        else:
            logger.warning(
                "Observer cache lacks weighted_expert_frequency_sum; "
                "falling back to expert frequency for router importance."
            )
            raw = frequency / total_tokens
    else:  # source == "reap"
        if "weighted_ean_sum" in layer_data:
            raw = (
                torch.as_tensor(layer_data["weighted_ean_sum"])
                .float()
                .cpu()
                / total_tokens
            )
        elif "reap" in layer_data:
            conditional_reap = torch.as_tensor(layer_data["reap"]).float().cpu()
            raw = conditional_reap * frequency / total_tokens
        else:
            raise KeyError(
                "Observer cache has neither weighted_ean_sum nor reap; "
                "cannot construct REAP router importance."
            )

    raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    mean = raw.mean()
    if not torch.isfinite(mean) or mean <= 0:
        logger.warning(
            "Router importance is empty/non-finite; using uniform importance."
        )
        return torch.ones_like(frequency)
    return (raw / mean).clamp_min(_EPS)


def _sample_columns(
    width: int,
    sketch_size: int,
    seed: int,
) -> tuple[torch.Tensor, float]:
    """Select deterministic weight columns and return their Horvitz scale."""
    if sketch_size <= 0 or sketch_size >= width:
        return torch.arange(width, dtype=torch.long), 1.0
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    columns = torch.randperm(width, generator=generator)[:sketch_size]
    columns, _ = torch.sort(columns)
    return columns, float(width) / float(sketch_size)


def _sample_weight_columns(weight: torch.Tensor, columns: torch.Tensor) -> torch.Tensor:
    """Copy selected columns from an arbitrary model device to CPU float32."""
    device_columns = columns.to(weight.device)
    return weight.detach().index_select(1, device_columns).float().cpu()


def _sample_fisher_columns(
    fisher: torch.Tensor,
    columns: torch.Tensor,
    out_features: int,
    sampled_width: int,
) -> torch.Tensor:
    """Expand row Fisher or sample an element-wise Fisher tensor."""
    fisher = fisher.detach().float().cpu()
    if fisher.ndim == 1:
        if fisher.shape[0] != out_features:
            raise ValueError(
                f"Per-unit Fisher shape {tuple(fisher.shape)} does not match "
                f"projection output width {out_features}."
            )
        return fisher[:, None].expand(out_features, sampled_width)
    if fisher.ndim == 2:
        if fisher.shape[0] != out_features:
            raise ValueError(
                f"Element-wise Fisher shape {tuple(fisher.shape)} does not match "
                f"projection output width {out_features}."
            )
        return fisher.index_select(1, columns)
    raise ValueError(
        f"Expected per-unit [out] or element-wise [out, in] Fisher, got "
        f"shape {tuple(fisher.shape)}."
    )


@torch.no_grad()
def pairwise_local_fisher_distance(
    experts: nn.ModuleList | list[nn.Module],
    expert_fisher: dict[int, dict[str, torch.Tensor]],
    model_attrs: dict[str, Any],
    *,
    router_importance: torch.Tensor | None = None,
    sketch_size: int = 32,
    seed: int = 42,
) -> torch.Tensor:
    r"""Compute the endpoint-average local Fisher distance between all experts.

    For experts ``i`` and ``j`` this estimates

    .. math::

        d^2(i,j) = \tfrac12 \sum_k (F_{i,k}+F_{j,k})
                   (\theta_{i,k}-\theta_{j,k})^2.

    If ``sketch_size`` is smaller than a projection's input width, the same
    deterministic subset of input coordinates is sampled for every expert and
    scaled by ``input_width / sketch_size``.  This is an unbiased coordinate
    estimate of the squared distance and avoids materialising every pairwise
    full-parameter difference.  Set ``sketch_size <= 0`` for the exact local
    metric when memory permits.
    """
    num_experts = len(experts)
    if num_experts < 1:
        raise ValueError("At least one expert is required.")
    missing = sorted(set(range(num_experts)) - set(expert_fisher))
    if missing:
        raise KeyError(f"Fisher cache is missing expert indices: {missing}")

    if router_importance is None:
        importance = torch.ones(num_experts, dtype=torch.float32)
    else:
        importance = torch.as_tensor(router_importance).float().cpu()
        if tuple(importance.shape) != (num_experts,):
            raise ValueError(
                f"router_importance shape {tuple(importance.shape)}; expected "
                f"({num_experts},)."
            )
        importance = torch.nan_to_num(
            importance, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp_min(_EPS)

    distance_sq = torch.zeros((num_experts, num_experts), dtype=torch.float64)

    for param_name in _PARAM_NAMES:
        linears = [
            _expert_linears(expert, model_attrs).get(param_name) for expert in experts
        ]
        if any(linear is None for linear in linears):
            continue
        first = linears[0]
        out_features, in_features = first.weight.shape
        if any(
            tuple(linear.weight.shape) != (out_features, in_features)
            for linear in linears
        ):
            raise ValueError(f"Experts have inconsistent {param_name} shapes.")

        columns, coordinate_scale = _sample_columns(
            in_features,
            sketch_size,
            seed + _PROJECTION_SEED_OFFSETS[param_name],
        )
        sampled_width = int(columns.numel())
        weights = torch.stack(
            [_sample_weight_columns(linear.weight, columns) for linear in linears],
            dim=0,
        )  # [N, out, sampled_in]
        fishers = torch.stack(
            [
                _sample_fisher_columns(
                    expert_fisher[idx][param_name],
                    columns,
                    out_features,
                    sampled_width,
                )
                for idx in range(num_experts)
            ],
            dim=0,
        )
        fishers = fishers * importance[:, None, None]

        # Efficient variable-metric pairwise expansion.  For one-sided metric i:
        # M_ij = sum_k F_i,k (W_i,k - W_j,k)^2
        #      = A_i + B_ij - 2 C_ij.
        x = weights.flatten(1)
        metric = fishers.flatten(1)
        x_sq = x.square()
        a = (metric * x_sq).sum(dim=1)
        b = metric @ x_sq.T
        c = (metric * x) @ x.T
        one_sided = a[:, None] + b - 2.0 * c
        symmetric = 0.5 * (one_sided + one_sided.T)
        distance_sq.add_(
            symmetric.clamp_min(0.0).to(torch.float64),
            alpha=coordinate_scale,
        )

        del weights, fishers, x, metric, x_sq, a, b, c, one_sided, symmetric

    distance = distance_sq.clamp_min(0.0).sqrt().to(torch.float32)
    distance = 0.5 * (distance + distance.T)
    distance.fill_diagonal_(0.0)
    return distance


def _validate_distance_matrix(distance: torch.Tensor, name: str) -> torch.Tensor:
    distance = torch.as_tensor(distance).detach().float().cpu().clone()
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got {tuple(distance.shape)}.")
    distance = torch.nan_to_num(distance, nan=0.0, posinf=0.0, neginf=0.0)
    distance = 0.5 * (distance + distance.T)
    distance.clamp_min_(0.0)
    distance.fill_diagonal_(0.0)
    return distance


def normalise_distance_matrix(
    distance: torch.Tensor,
    name: str,
) -> tuple[torch.Tensor, float]:
    """Median-normalise positive off-diagonal entries of a distance matrix."""
    distance = _validate_distance_matrix(distance, name)
    n = distance.shape[0]
    off_diagonal = distance[~torch.eye(n, dtype=torch.bool)]
    positive = off_diagonal[off_diagonal > 0]
    if positive.numel() == 0:
        logger.warning(
            "%s distance is identically zero; component contributes 0.", name
        )
        return distance, 1.0
    scale = float(positive.median())
    return distance / max(scale, _EPS), scale


def _cosine_distance_from_signatures(signatures: torch.Tensor) -> torch.Tensor:
    signatures = torch.as_tensor(signatures).detach().float().cpu()
    signatures = F.normalize(signatures, p=2, dim=-1, eps=_EPS)
    distance = 1.0 - signatures @ signatures.T
    return _validate_distance_matrix(distance, "signature cosine")


def router_distance_from_observer(layer_data: dict[str, Any]) -> tuple[torch.Tensor, str]:
    """Get router geometry, preferring exact probability L2 from new caches."""
    if "router_weight_l2_distance" in layer_data:
        return (
            _validate_distance_matrix(
                layer_data["router_weight_l2_distance"], "router weight L2"
            ),
            "router_weight_l2_distance",
        )
    if "router_logit_similiarity" in layer_data:
        logger.warning(
            "Legacy observer cache lacks router_weight_l2_distance; using its "
            "mean pairwise router-logit cosine distance."
        )
        return (
            _validate_distance_matrix(
                layer_data["router_logit_similiarity"], "router logit cosine"
            ),
            "router_logit_similiarity",
        )
    if "router_logits_signatures" in layer_data:
        return (
            _cosine_distance_from_signatures(layer_data["router_logits_signatures"]),
            "router_logits_signatures",
        )
    raise KeyError("Observer cache contains no usable router-distance statistic.")


def activation_distance_from_observer(
    layer_data: dict[str, Any],
) -> tuple[torch.Tensor, str]:
    """Get the best available persisted activation-space distance."""
    if "online_characteristic_activation_dist" in layer_data:
        return (
            _validate_distance_matrix(
                layer_data["online_characteristic_activation_dist"],
                "activation distance",
            ),
            "online_characteristic_activation_dist",
        )
    if "ttm_similarity_matrix" in layer_data:
        return (
            _validate_distance_matrix(
                layer_data["ttm_similarity_matrix"], "routed activation distance"
            ),
            "ttm_similarity_matrix",
        )
    if "characteristic_activation" in layer_data:
        return (
            _cosine_distance_from_signatures(layer_data["characteristic_activation"]),
            "characteristic_activation",
        )
    raise KeyError("Observer cache contains no usable activation-distance statistic.")


def compose_moe_distance(
    fisher_distance: torch.Tensor,
    *,
    router_distance: torch.Tensor | None = None,
    activation_distance: torch.Tensor | None = None,
    fisher_weight: float = 1.0,
    router_weight: float = 0.0,
    activation_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Robustly normalise and combine Fisher/router/activation geometry."""
    weights = {
        "fisher": float(fisher_weight),
        "router": float(router_weight),
        "activation": float(activation_weight),
    }
    if any(value < 0 for value in weights.values()):
        raise ValueError(f"Distance weights must be non-negative, got {weights}.")
    if sum(weights.values()) <= 0:
        raise ValueError("At least one distance weight must be positive.")

    fisher_norm, fisher_scale = normalise_distance_matrix(
        fisher_distance, "Fisher"
    )
    total = fisher_weight * fisher_norm
    scales = {"fisher": fisher_scale}

    if router_weight > 0:
        if router_distance is None:
            raise ValueError("router_weight > 0 requires a router distance matrix.")
        router_norm, router_scale = normalise_distance_matrix(
            router_distance, "router"
        )
        if router_norm.shape != total.shape:
            raise ValueError("Router and Fisher distance shapes do not match.")
        total = total + router_weight * router_norm
        scales["router"] = router_scale

    if activation_weight > 0:
        if activation_distance is None:
            raise ValueError(
                "activation_weight > 0 requires an activation distance matrix."
            )
        activation_norm, activation_scale = normalise_distance_matrix(
            activation_distance, "activation"
        )
        if activation_norm.shape != total.shape:
            raise ValueError("Activation and Fisher distance shapes do not match.")
        total = total + activation_weight * activation_norm
        scales["activation"] = activation_scale

    total = _validate_distance_matrix(total, "combined MoE")
    return total, scales


@torch.no_grad()
def build_fisher_geodesic_distance_matrices(
    model: nn.Module,
    observer_data: dict[int, dict[str, Any]],
    expert_fisher: dict[int, dict[int, dict[str, torch.Tensor]]],
    model_attrs: dict[str, Any],
    *,
    importance_source: str = "reap",
    fisher_weight: float = 1.0,
    router_weight: float = 0.25,
    activation_weight: float = 0.25,
    sketch_size: int = 32,
    seed: int = 42,
) -> tuple[dict[int, torch.Tensor], dict[int, dict[str, Any]]]:
    """Build combined router-weighted Fisher distance matrices for all layers."""
    distances: dict[int, torch.Tensor] = {}
    diagnostics: dict[int, dict[str, Any]] = {}
    layers = model.model.layers

    for layer_idx in sorted(observer_data):
        if layer_idx not in expert_fisher:
            raise KeyError(f"Fisher cache is missing layer {layer_idx}.")
        layer_data = observer_data[layer_idx]
        importance = router_importance_from_observer(layer_data, importance_source)
        moe = getattr(layers[layer_idx], model_attrs["moe_block"])
        experts = getattr(moe, model_attrs["experts"])

        logger.info(
            "[Fisher-cluster] Layer %d: constructing %dx%d distance matrix "
            "(sketch=%s, importance=%s).",
            layer_idx,
            len(experts),
            len(experts),
            "exact" if sketch_size <= 0 else sketch_size,
            importance_source,
        )
        fisher_distance = pairwise_local_fisher_distance(
            experts,
            expert_fisher[layer_idx],
            model_attrs,
            router_importance=importance,
            sketch_size=sketch_size,
            seed=seed + 100_003 * layer_idx,
        )

        router_distance = None
        router_source = None
        if router_weight > 0:
            router_distance, router_source = router_distance_from_observer(layer_data)
        activation_distance = None
        activation_source = None
        if activation_weight > 0:
            activation_distance, activation_source = activation_distance_from_observer(
                layer_data
            )

        combined, scales = compose_moe_distance(
            fisher_distance,
            router_distance=router_distance,
            activation_distance=activation_distance,
            fisher_weight=fisher_weight,
            router_weight=router_weight,
            activation_weight=activation_weight,
        )
        distances[layer_idx] = combined
        diagnostics[layer_idx] = {
            "normalisation_scales": scales,
            "router_source": router_source,
            "activation_source": activation_source,
            "importance": importance,
            "fisher_distance": fisher_distance,
            "router_distance": router_distance,
            "activation_distance": activation_distance,
        }

    return distances, diagnostics