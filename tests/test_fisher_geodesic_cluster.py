"""CPU-only tests for router-weighted Fisher expert clustering."""

import math

import torch
import torch.nn as nn

from reap.fisher_geodesic_cluster import (
    compose_moe_distance,
    normalise_distance_matrix,
    pairwise_local_fisher_distance,
    router_distance_from_observer,
    router_importance_from_observer,
)
from reap.merge import MoEExpertMerger
from reap.metrics import pairwise_mean_squared_l2


MODEL_ATTRS = {
    "gate_proj": "gate_proj",
    "up_proj": "up_proj",
    "down_proj": "down_proj",
}


class TinyExpert(nn.Module):
    def __init__(self, gate_weight: list[float]):
        super().__init__()
        self.gate_proj = nn.Linear(2, 1, bias=False)
        self.up_proj = nn.Linear(2, 1, bias=False)
        self.down_proj = nn.Linear(2, 1, bias=False)
        self.gate_proj.weight.data.copy_(torch.tensor([gate_weight]))
        self.up_proj.weight.data.zero_()
        self.down_proj.weight.data.zero_()


def _unit_fisher(values: list[float]):
    return {
        idx: {
            "gate_proj": torch.tensor([value]),
            "up_proj": torch.tensor([1.0]),
            "down_proj": torch.tensor([1.0]),
        }
        for idx, value in enumerate(values)
    }


def test_pairwise_local_fisher_distance_matches_direct_formula():
    experts = nn.ModuleList(
        [TinyExpert([0.0, 0.0]), TinyExpert([1.0, 0.0]), TinyExpert([0.0, 2.0])]
    )
    fisher = _unit_fisher([1.0, 3.0, 1.0])

    distance = pairwise_local_fisher_distance(
        experts,
        fisher,
        MODEL_ATTRS,
        sketch_size=0,
    )

    # d(0,1)^2 = (1+3)/2 * ||[0,0]-[1,0]||^2 = 2
    assert math.isclose(distance[0, 1].item(), math.sqrt(2.0), rel_tol=1e-6)
    # d(0,2)^2 = (1+1)/2 * ||[0,0]-[0,2]||^2 = 4
    assert math.isclose(distance[0, 2].item(), 2.0, rel_tol=1e-6)
    assert torch.allclose(distance, distance.T)
    assert torch.equal(distance.diag(), torch.zeros(3))


def test_router_importance_conditions_the_fisher_metric():
    experts = nn.ModuleList([TinyExpert([0.0, 0.0]), TinyExpert([1.0, 0.0])])
    fisher = _unit_fisher([1.0, 1.0])

    unweighted = pairwise_local_fisher_distance(
        experts, fisher, MODEL_ATTRS, sketch_size=0
    )
    weighted = pairwise_local_fisher_distance(
        experts,
        fisher,
        MODEL_ATTRS,
        router_importance=torch.tensor([1.0, 3.0]),
        sketch_size=0,
    )

    assert math.isclose(unweighted[0, 1].item(), 1.0, rel_tol=1e-6)
    assert math.isclose(weighted[0, 1].item(), math.sqrt(2.0), rel_tol=1e-6)


def test_coordinate_sketch_is_deterministic():
    experts = nn.ModuleList(
        [
            TinyExpert([0.0, 1.0]),
            TinyExpert([2.0, 3.0]),
            TinyExpert([4.0, 5.0]),
        ]
    )
    fisher = _unit_fisher([1.0, 2.0, 3.0])

    first = pairwise_local_fisher_distance(
        experts, fisher, MODEL_ATTRS, sketch_size=1, seed=17
    )
    second = pairwise_local_fisher_distance(
        experts, fisher, MODEL_ATTRS, sketch_size=1, seed=17
    )
    assert torch.equal(first, second)


def test_reap_importance_is_corpus_router_weighted_activation():
    layer_data = {
        "total_tokens": torch.tensor(100),
        "expert_frequency": torch.tensor([20, 80]),
        "weighted_ean_sum": torch.tensor([10.0, 20.0]),
    }
    importance = router_importance_from_observer(layer_data, "reap")
    # Raw corpus means [0.1, 0.2], then normalized to mean one.
    assert torch.allclose(importance, torch.tensor([2.0 / 3.0, 4.0 / 3.0]))


def test_reap_importance_supports_legacy_conditional_cache():
    layer_data = {
        "total_tokens": torch.tensor(10),
        "expert_frequency": torch.tensor([2, 8]),
        "reap": torch.tensor([5.0, 1.0]),
    }
    importance = router_importance_from_observer(layer_data, "reap")
    # Conditional REAP * routing frequency / total_tokens = [1.0, 0.8].
    assert torch.allclose(importance, torch.tensor([10.0 / 9.0, 8.0 / 9.0]))


def test_pairwise_router_probability_distance():
    # Experts are rows, observations/tokens are columns.
    router_probabilities = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
    distance = pairwise_mean_squared_l2(router_probabilities)
    assert torch.allclose(distance, torch.tensor([[0.0, 0.5], [0.5, 0.0]]))


def test_router_distance_prefers_dispatched_weight_geometry():
    dispatched = torch.tensor([[0.0, 0.25], [0.25, 0.0]])
    legacy = torch.tensor([[0.0, 1.5], [1.5, 0.0]])
    distance, source = router_distance_from_observer(
        {
            "router_weight_l2_distance": dispatched,
            "router_logit_similiarity": legacy,
        }
    )
    assert source == "router_weight_l2_distance"
    assert torch.equal(distance, dispatched)


def test_compose_moe_distance_median_normalizes_components():
    fisher = torch.tensor([[0.0, 2.0, 4.0], [2.0, 0.0, 6.0], [4.0, 6.0, 0.0]])
    router = fisher * 10.0
    fisher_normalized, scale = normalise_distance_matrix(fisher, "Fisher")
    combined, scales = compose_moe_distance(
        fisher,
        router_distance=router,
        fisher_weight=1.0,
        router_weight=1.0,
    )

    assert scale == 4.0
    assert scales == {"fisher": 4.0, "router": 40.0}
    assert torch.allclose(combined, 2.0 * fisher_normalized)


def test_fisher_importance_moves_the_new_expert_barycenter(monkeypatch):
    for name in (
        "FISHER_BETA",
        "KARCHER_ITERS",
        "KARCHER_ETA",
        "V28_FREQ_MIX",
        "GL_ROUTER_GATED",
    ):
        monkeypatch.delenv(name, raising=False)
    tensors = [torch.zeros(1, 2), torch.full((1, 2), 10.0)]
    unit_fisher = [torch.ones(1), torch.ones(1)]
    merged = MoEExpertMerger._v27_fisher_geodesic_merge(
        tensors,
        unit_fisher=unit_fisher,
        fisher_importance=torch.tensor([1.0, 3.0]),
    )
    assert torch.allclose(merged, torch.full((1, 2), 7.5))