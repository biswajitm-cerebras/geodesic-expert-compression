"""Unit tests for the v27-GL path-integrated Fisher helpers (no model / GPU needed)."""

import math

import numpy as np
import torch

from reap.fisher_gl_v27 import (
    cluster_relative_change,
    clustered_expert_keys,
    floor_fisher,
    gauss_legendre_nodes_01,
    gl_accumulate_fisher,
)
from reap.fisher_v27 import F_FLOOR


def test_gl_nodes_3_matches_v27_reference():
    nodes = gauss_legendre_nodes_01(3)
    h = 0.5 * math.sqrt(3.0 / 5.0)
    ref = [
        (0.5 - h, 5.0 / 18.0),
        (0.5, 8.0 / 18.0),
        (0.5 + h, 5.0 / 18.0),
    ]
    assert len(nodes) == 3
    for (t, w), (rt, rw) in zip(sorted(nodes), sorted(ref)):
        assert abs(t - rt) < 1e-12, (t, rt)
        assert abs(w - rw) < 1e-12, (w, rw)
    # Weights integrate constants exactly (sum to 1 on [0, 1]).
    assert abs(sum(w for _, w in nodes) - 1.0) < 1e-12


def test_gl_nodes_1_is_endpoint():
    # 1 node must be the geodesic endpoint t=0 with weight 1 so GL reduces to endpoint Fisher.
    nodes = gauss_legendre_nodes_01(1)
    assert nodes == [(0.0, 1.0)]


def test_gl_nodes_weights_sum_to_one():
    for n in (2, 3, 4, 5):
        nodes = gauss_legendre_nodes_01(n)
        assert len(nodes) == n
        assert abs(sum(w for _, w in nodes) - 1.0) < 1e-12
        assert all(0.0 <= t <= 1.0 for t, _ in nodes)


def test_gl_nodes_integrates_polynomial():
    # n-point GL is exact for polynomials up to degree 2n-1. Check f(t)=t^3 with n=2.
    nodes = gauss_legendre_nodes_01(2)
    approx = sum(w * (t ** 3) for t, w in nodes)
    assert abs(approx - 0.25) < 1e-12  # integral of t^3 on [0,1] = 1/4


def test_clustered_expert_keys_skips_singletons():
    labels = {
        0: torch.tensor([0, 0, 1, 2, 2, 2]),  # clusters: {0,1}, {2}, {3,4,5}
        1: torch.tensor([0, 1, 2, 3]),  # all singletons
    }
    keys = clustered_expert_keys(labels)
    assert keys[0] == [0, 1, 3, 4, 5]  # experts in size>1 clusters only
    assert 1 not in keys  # layer with only singletons excluded


def test_gl_accumulate_reduces_to_single_node():
    # One node with weight 1 => accumulator equals the node Fisher exactly.
    node = {0: {2: {"gate_proj": torch.tensor([1.0, 4.0, 9.0])}}}
    acc = gl_accumulate_fisher(None, node, 1.0)
    assert torch.allclose(acc[0][2]["gate_proj"], node[0][2]["gate_proj"])


def test_gl_accumulate_weighted_sum():
    n0 = {0: {0: {"up_proj": torch.tensor([1.0, 2.0])}}}
    n1 = {0: {0: {"up_proj": torch.tensor([3.0, 4.0])}}}
    acc = gl_accumulate_fisher(None, n0, 0.25)
    acc = gl_accumulate_fisher(acc, n1, 0.75)
    expected = 0.25 * n0[0][0]["up_proj"] + 0.75 * n1[0][0]["up_proj"]
    assert torch.allclose(acc[0][0]["up_proj"], expected)


def test_floor_fisher_clamps():
    f = {0: {0: {"down_proj": torch.tensor([0.0, 1e-20, 5.0])}}}
    floor_fisher(f)
    assert torch.all(f[0][0]["down_proj"] >= F_FLOOR)
    assert f[0][0]["down_proj"][2].item() == 5.0


def test_cluster_relative_change_zero_when_identical():
    prev = {(0, 0, "gate_proj"): torch.randn(4, 3)}
    curr = {(0, 0, "gate_proj"): prev[(0, 0, "gate_proj")].clone()}
    assert cluster_relative_change(prev, curr) == 0.0


def test_cluster_relative_change_ratio():
    w = torch.ones(2, 2)
    prev = {(0, 0, "gate_proj"): w.clone()}
    curr = {(0, 0, "gate_proj"): w.clone() * 2.0}  # diff norm == prev norm
    assert abs(cluster_relative_change(prev, curr) - 1.0) < 1e-6


def test_gl_fisher_merge_equals_endpoint_at_single_node():
    """1-node GL (t=0) integrated Fisher must equal the raw endpoint Fisher tensor."""
    endpoint = {0: {0: {"gate_proj": torch.tensor([0.5, 2.0, 3.0])}}}
    nodes = gauss_legendre_nodes_01(1)
    acc = None
    for t, w in nodes:
        # At t=0 the interpolant is the original expert, so the "node Fisher" is the
        # endpoint Fisher; weight is 1.0.
        assert t == 0.0 and w == 1.0
        acc = gl_accumulate_fisher(acc, endpoint, w)
    acc = floor_fisher(acc)
    assert torch.allclose(acc[0][0]["gate_proj"], endpoint[0][0]["gate_proj"])
