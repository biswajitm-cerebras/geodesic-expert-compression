#!/usr/bin/env python
"""Plot pairwise cosine similarity of MoE expert weights for the base model.

For each requested decoder layer we build, for every expert, the flattened
concatenation of its gate_proj / up_proj / down_proj weights, L2-normalise it,
and compute the 128x128 cosine-similarity matrix. High off-diagonal similarity
=> redundant experts (good merge candidates); low => distinct (better pruned).

Reads tensors lazily from the sharded safetensors checkpoint (no full model
load), so it runs comfortably on the login node.

Usage:
  python scripts/plot_expert_cosine.py [--layers 0,11,23,35,47]
                                       [--proj all|gate|up|down]
                                       [--out artifacts/analysis/expert_cosine.png]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
import torch
from safetensors import safe_open

BASE_SNAP = (
    "/home/biswajit.mishra/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/"
    "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
)
NUM_EXPERTS = 128
PROJ_KEYS = {
    "all": ("gate_proj", "up_proj", "down_proj"),
    "gate": ("gate_proj",),
    "up": ("up_proj",),
    "down": ("down_proj",),
}


def build_weight_map(snap: str) -> dict[str, str]:
    idx = json.load(open(os.path.join(snap, "model.safetensors.index.json")))
    return idx["weight_map"]


def load_layer_expert_matrix(
    snap: str, weight_map: dict[str, str], layer: int, projs: tuple[str, ...]
) -> np.ndarray:
    """Return (NUM_EXPERTS, D) fp32 matrix of flattened expert weights."""
    # Collect the keys we need and group them by shard file for efficient reads.
    needed: dict[int, list[tuple[int, str]]] = defaultdict(list)  # expert -> [(order, key)]
    shard_of: dict[str, str] = {}
    for e in range(NUM_EXPERTS):
        for order, p in enumerate(projs):
            k = f"model.layers.{layer}.mlp.experts.{e}.{p}.weight"
            needed[e].append((order, k))
            shard_of[k] = weight_map[k]

    # Open each shard once; cache tensors we need.
    keys_by_shard: dict[str, list[str]] = defaultdict(list)
    for k, sh in shard_of.items():
        keys_by_shard[sh].append(k)

    tensor_cache: dict[str, torch.Tensor] = {}
    for sh, keys in keys_by_shard.items():
        path = os.path.join(snap, sh)
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in keys:
                tensor_cache[k] = f.get_tensor(k).to(torch.float32).reshape(-1)

    rows = []
    for e in range(NUM_EXPERTS):
        parts = [tensor_cache[k] for _, k in sorted(needed[e])]
        rows.append(torch.cat(parts))
    X = torch.stack(rows).numpy()  # (128, D)
    return X


def cosine_matrix(X: np.ndarray) -> np.ndarray:
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return Xn @ Xn.T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,11,23,35,47",
                    help="comma-separated decoder layer indices")
    ap.add_argument("--proj", default="all", choices=list(PROJ_KEYS))
    ap.add_argument("--snap", default=BASE_SNAP)
    ap.add_argument("--out", default="artifacts/analysis/expert_cosine.png")
    ap.add_argument("--stats-out", default="artifacts/analysis/expert_cosine_stats.txt")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = [int(x) for x in args.layers.split(",") if x.strip() != ""]
    projs = PROJ_KEYS[args.proj]
    weight_map = build_weight_map(args.snap)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ncol = len(layers)
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4.6), squeeze=False)

    stats_lines = [f"Expert weight cosine similarity ({args.proj} proj)", ""]
    for j, L in enumerate(layers):
        X = load_layer_expert_matrix(args.snap, weight_map, L, projs)
        C = cosine_matrix(X)
        off = C[~np.eye(NUM_EXPERTS, dtype=bool)]
        mean_off, max_off = float(off.mean()), float(off.max())
        p95 = float(np.percentile(off, 95))
        stats_lines.append(
            f"layer {L:2d}: off-diag cosine  mean={mean_off:+.3f}  "
            f"p95={p95:+.3f}  max={max_off:+.3f}"
        )
        ax = axes[0][j]
        im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(f"layer {L}\nmean off-diag={mean_off:+.3f}", fontsize=10)
        ax.set_xlabel("expert")
        if j == 0:
            ax.set_ylabel("expert")
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, pad=0.02, label="cosine")
    fig.suptitle(
        f"Qwen3-30B-A3B-Instruct  MoE expert weight cosine similarity  "
        f"({args.proj} proj, {NUM_EXPERTS} experts/layer)",
        fontsize=12,
    )
    fig.savefig(args.out, dpi=130, bbox_inches="tight")

    os.makedirs(os.path.dirname(args.stats_out), exist_ok=True)
    with open(args.stats_out, "w") as f:
        f.write("\n".join(stats_lines) + "\n")
    print("\n".join(stats_lines))
    print(f"\nSaved heatmap -> {args.out}")
    print(f"Saved stats   -> {args.stats_out}")


if __name__ == "__main__":
    main()
