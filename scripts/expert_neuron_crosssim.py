#!/usr/bin/env python
"""Per-neuron cross-expert similarity diagnostic.

Whole-expert cosine is ~0 (experts orthogonal as vectors). But a useful feature
may be scattered across experts in arbitrary intermediate-neuron slots. This
script tests whether INDIVIDUAL neuron units have near-duplicate twins in OTHER
experts, which is what a per-head (per-neuron) align-then-merge would exploit.

An expert's intermediate neuron j is the functional unit
    u_{e,j} = concat( gate_proj[j, :], up_proj[j, :], down_proj[:, j] )   (dim 3H)
i.e. the row j of gate/up and the column j of down. We build all 128*I units
for a layer, L2-normalise, and for a random sample of query units measure the
maximum cosine to units belonging to OTHER experts (cross-expert nearest
neighbour). A high NN cosine for many neurons => scattered redundancy that
per-neuron merging can recover; near-zero => neurons are orthogonal too and
per-neuron merging won't help.

Reads lazily from sharded safetensors (login-node friendly).

Usage:
  python scripts/expert_neuron_crosssim.py --layers 0,11,23,35,47 --sample 4000
"""
from __future__ import annotations

import argparse
import json
import os
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


def build_weight_map(snap: str) -> dict[str, str]:
    idx = json.load(open(os.path.join(snap, "model.safetensors.index.json")))
    return idx["weight_map"]


def load_layer_neuron_units(
    snap: str, weight_map: dict[str, str], layer: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (U, expert_id):
    U          (128*I, 3H) fp32 matrix of per-neuron unit vectors.
    expert_id  (128*I,)    which expert each unit came from.
    """
    # Gather keys grouped by shard for efficient reads.
    keys = {}
    for e in range(NUM_EXPERTS):
        for p in ("gate_proj", "up_proj", "down_proj"):
            keys[(e, p)] = f"model.layers.{layer}.mlp.experts.{e}.{p}.weight"
    keys_by_shard: dict[str, list[tuple[tuple[int, str], str]]] = defaultdict(list)
    for ep, k in keys.items():
        keys_by_shard[weight_map[k]].append((ep, k))

    tensors: dict[tuple[int, str], torch.Tensor] = {}
    for sh, items in keys_by_shard.items():
        with safe_open(os.path.join(snap, sh), framework="pt", device="cpu") as f:
            for ep, k in items:
                tensors[ep] = f.get_tensor(k).to(torch.float32)

    units = []
    expert_id = []
    for e in range(NUM_EXPERTS):
        g = tensors[(e, "gate_proj")]   # (I, H)
        u = tensors[(e, "up_proj")]     # (I, H)
        d = tensors[(e, "down_proj")]   # (H, I)
        I = g.shape[0]
        # unit j = [gate row j | up row j | down col j]  -> (I, 3H)
        blk = torch.cat([g, u, d.transpose(0, 1)], dim=1)  # (I, 3H)
        units.append(blk)
        expert_id.append(np.full(I, e, dtype=np.int32))
    U = torch.cat(units, dim=0).numpy()          # (128*I, 3H)
    return U, np.concatenate(expert_id)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,11,23,35,47")
    ap.add_argument("--sample", type=int, default=4000,
                    help="number of random query neurons per layer")
    ap.add_argument("--snap", default=BASE_SNAP)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="artifacts/analysis/expert_neuron_crosssim.png")
    ap.add_argument("--stats-out",
                    default="artifacts/analysis/expert_neuron_crosssim_stats.txt")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(args.seed)
    layers = [int(x) for x in args.layers.split(",") if x.strip() != ""]
    weight_map = build_weight_map(args.snap)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig, axes = plt.subplots(1, len(layers), figsize=(4.0 * len(layers), 4.2),
                             squeeze=False)

    stats = ["Per-neuron cross-expert nearest-neighbour cosine", ""]
    for j, L in enumerate(layers):
        U, eid = load_layer_neuron_units(args.snap, weight_map, L)
        n, _ = U.shape
        # L2 normalise
        Un = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
        Un_t = torch.from_numpy(Un)  # (n, 3H)

        q_idx = rng.choice(n, size=min(args.sample, n), replace=False)
        Q = Un_t[q_idx]                                   # (S, 3H)
        # cosine of every query against all units, in chunks to bound memory
        nn_cos = np.empty(len(q_idx), dtype=np.float32)
        q_eid = eid[q_idx]
        CH = 512
        for s in range(0, len(q_idx), CH):
            qb = Q[s:s + CH]                              # (b, 3H)
            sims = (qb @ Un_t.T).numpy()                  # (b, n)
            # mask self-expert (so NN is cross-expert only)
            for r in range(qb.shape[0]):
                same = eid == q_eid[s + r]
                sims[r, same] = -2.0
            nn_cos[s:s + qb.shape[0]] = sims.max(axis=1)

        mean = float(nn_cos.mean())
        med = float(np.median(nn_cos))
        p95 = float(np.percentile(nn_cos, 95))
        frac05 = float((nn_cos >= 0.5).mean())
        frac07 = float((nn_cos >= 0.7).mean())
        stats.append(
            f"layer {L:2d}: NN cos  mean={mean:+.3f} median={med:+.3f} "
            f"p95={p95:+.3f}  frac>=0.5={frac05:.3f}  frac>=0.7={frac07:.3f}"
        )

        ax = axes[0][j]
        ax.hist(nn_cos, bins=60, range=(-0.1, 1.0), color="#4477aa")
        ax.axvline(0.5, color="red", ls="--", lw=1)
        ax.set_title(f"layer {L}\nmean={mean:.2f}, >=0.5: {frac05*100:.0f}%", fontsize=9)
        ax.set_xlabel("cross-expert NN cosine")
        if j == 0:
            ax.set_ylabel("# neurons")

    fig.suptitle(
        "Qwen3-30B-A3B  per-neuron cross-expert nearest-neighbour cosine\n"
        "(does a useful feature have a twin in another expert?)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(args.out, dpi=130, bbox_inches="tight")

    with open(args.stats_out, "w") as f:
        f.write("\n".join(stats) + "\n")
    print("\n".join(stats))
    print(f"\nSaved histogram -> {args.out}")
    print(f"Saved stats     -> {args.stats_out}")


if __name__ == "__main__":
    main()
