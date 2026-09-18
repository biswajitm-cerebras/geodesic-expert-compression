#!/usr/bin/env python
"""Plot expert similarity in FUNCTIONAL space from the cached observer file.

Unlike raw-weight cosine (which is ~0 because experts are permutation/scale
symmetric), redundancy that M-SMoE / REAP actually exploit lives in router /
activation space. The observer cache stores, per decoder layer:
  - 'router_logit_similiarity'       (128,128)  cosine of per-expert router logits
  - 'characteristic_activation'      (128,2048) mean activation signature / expert
  - 'expert_frequency'               (128,)     how often each expert is routed to

We plot the router-logit similarity heatmap and the characteristic-activation
cosine heatmap side by side for the requested layers, and report off-diagonal
statistics (mean / p95 / max) so we can see which layers have merge-friendly
redundant experts vs. distinct ones.

Usage:
  python scripts/plot_expert_router_sim.py [--layers 0,11,23,35,47]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

OBS = (
    "artifacts/Qwen3-30B-A3B-Instruct-2507/evol-codealpaca-v1/all/"
    "observations_64_cosine-seed_42_v27_gl.pt"
)
NUM_EXPERTS = 128


def cosine_matrix(X: np.ndarray) -> np.ndarray:
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return Xn @ Xn.T


def off_diag_stats(C: np.ndarray) -> tuple[float, float, float]:
    off = C[~np.eye(C.shape[0], dtype=bool)]
    return float(off.mean()), float(np.percentile(off, 95)), float(off.max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,11,23,35,47")
    ap.add_argument("--obs", default=OBS)
    ap.add_argument("--out", default="artifacts/analysis/expert_router_sim.png")
    ap.add_argument("--stats-out", default="artifacts/analysis/expert_router_sim_stats.txt")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = [int(x) for x in args.layers.split(",") if x.strip() != ""]
    d = torch.load(args.obs, map_location="cpu", weights_only=False)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    nrow = len(layers)
    fig, axes = plt.subplots(nrow, 2, figsize=(9.0, 4.2 * nrow), squeeze=False)

    stats_lines = ["Expert functional-space similarity (from observer cache)", ""]
    im_router = im_act = None
    for i, L in enumerate(layers):
        rec = d[L]
        # 1) router-logit similarity (already a cosine-like matrix)
        R = rec["router_logit_similiarity"].to(torch.float32).numpy()
        # 2) characteristic-activation cosine (recompute from signatures)
        A = rec["characteristic_activation"].to(torch.float32).numpy()  # (128,2048)
        Cact = cosine_matrix(A)

        rm, rp95, rmax = off_diag_stats(R)
        am, ap95, amax = off_diag_stats(Cact)
        stats_lines.append(
            f"layer {L:2d}: router-sim  mean={rm:+.3f} p95={rp95:+.3f} max={rmax:+.3f}"
            f"   |  act-cosine  mean={am:+.3f} p95={ap95:+.3f} max={amax:+.3f}"
        )

        axr = axes[i][0]
        im_router = axr.imshow(R, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        axr.set_title(f"layer {L}  router-logit sim\nmean off-diag={rm:+.3f}, max={rmax:+.3f}", fontsize=9)
        axr.set_ylabel("expert")

        axa = axes[i][1]
        im_act = axa.imshow(Cact, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        axa.set_title(f"layer {L}  activation cosine\nmean off-diag={am:+.3f}, max={amax:+.3f}", fontsize=9)
        if i == nrow - 1:
            axr.set_xlabel("expert")
            axa.set_xlabel("expert")

    fig.colorbar(im_act, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label="similarity")
    fig.suptitle(
        "Qwen3-30B-A3B-Instruct  expert similarity in FUNCTIONAL space "
        "(router logits & activation signatures)",
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
