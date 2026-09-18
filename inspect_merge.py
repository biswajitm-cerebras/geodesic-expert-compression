"""Diagnostic: inspect merged MoE safetensors vs parent to explain random-chance scores.
CPU-only, reads a few layers' router + expert tensors from safetensors shards.
"""
import glob
import json
import os
import sys

import torch
from safetensors import safe_open

PARENT = glob.glob(
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/*"
    )
)[0]
MSMOE = "artifacts/Qwen3-30B-A3B-Instruct-2507/evol-codealpaca-v1/non_uniform_merged_models/m_smoe-seed_42_0.5/m_smoe"
V27 = "artifacts/Qwen3-30B-A3B-Instruct-2507/evol-codealpaca-v1/non_uniform_merged_models/v27geo-seed_42_0.5/v27_geodesic"

NUM_EXPERTS = 128
LAYERS_TO_CHECK = [0, 10, 23, 47]


def build_index(model_dir):
    """Return dict key -> shard file path."""
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        wm = json.load(open(idx_path))["weight_map"]
        return {k: os.path.join(model_dir, v) for k, v in wm.items()}
    # single shard
    shard = glob.glob(os.path.join(model_dir, "*.safetensors"))[0]
    with safe_open(shard, framework="pt") as f:
        return {k: shard for k in f.keys()}


def get_tensor(index, key):
    path = index.get(key)
    if path is None:
        return None
    with safe_open(path, framework="pt") as f:
        return f.get_tensor(key)


def analyze(name, model_dir):
    print(f"\n{'='*70}\n{name}: {model_dir}\n{'='*70}")
    index = build_index(model_dir)
    for L in LAYERS_TO_CHECK:
        pfx = f"model.layers.{L}.mlp"
        gate = get_tensor(index, f"{pfx}.gate.weight")
        # collect expert up_proj weights, hash each to find tie structure
        sigs = []
        nan_experts = 0
        norms = []
        for e in range(NUM_EXPERTS):
            w = get_tensor(index, f"{pfx}.experts.{e}.up_proj.weight")
            if w is None:
                sigs.append(None)
                continue
            if torch.isnan(w).any() or torch.isinf(w).any():
                nan_experts += 1
            # signature: cheap hash of a slice + norm rounded
            s = (round(float(w[:4, :4].float().sum()), 4), round(float(w.float().norm()), 2))
            sigs.append(s)
            norms.append(float(w.float().norm()))
        uniq = len(set(s for s in sigs if s is not None))
        gate_info = "MISSING" if gate is None else f"shape={tuple(gate.shape)} norm={float(gate.float().norm()):.2f} nan={bool(torch.isnan(gate).any())}"
        nmin = min(norms) if norms else 0
        nmax = max(norms) if norms else 0
        print(f"  L{L:2d}: unique_experts={uniq}/{NUM_EXPERTS}  nan_experts={nan_experts}  "
              f"up_proj_norm[min={nmin:.1f},max={nmax:.1f}]  gate: {gate_info}")


if __name__ == "__main__":
    analyze("PARENT", PARENT)
    analyze("MSMOE_0.5", MSMOE)
    analyze("V27GEO_0.5", V27)
