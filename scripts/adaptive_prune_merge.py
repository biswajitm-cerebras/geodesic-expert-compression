#!/usr/bin/env python
"""Adaptive prune-vs-merge at uniform expert count per layer.

Uses the observer cache (router-logit / activation similarity) to decide per layer:
  - HIGH redundancy (layer 47, activation-cos >= REDUNDANCY_THRESHOLD) → MERGE clusters
  - LOW redundancy (mid-stack, < threshold) → PRUNE by REAP saliency

All layers end up with the same K experts (servable). No distillation, no architecture changes.

Usage:
  python scripts/adaptive_prune_merge.py \
    --keep-experts <K> \
    --redundancy-threshold 0.15 \
    --output-dir artifacts/adaptive_prune_merge_v30
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoConfig
from safetensors.torch import save_file

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(name)s:%(levelname)s: %(message)s')

NUM_EXPERTS = 128
NUM_LAYERS = 48
DEFAULT_SNAPSHOT = "/home/biswajit.mishra/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"


def load_observer_cache(path: str) -> dict:
    """Load observer cache (contains router_logit_similiarity + characteristic_activation per layer)."""
    return torch.load(path, map_location="cpu", weights_only=False)


def compute_layer_redundancy(obs_cache: dict, layer_idx: int) -> float:
    """Compute mean off-diagonal activation cosine for a layer (proxy for redundancy)."""
    rec = obs_cache[layer_idx]
    act_sig = rec["characteristic_activation"].to(torch.float32).numpy()  # (128, 2048)
    
    # Compute cosine of activation signatures
    act_n = act_sig / (np.linalg.norm(act_sig, axis=1, keepdims=True) + 1e-12)
    act_cos = act_n @ act_n.T  # (128, 128)
    
    # Off-diagonal mean
    off_diag = act_cos[~np.eye(128, dtype=bool)]
    return float(off_diag.mean())


def compute_layer_saliency(obs_cache: dict, layer_idx: int) -> np.ndarray:
    """Compute saliency for each expert: routing frequency."""
    rec = obs_cache[layer_idx]
    freq = rec["expert_frequency"].numpy()  # (128,)
    return freq / (freq.sum() + 1e-12)  # Normalize to probability


def compute_activation_similarity(obs_cache: dict, layer_idx: int) -> np.ndarray:
    """Compute pairwise cosine similarity of expert activation signatures."""
    rec = obs_cache[layer_idx]
    act_sig = rec["characteristic_activation"].to(torch.float32).numpy()  # (128, 2048)
    
    # Normalize to unit vectors
    act_n = act_sig / (np.linalg.norm(act_sig, axis=1, keepdims=True) + 1e-12)
    # Cosine similarity matrix
    sim = act_n @ act_n.T  # (128, 128)
    return sim


def compute_router_logits_similarity(obs_cache: dict, layer_idx: int) -> np.ndarray:
    """Compute pairwise cosine similarity of expert router logits."""
    rec = obs_cache[layer_idx]
    if "router_logits_signatures" not in rec:
        logger.debug(f"  Layer {layer_idx}: router_logits_signatures not in cache, using activation only")
        return None
    
    logits_sig = rec["router_logits_signatures"].to(torch.float32).numpy()  # (128, D)
    
    # Normalize to unit vectors
    logits_n = logits_sig / (np.linalg.norm(logits_sig, axis=1, keepdims=True) + 1e-12)
    # Cosine similarity matrix
    sim = logits_n @ logits_n.T  # (128, 128)
    return sim


def compute_weight_similarity(model, layer_idx: int) -> np.ndarray:
    """Compute pairwise cosine similarity of expert weights (gate, up, down projections).
    Optimized using torch operations to avoid expensive numpy conversions."""
    import torch.nn.functional as F
    
    experts_layer = model.model.layers[layer_idx].mlp.experts
    device = next(experts_layer[0].parameters()).device
    
    # Flatten and stack all expert weights using torch operations
    weight_vecs = []
    for expert in experts_layer:
        vecs = []
        for param_name in ["gate_proj", "up_proj", "down_proj"]:
            # Get weight and flatten, keep on device
            w = getattr(expert, param_name).weight.data.to(torch.float32)
            vecs.append(w.flatten())
        weight_vec = torch.cat(vecs)
        weight_vecs.append(weight_vec)
    
    weight_vecs = torch.stack(weight_vecs)  # (128, total_params)
    
    # Normalize to unit vectors using torch operations
    weight_n = F.normalize(weight_vecs, p=2, dim=1)  # (128, 128)
    
    # Cosine similarity matrix using torch
    sim = torch.mm(weight_n, weight_n.t())  # (128, 128)
    
    # Convert to numpy
    return sim.cpu().detach().numpy()


def greedy_cluster_experts(model, layer_idx: int, obs_cache: dict, num_clusters: int) -> list:
    """Greedily cluster experts by repeatedly merging the most similar pair.
    
    Returns list of clusters, where each cluster is a list of expert indices.
    Optimized with vectorized operations for speed.
    """
    logger.info(f"Layer {layer_idx}: Starting greedy clustering (128 → {num_clusters})...")
    
    # Compute all three similarity metrics
    logger.info(f"  Layer {layer_idx}: Computing activation similarity...")
    act_sim = compute_activation_similarity(obs_cache, layer_idx)
    
    logger.info(f"  Layer {layer_idx}: Computing router logits similarity...")
    router_sim = compute_router_logits_similarity(obs_cache, layer_idx)
    
    logger.info(f"  Layer {layer_idx}: Computing weight similarity...")
    weight_sim = compute_weight_similarity(model, layer_idx)
    
    # Combine similarities: average the available metrics
    combined_sim = act_sim.copy()
    if router_sim is not None:
        combined_sim += router_sim
        combined_sim /= 2.0
    
    # Weight the combined (activation + router) by weight similarity
    combined_sim = 0.5 * combined_sim + 0.5 * weight_sim
    
    logger.info(f"  Layer {layer_idx}: Similarity matrix computed. Starting merges...")
    
    # Initialize: each expert is its own cluster
    clusters = [[i] for i in range(NUM_EXPERTS)]  # List of lists
    
    # Greedy merging: repeatedly merge the two most similar clusters
    merge_count = 0
    target_merges = NUM_EXPERTS - num_clusters
    
    while len(clusters) > num_clusters:
        merge_count += 1
        if merge_count % 5 == 0:
            logger.info(f"  Layer {layer_idx}: Merge iteration {merge_count}/{target_merges}, {len(clusters)} clusters remaining...")
        
        # Find the pair of clusters with maximum similarity
        best_sim = -2.0
        best_i, best_j = -1, -1
        
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Compute average inter-cluster similarity using vectorized indexing
                cluster_i = np.array(clusters[i])
                cluster_j = np.array(clusters[j])
                
                # Extract submatrix and compute mean
                sim_ij = combined_sim[np.ix_(cluster_i, cluster_j)].mean()
                
                if sim_ij > best_sim:
                    best_sim = sim_ij
                    best_i, best_j = i, j
        
        if best_i == -1:
            logger.warning(f"  Layer {layer_idx}: Could not find merge pair")
            break
        
        # Merge clusters[best_j] into clusters[best_i]
        clusters[best_i].extend(clusters[best_j])
        del clusters[best_j]
    
    logger.info(f"Layer {layer_idx}: Greedy clustering complete → {len(clusters)} clusters ({merge_count} merges)")
    return clusters


def merge_expert_cluster(model, layer_idx: int, cluster_indices: list):
    """Merge experts in a cluster via averaging."""
    if len(cluster_indices) <= 1:
        return
    
    logger.debug(f"  Merging cluster {cluster_indices[:3]}... in layer {layer_idx}")
    
    # Get expert gate, up, down weights
    experts_layer = model.model.layers[layer_idx].mlp.experts
    
    # Stack weights and average
    dom_idx = cluster_indices[0]
    dom_expert = experts_layer[dom_idx]
    
    for param_name in ["gate_proj", "up_proj", "down_proj"]:
        tensors = [getattr(experts_layer[idx], param_name).weight.data for idx in cluster_indices]
        avg_weight = torch.mean(torch.stack(tensors), dim=0)
        getattr(dom_expert, param_name).weight.data = avg_weight


def prune_expert_layer(model, layer_idx: int, keep_indices: list):
    """Prune experts in a layer: copy selected experts to positions 0:K."""
    experts_layer = model.model.layers[layer_idx].mlp.experts
    
    logger.debug(f"  Pruning layer {layer_idx}: keeping {keep_indices}")
    
    # Clone ALL source weights first to avoid aliasing bugs when source and
    # destination index ranges overlap (e.g., keep_indices=[50,3,1,7,...] means
    # step 2 would read experts[1] which was already overwritten at step 1).
    cloned = {}
    for old_idx in keep_indices:
        if old_idx not in cloned:
            cloned[old_idx] = {
                p: getattr(experts_layer[old_idx], p).weight.data.clone()
                for p in ["gate_proj", "up_proj", "down_proj"]
            }
    
    for new_idx, old_idx in enumerate(keep_indices):
        for param_name in ["gate_proj", "up_proj", "down_proj"]:
            getattr(experts_layer[new_idx], param_name).weight.data.copy_(cloned[old_idx][param_name])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    ap.add_argument("--keep-experts", type=int, default=64, help="Number of experts to keep per layer")
    ap.add_argument("--redundancy-threshold", type=float, default=0.15, help="Activation-cosine threshold for deciding merge vs. prune")
    ap.add_argument("--obs-cache", default="artifacts/Qwen3-30B-A3B-Instruct-2507/evol-codealpaca-v1/all/observations_64_cosine-seed_42_v27_gl.pt")
    ap.add_argument("--output-dir", default="artifacts/adaptive_prune_merge_v30")
    ap.add_argument("--decisions-only", action="store_true", help="Only compute decisions, don't process weights")
    args = ap.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Loading observer cache from {args.obs_cache}")
    obs_cache = load_observer_cache(args.obs_cache)
    
    # Per-layer decision: redundancy-driven merge vs. orthogonal prune
    layer_decisions = {}
    merge_count = 0
    prune_count = 0
    for l in range(NUM_LAYERS):
        redundancy = compute_layer_redundancy(obs_cache, l)
        decision = "merge" if redundancy >= args.redundancy_threshold else "prune"
        layer_decisions[l] = {"redundancy": redundancy, "decision": decision}
        if decision == "merge":
            merge_count += 1
        else:
            prune_count += 1
        logger.info(f"Layer {l:2d}: redundancy={redundancy:.3f} → {decision}")
    
    logger.info(f"\nSummary: {merge_count} layers MERGE, {prune_count} layers PRUNE")
    
    # Log decisions
    decisions_path = os.path.join(args.output_dir, "adaptive_decisions.json")
    with open(decisions_path, "w") as f:
        json.dump(layer_decisions, f, indent=2)
    logger.info(f"Saved layer decisions to {decisions_path}")
    
    if args.decisions_only:
        logger.info("--decisions-only: stopping here. Run without flag to process weights.")
        return
    
    # Load model
    logger.info(f"Loading model from {args.snapshot}")
    config = AutoConfig.from_pretrained(args.snapshot, trust_remote_code=True)
    
    # Force load with 128 experts (original size)
    config.num_experts = NUM_EXPERTS
    model = AutoModelForCausalLM.from_pretrained(
        args.snapshot,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True
    )
    model.eval()
    
    logger.info(f"Model loaded. Total params: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
    
    # Apply per-layer decisions
    # Store keep_indices for each layer (needed for gate weight reordering during prune)
    layer_keep_indices = {}
    
    with torch.no_grad():
        for l in range(NUM_LAYERS):
            decision = layer_decisions[l]["decision"]
            
            if decision == "merge":
                logger.info(f"Layer {l}: Merging experts via greedy similarity-based clustering...")
                # Greedy clustering: use activation, router logits, and weight similarity
                clusters = greedy_cluster_experts(model, l, obs_cache, args.keep_experts)
                
                merge_positions = []
                for cluster in clusters:
                    merge_expert_cluster(model, l, cluster)
                    # The merged expert stays at position of the first expert in the cluster
                    merge_positions.append(cluster[0])
                
                # Store the positions where merged experts are located
                layer_keep_indices[l] = merge_positions
                
                logger.debug(f"Layer {l}: Merged into {len(clusters)} clusters at positions {merge_positions[:5]}...")
            else:
                logger.info(f"Layer {l}: Pruning experts by saliency...")
                saliency = compute_layer_saliency(obs_cache, l)
                keep_indices = np.argsort(-saliency)[:args.keep_experts].tolist()
                prune_expert_layer(model, l, keep_indices)
                # Store for gate weight reordering
                layer_keep_indices[l] = keep_indices
    
    # Update config
    logger.info(f"Updating config: num_experts {config.num_experts} → {args.keep_experts}")
    config.num_experts = args.keep_experts
    
    # Update config first
    logger.info(f"Updating config: num_experts {config.num_experts} → {args.keep_experts}")
    config.num_experts = args.keep_experts
    model.config.num_experts = args.keep_experts
    
    # FIRST: Compact merged experts from sparse to dense positions
    # Merged experts are at positions [0*2, 1*2, 2*2, ..., 63*2] = [0, 2, 4, ..., 126]
    # We need to move them to [0, 1, 2, ..., 63] before gate weight handling
    logger.info("Compacting merged experts to dense positions...")
    for l in range(NUM_LAYERS):
        if layer_decisions[l]["decision"] == "merge":
            mlp_layer = model.model.layers[l].mlp
            experts_layer = mlp_layer.experts
            
            # Get sparse positions of merged experts
            sparse_positions = layer_keep_indices[l]
            
            # Copy experts from sparse to dense positions
            for new_idx, old_idx in enumerate(sparse_positions):
                for param_name in ["gate_proj", "up_proj", "down_proj"]:
                    src_weight = getattr(experts_layer[old_idx], param_name).weight.data
                    getattr(experts_layer[new_idx], param_name).weight.data.copy_(src_weight)
            
            # After compaction, experts are at dense positions [0, 1, ..., keep_experts-1]
            # Gate columns are already in this order (reordered in previous step)
            layer_keep_indices[l] = list(range(args.keep_experts))
            
            logger.debug(f"Layer {l}: compacted merged experts from sparse to dense")
    
    logger.info("Expert compaction complete.")
    
    # SECOND: Prune and reorder gate weights to match expert arrangement
    # Gate weight shape is [num_experts, hidden_dim]
    # layer_keep_indices[l] contains:
    # - For merged layers: sparse positions [0, 2, 4, ..., 126] where merged experts are
    # - For pruned layers: original expert indices that were selected [50, 23, 45, ...]
    # We need to reorder gate columns by these indices before compacting
    logger.info("Reordering router gate weights before expert compaction...")
    for l in range(NUM_LAYERS):
        mlp_layer = model.model.layers[l].mlp
        if hasattr(mlp_layer, 'gate') and hasattr(mlp_layer.gate, 'weight'):
            gate_weight = mlp_layer.gate.weight.data  # [128, 2048]
            
            # Reorder gate columns based on selected/merged expert positions
            keep_indices = layer_keep_indices[l]
            reordered_gate = gate_weight[keep_indices, :].clone()  # [64, 2048]
            mlp_layer.gate.weight.data = reordered_gate
            
            logger.debug(f"Layer {l}: gate weight reordered from {gate_weight.shape} to {reordered_gate.shape}")
    
    logger.info("Gate weight reordering complete.")
    
    # CRITICAL: Physically remove unused experts from ModuleLists before saving
    # This ensures save_pretrained() only serializes experts 0:keep_experts
    logger.info("Removing unused expert modules from all layers...")
    for l in range(NUM_LAYERS):
        experts_layer = model.model.layers[l].mlp.experts
        # Remove experts in reverse order to avoid index shifting
        while len(experts_layer) > args.keep_experts:
            idx_to_remove = len(experts_layer) - 1
            del experts_layer[idx_to_remove]
            logger.debug(f"Layer {l}: removed expert {idx_to_remove}")
    
    logger.info("Expert module removal complete.")
    
    # Extract state dict from truncated model
    logger.info("Extracting state dict from truncated model...")
    state_dict = model.state_dict()
    logger.info(f"State dict has {len(state_dict)} parameters")
    
    # Verify state dict only has experts 0:keep_experts
    expert_indices = set()
    for key in state_dict.keys():
        if '.mlp.experts.' in key:
            parts = key.split('.')
            try:
                expert_idx_pos = parts.index('experts') + 1
                expert_idx = int(parts[expert_idx_pos])
                expert_indices.add(expert_idx)
            except (ValueError, IndexError):
                pass
    
    if expert_indices:
        logger.info(f"State dict contains experts: {min(expert_indices)} to {max(expert_indices)}")
        assert max(expert_indices) < args.keep_experts, f"Found expert {max(expert_indices)} >= {args.keep_experts}!"
    
    # Save using safetensors directly to avoid save_pretrained() rebuilding the state dict
    logger.info(f"Saving model to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save config and generation_config
    model.config.save_pretrained(args.output_dir)
    if model.generation_config is not None:
        model.generation_config.save_pretrained(args.output_dir)
    
    # Save state dict as safetensors using the truncated model's state_dict
    # This ensures only experts 0:keep_experts are saved
    from safetensors.torch import save_file
    save_file(state_dict, os.path.join(args.output_dir, "model.safetensors"))
    
    # Copy tokenizer files from snapshot so vLLM can load without HF online access
    import shutil
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                       "special_tokens_map.json"]
    for fname in tokenizer_files:
        src = os.path.join(args.snapshot, fname)
        dst = os.path.join(args.output_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            logger.info(f"  Copied tokenizer file: {fname}")
    
    logger.info(f"✓ Model saved to {args.output_dir}")
    logger.info(f"  Config: num_experts={args.keep_experts}")
    logger.info(f"  Ready for inference with vLLM or transformers.")


if __name__ == "__main__":
    main()

