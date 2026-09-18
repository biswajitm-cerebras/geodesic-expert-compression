from __future__ import annotations
import os
import time
import pickle
import logging
import dataclasses
import pathlib
import re
import time
from typing import Any
import gc
import yaml
import shutil

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser

from accelerate.utils import set_seed
from accelerate.hooks import remove_hook_from_module


from reap.args import (
    ReapArgs,
    ModelArgs,
    DatasetArgs,
    ObserverArgs,
    ClusterArgs,
    KdArgs,
    EvalArgs,
    MergeArgs,
)
from reap.merge import MergeMethod, MoEExpertMerger
from reap.data import load_category_batches, parse_composite_dataset_spec
from reap.fisher_v27 import collect_expert_unit_fisher
from reap.fisher_geodesic_cluster import (
    build_fisher_geodesic_distance_matrices,
    router_importance_from_observer,
)
from reap.observer import OBSERVER_CONFIG_REGISTRY, MoETransformerObserver
from reap.cluster import (
    get_penalty_vector,
    hierarchical_clustering,
    dynamic_frequency_penalized_clustering,
    multi_layer_hierarchical_clustering,
    mc_smoe_clustering,
    multi_layer_kmeans_clustering,
    multi_layer_kmeans_clustering_on_ca,
    restricted_hierarchical_clustering,
    kmeans_clustering,
)
from reap.model_util import (
    get_moe,
    assert_merge,
    MODEL_ATTRS,
    patched_model_map,
    get_super_expert_indices,
)
from reap.eval import run_evaluate
from reap.cluster_plots import plot_cluster_analysis
from reap.metrics import get_distance_fn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args() -> tuple[Any, ...]:
    parser = HfArgumentParser(
        (
            ReapArgs,
            ModelArgs,
            DatasetArgs,
            ObserverArgs,
            ClusterArgs,
            KdArgs,
            EvalArgs,
            MergeArgs,
        )
    )
    args = parser.parse_args_into_dataclasses()
    return args


def str_to_directory_name(s: str) -> str:
    """Convert a string to a valid directory name by replacing special characters."""
    return re.sub(r"[^\w\-_.]", "_", s)


def create_results_directory(model_name: str, dataset_name: str) -> pathlib.Path:
    """Create a clean directory name from model and dataset names.

    For composite dataset specs (comma-separated), uses a short hash of the
    full spec as the directory name: ``composite_<md5[:8]>``.
    """
    import hashlib

    model_clean = model_name.split("/")[-1]
    model_clean = str_to_directory_name(model_clean)

    if "," in dataset_name:
        # Composite dataset spec — use hash-based directory name
        spec_hash = hashlib.md5(dataset_name.encode()).hexdigest()[:8]
        dataset_clean = f"composite_{spec_hash}"
        logger.info(
            f"Composite dataset spec detected. Using directory name: {dataset_clean}"
        )
    else:
        dataset_clean = dataset_name.split("/")[-1]
        dataset_clean = str_to_directory_name(dataset_clean)

    results_dir = pathlib.Path("./artifacts") / model_clean / dataset_clean

    if results_dir.exists():
        logger.warning(f"Directory '{results_dir}' already exists")
    else:
        results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created artifacts directory: {results_dir}")

    return results_dir


def _setup_observer(model, obs_args):
    """Create and return an MoETransformerObserver for the given model."""
    try:
        renormalize_router_weights = (
            getattr(model.config, "norm_topk_prob", False)
            and obs_args.renormalize_router_weights
        )
        if renormalize_router_weights:
            logger.info("Renormalizing topk router weights to sum to 1.")
        observer_config = OBSERVER_CONFIG_REGISTRY[model.__class__.__name__](
            distance_measure="cosine",
            renormalize_router_weights=renormalize_router_weights,
            record_pruning_metrics_only=obs_args.record_pruning_metrics_only,
        )
    except KeyError:
        raise ValueError(
            f"No observer configuration registered for model '{model.__class__.__name__}'. "
            f"Supported: {list(OBSERVER_CONFIG_REGISTRY.keys())}"
        )
    return MoETransformerObserver(
        model=model,
        hook_config=observer_config,
    )


def _profile_model(model, tokenizer, model_args, obs_args, observer):
    """Run a profiling forward pass to avoid OOM at inference time."""
    with torch.no_grad():
        try:
            model_max_length = obs_args.model_max_length
            if model_max_length is None:
                model_max_length = tokenizer.model_max_length
            logger.info(f"Profiling at model max length: {model_max_length}.")
            s = "hello " * model_max_length
            tokenized = tokenizer(
                [s],
                return_tensors="pt",
                truncation=True,
                max_length=model_max_length,
            )
            tokenized = {k: v.to(model.device) for k, v in tokenized.items()}
            for _ in range(2):
                _ = model(**tokenized)
        except Exception as e:
            raise RuntimeError(
                f"Failed to run model with max input length {model_max_length}: {e}"
            )
    logger.info(
        f"Model {model_args.model_name} successfully loaded and profiled at max length {model_max_length}."
    )
    observer.reset()


def record_activations(
    model, tokenizer, reap_args, model_args, ds_args, obs_args, results_dir
):
    if ds_args.dataset_name == "combined":
        # just return the combined data
        cat_dir = results_dir / "all"
        f_name = cat_dir / obs_args.output_file_name
        if f_name.exists():
            return torch.load(f_name, weights_only=False)
        else:
            raise RuntimeError(
                f"Combined dataset requested but no pre-recorded data found at {f_name}"
            )

    # check for composite dataset specification
    composite_components = parse_composite_dataset_spec(
        ds_args.dataset_name,
        default_split=ds_args.split,
    )
    if composite_components is not None:
        combined_batches = []
        total_batches = sum(c.num_batches for c in composite_components)
        logger.info(
            f"Composite dataset specified, overwriting given batches_per_category={obs_args.batches_per_category} "
            f"with values in composite dataset spec."
        )
        logger.info(
            f"Loading composite dataset with {len(composite_components)} "
            f"components, {total_batches} total data batches."
        )

        for comp_idx, component in enumerate(composite_components):
            comp_label = (
                f"{component.name}"
                f"{f'[{component.subset}]' if component.subset is not None else ''}"
                f"[{component.split}]"
            )
            logger.info(
                f"[{comp_idx + 1}/{len(composite_components)}] Loading component: "
                f"{comp_label} ({component.num_batches} batches)"
            )
            component_batches = load_category_batches(
                dataset_name=component.name,
                split=component.split,
                subset=component.subset,
                tokenizer=tokenizer,
                model_max_length=obs_args.model_max_length,
                split_by_category=False,
                return_vllm_tokens_prompt=obs_args.return_vllm_tokens_prompt,
                truncate=obs_args.truncate,
                batches_per_category=component.num_batches,
                batch_size=obs_args.batch_size,
            )
            combined_batches.extend(component_batches["all"])

        category_data_batches = {"all": combined_batches}
    else:
        category_data_batches = load_category_batches(
            dataset_name=ds_args.dataset_name,
            split=ds_args.split,
            subset=ds_args.dataset_config_name,
            tokenizer=tokenizer,
            model_max_length=obs_args.model_max_length,
            split_by_category=obs_args.split_by_category,
            return_vllm_tokens_prompt=obs_args.return_vllm_tokens_prompt,
            truncate=obs_args.truncate,
            batches_per_category=obs_args.batches_per_category,
            batch_size=obs_args.batch_size,
        )

    logger.info(
        "Loaded and processed data for categories: %s",
        str(list(category_data_batches.keys())),
    )
    
    # load observer and hook model
    observer = _setup_observer(model, obs_args)

    if reap_args.profile:
        _profile_model(model, tokenizer, model_args, obs_args, observer)

    # run samples over model and save observer state
    with torch.no_grad():
        for category, cat_data in category_data_batches.items():
            logger.info(f"Processing category: {category}...")
            cat_dir = results_dir / str_to_directory_name(category)
            cat_dir.mkdir(parents=True, exist_ok=True)
            f_name = cat_dir / obs_args.output_file_name
            if f_name.exists() and not obs_args.overwrite_observations:
                logger.info(
                    f"Category '{category}' previously processed. Skipping to next category..."
                )
                continue
            try:
                logger.info("No previous data found @ %s", f_name)
                for sample in tqdm(cat_data, desc=f"Processing {category} samples"):
                    attention_mask = sample.get("attention_mask", None)
                    sample = {
                        k: v.to(model.device) if torch.is_tensor(v) else v
                        for k, v in sample.items()
                    }
                    with observer.set_attention_mask(attention_mask):
                        model(**sample)
            except Exception as e:
                logger.error(f"Error processing category '{category}'")
                logger.info(
                    f"Saving partial results for category '{category}' and exiting"
                )
                observer.save_state(cat_dir / "partial.pkl")
                logger.info(
                    f"{category} data processed and saved to "
                    f"{cat_dir / obs_args.output_file_name}"
                )
                raise e
            observer.save_state(cat_dir / obs_args.output_file_name)
            observer.reset()
            logger.info(
                f"{category} data processed and saved to "
                f"{cat_dir / obs_args.output_file_name}"
            )
    observer.close_hooks()
    with open(f"{cat_dir / obs_args.output_file_name}", "rb") as f:
        observer_data = torch.load(f, weights_only=False)
    return observer_data


def cluster(
    data: dict[int, dict[str, Any]],
    num_clusters: int,
    cluster_args: ClusterArgs,
    distance_measure: str,
    results_dir: pathlib.Path,
    precomputed_distances: dict[int, torch.Tensor] | None = None,
) -> dict[int, torch.Tensor]:
    """Cluster the model's experts based on the specified clustering method."""
    logger.info(f"Clustering experts using settings:\n{cluster_args.__str__()}\n")

    cluster_labels = {}
    distances = {}
    all_layer_expert_proba = {}
    if cluster_args.singleton_super_experts or cluster_args.singleton_outlier_experts:
        super_expert_idx = get_super_expert_indices(
            data, include_last_layers=cluster_args.singleton_outlier_experts
        )
    for layer in tqdm(data, "Clustering experts..."):
        expert_prob = data[layer]["expert_frequency"] / data[layer]["total_tokens"]
        if cluster_args.expert_sim == "fisher_geodesic":
            if precomputed_distances is None or layer not in precomputed_distances:
                raise ValueError(
                    "fisher_geodesic clustering requires a precomputed distance "
                    f"matrix for layer {layer}."
                )
            distance = precomputed_distances[layer].clone()
        else:
            ttm_sim_matrix = data[layer].get("ttm_similarity_matrix")
            online_characteristic_activation_dist = data[layer].get(
                "online_characteristic_activation_dist"
            )
            ca = data[layer]["characteristic_activation"]
            routed_ca = data[layer].get("routed_characteristic_activation")
            router_logits = data[layer]["router_logit_similiarity"]

            expert_similarity_scores = {
                "ttm": ttm_sim_matrix,
                "dynamic_ttm": ttm_sim_matrix,
                "characteristic_activation": ca,
                "routed_characteristic_activation": routed_ca,
                "router_logits": router_logits,
                "online_characteristic_activation_dist": online_characteristic_activation_dist,
            }
            distance = expert_similarity_scores[cluster_args.expert_sim]

            if (
                cluster_args.expert_sim
                in [
                    "characteristic_activation",
                    "routed_characteristic_activation",
                    "router_logits",
                ]
                and cluster_args.cluster_method != "kmeans"
            ):
                # get NxN similarity matrix for vector metrics
                distance_fn = get_distance_fn(distance_measure)
                distance = distance_fn(distance.unsqueeze(0), distance.unsqueeze(1))

        if cluster_args.singleton_super_experts:
            # set super expert distance to max
            super_experts_in_layer = super_expert_idx[super_expert_idx[:, 0] == layer][
                :, 1
            ]
            if len(super_experts_in_layer) > 0:
                max_value = torch.finfo(distance.dtype).max
                distance[:, super_experts_in_layer] = max_value
                distance[super_experts_in_layer, :] = max_value

        distances[layer] = distance
        all_layer_expert_proba[layer] = expert_prob
        if cluster_args.multi_layer or cluster_args.cluster_method == "mc_smoe":
            continue
        if cluster_args.frequency_penalty and cluster_args.expert_sim != "dynamic_ttm":
            penalty = get_penalty_vector(
                expert_prob,
                cluster_args.softmax_temperature,
            )
            penalty_matrix = penalty.unsqueeze(0) + penalty.unsqueeze(1)
            penalized_distance = distance * penalty_matrix
            penalized_distance[penalized_distance.isnan()] = float("inf")
            distance = penalized_distance

        if cluster_args.expert_sim == "dynamic_ttm":
            cluster_label = dynamic_frequency_penalized_clustering(
                distance,
                expert_prob,
                num_clusters,
                cluster_args.softmax_temperature,
            )

        elif cluster_args.cluster_method == "agglomerative":
            if (
                hasattr(cluster_args, "max_cluster_size")
                and cluster_args.max_cluster_size is None
            ):
                cluster_label = hierarchical_clustering(
                    distance,
                    cluster_args.linkage_method,
                    num_clusters,
                )
            else:
                cluster_label = restricted_hierarchical_clustering(
                    distance,
                    cluster_args.linkage_method,
                    num_clusters,
                    max_cluster_size=cluster_args.max_cluster_size,
                )
            if isinstance(cluster_label, np.ndarray):
                cluster_label = torch.tensor(cluster_label)

        elif cluster_args.cluster_method == "kmeans":
            cluster_label = kmeans_clustering(distance, num_clusters)

        else:
            raise NotImplementedError(
                f"Clustering method '{cluster_args.cluster_method}' is not implemented."
            )
        cluster_labels[layer] = cluster_label

    if cluster_args.multi_layer:
        # we have parsed distances, time to cluster across layers]
        logger.info(
            f"Multi layer clustering with multi_layer={cluster_args.multi_layer}"
        )
        if cluster_args.cluster_method == "agglomerative":
            cluster_labels = multi_layer_hierarchical_clustering(
                distances,
                cluster_args.multi_layer,
                cluster_args.linkage_method,
                num_clusters,
            )
        elif cluster_args.cluster_method == "kmeans":
            # try v2:
            if cluster_args.expert_sim != "characteristic_activation":
                raise ValueError(
                    "multi_layer kmeans clustering on ca only implemented for characteristic_activation expert sim"
                )
            cluster_labels = multi_layer_kmeans_clustering_on_ca(
                distances,
                num_layers=cluster_args.multi_layer,
                n_clusters=num_clusters,
            )

            # cluster_labels = multi_layer_kmeans_clustering(
            #     distances,
            #     num_layers=cluster_args.multi_layer,
            #     n_clusters=num_clusters,
            # )

    if cluster_args.cluster_method == "mc_smoe":
        logger.info(f"Performing MC-SMoE adpative layer-wise merging...")
        cluster_labels = mc_smoe_clustering(
            distances,
            all_layer_expert_proba,
            total_clusters=len(distances) * num_clusters,
        )
    return cluster_labels


def _load_v27_calibration_batches(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    cluster_labels: dict[int, torch.Tensor],
    ds_args,
    obs_args,
    merge_args: MergeArgs,
) -> tuple[list, dict, list[int], int]:
    """Load calibration batches and derive model attrs / sparse layer indices for v27.

    Returns ``(batches, model_attrs, sparse_layer_indices, n_batches)``.
    """
    model_attrs = MODEL_ATTRS[model.__class__.__name__]
    sparse_layer_indices = sorted(cluster_labels.keys())

    n_batches = merge_args.fisher_num_batches
    logger.info(
        f"[v27-Fisher] Loading up to {n_batches} calibration batches from "
        f"'{ds_args.dataset_name}' for Fisher estimation."
    )
    category_data_batches = load_category_batches(
        dataset_name=ds_args.dataset_name,
        split=ds_args.split,
        subset=ds_args.dataset_config_name,
        tokenizer=tokenizer,
        model_max_length=obs_args.model_max_length,
        split_by_category=False,
        return_vllm_tokens_prompt=False,
        truncate=obs_args.truncate,
        batches_per_category=n_batches,
        # Backprop through the full MoE model is memory-heavy: use micro-batch=1.
        batch_size=1,
    )
    return category_data_batches["all"], model_attrs, sparse_layer_indices, n_batches


def collect_v27_fisher(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    cluster_labels: dict[int, torch.Tensor],
    ds_args,
    obs_args,
    merge_args: MergeArgs,
    results_dir: pathlib.Path,
) -> dict[int, dict[int, dict[str, torch.Tensor]]]:
    """Load calibration batches and collect per-expert diagonal Fisher for v27."""
    batches, model_attrs, sparse_layer_indices, n_batches = (
        _load_v27_calibration_batches(
            model, tokenizer, cluster_labels, ds_args, obs_args, merge_args
        )
    )

    cache_path = merge_args.fisher_cache_path
    if cache_path is None:
        cache_path = str(results_dir / "v27_expert_fisher.pt")

    return collect_expert_unit_fisher(
        model,
        batches,
        model_attrs,
        sparse_layer_indices,
        cache_path=cache_path,
        max_batches=n_batches,
        l1_normalize=True,
        overwrite=merge_args.fisher_overwrite,
    )


def prepare_fisher_geodesic_clustering(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    observer_data: dict[int, dict[str, Any]],
    ds_args,
    obs_args,
    cluster_args: ClusterArgs,
    merge_args: MergeArgs,
    results_dir: pathlib.Path,
    seed: int,
) -> tuple[
    dict[int, torch.Tensor],
    dict[int, dict[int, dict[str, torch.Tensor]]],
]:
    """Collect endpoint Fisher and build/cache the clustering distance matrices."""
    if cluster_args.cluster_method != "agglomerative":
        raise ValueError(
            "--expert-sim=fisher_geodesic currently requires "
            "--cluster-method=agglomerative."
        )
    if cluster_args.multi_layer is not None:
        raise ValueError(
            "Multi-layer allocation is not yet defined for Fisher-geodesic "
            "clustering; cluster each MoE layer independently."
        )

    # Fisher collection only needs the layer keys; identity labels ensure every
    # observed sparse layer and every original expert is represented.
    identity_labels = {
        layer: torch.arange(len(layer_data["expert_frequency"]))
        for layer, layer_data in observer_data.items()
    }
    expert_fisher = collect_v27_fisher(
        model,
        tokenizer,
        identity_labels,
        ds_args,
        obs_args,
        merge_args,
        results_dir,
    )

    cache_path = cluster_args.fisher_distance_cache_path
    if cache_path is None:
        cache_path = str(results_dir / "fisher_geodesic_distances.pt")
    cache_path = pathlib.Path(cache_path)
    fisher_elementwise = os.environ.get("FISHER_ELEMENTWISE", "0") not in (
        "0",
        "",
        "false",
        "False",
    )
    fisher_cache_path = pathlib.Path(
        merge_args.fisher_cache_path
        or (results_dir / "v27_expert_fisher.pt")
    )
    if fisher_elementwise:
        fisher_cache_path = fisher_cache_path.with_name(
            f"{fisher_cache_path.stem}_ew{fisher_cache_path.suffix}"
        )
    cache_config = {
        "version": 1,
        "model_class": model.__class__.__name__,
        "layers": {
            int(layer): int(len(layer_data["expert_frequency"]))
            for layer, layer_data in observer_data.items()
        },
        "importance_source": merge_args.fisher_router_weight_source,
        "fisher_weight": float(cluster_args.fisher_distance_weight),
        "router_weight": float(cluster_args.router_distance_weight),
        "activation_weight": float(cluster_args.activation_distance_weight),
        "sketch_size": int(cluster_args.fisher_sketch_size),
        "seed": int(seed),
        "fisher_cache_path": str(fisher_cache_path),
        "fisher_elementwise": fisher_elementwise,
        "fisher_num_batches": int(merge_args.fisher_num_batches),
        "observer_output_file": str(obs_args.output_file_name),
        "observer_total_tokens": {
            int(layer): int(torch.as_tensor(layer_data["total_tokens"]).item())
            for layer, layer_data in observer_data.items()
        },
    }

    if (
        cache_path.exists()
        and not cluster_args.fisher_distance_overwrite
        and not merge_args.fisher_overwrite
    ):
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("config") == cache_config:
            logger.info(
                "[Fisher-cluster] Loading cached distance matrices: %s", cache_path
            )
            return payload["distances"], expert_fisher
        logger.warning(
            "[Fisher-cluster] Ignoring cache with mismatched configuration: %s",
            cache_path,
        )

    model_attrs = MODEL_ATTRS[model.__class__.__name__]
    distances, diagnostics = build_fisher_geodesic_distance_matrices(
        model,
        observer_data,
        expert_fisher,
        model_attrs,
        importance_source=merge_args.fisher_router_weight_source,
        fisher_weight=cluster_args.fisher_distance_weight,
        router_weight=cluster_args.router_distance_weight,
        activation_weight=cluster_args.activation_distance_weight,
        sketch_size=cluster_args.fisher_sketch_size,
        seed=seed,
    )

    is_rank0 = not (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_rank() != 0
    )
    if is_rank0:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": cache_config,
                "distances": distances,
                "diagnostics": diagnostics,
            },
            cache_path,
        )
        logger.info(
            "[Fisher-cluster] Saved distance matrices and diagnostics: %s",
            cache_path,
        )
    return distances, expert_fisher


def _gl_config_from_env() -> tuple[int, int, float]:
    """Read GL path-integrated Fisher config from the environment.

    Returns ``(gl_nodes, gl_iters, gl_conv_eps)``.

    - ``GL_ITERS`` (default 0): number of Picard fixed-point iterations. 0 disables the
      GL path integration entirely (pure endpoint Fisher, bit-identical to before).
    - ``GL_NODES`` (default 3): number of Gauss-Legendre quadrature nodes on ``[0, 1]``.
    - ``GL_CONV_EPS`` (default 1e-4): relative-change convergence threshold.
    """
    gl_iters = int(os.environ.get("GL_ITERS", "0"))
    gl_nodes = int(os.environ.get("GL_NODES", "3"))
    gl_conv_eps = float(os.environ.get("GL_CONV_EPS", "1e-4"))
    return gl_nodes, gl_iters, gl_conv_eps


def gl_fisher_geodesic_merge(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    cluster_labels: dict[int, torch.Tensor],
    observer_data: dict[int, dict[str, Any]],
    ds_args,
    obs_args,
    merge_args: MergeArgs,
    results_dir: pathlib.Path,
    gl_nodes: int,
    gl_iters: int,
    gl_conv_eps: float,
) -> None:
    """Gauss-Legendre *path-integrated* Fisher-Geodesic expert merge (v27-GL port).

    Runs a Picard fixed-point iteration: seed with the endpoint-Fisher merge, then
    repeatedly (a) GL-quadrature the per-expert Fisher along each expert's geodesic toward
    the running merged centroid, and (b) re-merge with that path-integrated Fisher. Mutates
    ``model`` in place (final tied experts hold the converged merge), mirroring
    :func:`merge`.
    """
    from reap.fisher_gl_v27 import (
        clustered_expert_keys,
        cluster_relative_change,
        floor_fisher,
        gauss_legendre_nodes_01,
        gl_accumulate_fisher,
        restore_expert_weights,
        set_expert_interpolants,
        snapshot_expert_weights,
    )

    batches, model_attrs, sparse_layer_indices, n_batches = (
        _load_v27_calibration_batches(
            model, tokenizer, cluster_labels, ds_args, obs_args, merge_args
        )
    )

    keys = clustered_expert_keys(cluster_labels)
    # Original per-expert weights W_i (needed as fixed geodesic endpoints and as the
    # tensors the merge blends every iteration).
    originals = snapshot_expert_weights(model, keys, model_attrs)

    # ── Iteration 0: endpoint Fisher → one-shot merge (seed) ─────────────────────
    logger.info("[v27-GL] Iteration 0: endpoint Fisher seed merge.")
    cache_path = merge_args.fisher_cache_path
    if cache_path is None:
        cache_path = str(results_dir / "v27_expert_fisher.pt")
    endpoint_fisher = collect_expert_unit_fisher(
        model,
        batches,
        model_attrs,
        sparse_layer_indices,
        cache_path=cache_path,
        max_batches=n_batches,
        l1_normalize=True,
        overwrite=merge_args.fisher_overwrite,
    )
    merge(model, cluster_labels, observer_data, merge_args, expert_fisher=endpoint_fisher)
    # Free the endpoint (seed) Fisher before the GL iterations allocate their own.
    # For the weight-wise (FISHER_ELEMENTWISE=1) variant this [out, in] Fisher is
    # multi-GB, and it is only needed for the seed merge above.
    del endpoint_fisher
    gc.collect()

    nodes = gauss_legendre_nodes_01(gl_nodes)
    logger.info(
        f"[v27-GL] {gl_iters} fixed-point iterations, {gl_nodes}-point GL "
        f"quadrature nodes={[round(t, 4) for t, _ in nodes]}."
    )

    # ── Picard fixed-point iterations ────────────────────────────────────────────
    for iteration in range(1, gl_iters + 1):
        # Current merged centroids θ*_c (experts are tied → each holds its cluster's θ*).
        merged_now = snapshot_expert_weights(model, keys, model_attrs)
        prev = merged_now

        gl_fisher = None
        for node_i, (t, w) in enumerate(nodes):
            # Place every clustered expert at its geodesic node γ_i(t) and collect Fisher.
            set_expert_interpolants(
                model, originals, merged_now, t, keys, model_attrs
            )
            node_fisher = collect_expert_unit_fisher(
                model,
                batches,
                model_attrs,
                sparse_layer_indices,
                cache_path=None,
                max_batches=n_batches,
                l1_normalize=True,
                overwrite=True,
            )
            gl_fisher = gl_accumulate_fisher(gl_fisher, node_fisher, w)
            # Free the per-node Fisher before the next backward pass (host RAM guard:
            # the 30B expert set snapshots are multi-GB float32 on CPU).
            del node_fisher
            logger.info(
                f"[v27-GL] iter {iteration}: collected node {node_i + 1}/{len(nodes)} "
                f"(t={t:.4f}, w={w:.4f})."
            )
        gl_fisher = floor_fisher(gl_fisher)

        # Blend the *original* expert weights with the GL-integrated Fisher.
        restore_expert_weights(model, originals, model_attrs)
        merge(model, cluster_labels, observer_data, merge_args, expert_fisher=gl_fisher)

        curr = snapshot_expert_weights(model, keys, model_attrs)
        rel_change = cluster_relative_change(prev, curr)
        logger.info(
            f"[v27-GL] Iteration {iteration}/{gl_iters} relative change: {rel_change:.6f}"
        )
        # Release this iteration's full-weight CPU snapshots and Fisher before the next
        # iteration re-snapshots the model (keeps peak host RAM ≈ originals + 1 snapshot,
        # not a growing pile — see OOM on job 188051 at iter 2 under --mem=256G).
        del merged_now, prev, curr, gl_fisher
        gc.collect()
        if rel_change < gl_conv_eps:
            logger.info(f"[v27-GL] Converged at iteration {iteration}.")
            break


def compact_router(
    model: nn.Module,
    cluster_labels: dict[int, torch.Tensor],
    observer_data: dict[int, dict[str, Any]],
    model_attrs: dict,
    mode: str,
) -> None:
    """Fix the router after expert merging so routing diversity is preserved.

    The expert FFN weights within each cluster are identical after merging, but
    by default the router gate still has 128 output rows (one per original expert)
    and routes to all of them — burning the per-token expert budget on duplicates
    and collapsing MC/general knowledge performance.

    Two modes (selected via the ROUTER_COMPACT env-var):

    ``slice``  — REAP-style compaction.  For each cluster the router rows are
        averaged and written into the dominant-expert slot, then the router is
        sliced to ``num_clusters`` rows and ``config.num_experts`` is updated.
        The model shrinks its routing table to match the reduced expert set.
        Produces a smaller, clean model — same approach as prune.py.

    ``zero``   — Soft masking (no shape change).  All non-dominant router rows
        within each cluster are zeroed out so the softmax always routes tokens
        exclusively to the dominant expert slot of that cluster.  The model
        keeps 128 slots but only num_clusters of them are ever activated.
        Easier to reload (identical config / weight shapes).
    """
    if mode not in ("slice", "zero"):
        return

    for layer_idx, layer in enumerate(cluster_labels):
        moe = get_moe(model, layer)
        cluster_label = cluster_labels[layer]

        # expert_frequency gives usage counts → dominant = highest-frequency slot
        expert_proba = (
            observer_data[layer]["expert_frequency"]
            / observer_data[layer]["total_tokens"]
        )

        router = getattr(moe, model_attrs["router"])
        W = router.weight.data  # [num_experts, hidden]

        if mode == "zero":
            # Zero all non-dominant rows in each cluster; dominant row keeps its
            # original router logit so the temperature / softmax scale is unchanged.
            for cluster_id in cluster_label.unique():
                expert_indices = torch.where(cluster_label == cluster_id)[0].tolist()
                if len(expert_indices) <= 1:
                    continue
                dom = int(expert_proba[expert_indices].argmax())
                dom_global = expert_indices[dom]
                for idx in expert_indices:
                    if idx != dom_global:
                        W[idx] = 0.0

        elif mode == "slice":
            # Average router rows within each cluster into the dominant slot, then
            # compact the router and expert list to num_clusters rows/experts.
            cluster_ids = cluster_label.unique()
            dom_indices = []
            for cluster_id in cluster_ids:
                expert_indices = torch.where(cluster_label == cluster_id)[0].tolist()
                dom = int(expert_proba[expert_indices].argmax())
                dom_global = expert_indices[dom]
                # Replace dominant row with the mean of the cluster's router rows.
                W[dom_global] = W[expert_indices].mean(dim=0)
                dom_indices.append(dom_global)

            # Slice router to num_clusters rows.
            router.weight.data = W[dom_indices]
            if getattr(router, "bias", None) is not None:
                router.bias.data = router.bias.data[dom_indices]
            if hasattr(router, "e_score_correction_bias"):
                router.e_score_correction_bias.data = (
                    router.e_score_correction_bias.data[dom_indices]
                )
            router.out_features = len(dom_indices)
            if hasattr(router, "num_experts"):  # transformers >= 4.54
                router.num_experts = len(dom_indices)
            setattr(moe, model_attrs["router"], router)

            # Slice expert list to keep only the dominant expert per cluster.
            if not model_attrs.get("fused", False):
                all_experts = getattr(moe, model_attrs["experts"])
                retained = torch.nn.ModuleList([all_experts[i] for i in dom_indices])
                setattr(moe, model_attrs["experts"], retained)
            # fused experts not handled (Qwen3 uses non-fused)

            moe.num_experts = len(dom_indices)

    if mode == "slice":
        num_clusters = len(cluster_labels[list(cluster_labels.keys())[0]].unique())
        setattr(model.config, model_attrs["num_experts"], num_clusters)
        logger.info(
            f"[compact_router] Sliced router + experts to {num_clusters} per layer "
            f"(config.num_experts updated)."
        )
    else:
        logger.info("[compact_router] Zeroed non-dominant router rows (128 slots kept).")


def merge(
    model: nn.Module,
    cluster_labels: dict[int, torch.Tensor],
    observer_data: dict[int, dict[str, Any]],
    merge_args: MergeArgs,
    expert_fisher: dict[int, dict[int, dict[str, torch.Tensor]]] | None = None,
):
    """Merge experts based on the clustering results."""
    logger.info(f"Merging experts using method '{merge_args.merge_method}'")
    model_attrs = MODEL_ATTRS[model.__class__.__name__]

    try:
        merge_method = MergeMethod(merge_args.merge_method)
    except ValueError:
        raise NotImplementedError(
            f"Merge method '{merge_args.merge_method}' is not implemented. "
            f"Supported methods: {[method.value for method in MergeMethod]}"
        )

    if merge_method == MergeMethod.V27_FISHER_GEODESIC:
        if expert_fisher is None:
            raise ValueError(
                "expert_fisher must be provided for the v27_fisher_geodesic merge."
            )
        if merge_args.permute:
            logger.warning(
                "v27_fisher_geodesic ignores weight permutation (permute=%s); the "
                "per-unit Fisher blend performs neuron alignment implicitly. "
                "Disabling permutation for this merge.",
                merge_args.permute,
            )

    for layer_idx, layer in enumerate(tqdm(cluster_labels, "Merging layers...")):
        if merge_args.skip_first and layer_idx == 0:
            logger.info(
                f"Skipping merging for layer {layer_idx} as per 'skip_first' argument."
            )
            continue

        if merge_args.skip_last and layer_idx == len(cluster_labels) - 1:
            logger.info(
                f"Skipping merging for layer {layer_idx} as per 'skip_last' argument."
            )
            continue

        expert_proba = (
            observer_data[layer]["expert_frequency"]
            / observer_data[layer]["total_tokens"]
        )
        fisher_importance = None
        if merge_method == MergeMethod.V27_FISHER_GEODESIC:
            fisher_importance = router_importance_from_observer(
                observer_data[layer],
                merge_args.fisher_router_weight_source,
            )
        cluster_label = cluster_labels[layer]
        moe = get_moe(model, layer)
        permutation = (
            None
            if merge_method == MergeMethod.V27_FISHER_GEODESIC
            else merge_args.permute
        )
        layer_fisher = (
            expert_fisher.get(layer) if expert_fisher is not None else None
        )
        merger = MoEExpertMerger(
            moe=moe,
            cluster_label=cluster_label,
            expert_proba=expert_proba,
            model_attrs=model_attrs,
            merge_method=merge_method,
            dom_as_base=merge_args.dom_as_base,
            select_top_k=merge_args.select_top_k,
            permute=permutation,
            tie_tensors=merge_args.save_as_tied_params,
            expert_fisher=layer_fisher,
            fisher_importance=fisher_importance,
        )
        merger.merge_experts()
        assert_merge(model, moe, cluster_label)


def save_merged_model(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    merged_model_dir: pathlib.Path,
    safe_serialization,
) -> pathlib.Path:
    logger.info("Saving merged model...")
    merged_model_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    try:
        model.save_pretrained(merged_model_dir, safe_serialization=safe_serialization)
        tokenizer.save_pretrained(merged_model_dir)
    except Exception as e:
        raise e

    end = time.time()
    logger.info(
        f"Merged model saved to {merged_model_dir} in {end - start:.2f} seconds"
    )
    return merged_model_dir


@torch.no_grad()
def smoke_test(model: nn.Module, tokenizer: AutoTokenizer):
    """Run a smoke test to ensure the model is functioning correctly."""
    prompt = "What is your name?"
    test_input = [
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        test_input,
        return_tensors="pt",
        add_generation_prompt=True,
        tokenize=True,
        # enable_thinking=False,
    ).to(model.device)
    outputs = model.generate(
        inputs,
        max_new_tokens=50,
        do_sample=True,
    )
    response = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    logger.info("Smoke test response: %s", response[0])


def get_model_dir(
    results_dir, num_clusters, cluster_labels, cluster_args, obs_args, merge_args
) -> pathlib.Path:
    cluster_desc = cluster_args.cluster_description
    if not cluster_desc:
        cluster_desc = (
            f"{cluster_args.expert_sim}_{obs_args.distance_measure}_{num_clusters}_"
            f"{cluster_args.linkage_method}_freq-penalty-{cluster_args.frequency_penalty}"
            f"_softmax-{cluster_args.softmax_temperature}_multi_layer-{cluster_args.multi_layer}"
        )
        if cluster_args.max_cluster_size is not None:
            cluster_desc += f"_max_size-{cluster_args.max_cluster_size}"
    merge_model_subdir_name = merge_args.merged_model_dir_name

    if not merge_model_subdir_name:
        merge_model_subdir_name = f"{merge_args.merge_method}-permute_{merge_args.permute}-skip_first_{merge_args.skip_first}-skip_last_{merge_args.skip_last}-multilayer_{cluster_args.multi_layer}"

    # Check for non uniform compression
    non_uniform_cluster_labels = (
        len(
            torch.unique(
                torch.tensor(
                    [
                        len(torch.unique(clusters))
                        for clusters in cluster_labels.values()
                    ]
                )
            )
        )
        > 1
    )
    if (
        non_uniform_cluster_labels
        or cluster_args.multi_layer
        or merge_args.skip_first
        or merge_args.skip_last
    ):
        logger.info("Detected non-uniform compression across layers.")
        merge_model_parent_dir_name = "non_uniform_merged_models"
    else:
        merge_model_parent_dir_name = "merged_models"

    merged_model_dir = (
        results_dir
        / merge_model_parent_dir_name
        / merge_model_subdir_name
        / cluster_desc
    )
    return merged_model_dir


def dump_args_to_yaml(
    pruned_model_dir: pathlib.Path,
    **all_args,
):
    """Dump all arguments to a YAML file."""

    def convert_paths_to_str(data):
        if isinstance(data, dict):
            return {k: convert_paths_to_str(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [convert_paths_to_str(i) for i in data]
        elif isinstance(data, pathlib.Path):
            return str(data)
        else:
            return data

    serializable_args = {}
    for name, arg in all_args.items():
        if dataclasses.is_dataclass(arg):
            serializable_args[name] = convert_paths_to_str(dataclasses.asdict(arg))
        else:
            serializable_args[name] = convert_paths_to_str(arg)

    output_path = pruned_model_dir / "reap_args.yaml"
    with open(output_path, "w") as f:
        yaml.dump(serializable_args, f, default_flow_style=False)
    logger.info(f"Arguments saved to {output_path}")


def _init_dist_group() -> int:
    """Initialise a Gloo process group from SLURM environment if SLURM_NTASKS > 1.

    Returns the local rank (0 when running single-node / non-distributed).
    Uses Gloo so that the all_reduce calls in fisher_v27 work on CPU tensors
    regardless of how many GPUs each rank uses internally via device_map="auto".
    """
    import subprocess
    import torch.distributed as dist

    ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    if ntasks <= 1:
        return 0

    rank = int(os.environ.get("SLURM_PROCID", "0"))
    world_size = ntasks

    # Resolve SLURM node-list to first hostname then to an IPv4 address via
    # InfiniBand (ibs9) to avoid Gloo binding to link-local IPv6.
    node_list = os.environ.get("SLURM_NODELIST", "localhost")
    try:
        first_host = subprocess.check_output(
            ["scontrol", "show", "hostnames", node_list],
            text=True,
        ).splitlines()[0].strip()
        ipv4_out = subprocess.check_output(
            ["getent", "ahostsv4", first_host],
            text=True,
        )
        master_addr = "127.0.0.1"
        for line in ipv4_out.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "STREAM":
                master_addr = parts[0]
                break
    except Exception:
        master_addr = "127.0.0.1"

    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", "29600")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "ibs9")

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    logger.info(
        f"[dist-Fisher] Gloo group initialised: rank={rank}/{world_size} "
        f"master={master_addr}"
    )
    return rank


def main():
    (
        reap_args,
        model_args,
        ds_args,
        obs_args,
        cluster_args,
        kd_args,
        eval_args,
        merge_args,
    ) = parse_args()
    set_seed(reap_args.seed)

    # Distributed Fisher: init process group from SLURM env when SLURM_NTASKS > 1.
    # rank==0 is the coordinator; all ranks run the full GL loop (deterministic
    # operations are identical on every rank), but only rank-0 writes to disk.
    _dist_rank = _init_dist_group()
    _is_rank0 = (_dist_rank == 0)
    results_dir = create_results_directory(model_args.model_name, ds_args.dataset_name)

    if cluster_args.singleton_super_experts and cluster_args.singleton_outlier_experts:
        raise ValueError(
            "Both 'singleton_super_experts' in clustering and 'perserve_super_experts' in merging cannot be set to True."
        )
    # get local patched model if req'd
    model_name = patched_model_map(model_args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
        # local_files_only=True,
    )

    # record activations or load previously recorded activations
    logger.info(
        f"Running observer to collect activation data for model {model_args.model_name} on dataset {ds_args.dataset_name}."
    )
    observer_data = record_activations(
        model,
        tokenizer,
        reap_args,
        model_args,
        ds_args,
        obs_args,
        results_dir,
    )
    if reap_args.run_observer_only:
        logger.info(
            "Observer run completed. Exiting after collecting activation data since "
            "`run_observer_only` is set to True."
        )
        return

    # clustering
    logger.info("Start of clustering")
    num_clusters = cluster_args.num_clusters
    if num_clusters is None:
        if cluster_args.compression_ratio is None:
            raise ValueError(
                "Either num_clusters or compression_ratio must be set for clustering."
            )
        else:
            # Calculate num_clusters from compression_ratio
            if not merge_args.skip_first and not merge_args.skip_last:
                total_experts = len(
                    observer_data[next(iter(observer_data))]["expert_frequency"]
                )
                num_clusters = int(total_experts * (1 - cluster_args.compression_ratio))
            else:
                # If skipping first or last layer, adjust total_experts accordingly
                experts_per_layer = len(
                    observer_data[next(iter(observer_data))]["expert_frequency"]
                )
                layers = len(observer_data)
                total_experts = layers * experts_per_layer
                total_clusters = int(
                    total_experts * (1 - cluster_args.compression_ratio)
                )
                total_layers = len(observer_data)
                if merge_args.skip_first:
                    total_layers -= 1
                if merge_args.skip_last:
                    total_layers -= 1
                num_clusters = int(total_clusters / total_layers)
            logger.info(
                f"Calculated num_clusters: {num_clusters} from compression_ratio: {cluster_args.compression_ratio}"
            )
    precomputed_distances = None
    precluster_fisher = None
    if cluster_args.expert_sim == "fisher_geodesic":
        precomputed_distances, precluster_fisher = (
            prepare_fisher_geodesic_clustering(
                model,
                tokenizer,
                observer_data,
                ds_args,
                obs_args,
                cluster_args,
                merge_args,
                results_dir,
                reap_args.seed,
            )
        )

    cluster_labels = cluster(
        observer_data,
        num_clusters,
        cluster_args,
        obs_args.distance_measure,
        results_dir,
        precomputed_distances=precomputed_distances,
    )
    logger.info("Clustering completed.")

    # merging
    logging.info("Start of merging")
    merged_model_dir = get_model_dir(
        results_dir,
        num_clusters,
        cluster_labels,
        cluster_args,
        obs_args,
        merge_args,
    )
    if (
        merged_model_dir.exists()
        and list(merged_model_dir.glob("*.safetensors"))
        and not merge_args.overwrite_merged_model
    ):
        logger.info(
            f"Merged model files already exist in {merged_model_dir}. Skipping merging."
        )
    else:
        is_v27 = (
            MergeMethod(merge_args.merge_method) == MergeMethod.V27_FISHER_GEODESIC
        )
        gl_nodes, gl_iters, gl_conv_eps = _gl_config_from_env()
        if is_v27 and gl_iters > 0:
            if precluster_fisher is not None:
                # The clustering pass has already persisted this Fisher cache.
                # GL reloads it for the seed merge; drop this reference first so
                # element-wise caches do not occupy host RAM twice.
                del precluster_fisher
                precluster_fisher = None
                gc.collect()
            logger.info(
                "Running v27-GL path-integrated Fisher-Geodesic merge "
                f"(GL_ITERS={gl_iters}, GL_NODES={gl_nodes})..."
            )
            gl_fisher_geodesic_merge(
                model,
                tokenizer,
                cluster_labels,
                observer_data,
                ds_args,
                obs_args,
                merge_args,
                results_dir,
                gl_nodes=gl_nodes,
                gl_iters=gl_iters,
                gl_conv_eps=gl_conv_eps,
            )
        else:
            expert_fisher = None
            if is_v27:
                if precluster_fisher is not None:
                    logger.info(
                        "Reusing the endpoint Fisher collected for clustering."
                    )
                    expert_fisher = precluster_fisher
                else:
                    logger.info(
                        "Collecting per-expert Fisher for "
                        "v27_fisher_geodesic merge..."
                    )
                    expert_fisher = collect_v27_fisher(
                        model,
                        tokenizer,
                        cluster_labels,
                        ds_args,
                        obs_args,
                        merge_args,
                        results_dir,
                    )
            merge(
                model,
                cluster_labels,
                # num_clusters,
                observer_data,
                merge_args,
                expert_fisher=expert_fisher,
            )
        logger.info("Merging completed.")

        # Router compaction: fix routing diversity lost by expert merging.
        # ROUTER_COMPACT=slice -> REAP-style, compact to num_clusters experts.
        # ROUTER_COMPACT=zero  -> zero non-dominant router rows, keep 128 slots.
        # Default (unset / empty) -> no router update (original behaviour).
        router_compact_mode = os.environ.get("ROUTER_COMPACT", "").strip().lower()
        if router_compact_mode in ("slice", "zero"):
            model_attrs = MODEL_ATTRS[model.__class__.__name__]
            compact_router(model, cluster_labels, observer_data, model_attrs, router_compact_mode)

        if _is_rank0:
            logger.info("Saving merged model...")
            merged_model_dir = save_merged_model(
                model,
                tokenizer,
                merged_model_dir,
                safe_serialization=True if not merge_args.save_as_tied_params else False,
            )
            logger.info(f"Merged model saved to {merged_model_dir}.")

            # save clustering results
            logger.info("Saving clustering results...")
            cluster_analysis_dir = merged_model_dir / "clusters"
            cluster_analysis_dir.mkdir(parents=True, exist_ok=True)
            with open(cluster_analysis_dir / "clusters.pkl", "wb") as f:
                pickle.dump(cluster_labels, f)

            if reap_args.plot_clusters:
                logger.info("Plotting clusters analysis...")
                plot_cluster_analysis(
                    cluster_labels,
                    cluster_analysis_dir,
                    merge_args.skip_first,
                    merge_args.skip_last,
                )
            logger.info(
                f"Clustering results saved to {merged_model_dir / cluster_analysis_dir}"
            )

            # smoke test
            if reap_args.smoke_test:
                logger.info("Running smoke test on the merged model...")
                try:
                    smoke_test(model, tokenizer)
                except Exception as e:
                    logger.error(f"Smoke test failed: {e}")
                    pass

            dump_args_to_yaml(
                merged_model_dir,
                reap_args=reap_args,
                model_args=model_args,
                ds_args=ds_args,
                obs_args=obs_args,
                cluster_args=cluster_args,
                kd_args=kd_args,
                eval_args=eval_args,
                merge_args=merge_args,
            )

            if model_name == "artifacts/models/GLM-4.5-Air":
                # move modelling file
                source_file = pathlib.Path(model_name) / "modeling_glm4_moe.py"
                target_file = merged_model_dir / "modeling_glm4_moe.py"
                if source_file.exists():
                    shutil.copy2(source_file, target_file)
                    logger.info(f"Copied modeling_glm4_moe.py to {merged_model_dir}")
                else:
                    raise RuntimeError(
                        f"Source file {source_file} does not exist. Cannot copy to {target_file}."
                    )

        # All ranks sync after merge/save so non-rank-0 nodes wait for rank-0 writes.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
            logger.info(
                f"[dist-Fisher] Rank {_dist_rank}: barrier passed, save complete."
            )

    # eval
    if reap_args.do_eval:
        remove_hook_from_module(model, recurse=True)
        model.to("cpu")
        del model
        del observer_data
        del cluster_labels
        torch.cuda.empty_cache()
        gc.collect()
        model_args.model_name = merged_model_dir
        run_evaluate(model_args, merged_model_dir / "eval", eval_args, reap_args.seed)

    # Tear down distributed group if it was initialised.
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
