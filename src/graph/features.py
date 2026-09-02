"""
src/graph/features.py
──────────────────────────────────────────────────────────────────────────────
Compute graph-structural and temporal features for all nodes in the
transaction graph.

Two backends:
  • NetworkX  — CPU-only, works on Windows, always available.
  • cuGraph   — GPU-accelerated via RAPIDS, requires Linux/WSL2.
                Automatically falls back to NetworkX if not available.

Features computed:
──────────────────
  Structural (via PageRank, k-core, clustering):
    - in_degree, out_degree, degree_ratio (out / (in+1))
    - pagerank                            (measures node centrality)
    - k_core_number                       (measures embeddedness in dense subgraph)
    - local_clustering_coefficient        (how clique-like the neighbourhood is)

  Temporal rolling (computed directly from transaction DataFrame):
    - tx_velocity_24h    — tx count in last 24 steps (hours)
    - tx_velocity_7d     — tx count in last 7*24 = 168 steps
    - amount_velocity_24h — total amount sent in last 24 steps (log-scaled)
    - amount_spike       — ratio of 24h amount to 7d amount (sudden spikes)
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Dict, Optional

import networkx as nx
import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

# Try importing cuGraph — silently fall back to NetworkX if unavailable
try:
    import cugraph
    import cudf
    CUGRAPH_AVAILABLE = True
    logger.info("cuGraph available — GPU acceleration enabled.")
except ImportError:
    CUGRAPH_AVAILABLE = False
    logger.warning("cuGraph not found. Using NetworkX (CPU). "
                   "For GPU acceleration, set up the RAPIDS environment.")


# ── Structural Features ───────────────────────────────────────────────────────

class StructuralFeatureComputer:
    """
    Computes graph-structural node features using either NetworkX or cuGraph.

    Parameters
    ----------
    use_gpu : bool
        Force GPU usage. If cuGraph is unavailable, raises an error.
        Default: auto-detect.
    """

    def __init__(self, use_gpu: Optional[bool] = None) -> None:
        if use_gpu is True and not CUGRAPH_AVAILABLE:
            raise RuntimeError(
                "use_gpu=True but cuGraph is not installed. "
                "Activate the fraud-detection-rapids conda environment."
            )
        self.use_gpu = CUGRAPH_AVAILABLE if use_gpu is None else use_gpu

    def compute(self, G: nx.DiGraph) -> pd.DataFrame:
        """
        Compute structural features for all nodes.

        Returns
        -------
        pd.DataFrame with index = node_id and columns for each feature.
        Also benchmarks GPU vs CPU time if cuGraph is available.
        """
        if self.use_gpu:
            return self._compute_cugraph(G)
        else:
            return self._compute_networkx(G)

    # ── NetworkX (CPU) ────────────────────────────────────────────

    def _compute_networkx(self, G: nx.DiGraph) -> pd.DataFrame:
        logger.info("Computing structural features with NetworkX (CPU) …")
        n_nodes = G.number_of_nodes()

        features: Dict[str, np.ndarray] = {}

        # Degree
        t0 = time.perf_counter()
        in_deg = dict(G.in_degree())
        out_deg = dict(G.out_degree())
        features["in_degree"] = np.array([in_deg.get(i, 0) for i in range(n_nodes)], dtype=np.float32)
        features["out_degree"] = np.array([out_deg.get(i, 0) for i in range(n_nodes)], dtype=np.float32)
        features["degree_ratio"] = features["out_degree"] / (features["in_degree"] + 1)
        logger.info(f"  Degree: {time.perf_counter() - t0:.2f}s")

        # PageRank
        t0 = time.perf_counter()
        pr = nx.pagerank(G, alpha=0.85, weight="weight", max_iter=100)
        features["pagerank"] = np.array([pr.get(i, 0.0) for i in range(n_nodes)], dtype=np.float32)
        logger.info(f"  PageRank: {time.perf_counter() - t0:.2f}s")

        # K-core (on undirected version)
        t0 = time.perf_counter()
        G_undirected = G.to_undirected()
        core_numbers = nx.core_number(G_undirected)
        features["k_core_number"] = np.array(
            [core_numbers.get(i, 0) for i in range(n_nodes)], dtype=np.float32
        )
        logger.info(f"  K-core: {time.perf_counter() - t0:.2f}s")

        # Clustering coefficient
        t0 = time.perf_counter()
        # Use undirected for clustering (standard practice)
        clustering = nx.clustering(G_undirected)
        features["local_clustering_coefficient"] = np.array(
            [clustering.get(i, 0.0) for i in range(n_nodes)], dtype=np.float32
        )
        logger.info(f"  Clustering: {time.perf_counter() - t0:.2f}s")

        return pd.DataFrame(features)

    # ── cuGraph (GPU) ─────────────────────────────────────────────

    def _compute_cugraph(self, G: nx.DiGraph) -> pd.DataFrame:
        logger.info("Computing structural features with cuGraph (GPU) …")

        # Convert NetworkX edge list to cuDF DataFrame
        edges = list(G.edges(data=True))
        src = [e[0] for e in edges]
        dst = [e[1] for e in edges]
        weights = [e[2].get("weight", 1.0) for e in edges]

        t_transfer = time.perf_counter()
        edge_df = cudf.DataFrame({"src": src, "dst": dst, "weight": weights})
        logger.info(f"  Host→GPU transfer: {time.perf_counter() - t_transfer:.3f}s")

        G_cu = cugraph.Graph(directed=True)
        G_cu.from_cudf_edgelist(edge_df, source="src", destination="dst", edge_attr="weight")

        features: Dict[str, np.ndarray] = {}
        n_nodes = G.number_of_nodes()

        # Degree
        t0 = time.perf_counter()
        deg_df = G_cu.in_degree()
        in_deg = dict(zip(deg_df["vertex"].to_pandas(), deg_df["in_degree"].to_pandas()))
        deg_df = G_cu.out_degree()
        out_deg = dict(zip(deg_df["vertex"].to_pandas(), deg_df["out_degree"].to_pandas()))
        features["in_degree"] = np.array([in_deg.get(i, 0) for i in range(n_nodes)], dtype=np.float32)
        features["out_degree"] = np.array([out_deg.get(i, 0) for i in range(n_nodes)], dtype=np.float32)
        features["degree_ratio"] = features["out_degree"] / (features["in_degree"] + 1)
        logger.info(f"  Degree (GPU): {time.perf_counter() - t0:.3f}s")

        # PageRank
        t0 = time.perf_counter()
        pr_df = cugraph.pagerank(G_cu, alpha=0.85, max_iter=100)
        pr_dict = dict(zip(pr_df["vertex"].to_pandas(), pr_df["pagerank"].to_pandas()))
        features["pagerank"] = np.array([pr_dict.get(i, 0.0) for i in range(n_nodes)], dtype=np.float32)
        logger.info(f"  PageRank (GPU): {time.perf_counter() - t0:.3f}s")

        # K-core
        t0 = time.perf_counter()
        G_cu_undir = cugraph.Graph(directed=False)
        G_cu_undir.from_cudf_edgelist(edge_df, source="src", destination="dst")
        core_df = cugraph.k_core(G_cu_undir)
        core_dict = dict(zip(core_df["vertex"].to_pandas(), core_df["core_number"].to_pandas()))
        features["k_core_number"] = np.array(
            [core_dict.get(i, 0) for i in range(n_nodes)], dtype=np.float32
        )
        logger.info(f"  K-core (GPU): {time.perf_counter() - t0:.3f}s")

        # Clustering
        t0 = time.perf_counter()
        tri_df = cugraph.triangle_count(G_cu_undir)
        # Approximate clustering coefficient from triangle counts
        # clustering ≈ 2T / (d*(d-1)) where T = triangles, d = degree
        deg_arr = features["in_degree"] + features["out_degree"]
        tri_dict = dict(zip(tri_df["vertex"].to_pandas(), tri_df["counts"].to_pandas()))
        triangles = np.array([tri_dict.get(i, 0) for i in range(n_nodes)], dtype=np.float32)
        denom = deg_arr * (deg_arr - 1)
        denom = np.where(denom > 0, denom, 1)
        features["local_clustering_coefficient"] = (2.0 * triangles / denom).astype(np.float32)
        logger.info(f"  Clustering (GPU): {time.perf_counter() - t0:.3f}s")

        return pd.DataFrame(features)


# ── Temporal Features ─────────────────────────────────────────────────────────

def compute_temporal_features(
    df: pd.DataFrame,
    account_to_idx: Dict[str, int],
) -> pd.DataFrame:
    """
    Compute rolling temporal features per account.

    Uses time steps (hours) in PaySim as the temporal axis.
    Rolling windows:
      - 24 steps  ≈ 24 hours (short-term velocity)
      - 168 steps ≈ 7 days   (medium-term baseline)

    Returns
    -------
    pd.DataFrame indexed by node_id (matches account_to_idx).
    """
    logger.info("Computing temporal rolling features …")

    n_nodes = len(account_to_idx)
    results = {
        "tx_velocity_24h": np.zeros(n_nodes, dtype=np.float32),
        "tx_velocity_7d": np.zeros(n_nodes, dtype=np.float32),
        "amount_velocity_24h": np.zeros(n_nodes, dtype=np.float32),
        "amount_velocity_7d": np.zeros(n_nodes, dtype=np.float32),
        "amount_spike_ratio": np.zeros(n_nodes, dtype=np.float32),
    }

    # Use the maximum step as "now" (latest hour in the dataset)
    t_max = int(df["step"].max())
    window_24h = 24
    window_7d = 168

    grp = df.groupby("nameOrig")
    for acc, g in tqdm(grp, desc="Temporal features", total=grp.ngroups):
        if acc not in account_to_idx:
            continue
        idx = account_to_idx[acc]

        steps = g["step"].values
        amounts = g["amount"].values

        mask_24h = steps >= (t_max - window_24h)
        mask_7d = steps >= (t_max - window_7d)

        tx_24h = int(mask_24h.sum())
        tx_7d = int(mask_7d.sum())
        amt_24h = float(np.log1p(amounts[mask_24h].sum()))
        amt_7d = float(np.log1p(amounts[mask_7d].sum()))

        results["tx_velocity_24h"][idx] = tx_24h
        results["tx_velocity_7d"][idx] = tx_7d
        results["amount_velocity_24h"][idx] = amt_24h
        results["amount_velocity_7d"][idx] = amt_7d
        # Spike ratio: 24h amount vs 7d amount — sudden spikes = mule behavior
        results["amount_spike_ratio"][idx] = amt_24h / (amt_7d + 1e-6)

    return pd.DataFrame(results)


# ── Benchmark Utility ─────────────────────────────────────────────────────────

def benchmark_networkx_vs_cugraph(G: nx.DiGraph) -> Dict[str, float]:
    """
    Run PageRank + K-core on both NetworkX and cuGraph and return timing dict.
    Used in notebooks/07_gpu_benchmark.ipynb.
    """
    results = {}

    # NetworkX
    computer_cpu = StructuralFeatureComputer(use_gpu=False)
    t0 = time.perf_counter()
    _ = computer_cpu.compute(G)
    results["networkx_total_s"] = time.perf_counter() - t0
    logger.info(f"NetworkX total: {results['networkx_total_s']:.2f}s")

    if CUGRAPH_AVAILABLE:
        computer_gpu = StructuralFeatureComputer(use_gpu=True)
        t0 = time.perf_counter()
        _ = computer_gpu.compute(G)
        results["cugraph_total_s"] = time.perf_counter() - t0
        results["speedup_x"] = results["networkx_total_s"] / results["cugraph_total_s"]
        logger.success(
            f"cuGraph total: {results['cugraph_total_s']:.2f}s  |  "
            f"Speedup: {results['speedup_x']:.1f}x"
        )
    else:
        results["cugraph_total_s"] = None
        results["speedup_x"] = None
        logger.warning("cuGraph not available — skipping GPU benchmark.")

    return results


# ── Combined Feature Matrix ───────────────────────────────────────────────────

def build_full_feature_matrix(
    structural_features: pd.DataFrame,
    temporal_features: pd.DataFrame,
    pyg_node_features: torch.Tensor,
) -> torch.Tensor:
    """
    Merge structural + temporal features with the tabular node features from
    the graph builder into a single feature matrix for baseline models.

    Returns
    -------
    torch.Tensor of shape (N, F_total).
    """
    tabular = pyg_node_features.numpy()  # (N, 11)
    structural = structural_features.values.astype(np.float32)  # (N, 6)
    temporal = temporal_features.values.astype(np.float32)  # (N, 5)

    # Fill any NaNs introduced by accounts with 0 transactions in the window
    structural = np.nan_to_num(structural, nan=0.0)
    temporal = np.nan_to_num(temporal, nan=0.0)

    combined = np.concatenate([tabular, structural, temporal], axis=1)
    logger.info(
        f"Full feature matrix: {combined.shape[1]} features "
        f"(11 tabular + {structural.shape[1]} structural + {temporal.shape[1]} temporal)"
    )
    return torch.tensor(combined, dtype=torch.float)
