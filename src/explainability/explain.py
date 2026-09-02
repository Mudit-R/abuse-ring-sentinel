"""
src/explainability/explain.py
──────────────────────────────────────────────────────────────────────────────
GNNExplainer + SHAP explainability for fraud predictions.

Why explainability matters in AML:
    AML regulations (FINRA, EU AMLD6) require that financial institutions be
    able to provide a human-readable rationale for why an account was flagged.
    Black-box models are legally insufficient. Being able to say "account X was
    flagged because 3 of its 5 direct counterparties have fraud scores > 0.8"
    is a concrete, auditable reason — exactly what GNNExplainer produces.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import shap
import torch
from loguru import logger
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.data import Data


# ── GNNExplainer ─────────────────────────────────────────────────────────────

class FraudExplainer:
    """
    Wraps PyG's GNNExplainer to produce per-node subgraph explanations.

    For each flagged account, GNNExplainer:
      1. Identifies the minimal subgraph of edges that most influenced the
         prediction (by maximising mutual information with the prediction)
      2. Returns edge masks (which connections matter) and node feature masks
         (which features matter)

    This directly answers the AML analyst's question:
      "Why was account #12345 flagged? Show me the network."
    """

    def __init__(
        self,
        model: torch.nn.Module,
        data: Data,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = device

        self.explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config={
                "mode": "binary_classification",
                "task_level": "node",
                "return_type": "probs",
            },
        )

    def explain_node(self, node_idx: int) -> Dict:
        """
        Explain the prediction for a single node.

        Returns
        -------
        dict with:
          - node_mask    : (F,) feature importance scores
          - edge_mask    : (E,) edge importance scores
          - subgraph_nx  : NetworkX subgraph of the important neighbourhood
          - fraud_prob   : Model's fraud probability for this node
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.data.x, self.data.edge_index)
            proba = torch.sigmoid(logits)[node_idx].item()

        explanation = self.explainer(
            x=self.data.x,
            edge_index=self.data.edge_index,
            index=node_idx,
        )

        return {
            "node_idx": node_idx,
            "fraud_prob": proba,
            "node_mask": explanation.node_mask.cpu().numpy(),
            "edge_mask": explanation.edge_mask.cpu().numpy(),
        }

    def explain_top_k_frauds(
        self,
        k: int = 5,
        test_proba: Optional[np.ndarray] = None,
        test_mask: Optional[torch.Tensor] = None,
    ) -> List[Dict]:
        """Explain the top-K highest-scoring nodes from the test set."""
        if test_proba is None:
            self.model.eval()
            with torch.no_grad():
                logits = self.model(self.data.x, self.data.edge_index)
                test_proba = torch.sigmoid(logits).cpu().numpy()

        if test_mask is not None:
            candidate_indices = test_mask.nonzero(as_tuple=True)[0].cpu().numpy()
            candidate_proba = test_proba[candidate_indices]
            top_k_local = np.argsort(candidate_proba)[::-1][:k]
            top_k_global = candidate_indices[top_k_local]
        else:
            top_k_global = np.argsort(test_proba)[::-1][:k]

        explanations = []
        for node_idx in top_k_global:
            logger.info(f"Explaining node {node_idx} (fraud_prob={test_proba[node_idx]:.4f}) …")
            exp = self.explain_node(int(node_idx))
            explanations.append(exp)

        return explanations

    def visualise_explanation(
        self,
        explanation: Dict,
        idx_to_account: Dict[int, str],
        output_path: Optional[Path] = None,
    ) -> None:
        """
        Plot the explanation subgraph around the flagged node.
        Edges are coloured by importance (red = high, blue = low).
        """
        node_idx = explanation["node_idx"]
        fraud_prob = explanation["fraud_prob"]
        edge_mask = explanation["edge_mask"]

        ei = self.data.edge_index.cpu().numpy()
        # Find edges that touch this node's 2-hop neighbourhood
        mask1 = (ei[0] == node_idx) | (ei[1] == node_idx)
        neighbor1 = np.unique(ei[:, mask1])
        mask2 = np.isin(ei[0], neighbor1) | np.isin(ei[1], neighbor1)
        subgraph_edges = ei[:, mask2 & (edge_mask > 0.1)]

        G = nx.DiGraph()
        for src, dst in zip(subgraph_edges[0], subgraph_edges[1]):
            importance = float(edge_mask[mask2][
                (subgraph_edges[0] == src) & (subgraph_edges[1] == dst)
            ].mean() if len(subgraph_edges[0]) else 0)
            G.add_edge(int(src), int(dst), importance=importance)

        if node_idx not in G.nodes:
            G.add_node(node_idx)

        fig, ax = plt.subplots(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42)

        node_colors = [
            "#e74c3c" if n == node_idx else "#3498db" for n in G.nodes
        ]
        edge_colors = [
            G[u][v].get("importance", 0) for u, v in G.edges
        ]

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, ax=ax)
        edges = nx.draw_networkx_edges(
            G, pos, edge_color=edge_colors, edge_cmap=plt.cm.Reds,
            width=2, ax=ax, arrows=True, arrowsize=15,
        )
        labels = {n: idx_to_account.get(n, str(n))[:8] for n in G.nodes}
        nx.draw_networkx_labels(G, pos, labels, font_size=6, ax=ax)

        ax.set_title(
            f"GNNExplainer — Node {node_idx} | Fraud Prob: {fraud_prob:.4f}",
            fontsize=14, fontweight="bold",
        )
        fraud_patch = mpatches.Patch(color="#e74c3c", label="Flagged account")
        neighbor_patch = mpatches.Patch(color="#3498db", label="Neighbour accounts")
        ax.legend(handles=[fraud_patch, neighbor_patch])
        plt.tight_layout()

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.success(f"Explanation saved: {output_path}")
        plt.show()


# ── SHAP for Tabular Baseline ─────────────────────────────────────────────────

def shap_explain_xgboost(
    model,  # fitted XGBClassifier
    X_test: np.ndarray,
    feature_names: List[str],
    output_dir: Path,
    n_samples: int = 500,
) -> None:
    """
    Run SHAP TreeExplainer on XGBoost baseline.

    Produces:
      - Global feature importance (beeswarm plot)
      - Per-account waterfall plot for the top fraud prediction
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Running SHAP TreeExplainer …")

    explainer = shap.TreeExplainer(model)
    X_sample = X_test[:n_samples]
    shap_values = explainer.shap_values(X_sample)

    # Global importance
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        plot_type="dot",
        show=False,
    )
    plt.title("SHAP Feature Importance — XGBoost Fraud Detector", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / "shap_global.png", dpi=150, bbox_inches="tight")
    logger.success(f"SHAP global plot saved: {output_dir / 'shap_global.png'}")
    plt.close()

    # Per-account waterfall for the sample with highest fraud score
    model_proba = model.predict_proba(X_sample)[:, 1]
    top_idx = int(np.argmax(model_proba))
    explanation = shap.Explanation(
        values=shap_values[top_idx],
        base_values=explainer.expected_value,
        data=X_sample[top_idx],
        feature_names=feature_names,
    )
    shap.waterfall_plot(explanation, show=False)
    plt.title(f"SHAP Waterfall — Top Fraud Account (prob={model_proba[top_idx]:.4f})")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_waterfall.png", dpi=150, bbox_inches="tight")
    logger.success(f"SHAP waterfall saved: {output_dir / 'shap_waterfall.png'}")
    plt.close()
