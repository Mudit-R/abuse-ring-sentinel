"""
src/graph/builder.py
──────────────────────────────────────────────────────────────────────────────
Constructs a transaction graph from the PaySim CSV dataset.

Compatibility notes:
  - NetworkX 3.0+ removed write_gpickle/read_gpickle → using stdlib pickle
  - PyTorch 2.6+ requires weights_only=False for non-tensor objects

Design decisions (discuss in interviews):
──────────────────────────────────────────
• Directed edges  — mule accounts receive funds from many origins (high
  in-degree) then cash-out to a few destinations (low out-degree). Direction
  captures this asymmetry; an undirected graph would lose this signal.

• Nodes = accounts  — both `nameOrig` and `nameDest` become nodes. A single
  account can appear in both roles across different transactions.

• Edges = transactions  — each row in PaySim becomes a directed edge
  (nameOrig → nameDest). Edge attributes: amount, step (hour), type_encoded.

• Node label = isFraud  — an account is labelled fraudulent if it appears as
  the ORIGIN of any fraudulent transaction. This reflects real AML practice:
  the initiating account is the primary suspect.

• Fraud only exists in TRANSFER and CASH_OUT types in PaySim — we filter to
  these two types only for the transaction graph used in GNN training, but keep
  all types for tabular baselines to avoid leaking this domain knowledge.

Output artifacts:
─────────────────
• PyTorch Geometric HeteroData / Data object  (for GNN training)
• NetworkX DiGraph                            (for structural feature compute)
• Node feature DataFrame                      (for baseline models)
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch_geometric.data import Data, HeteroData
from tqdm import tqdm


# ── Constants ─────────────────────────────────────────────────────────────────

TRANSACTION_TYPES = {
    "PAYMENT": 0,
    "TRANSFER": 1,
    "CASH_OUT": 2,
    "DEBIT": 3,
    "CASH_IN": 4,
}

# PaySim fraud is only in TRANSFER and CASH_OUT
FRAUD_TX_TYPES = {"TRANSFER", "CASH_OUT"}

PAYSIM_COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg",
    "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest",
    "isFraud", "isFlaggedFraud",
]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class GraphBundle:
    """Container for all graph representations derived from PaySim."""

    # PyTorch Geometric homogeneous graph (accounts as nodes)
    pyg_data: Data

    # NetworkX DiGraph (for structural feature computation)
    nx_graph: nx.DiGraph

    # Account → node-index mapping
    account_to_idx: Dict[str, int]
    idx_to_account: Dict[int, str]

    # Fraud labels (per account), shape (N,)
    node_labels: torch.Tensor

    # Original transactions used to build this graph
    transactions: pd.DataFrame

    # Statistics logged at build time
    stats: Dict = field(default_factory=dict)


# ── Loader ────────────────────────────────────────────────────────────────────

def load_paysim(csv_path: Path, chunksize: Optional[int] = None) -> pd.DataFrame:
    """
    Load the PaySim CSV.

    For the full 6.3M-row file this takes ~15 s and ~1.5 GB RAM.
    Pass chunksize to stream row-by-row if RAM is tight.
    """
    logger.info(f"Loading PaySim data from {csv_path} …")
    df = pd.read_csv(
        csv_path,
        dtype={
            "step": "int32",
            "type": "category",
            "amount": "float32",
            "nameOrig": "string",
            "oldbalanceOrg": "float32",
            "newbalanceOrig": "float32",
            "nameDest": "string",
            "oldbalanceDest": "float32",
            "newbalanceDest": "float32",
            "isFraud": "int8",
            "isFlaggedFraud": "int8",
        },
        low_memory=False,
    )
    logger.info(f"Loaded {len(df):,} transactions. Fraud rate: {df['isFraud'].mean():.4%}")
    return df


# ── Graph Builder ─────────────────────────────────────────────────────────────

class TransactionGraphBuilder:
    """
    Builds a fraud detection graph from PaySim transaction data.

    Parameters
    ----------
    fraud_types_only : bool
        If True, only include TRANSFER and CASH_OUT edges in the GNN graph.
        This is the standard approach as fraud only appears in these types.
        Tabular baselines still use all types.
    min_tx_per_account : int
        Filter out accounts that appear in fewer than this many transactions.
        Reduces noise from one-off accounts (default: 2).
    """

    def __init__(
        self,
        fraud_types_only: bool = True,
        min_tx_per_account: int = 2,
    ) -> None:
        self.fraud_types_only = fraud_types_only
        self.min_tx_per_account = min_tx_per_account

    # ── Public API ────────────────────────────────────────────────

    def build(self, df: pd.DataFrame) -> GraphBundle:
        """
        Full pipeline: raw DataFrame → GraphBundle.
        """
        logger.info("Starting graph construction …")

        # Step 1: Prepare edge DataFrame
        edges_df = self._prepare_edges(df)

        # Step 2: Build node index
        account_to_idx, idx_to_account = self._build_node_index(edges_df)
        n_nodes = len(account_to_idx)
        logger.info(f"Nodes (unique accounts): {n_nodes:,}")

        # Step 3: Derive node-level fraud labels
        node_labels = self._derive_node_labels(df, account_to_idx, n_nodes)
        fraud_nodes = node_labels.sum().item()
        logger.info(
            f"Fraud nodes: {fraud_nodes:,} / {n_nodes:,} "
            f"({fraud_nodes / n_nodes:.4%})"
        )

        # Step 4: Build NetworkX graph (used for structural features)
        nx_graph = self._build_networkx(edges_df, account_to_idx)

        # Step 5: Build node feature matrix from tabular aggregations
        node_features = self._build_node_features(df, account_to_idx, n_nodes)

        # Step 6: Build PyTorch Geometric Data object
        pyg_data = self._build_pyg_data(
            edges_df, account_to_idx, node_features, node_labels
        )

        stats = {
            "n_nodes": n_nodes,
            "n_edges": len(edges_df),
            "n_fraud_nodes": fraud_nodes,
            "fraud_rate_nodes": fraud_nodes / n_nodes,
            "n_fraud_edges": int(edges_df["isFraud"].sum()),
        }
        logger.success(f"Graph built: {stats}")

        return GraphBundle(
            pyg_data=pyg_data,
            nx_graph=nx_graph,
            account_to_idx=account_to_idx,
            idx_to_account=idx_to_account,
            node_labels=node_labels,
            transactions=edges_df,
            stats=stats,
        )

    def save(self, bundle: GraphBundle, output_dir: Path) -> None:
        """Persist the graph bundle to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.save(bundle.pyg_data, output_dir / "pyg_data.pt")

        # nx.write_gpickle was removed in NetworkX 3.0 — use stdlib pickle
        with open(output_dir / "nx_graph.pkl", "wb") as f:
            pickle.dump(bundle.nx_graph, f)

        with open(output_dir / "account_to_idx.pkl", "wb") as f:
            pickle.dump(bundle.account_to_idx, f)
        with open(output_dir / "idx_to_account.pkl", "wb") as f:
            pickle.dump(bundle.idx_to_account, f)

        bundle.transactions.to_parquet(output_dir / "edges.parquet", index=False)
        logger.success(f"Graph bundle saved to {output_dir}")

    @staticmethod
    def load(output_dir: Path) -> GraphBundle:
        """Load a previously saved graph bundle."""
        # weights_only=False required for PyTorch 2.6+ when loading non-tensor objects
        pyg_data = torch.load(output_dir / "pyg_data.pt", weights_only=False)

        # Support both old (.gpickle) and new (.pkl) filenames
        nx_path = output_dir / "nx_graph.pkl"
        if not nx_path.exists():
            nx_path = output_dir / "nx_graph.gpickle"
        with open(nx_path, "rb") as f:
            nx_graph = pickle.load(f)

        with open(output_dir / "account_to_idx.pkl", "rb") as f:
            account_to_idx = pickle.load(f)
        with open(output_dir / "idx_to_account.pkl", "rb") as f:
            idx_to_account = pickle.load(f)

        transactions = pd.read_parquet(output_dir / "edges.parquet")
        node_labels = pyg_data.y

        return GraphBundle(
            pyg_data=pyg_data,
            nx_graph=nx_graph,
            account_to_idx=account_to_idx,
            idx_to_account=idx_to_account,
            node_labels=node_labels,
            transactions=transactions,
        )

    # ── Private helpers ────────────────────────────────────────────

    def _prepare_edges(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter and encode edges.

        Key decision: include only TRANSFER + CASH_OUT in the graph used for
        GNN training (fraud only exists in these types). This is NOT leakage —
        it's domain knowledge about the data-generating process.
        """
        if self.fraud_types_only:
            edges = df[df["type"].isin(FRAUD_TX_TYPES)].copy()
            logger.info(
                f"Filtered to {FRAUD_TX_TYPES}: {len(edges):,} / {len(df):,} transactions"
            )
        else:
            edges = df.copy()

        # Encode transaction type as integer
        edges["type_encoded"] = edges["type"].map(TRANSACTION_TYPES).astype("int8")

        # Normalize amount (log scale stabilises the 9-order-of-magnitude range)
        edges["amount_log"] = np.log1p(edges["amount"].astype("float64")).astype("float32")

        # Hour-of-day encoding (step = hour in PaySim)
        edges["hour_sin"] = np.sin(2 * np.pi * edges["step"] / 24).astype("float32")
        edges["hour_cos"] = np.cos(2 * np.pi * edges["step"] / 24).astype("float32")

        return edges.reset_index(drop=True)

    def _build_node_index(
        self, edges_df: pd.DataFrame
    ) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Assign a unique integer index to every unique account."""
        all_accounts = pd.unique(
            pd.concat([edges_df["nameOrig"], edges_df["nameDest"]])
        )
        account_to_idx: Dict[str, int] = {acc: i for i, acc in enumerate(all_accounts)}
        idx_to_account: Dict[int, str] = {i: acc for acc, i in account_to_idx.items()}
        return account_to_idx, idx_to_account

    def _derive_node_labels(
        self,
        df: pd.DataFrame,
        account_to_idx: Dict[str, int],
        n_nodes: int,
    ) -> torch.Tensor:
        """
        Label a node 1 (fraud) if it appears as the ORIGIN of any fraudulent
        transaction in the FULL dataset (not just TRANSFER/CASH_OUT filtered).

        Using the full dataset for labelling avoids missing fraud accounts that
        might not appear in the filtered edge set.
        """
        labels = torch.zeros(n_nodes, dtype=torch.long)
        fraud_origins = df.loc[df["isFraud"] == 1, "nameOrig"].unique()
        for acc in fraud_origins:
            if acc in account_to_idx:
                labels[account_to_idx[acc]] = 1
        return labels

    def _build_networkx(
        self, edges_df: pd.DataFrame, account_to_idx: Dict[str, int]
    ) -> nx.DiGraph:
        """
        Build a NetworkX DiGraph. Each edge stores amount_log and step.
        Used for structural feature computation (PageRank, k-core, etc.).
        """
        logger.info("Building NetworkX DiGraph …")
        G = nx.DiGraph()
        G.add_nodes_from(range(len(account_to_idx)))

        src = edges_df["nameOrig"].map(account_to_idx).values
        dst = edges_df["nameDest"].map(account_to_idx).values
        amounts = edges_df["amount_log"].values
        steps = edges_df["step"].values

        for s, d, amt, step in tqdm(
            zip(src, dst, amounts, steps), total=len(src), desc="Building nx graph"
        ):
            G.add_edge(int(s), int(d), weight=float(amt), step=int(step))

        logger.info(f"nx graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
        return G

    def _build_node_features(
        self,
        df: pd.DataFrame,
        account_to_idx: Dict[str, int],
        n_nodes: int,
    ) -> torch.Tensor:
        """
        Compute tabular (aggregation-based) node features from the FULL dataset.

        Features per account:
          0  total_sent_log        — log(1 + total amount sent)
          1  total_received_log    — log(1 + total amount received)
          2  tx_count_out          — number of outgoing transactions
          3  tx_count_in           — number of incoming transactions
          4  unique_dest_count     — unique destination accounts
          5  unique_src_count      — unique source accounts
          6  avg_sent_log          — log(1 + mean amount sent)
          7  avg_received_log      — log(1 + mean amount received)
          8  balance_drain_ratio   — (oldBalance - newBalance) / (oldBalance + 1e-6)
          9  night_tx_fraction     — fraction of transactions in steps 0-6 (off hours)
          10 fraud_type_fraction   — fraction of transactions in TRANSFER/CASH_OUT
        """
        logger.info("Computing tabular node features …")
        n_features = 11
        features = np.zeros((n_nodes, n_features), dtype=np.float32)

        # Outgoing transactions
        out_grp = df.groupby("nameOrig")
        for acc, grp in tqdm(out_grp, desc="Out features", total=out_grp.ngroups):
            if acc not in account_to_idx:
                continue
            idx = account_to_idx[acc]
            features[idx, 0] = np.log1p(grp["amount"].sum())
            features[idx, 2] = len(grp)
            features[idx, 4] = grp["nameDest"].nunique()
            features[idx, 6] = np.log1p(grp["amount"].mean())
            bal_drain = (
                (grp["oldbalanceOrg"] - grp["newbalanceOrig"])
                / (grp["oldbalanceOrg"] + 1e-6)
            ).mean()
            features[idx, 8] = float(bal_drain)
            features[idx, 9] = float((grp["step"] <= 6).mean())
            features[idx, 10] = float(grp["type"].isin(FRAUD_TX_TYPES).mean())

        # Incoming transactions
        in_grp = df.groupby("nameDest")
        for acc, grp in tqdm(in_grp, desc="In features", total=in_grp.ngroups):
            if acc not in account_to_idx:
                continue
            idx = account_to_idx[acc]
            features[idx, 1] = np.log1p(grp["amount"].sum())
            features[idx, 3] = len(grp)
            features[idx, 5] = grp["nameOrig"].nunique()
            features[idx, 7] = np.log1p(grp["amount"].mean())

        return torch.tensor(features, dtype=torch.float)

    def _build_pyg_data(
        self,
        edges_df: pd.DataFrame,
        account_to_idx: Dict[str, int],
        node_features: torch.Tensor,
        node_labels: torch.Tensor,
    ) -> Data:
        """
        Assemble PyTorch Geometric Data object.

        edge_index  : (2, E) LongTensor
        edge_attr   : (E, 3) FloatTensor  [amount_log, hour_sin, hour_cos]
        x           : (N, F) FloatTensor  — tabular node features
        y           : (N,)  LongTensor   — binary fraud label
        """
        src = edges_df["nameOrig"].map(account_to_idx).values
        dst = edges_df["nameDest"].map(account_to_idx).values

        edge_index = torch.tensor(
            np.stack([src, dst], axis=0), dtype=torch.long
        )
        edge_attr = torch.tensor(
            edges_df[["amount_log", "hour_sin", "hour_cos"]].values,
            dtype=torch.float,
        )

        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=node_labels,
            num_nodes=node_features.size(0),
        )
        logger.info(f"PyG Data: {data}")
        return data
