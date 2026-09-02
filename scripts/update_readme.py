"""
scripts/update_readme.py
──────────────────────────────────────────────────────────────────────────────
Reads outputs/final_results.json and outputs/benchmark_results.json (if exists)
and fills in the [X] placeholders in README.md with real numbers.

Usage:
    .venv\Scripts\python.exe scripts/update_readme.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def load_results():
    results = {}
    p = ROOT / "outputs" / "final_results.json"
    if p.exists():
        with open(p) as f:
            results = json.load(f)
    return results


def load_benchmark():
    p = ROOT / "outputs" / "benchmark_results.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def fmt(val, decimals=4):
    if val is None:
        return "[N/A]"
    return f"{float(val):.{decimals}f}"


def main():
    results = load_results()
    benchmark = load_benchmark()

    if not results:
        print("ERROR: outputs/final_results.json not found. Run run_pipeline.py first.")
        sys.exit(1)

    readme_path = ROOT / "README.md"
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Build replacement rows for the results table
    # Format: | Model | PR-AUC | ROC-AUC | F1 | Precision@500 | Recall@500 |
    model_display_map = {
        "LogisticRegression": "Logistic Regression",
        "XGBoost": "XGBoost",
        "LightGBM": "LightGBM",
        "GNN-GCN": "**GCN (PyG)**",
        "GNN-GRAPHSAGE": "**GraphSAGE (PyG)**",
        "GNN-GAT": "**GAT v2 (PyG)**",
    }

    # Replace the results table
    table_rows = []
    for key, display in model_display_map.items():
        m = results.get(key, {})
        if m:
            row = (
                f"| {display} "
                f"| {fmt(m.get('pr_auc', 0))} "
                f"| {fmt(m.get('roc_auc', 0))} "
                f"| {fmt(m.get('f1', 0))} "
                f"| {fmt(m.get('precision_at_500', 0))} "
                f"| {fmt(m.get('recall_at_500', 0))} |"
            )
            table_rows.append(row)

    if table_rows:
        # Replace placeholder rows in the table
        old_table_pattern = r"\| Logistic Regression.*?\n.*?\| XGBoost.*?\n.*?\| LightGBM.*?\n.*?\| \*\*GCN.*?\n.*?\| \*\*GraphSAGE.*?\n.*?\| \*\*GAT.*?\|"
        new_table_body = "\n".join(table_rows)
        content_updated = re.sub(old_table_pattern, new_table_body, content, flags=re.DOTALL)
        if content_updated != content:
            content = content_updated
            print("✅ Results table updated")
        else:
            print("⚠️  Could not auto-replace results table — appending at end")

    # Fill best model PR-AUC in resume bullet
    # Find best GNN
    best_pr_auc = 0.0
    best_model = "GAT"
    speedup = benchmark.get("speedup_x")

    for key in ["GNN-GAT", "GNN-GRAPHSAGE", "GNN-GCN"]:
        m = results.get(key, {})
        pr = m.get("pr_auc", 0)
        if pr > best_pr_auc:
            best_pr_auc = pr
            best_model = key.replace("GNN-", "")

    # Replace [X] in resume bullet
    resume_bullet_new = (
        f"> Built a graph-based fraud & mule account detection pipeline on 6.3M+ PaySim payment "
        f"transactions; constructed a directed transaction graph from scratch, engineered 22 "
        f"structural/temporal node features, and trained GCN/GraphSAGE/GATv2 (PyTorch Geometric) "
        f"models vs XGBoost/LightGBM baselines using time-based splits; achieved "
        f"**{best_pr_auc:.4f} PR-AUC** on the held-out test set"
    )
    if speedup:
        resume_bullet_new += f"; accelerated graph feature computation **~{speedup:.0f}x** via RAPIDS cuGraph GPU offloading"
    resume_bullet_new += (
        f"; deployed behind FastAPI + Docker with MLflow tracking and PSI-based drift detection; "
        f"implemented a 3-client Flower federated learning simulation for privacy-preserving "
        f"cross-institution training."
    )

    # Replace the old resume bullet
    old_bullet_pattern = r"> Built a graph-based fraud.*?training\."
    content = re.sub(old_bullet_pattern, resume_bullet_new, content, flags=re.DOTALL)

    # GPU benchmark table
    if benchmark:
        nx_total = benchmark.get("networkx_total_s")
        cu_total = benchmark.get("cugraph_total_s")
        spdup = benchmark.get("speedup_x")
        if nx_total and cu_total and spdup:
            gpu_note = (
                f"\n\n> **GPU Benchmark Result**: NetworkX {nx_total:.1f}s → "
                f"cuGraph {cu_total:.2f}s = **{spdup:.0f}x speedup** on RTX 4060"
            )
            content = content + gpu_note

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n✅ README.md updated with real results.")
    print(f"   Best GNN PR-AUC: {best_pr_auc:.4f} ({best_model})")
    if speedup:
        print(f"   GPU speedup: {speedup:.0f}x")

    # Show summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    for model, m in results.items():
        pr = m.get("pr_auc", 0)
        print(f"  {model:<22} PR-AUC = {pr:.4f}")


if __name__ == "__main__":
    main()
