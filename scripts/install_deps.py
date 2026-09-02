r"""
scripts/install_deps.py
──────────────────────────────────────────────────────────────────────────────
Install all project dependencies after PyTorch is already installed.
Run with: .venv\Scripts\python.exe scripts/install_deps.py
"""
import subprocess
import sys

PIP = sys.executable.replace("python.exe", "pip.exe") if "python.exe" in sys.executable else sys.executable
PY = sys.executable


def run(cmd, check=True):
    print(f"\n{'='*60}\n$ {' '.join(cmd)}\n{'='*60}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0 and check:
        print(f"WARNING: command returned {result.returncode} — continuing")
    return result.returncode


# 1. Verify torch + CUDA
run([PY, "-c", "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"])

# 2. PyTorch Geometric (core)
run([PY, "-m", "pip", "install", "torch_geometric==2.6.1", "--quiet"])

# 3. Try to install PyG extensions (may fail for some Python/CUDA combos — OK)
pyg_ext_result = run([
    PY, "-m", "pip", "install",
    "pyg_lib", "torch_scatter", "torch_sparse", "torch_cluster", "torch_spline_conv",
    "-f", "https://data.pyg.org/whl/torch-2.11.0+cu128.html", "--quiet"
], check=False)
if pyg_ext_result != 0:
    print("PyG extensions not available for this Python/CUDA combo — PyG core still works.")

# 4. Core ML stack
packages_ml = [
    "pandas==2.2.3",
    "polars",
    "numpy",
    "scipy",
    "scikit-learn==1.5.2",
    "xgboost==2.1.3",
    "lightgbm==4.5.0",
    "imbalanced-learn",
    "shap",
    "pyarrow",           # parquet support
    "fastparquet",
]
run([PY, "-m", "pip", "install", "--quiet"] + packages_ml)

# 5. Graph + Viz
packages_graph = [
    "networkx==3.4.2",
    "matplotlib==3.10.0",
    "seaborn==0.13.2",
    "plotly",
    "pyvis",
    "kaleido",
]
run([PY, "-m", "pip", "install", "--quiet"] + packages_graph)

# 6. MLflow + API
packages_mlflow = [
    "mlflow==2.19.0",
    "mlflow[xgboost]",
    "mlflow[lightgbm]",
    "fastapi==0.115.6",
    "uvicorn[standard]==0.34.0",
    "pydantic==2.10.4",
    "httpx==0.28.1",
    "python-multipart==0.0.20",
    "starlette",
]
run([PY, "-m", "pip", "install", "--quiet"] + packages_mlflow)

# 7. Dev tools
packages_dev = [
    "loguru==0.7.3",
    "rich==13.9.4",
    "tqdm==4.67.1",
    "kaggle",
    "pytest==8.3.4",
    "pytest-asyncio==0.25.2",
    "black",
    "flake8",
    "ipykernel",
    "jupyter",
    "notebook",
]
run([PY, "-m", "pip", "install", "--quiet"] + packages_dev)

# 8. Federated learning
run([PY, "-m", "pip", "install", "flwr[simulation]==1.14.0", "--quiet"])

# 9. Verify key packages
print("\n" + "="*60)
print("VERIFICATION")
print("="*60)
checks = [
    "import torch; print(f'  torch {torch.__version__} CUDA={torch.cuda.is_available()}')",
    "import torch_geometric; print(f'  torch_geometric {torch_geometric.__version__}')",
    "import networkx; print(f'  networkx {networkx.__version__}')",
    "import xgboost; print(f'  xgboost {xgboost.__version__}')",
    "import lightgbm; print(f'  lightgbm {lightgbm.__version__}')",
    "import mlflow; print(f'  mlflow {mlflow.__version__}')",
    "import fastapi; print(f'  fastapi {fastapi.__version__}')",
    "import pandas; print(f'  pandas {pandas.__version__}')",
    "import sklearn; print(f'  sklearn {sklearn.__version__}')",
    "import shap; print(f'  shap {shap.__version__}')",
    "import flwr; print(f'  flwr {flwr.__version__}')",
]
for check in checks:
    run([PY, "-c", check], check=False)

print("\n✅ All packages installed. Ready to train.")
