#!/bin/bash
# ============================================================
# Environment Setup Script
# Creates a conda environment and installs all required packages
# Usage: bash setup_env.sh
# ============================================================

ENV_NAME="scheduler_env"
PYTHON_VERSION="3.12"

echo "======================================"
echo " Creating environment: $ENV_NAME"
echo "======================================"

# --- Create conda environment ---
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# --- Activate ---
source activate $ENV_NAME || conda activate $ENV_NAME

# --- Core scientific stack ---
pip install numpy
pip install scipy
pip install pandas

# --- Machine learning ---
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn

# --- Optimization (Pyomo + Gurobi) ---
pip install pyomo
# Gurobi: install wheel then activate your license
pip install gurobipy

# --- Utilities ---
pip install tqdm
pip install psutil

echo ""
echo "======================================"
echo " Verifying installations"
echo "======================================"

python3 - << 'EOF'
packages = {
    "numpy":        "import numpy as np; print(f'  numpy       {np.__version__}')",
    "scipy":        "import scipy; print(f'  scipy       {scipy.__version__}')",
    "pandas":       "import pandas as pd; print(f'  pandas      {pd.__version__}')",
    "torch":        "import torch; print(f'  torch       {torch.__version__}')",
    "sklearn":      "import sklearn; print(f'  sklearn     {sklearn.__version__}')",
    "pyomo":        "import pyomo; print(f'  pyomo       {pyomo.__version__}')",
    "gurobipy":     "import gurobipy; print(f'  gurobipy    {gurobipy.gurobi.version()}')",
    "tqdm":         "import tqdm; print(f'  tqdm        {tqdm.__version__}')",
    "psutil":       "import psutil; print(f'  psutil      {psutil.__version__}')",
}

print("Package versions:")
for name, cmd in packages.items():
    try:
        exec(cmd)
    except Exception as e:
        print(f"  {name:<12} FAILED: {e}")

# Check scipy submodules used in project
print("\nScipy submodules:")
try:
    from scipy.stats import truncnorm, beta, norm, binom, bernoulli
    print("  truncnorm, beta, norm, binom, bernoulli  OK")
except Exception as e:
    print(f"  FAILED: {e}")

print("\nSetup complete.")
EOF

echo ""
echo "======================================"
echo " Done! Activate with:"
echo "   conda activate $ENV_NAME"
echo "======================================"
