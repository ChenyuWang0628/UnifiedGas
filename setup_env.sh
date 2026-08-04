#!/usr/bin/env bash
# Create a conda environment for UnifiedGas and verify the installation.
#
#   bash setup_env.sh              # environment name defaults to "unifiedgas"
#   bash setup_env.sh myenv        # or pass your own name
#
# If conda is unavailable the script falls back to a plain venv.
set -euo pipefail

ENV_NAME="${1:-unifiedgas}"
PY_VERSION="3.10"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== UnifiedGas environment setup ==="

if command -v conda >/dev/null 2>&1; then
    echo "[1/3] creating conda environment '$ENV_NAME' (python $PY_VERSION)"
    if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        echo "      environment already exists, reusing it"
    else
        conda create -y -n "$ENV_NAME" "python=$PY_VERSION"
    fi
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
    ACTIVATE_HINT="conda activate $ENV_NAME"
else
    echo "[1/3] conda not found, creating a venv at $HERE/.venv"
    if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
        echo "error: python3 is $(python3 -V 2>&1 | cut -d' ' -f2), but >= $PY_VERSION is required." >&2
        echo "       Install Python >= $PY_VERSION or use conda, then re-run this script." >&2
        exit 1
    fi
    python3 -m venv "$HERE/.venv"
    # shellcheck disable=SC1091
    source "$HERE/.venv/bin/activate"
    ACTIVATE_HINT="source $HERE/.venv/bin/activate"
fi

echo "[2/3] installing dependencies"
python -m pip install --upgrade pip >/dev/null
python -m pip install -r "$HERE/requirements.txt"

echo "[3/3] verifying the installation"
UNIFIEDGAS_HOME="$HERE" python - <<'PY'
import sys
import torch, numpy, sklearn, scipy, pandas, matplotlib
print(f"  python       {sys.version.split()[0]}")
print(f"  torch        {torch.__version__}  (cuda: {torch.cuda.is_available()})")
print(f"  numpy        {numpy.__version__}")
print(f"  scikit-learn {sklearn.__version__}")
print(f"  scipy        {scipy.__version__}")
print(f"  pandas       {pandas.__version__}")
print(f"  matplotlib   {matplotlib.__version__}")

sys.path.insert(0, __import__("os").environ["UNIFIEDGAS_HOME"])
from unifiedgas import GasClassifier
from unifiedgas.baselines import TRAINER_REGISTRY
n = sum(p.numel() for p in GasClassifier(num_classes=6, dataset="A").parameters())
print(f"  UnifiedGas   {n / 1e6:.3f} M parameters")
print(f"  baselines    {len(TRAINER_REGISTRY)} registered")
PY

echo
echo "=== done ==="
echo "Activate the environment with:  $ACTIVATE_HINT"
echo "Then download the datasets (see docs/DATA.md) and run:"
echo "  python scripts/predict.py --checkpoint checkpoints/unifiedgas_A_batch1to6.pt \\"
echo "      --input data/DataSetA/batch6.dat"
