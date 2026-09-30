#!/bin/bash
# Bootstrap a CPU-only conda env for ALFWorld + ScienceWorld + BabyAI via OpenRouter.
#
# Usage:
#   bash scripts/setup_openrouter.sh
#
# Optional:
#   CONDA_ENV=bbwm
#   SKIP_DATA=1
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-bbwm}"
MINIFORGE_ROOT="${MINIFORGE_ROOT:-$HOME/miniforge3}"

if [ ! -x "$MINIFORGE_ROOT/bin/conda" ]; then
    echo "Installing Miniforge to $MINIFORGE_ROOT ..."
    INSTALLER="/tmp/Miniforge3-Linux-x86_64.sh"
    curl -fsSL -o "$INSTALLER" \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    bash "$INSTALLER" -b -p "$MINIFORGE_ROOT"
    rm -f "$INSTALLER"
fi

# shellcheck disable=SC1091
source "$MINIFORGE_ROOT/etc/profile.d/conda.sh"
while [ -n "${CONDA_PREFIX:-}" ]; do
    conda deactivate || break
done

if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    echo "Creating conda env $CONDA_ENV (python 3.10 + openjdk 17) ..."
    conda create -n "$CONDA_ENV" python=3.10 openjdk=17 compilers make cmake libffi -y
fi
conda activate "$CONDA_ENV"
# jericho (TextWorld/ALFWorld) builds a C interpreter and needs make on PATH.
if ! command -v make >/dev/null 2>&1; then
    conda install -n "$CONDA_ENV" -y make cmake
fi

python -m pip install --upgrade pip
python -m pip install -r "$REPO_ROOT/requirements-api.txt"
# ALFWorld's unused THOR path imports OpenCV; the GUI wheel needs libGL.
python -m pip uninstall -y opencv-python
python -m pip install opencv-python-headless==4.11.0.86

if [ "${SKIP_DATA:-0}" != "1" ] && [ ! -f "$REPO_ROOT/data/alfworld/base_config.yaml" ]; then
    echo "Downloading ALFWorld game files..."
    (cd "$REPO_ROOT" && bash scripts/download_data.sh)
fi

if [ -x "$CONDA_PREFIX/bin/java" ]; then
    mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
    cat > "$CONDA_PREFIX/etc/conda/activate.d/java_home.sh" <<EOF
export JAVA_HOME="\$CONDA_PREFIX"
export PATH="\$CONDA_PREFIX/bin:\$PATH"
EOF
fi

echo
echo "Done."
echo "  conda activate $CONDA_ENV"
echo "  export OPENROUTER_API_KEY=...   # already set if ~/.config/secrets.env is loaded"
echo "  ALFWorld data: $REPO_ROOT/data/alfworld"
echo "  BabyAI indices: $REPO_ROOT/data/babyai"
echo "  java: $(command -v java) ($(java -version 2>&1 | head -1))"
