# Source this file to activate the CPU/OpenRouter eval env:
#   source scripts/env_openrouter.sh
MINIFORGE_ROOT="${MINIFORGE_ROOT:-$HOME/miniforge3}"
# shellcheck disable=SC1091
source "$MINIFORGE_ROOT/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-bbwm}"
export JAVA_HOME="${JAVA_HOME:-$CONDA_PREFIX}"
export PATH="$CONDA_PREFIX/bin:$PATH"
# MiniGrid/pygame is headless on this CPU box; dummy video avoids X11.
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
