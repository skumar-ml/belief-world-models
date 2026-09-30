#!/bin/bash
# Run Sonnet 4.6 paper cells (Figure 3): single run, API via LiteLLM.
#
# Grid: 3 benchmarks x 2 agents x 4 permutations = 24 cells (no Memory ablation)
#
# Usage:
#   bash scripts/run_paper_sonnet.sh
#   BENCHMARK=alfworld bash scripts/run_paper_sonnet.sh
#   BENCHMARK=babyai bash scripts/run_paper_sonnet.sh
set -uo pipefail
cd "$(dirname "$0")/.."

SPLIT="${SPLIT:-test}"
BENCHMARK="${BENCHMARK:-all}"

ALF_CELLS=(
    "react:alfworld"
    "react_walle:alfworld_react_walle"
    "react_wm:alfworld_react_wm"
    "react_walle_wm:alfworld_react_walle_wm"
    "reflact:alfworld_reflact"
    "reflact_walle:alfworld_reflact_walle"
    "reflact_wm:alfworld_reflact_wm"
    "reflact_walle_wm:alfworld_reflact_walle_wm"
)

SCI_CELLS=(
    "sciworld_react:sciworld_react"
    "sciworld_react_wm:sciworld_react_wm"
    "sciworld_react_walle_oracle:sciworld_react_walle_oracle"
    "sciworld_react_walle_oracle_wm:sciworld_react_walle_oracle_wm"
    "sciworld_reflact:sciworld_reflact"
    "sciworld_reflact_wm:sciworld_reflact_wm"
    "sciworld_reflact_walle_oracle:sciworld_reflact_walle_oracle"
    "sciworld_reflact_walle_oracle_wm:sciworld_reflact_walle_oracle_wm"
)

BABY_CELLS=(
    "react:babyai_react"
    "react_walle_oracle:babyai_react_walle_oracle"
    "react_wm:babyai_react_wm"
    "react_walle_oracle_wm:babyai_react_walle_oracle_wm"
    "reflact:babyai_reflact"
    "reflact_walle_oracle:babyai_reflact_walle_oracle"
    "reflact_wm:babyai_reflact_wm"
    "reflact_walle_oracle_wm:babyai_reflact_walle_oracle_wm"
)

run_cells() {
    local max_steps="$1"
    shift
    local cells=("$@")
    for cell in "${cells[@]}"; do
        METHOD="${cell%%:*}"
        EXP_CONFIG="${cell##*:}"
        echo "=== Sonnet $METHOD ==="
        if [ -n "$max_steps" ]; then
            METHOD="$METHOD" EXP_CONFIG="$EXP_CONFIG" SPLIT="$SPLIT" \
                MAX_STEPS="$max_steps" OVERRIDE=1 \
                bash scripts/run_api.sh || exit 1
        else
            METHOD="$METHOD" EXP_CONFIG="$EXP_CONFIG" SPLIT="$SPLIT" \
                OVERRIDE=1 \
                bash scripts/run_api.sh || exit 1
        fi
    done
}

if [ "$BENCHMARK" = "alfworld" ] || [ "$BENCHMARK" = "all" ]; then
    run_cells "${MAX_STEPS:-30}" "${ALF_CELLS[@]}"
fi

if [ "$BENCHMARK" = "sciworld" ] || [ "$BENCHMARK" = "all" ]; then
    run_cells "" "${SCI_CELLS[@]}"
fi

if [ "$BENCHMARK" = "babyai" ] || [ "$BENCHMARK" = "all" ]; then
    run_cells "${BABYAI_MAX_STEPS:-64}" "${BABY_CELLS[@]}"
fi

echo "Sonnet paper grid complete."
