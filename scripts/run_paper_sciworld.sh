#!/bin/bash
# Run the ScienceWorld paper grid (Figure 3): 3 seeds for Llama-3.1-8B and Qwen3-14B.
#
# Grid: 2 models x 2 agents x 4 permutations x 3 runs = 48 cells
#       + 2 models x 2 agents x Memory (wm_det) x 3 runs if SKIP_MEMORY=0
#
# Usage:
#   bash scripts/run_paper_sciworld.sh
#   BACKEND=openrouter bash scripts/run_paper_sciworld.sh
#   RUNS="1" bash scripts/run_paper_sciworld.sh
#   MODELS_FILTER=qwen14b bash scripts/run_paper_sciworld.sh
#   SKIP_MEMORY=0 bash scripts/run_paper_sciworld.sh  # Figure 4 Memory ablation
set -uo pipefail
cd "$(dirname "$0")/.."

SPLIT="${SPLIT:-test}"
RUNS="${RUNS:-1 2 3}"
SKIP_MEMORY="${SKIP_MEMORY:-1}"
PORT_BASE="${PORT_BASE:-8500}"
BACKEND="${BACKEND:-vllm}"  # vllm | openrouter

MODEL_IDS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen3-14B"
)
AGENT_CONFIGS=(
    "react_llama8b"
    "qwen3_14b"
)
SHORT_TAGS=(
    "llama8b"
    "qwen14b"
)
VLLM_ARGS_LIST=(
    "--enforce-eager"
    "--gpu-memory-utilization 0.92 --max-model-len 8192 --enforce-eager"
)
RUNNER="scripts/run_sciworld.sh"
if [ "$BACKEND" = "openrouter" ]; then
    AGENT_CONFIGS=(
        "openrouter_llama8b"
        "openrouter_qwen3_14b"
    )
    RUNNER="scripts/run_openrouter.sh"
fi

CELLS=(
    "sciworld_react:sciworld_react"
    "sciworld_react_wm:sciworld_react_wm"
    "sciworld_react_walle_oracle:sciworld_react_walle_oracle"
    "sciworld_react_walle_oracle_wm:sciworld_react_walle_oracle_wm"
    "sciworld_reflact:sciworld_reflact"
    "sciworld_reflact_wm:sciworld_reflact_wm"
    "sciworld_reflact_walle_oracle:sciworld_reflact_walle_oracle"
    "sciworld_reflact_walle_oracle_wm:sciworld_reflact_walle_oracle_wm"
)

MEMORY_CELLS=(
    "sciworld_react_wm_det:sciworld_react_wm_det"
    "sciworld_reflact_wm_det:sciworld_reflact_wm_det"
)

port_idx=0
for m in "${!MODEL_IDS[@]}"; do
    MODEL="${MODEL_IDS[$m]}"
    AGENT_CONFIG="${AGENT_CONFIGS[$m]}"
    TAG="${SHORT_TAGS[$m]}"
    VLLM_ARGS="${VLLM_ARGS_LIST[$m]}"

    if [ -n "${MODELS_FILTER:-}" ] && [[ " $MODELS_FILTER " != *" $TAG "* ]]; then
        continue
    fi

    for RUN in $RUNS; do
        for cell in "${CELLS[@]}"; do
            METHOD="${cell%%:*}"
            EXP_CONFIG="${cell##*:}"
            PORT=$((PORT_BASE + port_idx)); port_idx=$((port_idx + 1))
            echo "=== SciWorld $TAG $METHOD run$RUN ==="
            MODEL="$MODEL" AGENT_CONFIG="$AGENT_CONFIG" METHOD="$METHOD" \
                EXP_CONFIG="$EXP_CONFIG" SPLIT="$SPLIT" RUN="$RUN" \
                VLLM_ARGS="$VLLM_ARGS" PORT="$PORT" OVERRIDE=1 \
                bash "$RUNNER" || exit 1
        done

        if [ "$SKIP_MEMORY" != "1" ]; then
            for cell in "${MEMORY_CELLS[@]}"; do
                METHOD="${cell%%:*}"
                EXP_CONFIG="${cell##*:}"
                PORT=$((PORT_BASE + port_idx)); port_idx=$((port_idx + 1))
                echo "=== SciWorld Memory $TAG $METHOD run$RUN ==="
                MODEL="$MODEL" AGENT_CONFIG="$AGENT_CONFIG" METHOD="$METHOD" \
                    EXP_CONFIG="$EXP_CONFIG" SPLIT="$SPLIT" RUN="$RUN" \
                    VLLM_ARGS="$VLLM_ARGS" PORT="$PORT" OVERRIDE=1 \
                    bash "$RUNNER" || exit 1
            done
        fi
    done
done

echo "ScienceWorld paper grid complete."
