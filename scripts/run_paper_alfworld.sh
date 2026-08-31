#!/bin/bash
# Run the ALFWorld paper grid (Tables 1 and 3): 3 seeds for Llama-3.1-8B and Qwen3-14B.
#
# Grid: 2 models x 2 agents x 4 permutations x 3 runs = 48 main cells
#       + 2 models x 2 agents x Memory (wm_det) x 3 runs = 12 ablation cells
#
# Usage:
#   bash scripts/run_paper_alfworld.sh              # full grid
#   RUNS="1" bash scripts/run_paper_alfworld.sh     # single seed
#   MODELS_FILTER=llama8b bash scripts/run_paper_alfworld.sh
#   SKIP_MEMORY=1 bash scripts/run_paper_alfworld.sh  # Table 1 only
set -uo pipefail
cd "$(dirname "$0")/.."

SPLIT="${SPLIT:-test}"
MAX_STEPS="${MAX_STEPS:-30}"
RUNS="${RUNS:-1 2 3}"
SKIP_MEMORY="${SKIP_MEMORY:-0}"
PORT_BASE="${PORT_BASE:-8400}"

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

# Table 1 cells: base, Belief (wm), WALL-E (walle), BB-WM (walle_wm)
CELLS=(
    "react:alfworld"
    "react_walle:alfworld_react_walle"
    "react_wm:alfworld_react_wm"
    "react_walle_wm:alfworld_react_walle_wm"
    "reflact:alfworld_reflact"
    "reflact_walle:alfworld_reflact_walle"
    "reflact_wm:alfworld_reflact_wm"
    "reflact_walle_wm:alfworld_reflact_walle_wm"
)

# Table 3 Memory ablation
MEMORY_CELLS=(
    "react_wm_det:alfworld_react_wm_det"
    "reflact_wm_det:alfworld_reflact_wm_det"
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
            echo "=== ALFWorld $TAG $METHOD run$RUN ==="
            MODEL="$MODEL" AGENT_CONFIG="$AGENT_CONFIG" METHOD="$METHOD" \
                EXP_CONFIG="$EXP_CONFIG" SPLIT="$SPLIT" MAX_STEPS="$MAX_STEPS" \
                RUN="$RUN" VLLM_ARGS="$VLLM_ARGS" PORT="$PORT" OVERRIDE=1 \
                bash scripts/run_alfworld.sh || exit 1
        done

        if [ "$SKIP_MEMORY" != "1" ]; then
            for cell in "${MEMORY_CELLS[@]}"; do
                METHOD="${cell%%:*}"
                EXP_CONFIG="${cell##*:}"
                PORT=$((PORT_BASE + port_idx)); port_idx=$((port_idx + 1))
                echo "=== ALFWorld Memory $TAG $METHOD run$RUN ==="
                MODEL="$MODEL" AGENT_CONFIG="$AGENT_CONFIG" METHOD="$METHOD" \
                    EXP_CONFIG="$EXP_CONFIG" SPLIT="$SPLIT" MAX_STEPS="$MAX_STEPS" \
                    RUN="$RUN" VLLM_ARGS="$VLLM_ARGS" PORT="$PORT" OVERRIDE=1 \
                    bash scripts/run_alfworld.sh || exit 1
            done
        fi
    done
done

echo "ALFWorld paper grid complete."
