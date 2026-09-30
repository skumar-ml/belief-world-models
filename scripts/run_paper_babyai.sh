#!/bin/bash
# Run the BabyAI-Text paper grid (Figure 3): 3 seeds for Llama-3.1-8B and Qwen3-14B.
#
# Grid: 2 models x 2 agents x 4 permutations x 3 runs = 48 cells.
# Belief / BB-WM are the 6x6 placement-prior world model.
#
# Usage:
#   bash scripts/run_paper_babyai.sh
#   BACKEND=openrouter bash scripts/run_paper_babyai.sh
#   RUNS="1" bash scripts/run_paper_babyai.sh
#   MODELS_FILTER=llama8b bash scripts/run_paper_babyai.sh
set -uo pipefail
cd "$(dirname "$0")/.."

SPLIT="${SPLIT:-test}"
MAX_STEPS="${MAX_STEPS:-64}"
RUNS="${RUNS:-1 2 3}"
PORT_BASE="${PORT_BASE:-8600}"
BACKEND="${BACKEND:-vllm}"  # vllm | openrouter
FAMILIES="${FAMILIES:-goto+pickup+putnext+pick_up_seq_go_to}"

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
RUNNER="scripts/run_babyai.sh"
if [ "$BACKEND" = "openrouter" ]; then
    AGENT_CONFIGS=(
        "openrouter_llama8b"
        "openrouter_qwen3_14b"
    )
    RUNNER="scripts/run_openrouter.sh"
fi

CELLS=(
    "react:babyai_react"
    "react_walle_oracle:babyai_react_walle_oracle"
    "react_wm:babyai_react_wm"
    "react_walle_oracle_wm:babyai_react_walle_oracle_wm"
    "reflact:babyai_reflact"
    "reflact_walle_oracle:babyai_reflact_walle_oracle"
    "reflact_wm:babyai_reflact_wm"
    "reflact_walle_oracle_wm:babyai_reflact_walle_oracle_wm"
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
            echo "=== BabyAI $TAG $METHOD run$RUN ==="
            MODEL="$MODEL" AGENT_CONFIG="$AGENT_CONFIG" METHOD="$METHOD" \
                EXP_CONFIG="$EXP_CONFIG" SPLIT="$SPLIT" MAX_STEPS="$MAX_STEPS" \
                FAMILIES="$FAMILIES" RUN="$RUN" VLLM_ARGS="$VLLM_ARGS" \
                PORT="$PORT" OVERRIDE=1 \
                bash "$RUNNER" || exit 1
        done
    done
done

echo "BabyAI paper grid complete."
