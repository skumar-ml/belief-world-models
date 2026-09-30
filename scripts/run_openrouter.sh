#!/bin/bash
# Run one experiment cell against OpenRouter (no local GPU / vLLM).
#
# Required:
#   METHOD      metrics label
#   EXP_CONFIG  configs/task/<name>.json
#
# Optional:
#   AGENT_CONFIG  openrouter_llama8b | openrouter_qwen3_14b (default: openrouter_llama8b)
#   SPLIT         default test
#   MAX_STEPS     default 30 (ALFWorld) / 64 (BabyAI); omit for SciWorld
#   FAMILIES      BabyAI families; use + (default omits open/door)
#   RUN           run index
#   PLACEMENT     ALFWorld only: uniform (default) or zipf
#   OVERRIDE      default 1
#   DEBUG         1 for smoke test (DEBUG_N tasks)
#   DEBUG_N       tasks in debug mode (default: 3)
#
# Credentials: set OPENROUTER_API_KEY in the environment, or source it from
# ~/.config/secrets.env (already loaded by ~/.bashrc).
set -uo pipefail
cd "$(dirname "$0")/.."

METHOD="${METHOD:?set METHOD}"
EXP_CONFIG="${EXP_CONFIG:?set EXP_CONFIG}"
AGENT_CONFIG="${AGENT_CONFIG:-openrouter_llama8b}"
SPLIT="${SPLIT:-test}"
OVERRIDE="${OVERRIDE:-1}"
RUN="${RUN:-}"
PLACEMENT="${PLACEMENT:-}"
DEBUG="${DEBUG:-0}"
DEBUG_N="${DEBUG_N:-3}"
FAMILIES="${FAMILIES:-}"

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "[openrouter] ERROR: OPENROUTER_API_KEY is not set."
    exit 1
fi

if [ -n "${CONDA_PREFIX:-}" ]; then
    export JAVA_HOME="${JAVA_HOME:-$CONDA_PREFIX}"
    export PATH="$CONDA_PREFIX/bin:$PATH"
fi

mkdir -p logs outputs

OVERRIDE_FLAG=""
[ "$OVERRIDE" = "1" ] && OVERRIDE_FLAG="--override"

RUN_FLAG=""
[ -n "$RUN" ] && RUN_FLAG="--run $RUN"

PLACEMENT_FLAG=""
[ -n "$PLACEMENT" ] && PLACEMENT_FLAG="--placement $PLACEMENT"

DEBUG_FLAG=""
[ "$DEBUG" = "1" ] && DEBUG_FLAG="--debug --debug_n $DEBUG_N"

MAX_STEPS_FLAG=""
if [[ "$EXP_CONFIG" == alfworld* ]]; then
    MAX_STEPS="${MAX_STEPS:-30}"
    MAX_STEPS_FLAG="--max_steps $MAX_STEPS"
elif [[ "$EXP_CONFIG" == babyai* ]]; then
    MAX_STEPS="${MAX_STEPS:-64}"
    MAX_STEPS_FLAG="--max_steps $MAX_STEPS"
fi

FAMILIES_FLAG=""
if [[ "$EXP_CONFIG" == babyai* ]]; then
    FAMILIES="${FAMILIES:-goto+pickup+putnext+pick_up_seq_go_to}"
    FAMILIES_FLAG="--families ${FAMILIES}"
fi

echo "[openrouter] method=$METHOD exp=$EXP_CONFIG agent=$AGENT_CONFIG split=$SPLIT${PLACEMENT:+ placement=$PLACEMENT}${FAMILIES:+ families=$FAMILIES}"
python eval_baselines.py \
    --method "$METHOD" \
    --exp_config "$EXP_CONFIG" \
    --agent_config "$AGENT_CONFIG" \
    --split "$SPLIT" \
    $MAX_STEPS_FLAG \
    $FAMILIES_FLAG \
    $OVERRIDE_FLAG \
    $RUN_FLAG \
    $PLACEMENT_FLAG \
    $DEBUG_FLAG
exit $?
