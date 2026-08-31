#!/bin/bash
# Run one experiment cell using an API model (e.g. Claude Sonnet via LiteLLM).
#
# Required:
#   METHOD      metrics label
#   EXP_CONFIG  configs/task/<name>.json
#
# Optional:
#   AGENT_CONFIG  default litellm_claude_sonnet
#   SPLIT         default test
#   MAX_STEPS     default 30 (ALFWorld); omit for SciWorld per-task budgets
#   RUN           run index
#   OVERRIDE      default 1
#
# Credentials: set LITELLM_API_BASE and LITELLM_API_KEY in the environment,
# or source them from ~/.litellm_env.
set -uo pipefail

METHOD="${METHOD:?set METHOD}"
EXP_CONFIG="${EXP_CONFIG:?set EXP_CONFIG}"
AGENT_CONFIG="${AGENT_CONFIG:-litellm_claude_sonnet}"
SPLIT="${SPLIT:-test}"
MAX_STEPS="${MAX_STEPS:-30}"
OVERRIDE="${OVERRIDE:-1}"
RUN="${RUN:-}"

if [ -f "$HOME/.litellm_env" ]; then
    # shellcheck source=/dev/null
    source "$HOME/.litellm_env"
fi

if [ -z "${LITELLM_API_BASE:-}" ] || [ -z "${LITELLM_API_KEY:-}" ]; then
    echo "[run-api] ERROR: set LITELLM_API_BASE and LITELLM_API_KEY (or create ~/.litellm_env)."
    exit 1
fi

mkdir -p logs outputs

OVERRIDE_FLAG=""
[ "$OVERRIDE" = "1" ] && OVERRIDE_FLAG="--override"

RUN_FLAG=""
[ -n "$RUN" ] && RUN_FLAG="--run $RUN"

MAX_STEPS_FLAG=""
if [[ "$EXP_CONFIG" == alfworld* ]]; then
    MAX_STEPS_FLAG="--max_steps $MAX_STEPS"
fi

echo "[run-api] method=$METHOD exp=$EXP_CONFIG agent=$AGENT_CONFIG split=$SPLIT"
python eval_baselines.py \
    --method "$METHOD" \
    --exp_config "$EXP_CONFIG" \
    --agent_config "$AGENT_CONFIG" \
    --split "$SPLIT" \
    $MAX_STEPS_FLAG \
    $OVERRIDE_FLAG \
    $RUN_FLAG
exit $?
