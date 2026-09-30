#!/bin/bash
# Serve an HF model via vLLM and run one BabyAI-Text experiment cell.
#
# Required env vars:
#   METHOD      metrics label (e.g. react, reflact)
#   EXP_CONFIG  configs/task/<name>.json (default: same as METHOD)
#
# Optional:
#   MODEL           HF repo id (default: meta-llama/Llama-3.1-8B-Instruct)
#   AGENT_CONFIG    configs/model/<name>.json (default: react_llama8b)
#   SPLIT           test=unseen / dev=seen (default: test)
#   MAX_STEPS       env-step horizon (default: 64)
#   RUN             run index for multi-seed studies
#   OVERRIDE        1=fresh run, 0=resume (default: 1)
#   VLLM_ARGS       extra vLLM flags
#   PORT            vLLM port (default: 8000)
#   DEBUG           1 for smoke test (DEBUG_N tasks)
#   DEBUG_N         tasks in debug mode (default: 5)
#   FAMILIES        BabyAI families; use + for Slurm. Default omits open/door.
set -uo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
AGENT_CONFIG="${AGENT_CONFIG:-react_llama8b}"
METHOD="${METHOD:?set METHOD (e.g. react, reflact)}"
EXP_CONFIG="${EXP_CONFIG:-$METHOD}"
SPLIT="${SPLIT:-test}"
OVERRIDE="${OVERRIDE:-1}"
RUN="${RUN:-}"
VLLM_ARGS="${VLLM_ARGS:-}"
PORT="${PORT:-8000}"
DEBUG="${DEBUG:-0}"
DEBUG_N="${DEBUG_N:-5}"
FAMILIES="${FAMILIES:-goto+pickup+putnext+pick_up_seq_go_to}"
MAX_STEPS="${MAX_STEPS:-64}"
TAG="${METHOD}"

mkdir -p logs outputs

if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi

VLLM_LOG="logs/vllm_${TAG}.log"

cleanup() {
    echo "[$TAG] cleaning up vLLM (pid ${VLLM_PID:-none})..."
    [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null
    pkill -P $$ 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[$TAG] starting vLLM: $MODEL on :$PORT"
vllm serve "$MODEL" --port "$PORT" $VLLM_ARGS > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

echo "[$TAG] waiting for vLLM..."
for i in $(seq 1 240); do
    # /v1/models can 200 before the weights are loaded; wait until this MODEL id is listed.
    if curl -sf "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q "$MODEL"; then
        echo "[$TAG] vLLM ready after ${i}0s."
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[$TAG] ERROR: vLLM died during startup."
        tail -40 "$VLLM_LOG"
        exit 1
    fi
    sleep 10
done

if ! curl -sf "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q "$MODEL"; then
    echo "[$TAG] ERROR: vLLM not ready within ~40 min (model $MODEL not listed)."
    tail -40 "$VLLM_LOG"
    exit 1
fi

OVERRIDE_FLAG=""
[ "$OVERRIDE" = "1" ] && OVERRIDE_FLAG="--override"

RUN_FLAG=""
[ -n "${RUN:-}" ] && RUN_FLAG="--run $RUN"

DEBUG_FLAG=""
[ "$DEBUG" = "1" ] && DEBUG_FLAG="--debug --debug_n $DEBUG_N"

MAX_STEPS_FLAG=""
[ -n "${MAX_STEPS:-}" ] && MAX_STEPS_FLAG="--max_steps $MAX_STEPS"

FAMILIES_FLAG=""
[ -n "${FAMILIES:-}" ] && FAMILIES_FLAG="--families ${FAMILIES}"

echo "[$TAG] running eval (model=$MODEL, exp=$EXP_CONFIG, split=$SPLIT, run=${RUN:-flat}, families=$FAMILIES, max_steps=${MAX_STEPS})"
python eval_baselines.py \
    --method "$METHOD" \
    --exp_config "$EXP_CONFIG" \
    --agent_config "$AGENT_CONFIG" \
    --split "$SPLIT" \
    $OVERRIDE_FLAG \
    $RUN_FLAG \
    $DEBUG_FLAG \
    $MAX_STEPS_FLAG \
    $FAMILIES_FLAG \
    --api_base "http://localhost:$PORT/v1" \
    --api_key EMPTY
RC=$?

echo "[$TAG] eval exited with code $RC"
exit $RC
