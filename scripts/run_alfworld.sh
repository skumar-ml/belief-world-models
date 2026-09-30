#!/bin/bash
# Serve an HF model via vLLM and run one ALFWorld experiment cell.
#
# Required env vars:
#   METHOD      metrics label (e.g. react, reflact_walle_wm)
#   EXP_CONFIG  configs/task/<name>.json (e.g. alfworld, alfworld_reflact_wm)
#
# Optional:
#   MODEL           HF repo id (default: meta-llama/Llama-3.1-8B-Instruct)
#   AGENT_CONFIG    configs/model/<name>.json (default: react_llama8b)
#   SPLIT           test=unseen / dev=seen (default: test)
#   MAX_STEPS       env step budget (default: 30)
#   RUN             run index for multi-seed studies (appends run<N> to output path)
#   OVERRIDE        1=fresh run, 0=resume (default: 1)
#   PLACEMENT       uniform (default) or zipf
#   VLLM_ARGS       extra vLLM flags
#   PORT            vLLM port (default: 8000)
set -uo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
AGENT_CONFIG="${AGENT_CONFIG:-react_llama8b}"
METHOD="${METHOD:?set METHOD (e.g. react, reflact_walle_wm)}"
EXP_CONFIG="${EXP_CONFIG:?set EXP_CONFIG (e.g. alfworld, alfworld_reflact_wm)}"
SPLIT="${SPLIT:-test}"
MAX_STEPS="${MAX_STEPS:-30}"
OVERRIDE="${OVERRIDE:-1}"
RUN="${RUN:-}"
PLACEMENT="${PLACEMENT:-}"
VLLM_ARGS="${VLLM_ARGS:-}"
PORT="${PORT:-8000}"
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
for i in $(seq 1 360); do
    if curl -sf "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
        echo "[$TAG] vLLM ready after ${i}0s."
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[$TAG] ERROR: vLLM died during startup."
        tail -60 "$VLLM_LOG"
        exit 1
    fi
    sleep 10
done

if ! curl -sf "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
    echo "[$TAG] ERROR: vLLM not ready within ~60 min."
    tail -60 "$VLLM_LOG"
    exit 1
fi

OVERRIDE_FLAG=""
[ "$OVERRIDE" = "1" ] && OVERRIDE_FLAG="--override"

RUN_FLAG=""
[ -n "$RUN" ] && RUN_FLAG="--run $RUN"

PLACEMENT_FLAG=""
[ -n "$PLACEMENT" ] && PLACEMENT_FLAG="--placement $PLACEMENT"

echo "[$TAG] running eval (model=$MODEL, exp=$EXP_CONFIG, split=$SPLIT${PLACEMENT:+ placement=$PLACEMENT}, run=${RUN:-flat})"
python eval_baselines.py \
    --method "$METHOD" \
    --exp_config "$EXP_CONFIG" \
    --agent_config "$AGENT_CONFIG" \
    --split "$SPLIT" \
    --max_steps "$MAX_STEPS" \
    $OVERRIDE_FLAG \
    $RUN_FLAG \
    $PLACEMENT_FLAG \
    --api_base "http://localhost:$PORT/v1" \
    --api_key EMPTY
RC=$?

echo "[$TAG] eval exited with code $RC"
exit $RC
