#!/usr/bin/env bash
# Usage: ./run.sh <ID>    run one config (start vLLM, benchmark, stop vLLM)
#        ./run.sh --list  print config IDs in run order
# Env:   PORT (8000), SCENARIOS ("1024:256"), CONCURRENCIES (per config: "1 8 32 128", 72b "1 8 16"),
#        DRY_RUN=1 prints the commands without running anything.
set -euo pipefail
cd "$(dirname "$0")"

IDS=(14b-1gpu 14b-dp2 14b-tp2 14b-pp2 72b-tp2 72b-pp2 moe-tp2 moe-ep2)

PORT="${PORT:-8000}"
SCENARIOS="${SCENARIOS:-1024:256}"
DRY_RUN="${DRY_RUN:-0}"
COMMON_ARGS=(--max-model-len 8192 --gpu-memory-utilization 0.9 --no-enable-prefix-caching)
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"

die() {
  echo "error: $*" >&2
  exit 1
}

ID="${1:-}"
if [ "$ID" = "--list" ]; then
  printf '%s\n' "${IDS[@]}"
  exit 0
fi

# 72B leaves little KV cache (TP 18k, PP 11k tokens on H100 x2), so it stops at 16; above 8 requests PP queues.
DEFAULT_CONCURRENCIES="1 8 32 128"
case "$ID" in
  14b-1gpu) MODEL=Qwen/Qwen3-14B;              GPUS=1; CUDA_VISIBLE_DEVICES=0;   EXTRA_ARGS=() ;;
  14b-dp2)  MODEL=Qwen/Qwen3-14B;              GPUS=2; CUDA_VISIBLE_DEVICES=0,1; EXTRA_ARGS=(--data-parallel-size 2) ;;
  14b-tp2)  MODEL=Qwen/Qwen3-14B;              GPUS=2; CUDA_VISIBLE_DEVICES=0,1; EXTRA_ARGS=(--tensor-parallel-size 2) ;;
  14b-pp2)  MODEL=Qwen/Qwen3-14B;              GPUS=2; CUDA_VISIBLE_DEVICES=0,1; EXTRA_ARGS=(--pipeline-parallel-size 2) ;;
  72b-tp2)  MODEL=Qwen/Qwen2.5-72B-Instruct;   GPUS=2; CUDA_VISIBLE_DEVICES=0,1; EXTRA_ARGS=(--tensor-parallel-size 2 --gpu-memory-utilization 0.95); DEFAULT_CONCURRENCIES="1 8 16" ;;
  72b-pp2)  MODEL=Qwen/Qwen2.5-72B-Instruct;   GPUS=2; CUDA_VISIBLE_DEVICES=0,1; EXTRA_ARGS=(--pipeline-parallel-size 2 --gpu-memory-utilization 0.95); DEFAULT_CONCURRENCIES="1 8 16" ;;
  moe-tp2)  MODEL=Qwen/Qwen3-30B-A3B;          GPUS=2; CUDA_VISIBLE_DEVICES=0,1; EXTRA_ARGS=(--tensor-parallel-size 2) ;;
  moe-ep2)  MODEL=Qwen/Qwen3-30B-A3B;          GPUS=2; CUDA_VISIBLE_DEVICES=0,1; EXTRA_ARGS=(--data-parallel-size 2 --enable-expert-parallel) ;;
  *) die "usage: ./run.sh <ID> | --list. valid IDs: ${IDS[*]}" ;;
esac
export CUDA_VISIBLE_DEVICES
CONCURRENCIES="${CONCURRENCIES:-$DEFAULT_CONCURRENCIES}"
# Bash 3.2 (macOS) treats an empty array as unbound under set -u, hence the ${a[@]+...} form.
SERVE_ARGS=("${COMMON_ARGS[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})
DIR="results/$ID"

PENDING=()
for scenario in $SCENARIOS; do
  for c in $CONCURRENCIES; do
    name="in${scenario%%:*}_out${scenario##*:}_c$c.json"
    if [ -f "$DIR/$name" ]; then
      echo "skip $DIR/$name (exists)"
    else
      PENDING+=("$scenario:$c")
    fi
  done
done
if [ "${#PENDING[@]}" -eq 0 ]; then
  echo "$ID: all results exist, nothing to run"
  exit 0
fi

json_args=$(printf '"%s", ' "${SERVE_ARGS[@]}")
META=$(printf '{"id": "%s", "model": "%s", "gpus": %s, "serve_args": [%s]}' \
  "$ID" "$MODEL" "$GPUS" "${json_args%, }")
SERVE_CMD=(vllm serve "$MODEL" --port "$PORT" "${SERVE_ARGS[@]}")

SERVER_PID=""
stop_server() {
  [ -n "$SERVER_PID" ] || return 0
  # Signal the whole process group: vLLM's engine and worker processes hold GPU memory.
  # No output here, so a closed stdout (e.g. a dead `tee`) cannot abort the cleanup.
  kill -TERM -- "-$SERVER_PID" 2>/dev/null || return 0
  for _ in $(seq 60); do
    kill -0 -- "-$SERVER_PID" 2>/dev/null || return 0
    sleep 1
  done
  kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

echo "$ID: meta.json"
echo "$META"
echo "+ CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ${SERVE_CMD[*]}"

if [ "$DRY_RUN" != 1 ]; then
  if curl -sf -o /dev/null --max-time 5 "http://127.0.0.1:$PORT/health"; then
    die "a server already answers on port $PORT. stop it, or run with PORT=<other port>"
  fi
  mkdir -p "$DIR"
  echo "$META" >"$DIR/meta.json"

  # set -m puts the server in its own process group, so stop_server can signal all of it.
  set -m
  "${SERVE_CMD[@]}" </dev/null >"$DIR/server.log" 2>&1 &
  SERVER_PID=$!
  set +m

  echo "waiting for vLLM on port $PORT (log: $DIR/server.log)"
  deadline=$((SECONDS + HEALTH_TIMEOUT))
  until curl -sf -o /dev/null --max-time 5 "http://127.0.0.1:$PORT/health"; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      tail -n 40 "$DIR/server.log" >&2
      die "vLLM exited before it became healthy. see $DIR/server.log"
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
      tail -n 40 "$DIR/server.log" >&2
      die "vLLM not healthy after ${HEALTH_TIMEOUT}s. see $DIR/server.log"
    fi
    sleep 5
  done

  grep -E 'KV cache size|Maximum concurrency|Available KV cache memory|Model loading took' \
    "$DIR/server.log" >"$DIR/kv.txt" || echo "warning: no KV cache lines in $DIR/server.log" >&2
  cat "$DIR/kv.txt"
fi

for job in "${PENDING[@]}"; do
  IFS=: read -r input output c <<<"$job"
  name="in${input}_out${output}_c$c.json"
  bench=(vllm bench serve --backend vllm --model "$MODEL" --host 127.0.0.1 --port "$PORT"
    --dataset-name random --random-input-len "$input" --random-output-len "$output" --ignore-eos
    --num-prompts $((c * 4 > 32 ? c * 4 : 32)) --max-concurrency "$c" --num-warmups $((c < 8 ? c : 8))
    --percentile-metrics "ttft,itl" --metric-percentiles "50,99" --seed 0
    --save-result --result-dir "$DIR" --result-filename "$name")
  echo "+ ${bench[*]}"
  if [ "$DRY_RUN" = 1 ]; then
    continue
  fi
  "${bench[@]}"
  # A server crash mid-run still yields a saved JSON; keep it out of the "done" set.
  if ! python3 -c 'import json, sys; sys.exit(json.load(open(sys.argv[1]))["failed"] > 0)' "$DIR/$name"; then
    mv "$DIR/$name" "$DIR/$name.failed"
    die "requests failed in $name (moved to $name.failed). see $DIR/server.log"
  fi
done
if [ "$DRY_RUN" = 1 ]; then
  echo "$ID: dry run, nothing started"
else
  echo "$ID: done"
fi
