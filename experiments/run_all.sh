#!/usr/bin/env bash
# Run every config from `./run.sh --list` in order. A failed config is reported and the loop moves on.
# Env vars (PORT, SCENARIOS, CONCURRENCIES, DRY_RUN) pass through to run.sh.
set -euo pipefail
cd "$(dirname "$0")"

trap 'exit 130' INT
trap 'exit 143' TERM

ok=""
failed=""
for id in $(./run.sh --list); do
  echo "=== $id start $(date '+%F %T') ==="
  if ./run.sh "$id"; then
    ok="$ok $id"
    echo "=== $id ok $(date '+%F %T') ==="
  else
    status=$?
    failed="$failed $id"
    echo "=== $id FAILED (exit $status) $(date '+%F %T'), continuing ===" >&2
  fi
done

echo "ok:${ok:- none}"
echo "failed:${failed:- none}"
[ -z "$failed" ]
