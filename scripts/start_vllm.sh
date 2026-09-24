#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible server on port 8000 in the background and wait until it is ready.
# Logs go to outputs/vllm.log. Stop it with: kill "$(cat outputs/vllm.pid)"
#
# Usage: scripts/start_vllm.sh <model> [extra `vllm serve` args...]
#   scripts/start_vllm.sh Qwen/Qwen3-4B-Instruct-2507 --dtype half --tensor-parallel-size 2
set -euo pipefail

model=$1
shift
mkdir -p outputs
nohup vllm serve "$model" --port 8000 --enable-prefix-caching "$@" > outputs/vllm.log 2>&1 &
echo $! > outputs/vllm.pid

for _ in $(seq 1 180); do
  if curl -sf localhost:8000/v1/models > /dev/null; then
    echo "vLLM ready: $model"
    curl -s localhost:8000/v1/models | python -c "import json,sys; print([m['id'] for m in json.load(sys.stdin)['data']])"
    exit 0
  fi
  if ! kill -0 "$(cat outputs/vllm.pid)" 2> /dev/null; then
    echo "vLLM exited during startup:"
    tail -50 outputs/vllm.log
    exit 1
  fi
  sleep 5
done
echo "vLLM not ready after 15 minutes:"
tail -50 outputs/vllm.log
exit 1
