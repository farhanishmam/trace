#!/usr/bin/env bash
# Serve a Qwen3-VL checkpoint for trace-tool-eval.
#
#   MODEL=Qwen/Qwen3-VL-8B-Instruct PORT=18100 scripts/serve_vllm.sh
#
# SERVED (the name clients use) defaults to the lowercased basename of MODEL,
# e.g. qwen3-vl-8b-instruct; run_phase1.sh and smoke_test.sh derive it the same way.
# Images are resized client-side to the condition's pixel bounds (multiples of
# 32), so the server's own resizing is a no-op. --generation-config vllm stops
# the checkpoint's generation_config.json (top_k=20 etc.) from overriding the
# sampling parameters each request sends. --chat-template-content-format openai
# keeps content lists intact so <tool_response> text and images render in order.
# The image limit covers 1 input image + 5 tool rounds x 2 calls.
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-VL-8B-Instruct}
SERVED=${SERVED:-$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')}
PORT=${PORT:-18100}
TP=${TP:-1}
DP=${DP:-1}
MAX_LEN=${MAX_LEN:-32768}
MAX_IMAGES=${MAX_IMAGES:-12}

exec vllm serve "$MODEL" \
  --served-model-name "$SERVED" \
  --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --data-parallel-size "$DP" \
  --gpu-memory-utilization 0.90 \
  --max-model-len "$MAX_LEN" \
  --limit-mm-per-prompt "{\"image\": $MAX_IMAGES, \"video\": 0}" \
  --chat-template-content-format openai \
  --generation-config vllm \
  "$@"
