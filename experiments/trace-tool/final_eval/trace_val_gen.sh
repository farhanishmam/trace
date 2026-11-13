#!/usr/bin/env bash
# final_eval/trace_val_gen.sh -- TRACE validation under TRACE's own protocol (condition no_tool: TRACE system
# prompt, temperature 0.6, top_p 0.95, 2048 tokens, one turn, image pixels 262,144..4,194,304), 2000 rows x
# seeds 42,43,44, for ONE model, on this step's GPUS (one vLLM data-parallel replica per TP group).
#
#   MODEL=Qwen/Qwen3.8-27B GPUS=0,1 PORT=18400 bash final_eval/trace_val_gen.sh
#
# Writes runs/final_eval/trace_val/<served>/no_tool/seed{42,43,44}/ and a deterministic-extraction score; rows
# whose answer needs TRACE's Qwen3-32B extraction stay in judge_pending.jsonl for final_eval/trace_val_judge.sh.
# Resumable: finished rows are kept (trace-tool-eval resumes on run_config.json). The three seeds run
# concurrently against the same server so it stays saturated through each seed's tail.
# Env: TP (default 1), SEEDS (42,43,44), CONC_PER_GPU (episodes in flight per GPU per seed, default 48), LIMIT (smoke).
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
# shellcheck disable=SC1091
source .venv/bin/activate
source scripts/_model_env.sh
: "${GPUS:?GPUS is required (the device tokens of this step)}"
PORT=${PORT:-18400}; TP=${TP:-1}; SEEDS=${SEEDS:-42,43,44}; CONC_PER_GPU=${CONC_PER_GPU:-48}
N=$(tr ',' '\n' <<<"$GPUS" | grep -c .)
[ $((N % TP)) -eq 0 ] || { echo "FATAL: config error: $N GPUs not divisible by TP=$TP"; exit 4; }
DP=$((N / TP))
OUT_ROOT=${OUT_ROOT:-runs/final_eval/trace_val}
OUT=$OUT_ROOT/$SERVED
LOGD=$REPO/logs/final_eval; mkdir -p "$LOGD" "$OUT"
SLOG=$LOGD/vllm-$SERVED-$(hostname)-$PORT.log
SEL=(); if [ -n "${LIMIT:-}" ]; then SEL=(--limit "$LIMIT"); fi

echo "==== $(date '+%F %T') $MODEL@$REVISION served=$SERVED GPUS=$GPUS TP=$TP DP=$DP port=$PORT out=$OUT"
python -c 'import sys, torch, vllm; n=torch.cuda.device_count(); print("vllm", vllm.__version__, "torch", torch.__version__, "GPUs visible", n); sys.exit(0 if n >= 1 else 6)'
# TP and DP must reach serve_vllm.sh (it defaults both to 1): before 2026-11-12 01:40 they did not, so the
# 2-GPU launches of that night served from one card (time only; no request or score changes).
CUDA_VISIBLE_DEVICES=$GPUS TP=$TP DP=$DP setsid bash scripts/serve_vllm.sh >>"$SLOG" 2>&1 &
SPID=$!
cleanup(){ kill -TERM -- -"$SPID" 2>/dev/null || kill -TERM "$SPID" 2>/dev/null; for _ in $(seq 60); do kill -0 "$SPID" 2>/dev/null || break; sleep 2; done; kill -KILL -- -"$SPID" 2>/dev/null || true; }
trap cleanup EXIT
trap 'echo interrupted; exit 143' INT TERM
waited=0
until curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; do
  kill -0 "$SPID" 2>/dev/null || { echo "vLLM exited during startup:"; tail -40 "$SLOG"; exit 1; }
  [ "$waited" -lt 2400 ] || { echo "vLLM not ready after ${waited}s"; tail -40 "$SLOG"; exit 1; }
  sleep 10; waited=$((waited + 10))
done
echo "==== $(date '+%F %T') vLLM ready after ${waited}s"

CONC=$((CONC_PER_GPU * N))
# run_config.json records the client side only; the server side goes next to it
for seed in ${SEEDS//,/ }; do
  mkdir -p "$OUT/no_tool/seed$seed"
  python - "$OUT/no_tool/seed$seed/server.json" <<PY
import json, sys, subprocess
gpus = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", "$GPUS"], capture_output=True, text=True).stdout.split("\n")
json.dump({"model": "$MODEL", "revision": "$REVISION", "served": "$SERVED", "host": "$(hostname)", "gpus": [g for g in gpus if g],
           "tp": $TP, "dp": $DP, "max_model_len": "${MAX_LEN:-}", "serve_extra": "${SERVE_EXTRA:-}", "mm_extra": """${MM_EXTRA:-}""",
           "started": "$(date -Is)"}, open(sys.argv[1], "w"), indent=1)
PY
done
pids=()
for seed in ${SEEDS//,/ }; do
  trace-tool-eval --endpoint "http://127.0.0.1:$PORT/v1" --model "$SERVED" --model-id "$MODEL" \
    ${REVISION:+--model-revision "$REVISION"} --conditions no_tool --seeds "$seed" ${SEL[@]+"${SEL[@]}"} \
    --concurrency "$CONC" --out "$OUT" >"$LOGD/gen-$SERVED-seed$seed.log" 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=$?; done
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null || { echo "vLLM stopped responding during the run"; tail -40 "$SLOG"; exit 1; }
[ "$rc" -eq 0 ] || { echo "trace-tool-eval failed (rc=$rc); see $LOGD/gen-$SERVED-seed*.log"; exit 1; }
trace-tool-score "$OUT" --report "$OUT/report.md" >"$LOGD/score-$SERVED.log" 2>&1
python - "$OUT" <<'PY'
import json, pathlib, sys
bad = 0
for p in sorted(pathlib.Path(sys.argv[1]).glob("no_tool/seed*/summary.json")):
    s = json.loads(p.read_text())
    print(p.parent.name, {k: s.get(k) for k in ("n", "official_acc", "deterministic_acc", "judge_pending", "excluded_endpoint_errors") if k in s})
    bad += s.get("excluded_endpoint_errors") or 0
sys.exit(1 if bad else 0)
PY
echo "==== $(date '+%F %T') generation done: $OUT"
