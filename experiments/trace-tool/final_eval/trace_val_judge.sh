#!/usr/bin/env bash
# final_eval/trace_val_judge.sh -- TRACE's Qwen3-32B extraction pass (pinned revision 9216db57) for every
# final-evaluation model whose generation is complete, then re-score. Serve the judge at TP1 on a >= 90 GB card
# (the TRACE-VERO study saw degraded run-on extractions at TP4 on 40 GB A800s; see its judge-agreement check).
#
#   GPUS=0 PORT=18500 bash final_eval/trace_val_judge.sh [served ...]   (default: every dir under OUT_ROOT)
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
# shellcheck disable=SC1091
source .venv/bin/activate
: "${GPUS:?GPUS is required}"
PORT=${PORT:-18500}
OUT_ROOT=${OUT_ROOT:-runs/final_eval/trace_val}
LOGD=$REPO/logs/final_eval; mkdir -p "$LOGD"
TP=$(tr ',' '\n' <<<"$GPUS" | grep -c .)
if [ $# -gt 0 ]; then SERVED_LIST=("$@"); else mapfile -t SERVED_LIST < <(ls "$OUT_ROOT"); fi
OUTS=()
for s in "${SERVED_LIST[@]}"; do
  d=$OUT_ROOT/$s
  n=$(ls -d "$d"/no_tool/seed*/scores.jsonl 2>/dev/null | wc -l)
  [ "$n" -ge 1 ] || { echo "skip $s: no scored seeds"; continue; }
  trace-tool-score "$d" ${JUDGE_EXTRA:-} $( [ -f "$d/judge_results.jsonl" ] && echo --judge-results "$d/judge_results.jsonl" ) \
    --report "$d/report.md" >>"$LOGD/score-$s.log" 2>&1
  OUTS+=("$d")
done
any_pending(){ for d in "${OUTS[@]}"; do [ -s "$d/judge_pending.jsonl" ] && return 0; done; return 1; }
if [ ${#OUTS[@]} -eq 0 ] || ! any_pending; then echo "nothing pending"; exit 0; fi
SLOG=$LOGD/vllm-judge-$(hostname)-$PORT.log
CUDA_VISIBLE_DEVICES=$GPUS MODEL=Qwen/Qwen3-32B SERVED=qwen3-32b-judge REVISION=9216db5781bf21249d130ec9da846c4624c16137 \
  PORT=$PORT TP=$TP DP=1 MAX_IMAGES=0 MAX_LEN=16384 setsid bash scripts/serve_vllm.sh >>"$SLOG" 2>&1 &
SPID=$!
cleanup(){ kill -TERM -- -"$SPID" 2>/dev/null || kill -TERM "$SPID" 2>/dev/null; for _ in $(seq 60); do kill -0 "$SPID" 2>/dev/null || break; sleep 2; done; kill -KILL -- -"$SPID" 2>/dev/null || true; }
trap cleanup EXIT
trap 'echo interrupted; exit 143' INT TERM
waited=0
until curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; do
  kill -0 "$SPID" 2>/dev/null || { echo "judge vLLM exited during startup:"; tail -40 "$SLOG"; exit 1; }
  [ "$waited" -lt 2400 ] || { echo "judge not ready after ${waited}s"; exit 1; }
  sleep 10; waited=$((waited + 10))
done
echo "==== $(date '+%F %T') judge ready after ${waited}s; outs: ${OUTS[*]}"
for attempt in 1 2; do
  for d in "${OUTS[@]}"; do
    [ -s "$d/judge_pending.jsonl" ] || continue
    echo "==== $(date '+%F %T') judging $d ($(wc -l <"$d/judge_pending.jsonl") pending, attempt $attempt)"
    trace-tool-judge --pending "$d/judge_pending.jsonl" --out "$d/judge_results.jsonl" \
      --endpoint "http://127.0.0.1:$PORT/v1" --model qwen3-32b-judge >>"$LOGD/judge-$(basename "$d").log" 2>&1
    trace-tool-score "$d" --judge-results "$d/judge_results.jsonl" --report "$d/report.md" >>"$LOGD/score-$(basename "$d").log" 2>&1
  done
  any_pending || break
done
for d in "${OUTS[@]}"; do
  p=$( [ -s "$d/judge_pending.jsonl" ] && wc -l <"$d/judge_pending.jsonl" || echo 0)
  echo "$d: still pending $p"
done
any_pending && { echo "INCOMPLETE: items still await the judge"; exit 1; }
echo "==== $(date '+%F %T') judge done"
