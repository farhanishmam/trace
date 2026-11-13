#!/usr/bin/env bash
# final_eval/run_mmlu_pro_api.sh -- MMLU-Pro, secondary chat-format protocol (mmlu_pro/mmlu_pro_api_eval.py), one
# greedy engine per GPU of this step. NSHARDS/SHARD_START let two allocations split one model (score separately).
#   MODEL=... REVISION=... NAME=... GPUS=0,1,2,3 [NSHARDS=8 SHARD_START=4] bash final_eval/run_mmlu_pro_api.sh
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
# shellcheck disable=SC1091
source .venv/bin/activate
: "${MODEL:?}" "${NAME:?}" "${GPUS:?}"
REVISION=${REVISION:-}
export VLLM_USE_FLASHINFER_SAMPLER=0   # greedy: never used, and its JIT build races on the shared NFS cache
OUT=$REPO/runs/final_eval/mmlu_pro_api/$NAME
LOGD=$REPO/logs/final_eval; mkdir -p "$OUT" "$LOGD"
IFS=, read -ra TOK <<<"$GPUS"
n=${#TOK[@]}; NSHARDS=${NSHARDS:-$n}; SHARD_START=${SHARD_START:-0}
echo "==== $(date '+%F %T') $NAME ($MODEL@$REVISION): shards $SHARD_START..$((SHARD_START + n - 1)) of $NSHARDS on GPUS=$GPUS"
pids=()
for k in $(seq 0 $((n - 1))); do
  i=$((SHARD_START + k))
  CUDA_VISIBLE_DEVICES=${TOK[$k]} python final_eval/mmlu_pro/mmlu_pro_api_eval.py generate --model "$MODEL" \
    ${REVISION:+--revision "$REVISION"} --shard "$i/$NSHARDS" --out "$OUT" >"$LOGD/mmluapi-$NAME-shard$i.log" 2>&1 &
  pids+=($!)
  sleep 20   # stagger engine start-up (JIT caches live on shared NFS)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
[ "$rc" -eq 0 ] || { echo "a shard failed; see $LOGD/mmluapi-$NAME-shard*.log"; exit 1; }
if [ "$NSHARDS" -eq "$n" ]; then python final_eval/mmlu_pro/mmlu_pro_api_eval.py score --out "$OUT" --model-name "$NAME"; fi
echo "==== $(date '+%F %T') done $NAME"
