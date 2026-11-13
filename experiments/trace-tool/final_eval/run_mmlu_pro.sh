#!/usr/bin/env bash
# final_eval/run_mmlu_pro.sh -- MMLU-Pro (official TIGER-AI-Lab protocol, mmlu_pro/mmlu_pro_eval.py) for ONE model,
# one greedy vLLM engine per GPU of this step (data parallel over question shards), then the official scoring.
#
#   MODEL=zlab-princeton/Vero-Qwen25-7B REVISION=180e84be... NAME=vero-qwen25-7b GPUS=0,1,2,3 bash final_eval/run_mmlu_pro.sh
#
# Writes runs/final_eval/mmlu_pro/<NAME>/{shardXXofYY.jsonl, meta.json, summary.json, official_results/}.
# Resumable per shard (finished questions are kept). GPU_UTIL (default 0.92: a 7B VL checkpoint on 24 GB).
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
# shellcheck disable=SC1091
source .venv/bin/activate
: "${MODEL:?}" "${NAME:?}" "${GPUS:?}"
REVISION=${REVISION:-}
GPU_UTIL=${GPU_UTIL:-0.92}
# Greedy decoding never uses FlashInfer's top-k/top-p sampler, but vLLM JIT-builds it at startup into one shared
# NFS cache dir, and 20 engines building it at once broke 3 shards (ninja races, 2026-11-12). Off: no token changes.
export VLLM_USE_FLASHINFER_SAMPLER=0
OUT=$REPO/runs/final_eval/mmlu_pro/$NAME
LOGD=$REPO/logs/final_eval; mkdir -p "$OUT" "$LOGD"
IFS=, read -ra TOK <<<"$GPUS"
n=${#TOK[@]}
python - "$OUT/meta.json" "$MODEL" "$REVISION" "$n" "$GPU_UTIL" <<'PY'
import json, sys, vllm, torch, transformers
path, model, rev, n, util = sys.argv[1:]
json.dump({"model": model, "revision": rev or None, "shards": int(n), "gpu_util": float(util), "vllm": vllm.__version__,
           "torch": torch.__version__, "transformers": transformers.__version__}, open(path, "w"), indent=1)
PY
echo "==== $(date '+%F %T') $NAME ($MODEL@$REVISION): $n shards on GPUS=$GPUS"
pids=()
for i in $(seq 0 $((n - 1))); do
  CUDA_VISIBLE_DEVICES=${TOK[$i]} python final_eval/mmlu_pro/mmlu_pro_eval.py generate --model "$MODEL" \
    ${REVISION:+--revision "$REVISION"} --shard "$i/$n" --out "$OUT" --gpu-util "$GPU_UTIL" \
    >"$LOGD/mmlu-$NAME-shard$i.log" 2>&1 &
  pids+=($!)
done
trap 'kill "${pids[@]}" 2>/dev/null; exit 143' INT TERM
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
[ "$rc" -eq 0 ] || { echo "a shard failed; see $LOGD/mmlu-$NAME-shard*.log"; exit 1; }
python final_eval/mmlu_pro/mmlu_pro_eval.py score --out "$OUT" --model-name "$NAME"
echo "==== $(date '+%F %T') done $NAME"
