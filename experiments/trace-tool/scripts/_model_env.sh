# Sourced by the scripts: checkpoint, served name, revision and extra conditions.
#   MODEL     HF id (default: TRACE's 7B base model)
#   SERVED    name the endpoint serves it under (default: lowercased basename of MODEL)
#   REVISION  checkpoint revision (default: the pinned revision for the models below;
#             set REVISION= to use the latest)
#   EXTRA_CONDITIONS  comma-separated conditions appended to the scripts' default condition
#             lists (default: the five *_forced conditions for Qwen/Qwen3.5-*, none otherwise;
#             set EXTRA_CONDITIONS= to add none). An explicit CONDITIONS ignores it.
MODEL=${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
SERVED=${SERVED:-$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')}
if [ -z "${REVISION+set}" ]; then
  case "$MODEL" in
    # rlvr/evaluation/trace_validation/suite.v1.json in github.com/maveryn/trace
    Qwen/Qwen2.5-VL-3B-Instruct) REVISION=66285546d2b821cf421d4f5eb2576359d3770cd3 ;;
    Qwen/Qwen2.5-VL-7B-Instruct) REVISION=cc594898137f460bfe9f0759e9844b3ce807cfb5 ;;
    maveryn/trace-qwen2.5-vl-3b) REVISION=2ec2374d5c219e6b12e26bda93d3b3adeb1e30c5 ;;
    maveryn/trace-qwen2.5-vl-7b) REVISION=4d0f1ae8ee25022058090dbdbff61957ece7331d ;;
    # ours: the snapshots whose chat template and vision config this repo was checked against
    Qwen/Qwen3.5-4B) REVISION=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a ;;
    Qwen/Qwen3.5-9B) REVISION=c202236235762e1c871ad0ccb60c8ee5ba337b9a ;;
    # final evaluation (2026-11-12): the snapshots evaluated (no_tool only, see src/trace_tool/models.py)
    Qwen/Qwen3.8-27B) REVISION=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 ;;
    google/gemma-4-12B-it) REVISION=707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 ;;
    lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct) REVISION=878cde7e04fd962a6082951d8f83083a73fea588 ;;
    *) REVISION= ;;
  esac
fi
if [ -z "${EXTRA_CONDITIONS+set}" ]; then
  case "$MODEL" in
    Qwen/Qwen3.5-*) EXTRA_CONDITIONS=zoom_forced,visual_ops_forced,calculator_forced,ocr_forced,all_forced ;;
    *) EXTRA_CONDITIONS= ;;
  esac
fi
# Server context for scripts/serve_vllm.sh (an explicit MAX_LEN, e.g. the judge's 16384, wins). Qwen3.5
# (32-px grid, up to ~4K tokens per 4.19 MP tool image, 11 images per episode) gets 65536 so a long
# visual_ops/all episode is not rejected for prompt length; the Qwen2.5-VL runs used the 32768 default.
if [ -z "${MAX_LEN:-}" ]; then
  case "$MODEL" in
    Qwen/Qwen3.5-*) MAX_LEN=65536 ;;
  esac
fi
# Final-evaluation models (no_tool only): 16384 tokens covers a 2048-token answer after the largest
# TRACE image (<= 1.28 MP, so <= ~1,700 visual tokens on a 28/32-px grid, <= 1,120 for Gemma 4).
# SERVE_EXTRA holds extra `vllm serve` flags (scripts/serve_vllm.sh); MM_EXTRA extends --limit-mm-per-prompt.
#   Gemma 4: the checkpoint's own image budget (max_soft_tokens 280); no audio inputs. Gemma's processor resizes every
#            image to fill its budget (at 280 most TRACE images are downscaled, small ones upscaled), so TRACE's pixel
#            bounds do not set what Gemma sees. Budget 1120 was tried first: it upscales every TRACE image (median 1.8x)
#            and collapsed accuracy under vLLM 0.30 (seed 42: 19.4% vs 61.3% deterministic; results/final_eval/
#            FINDINGS.md), so the default is kept.
#   LLaVA-OneVision-2: the checkpoint's processor is custom code.
case "$MODEL" in
  Qwen/Qwen3.8-*|google/gemma-4-*|lmms-lab-encoder/LLaVA-OneVision-2-*) MAX_LEN=${MAX_LEN:-16384} ;;
esac
#   Qwen3.8-27B: vLLM's default max_num_seqs (1024) exceeds the Mamba (linear-attention) state blocks that fit
#            next to 51 GiB of weights on a 96 GB card (629), so engine startup fails; 256 is above the client's
#            concurrency (final_eval/trace_val_gen.sh: 3 seeds x 48 per GPU) and does not change any request.
case "$MODEL" in
  Qwen/Qwen3.8-*) SERVE_EXTRA=${SERVE_EXTRA---max-num-seqs=256} ;;
  google/gemma-4-*) MM_EXTRA=${MM_EXTRA-', "audio": 0'} ;;
  lmms-lab-encoder/LLaVA-OneVision-2-*) SERVE_EXTRA=${SERVE_EXTRA---trust-remote-code} ;;
esac
