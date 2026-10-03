# Sourced by the scripts: checkpoint, served name and revision.
#   MODEL     HF id (default: TRACE's 7B base model)
#   SERVED    name the endpoint serves it under (default: lowercased basename of MODEL)
#   REVISION  checkpoint revision (default: TRACE's pinned revision for the models below;
#             set REVISION= to use the latest)
MODEL=${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
SERVED=${SERVED:-$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')}
if [ -z "${REVISION+set}" ]; then
  case "$MODEL" in
    # rlvr/evaluation/trace_validation/suite.v1.json in github.com/maveryn/trace
    Qwen/Qwen2.5-VL-3B-Instruct) REVISION=66285546d2b821cf421d4f5eb2576359d3770cd3 ;;
    Qwen/Qwen2.5-VL-7B-Instruct) REVISION=cc594898137f460bfe9f0759e9844b3ce807cfb5 ;;
    maveryn/trace-qwen2.5-vl-3b) REVISION=2ec2374d5c219e6b12e26bda93d3b3adeb1e30c5 ;;
    maveryn/trace-qwen2.5-vl-7b) REVISION=4d0f1ae8ee25022058090dbdbff61957ece7331d ;;
    *) REVISION= ;;
  esac
fi
