# Final evaluation (2026-11-12)

Two evaluations, requested together:

1. **MMLU-Pro** for Vero-Qwen25-7B and TRACE-Qwen2.5-VL-7B, with their shared base Qwen2.5-VL-7B-Instruct as the
   reference. The official [TIGER-AI-Lab/MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro) protocol is used:
   5-shot chain of thought, greedy decoding, 2,048 new tokens, 4,096-token context, raw-text prompts, and the
   official answer regexes and scorer.
2. **TRACE validation** for Qwen3.8-27B, Qwen3.5-9B, Gemma-4-12B-it and LLaVA-OneVision-2-8B-Instruct, under
   TRACE's own protocol (this repo's `no_tool` condition). That means 2,000 rows × seeds 42, 43, 44, with
   thinking off.

The results and the findings are in [`../results/final_eval/`](../results/final_eval/) (start with `FINDINGS.md`).
A secondary MMLU-Pro run uses the repository's other official entry point, `evaluate_from_api.py` (the same 5-shot
prompt as one chat message), to separate format effects from knowledge.

## Files

| file | what |
| --- | --- |
| `mmlu_pro/official/` | `evaluate_from_local.py` and `cot_prompt_lib/initial_prompt.txt`, copied byte for byte from MMLU-Pro @ `f418b116` (Apache-2.0; `NOTICE.md` has the hashes) |
| `mmlu_pro/mmlu_pro_eval.py` | `generate` (one greedy vLLM engine per GPU on a question shard) and `score` (merges the shards, then applies the official `save_res`) |
| `run_mmlu_pro.sh` | one model on all GPUs of a node: shards, then score |
| `mmlu_pro/mmlu_pro_api_eval.py`, `run_mmlu_pro_api.sh` | secondary protocol: the official `evaluate_from_api.py` prompts (captured from its own `single_request`) as one chat message, greedy, 4,000 tokens; strict accuracy |
| `trace_val_gen.sh` | one model: vLLM data-parallel server, `trace-tool-eval --conditions no_tool` per seed, deterministic scoring |
| `trace_val_judge.sh` | TRACE's Qwen3-32B extraction pass (TP1 on a ≥ 90 GB card) for every finished model, then re-score |
| (cluster only, untracked) `launch.sh`, `step.sh`, `keeper.sh`, `after_gen.sh`, `status.sh` | placement on held CHPC allocations: probe, shared GPU claims, a named `srun` step per task, claim heartbeat and release, the judge trigger, a status line per task. They hold machine paths, so they stay on the cluster like the earlier studies' dispatch tooling. Any launcher works: each run script only needs `GPUS` (device tokens) and the env in its header. |
| `summarize.py` | aggregates into `results/final_eval/` (tables, JSON summaries, report copies) |

## What differs from the official MMLU-Pro script, and why it cannot change a score

- **Data parallel instead of tensor parallel.** Each GPU runs one engine on every n-th question. Greedy decoding
  makes each answer independent of its batch, up to floating-point non-determinism.
- **Data loading.** The same parquet files are read from a pinned Hub snapshot
  (`TIGER-Lab/MMLU-Pro@b189ec76`), because the compute nodes run offline. They then go through the official
  `preprocess`.
- **Engine settings.** VL checkpoints skip multimodal profiling (`limit_mm_per_prompt` 0), and
  `gpu_memory_utilization` is 0.92 instead of 0.8, because a 7B VL checkpoint does not fit at 0.8 on a 24 GB card.
  Neither setting changes a prompt or a sampled token.
- **Scoring.** Scoring calls the official `save_res` per subject in sorted order, twice per subject as `eval_cot`
  does, with the module's `random` seeded 12345 at import. An answer the regexes cannot extract therefore gets
  the same random letter the official script would give it. Strict accuracy (unextractable = wrong) is reported
  as well.

## New model families (no_tool only)

`src/trace_tool/models.py` adds `qwen3.8`, `gemma4` and `llava-onevision-2` with `tool_format="none"`, so a tool
condition refuses to start for them. Per-model serving settings are in `scripts/_model_env.sh`.

- **Qwen3.8-27B:** thinking is off. The chat template then adds no reasoning-effort text, and the system and user
  turns render as they do for Qwen3.5.
- **Gemma-4-12B-it:** the checkpoint's own image budget (`max_soft_tokens` 280). Gemma's processor resizes every
  image to fill that budget, so the client applies only TRACE's pixel bounds (family patch factor 1, no grid
  rounding). Audio inputs are off and thinking is off (the template default). Budget 1120 was tried first; it
  collapsed accuracy under vLLM 0.30 (19.5% vs 62.3%) and is kept as a diagnostic in `runs/final_eval/trace_val_diag/`
  (see the findings).
- **LLaVA-OneVision-2-8B:** its processor is custom code (`--trust-remote-code`). The processor's
  `max_pixels` is 4.0 MP, above every TRACE image. The template has no thinking switch.

## Rerun

Each task resumes where it stopped. On a GPU node, from the repository root with `.venv` set up as in the top-level
README:

```bash
GPUS=0,1 MODEL=Qwen/Qwen3.8-27B bash final_eval/trace_val_gen.sh
```

```bash
GPUS=0 bash final_eval/trace_val_judge.sh
```

```bash
GPUS=0,1,2,3 MODEL=zlab-princeton/Vero-Qwen25-7B REVISION=180e84be5acb2aa887cf51015b84b6a6e453ee90 NAME=vero-qwen25-7b bash final_eval/run_mmlu_pro.sh
```

The judge needs a card of 90 GB or more. Then run:

```bash
python final_eval/summarize.py
```
