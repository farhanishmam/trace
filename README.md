# trace-tool

Tests whether giving tools to Qwen2.5-VL-3B/7B, TRACE's own base models, improves them on [TRACE](https://huggingface.co/datasets/maveryn/trace). Qwen3-VL is also supported.

The tools:
- the zoom-in tool from the "Zooming without Zooming" (ZwZ) tool-use baseline;
- the other image operations in ZwZ Table 9;
- a calculator;
- OCR.

The hypotheses, conditions, statistics and the conditional RL phase are in **[EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md)**.

## Layout

```
configs/conditions.yaml          evaluation conditions (tools, budgets, pixel bounds)
configs/verl/                    tool configs for the Phase-2 RL arms
prompts/                         TRACE system prompt (sha-checked), Qwen3-VL cookbook prompt
src/trace_tool/tools/            zoom (Qwen-Agent port), rotate/flip/draw, calculator, ocr
src/trace_tool/models.py         model families: patch grid and grounding-coordinate convention
src/trace_tool/prompting.py      tool prompts in Qwen-Agent's Nous format (= Qwen3-VL's native tool template)
src/trace_tool/agent.py          multi-turn episode loop against an OpenAI-compatible endpoint
src/trace_tool/run_eval.py       CLI: generate (resumable)
src/trace_tool/score_runs.py     CLI: score with TRACE's own code, per-run summaries, report
src/trace_tool/judge.py          CLI: TRACE's Qwen3-32B fallback extraction
src/trace_tool/compare.py        CLI: paired deltas, bootstrap CI, McNemar + Holm
src/trace_tool/rl/               Phase-2 scaffold for verl (reward, data prep, tool adapters, tool chat template for Qwen2.5-VL)
src/trace_tool/vendor/trace_rlvr TRACE scorer + extractor, vendored (Apache-2.0, pinned commit)
scripts/                         serve_vllm.sh, smoke_test.sh, run_phase1.sh, check_template.py, rl/
tests/                           CPU tests (no GPU or network needed)
```

## Setup (server)

```bash
python3 -m venv .venv && source .venv/bin/activate
```

```bash
pip install vllm && pip install -e ".[ocr,json5]"
```

RapidOCR pulls `opencv-python`, which needs GUI libraries (`libGL`, `libxcb`) that headless servers lack. vLLM pulls `opencv-python-headless`. Both packages own the same `cv2/` directory, so removing just one deletes files the other needs. Remove both, then reinstall the headless build:

```bash
pip uninstall -y opencv-python opencv-python-headless && pip install opencv-python-headless
```

If the OCR backend cannot load, `trace-tool-eval` stops at startup with the error, rather than running the `ocr` and `all` conditions with a broken tool.

The dataset is fetched from the Hub on first use and pinned to `dataset-v1` (commit `4e5b5436…`, sha256-checked). For a local copy, pass `--dataset-path` to the CLIs, or set `DATASET_PATH=...` for the scripts.

## Run everything with one script

[`run_on_server.sh`](run_on_server.sh) runs the whole of Phase 1 on a Linux GPU server:
1. setup, check, smoke;
2. pilot and full, for 7B then 3B;
3. the judge pass;
4. comparisons;
5. `results_summary.tgz` to send back.

It starts and stops vLLM itself, logs to `logs/`, and resumes after a crash if you rerun it:

```bash
nohup bash run_on_server.sh > run.log 2>&1 &
```

Useful controls:
- **Stop:** `kill <pid>` stops the current step and vLLM within seconds. The script prints its pid at the start. Rerun the same command to resume.
- **Failures:** if vLLM dies, rows end in endpoint errors, or judge answers stay pending, the script stops with `INCOMPLETE` instead of producing partial results.
- **GPUs:** `GPUS=0,1,2,3` gives one replica per GPU. `JUDGE_GPUS` takes 1, 2, 4 or 8 GPUs, about 80 GB in total, for Qwen3-32B.
- **Pilot first:** to follow the design's gate, run `STEPS="setup check smoke pilot judge compare"`, read the pilot, then run `STEPS="full judge compare"`.

The other settings are listed at the top of the file, including `VLLM_SPEC` to pin a vLLM build that matches your CUDA driver; setup checks that torch can see the GPU. The sections below are the same steps by hand.

## Run Phase 1

All scripts take the checkpoint as `MODEL` (default `Qwen/Qwen2.5-VL-7B-Instruct`) and derive the rest from it in `scripts/_model_env.sh`:
- the served name, `qwen2.5-vl-7b-instruct`, which is also the output directory, so 7B and 3B runs never mix;
- TRACE's pinned revision for its models, which you can override with `REVISION`.

The model family, and with it the tools' coordinate convention, is detected from the model id.

### 1. Serve the model

Each model and port pair runs in its own shell. Pass `DP=<gpus>` for data parallelism.

```bash
MODEL=Qwen/Qwen2.5-VL-7B-Instruct PORT=18100 scripts/serve_vllm.sh
```

### 2. Sanity-check template rendering

Run this in the vLLM environment, which has transformers and torch; otherwise `pip install -e ".[check]"`.

```bash
python scripts/check_template.py --model Qwen/Qwen2.5-VL-7B-Instruct --revision cc594898137f460bfe9f0759e9844b3ce807cfb5 --endpoint http://127.0.0.1:18100 --served qwen2.5-vl-7b-instruct
```

### 3. Smoke test (20 rows × every condition)

```bash
MODEL=Qwen/Qwen2.5-VL-7B-Instruct scripts/smoke_test.sh
```

### 4. Pilot (1000 rows, seed 42, all conditions), then the full run (2000 rows × 3 seeds, core conditions)

```bash
STAGE=pilot MODEL=Qwen/Qwen2.5-VL-7B-Instruct scripts/run_phase1.sh
```

```bash
STAGE=full MODEL=Qwen/Qwen2.5-VL-7B-Instruct scripts/run_phase1.sh
```

### 5. Judge pass (TRACE protocol; only missing/ambiguous extractions)

TRACE used an 8k judge context for 2k-token responses. 16k fits the 8k-token responses of `no_tool_budget`. A response that still doesn't fit is recorded as `rejected` and scores 0.

```bash
MODEL=Qwen/Qwen3-32B SERVED=qwen3-32b-judge REVISION=9216db5781bf21249d130ec9da846c4624c16137 PORT=18200 MAX_IMAGES=0 MAX_LEN=16384 scripts/serve_vllm.sh
```

```bash
trace-tool-judge --pending runs/full/qwen2.5-vl-7b-instruct/judge_pending.jsonl --out runs/full/qwen2.5-vl-7b-instruct/judge_results.jsonl
```

```bash
trace-tool-score runs/full/qwen2.5-vl-7b-instruct --judge-results runs/full/qwen2.5-vl-7b-instruct/judge_results.jsonl --report runs/full/qwen2.5-vl-7b-instruct/report.md
```

### 6. Paired comparison against the TRACE-protocol baseline

```bash
trace-tool-compare runs/full/qwen2.5-vl-7b-instruct --baseline no_tool --conditions no_tool_budget,zoom,visual_ops,calculator,ocr,all --out runs/full/qwen2.5-vl-7b-instruct/compare.md
```

Repeat with `MODEL=Qwen/Qwen2.5-VL-3B-Instruct` (serve, smoke, pilot, full).

**Sanity check:** on the full run, `no_tool` seed 42 should land near TRACE's published base scores: 24.45 (3B) and 34.25 (7B), under the same protocol and judge. Expect small differences from the vLLM version and from client-side resizing. A large gap means something is off before any tool result can be trusted.

**Optional:** `MODEL=maveryn/trace-qwen2.5-vl-7b` (and 3b) are TRACE's GRPO-trained checkpoints, published at 51.55 and 41.05. They show whether tools still help after RL without tools.

Useful knobs:
- `trace-tool-eval --model qwen2.5-vl-7b-instruct --conditions all --limit 1 --dry-run` prints the exact first prompt per condition (no endpoint or `--out` needed).
- `--conditions` picks a subset.
- `--domains charts,pages` filters domains.
- `--concurrency` sets episodes in flight.
- `--tool-threads` sets the CPU threads for tools.
- `TRACE_TOOL_OCR_THREADS` sets onnxruntime threads per OCR engine.

## Outputs

`runs/<stage>/<model>/<condition>/seed<seed>/` contains:

| file | contents |
| --- | --- |
| `run_config.json` | model, condition, every setting, system-prompt sha256, tool schemas, dataset revision, git commit |
| `responses.jsonl` | one episode per row: transcript (images as index+size), every requested tool call (args, result, latency, whether it executed), stop reason, tokens |
| `scores.jsonl` | strict (RL-reward) and official (TRACE validation) correctness per row |
| `summary.json` | accuracy overall / by domain / by answer type, tool use, calls, errors, turns, tokens, latency |

Tool-use metrics:
- `tool_use_rate` is the share of episodes with at least one *successful* call to a tool the condition offers.
- `tool_attempt_rate` counts any request, including malformed calls, calls to unknown tools, and calls skipped by the turn or per-turn limits.
- `tool_error_rate` is failed calls divided by executed calls.

Robustness:
- A run directory is locked while it is being written, so a second writer is refused.
- Resuming with a different model, model id, condition settings, dataset selection or OCR backend is refused.
- A changed code version only produces a warning.

At the root of each run you also get:
- `report.md`, one row per condition and seed;
- `compare.md`, paired deltas;
- `judge_pending.jsonl` / `judge_results.jsonl`.

## Tests

```bash
pip install -e ".[ocr,dev,json5,scipy]"
pytest
```

| Test file | What it checks |
| --- | --- |
| `test_zoom.py` | Pixel- and box-level parity with Qwen-Agent's zoom code (`tests/reference/`) |
| `test_prompting.py`, `test_qwen25.py` | Byte-identical rendering: against the native `tools=` / `role: tool` path (Qwen3-VL template), and against Qwen-Agent's Nous layout (Qwen2.5-VL template). Also absolute-pixel coordinate mapping on the 28-px grid. |
| `test_calculator.py` | Injection and blow-up inputs are rejected |
| `test_agent.py`, `test_cli_e2e.py` | Multi-turn loop, budgets, resume and the whole CLI chain, against a fake vLLM endpoint |
| `test_rl.py` | RL reward and data prep |

`TRACE_VALIDATION_PARQUET=/path/to/validation.parquet pytest` also checks that the gold answers of all 2,000 real rows score 1.0.

**Not tested here:**
- anything that needs a GPU, meaning real model behaviour and vLLM's own rendering (`scripts/check_template.py` and the smoke test cover these on the server). The Qwen2.5-VL rendering was also checked against the real Hugging Face tokenizers at TRACE's pinned revisions;
- the verl adapters in `src/trace_tool/rl/verl_tools.py` and `scripts/rl/run_grpo_template.sh`, which are templates for Phase 2.

## Review before merging

Two parts are security-sensitive and should get a human review:
- `src/trace_tool/tools/calculator.py` evaluates model-written expressions. It uses an AST whitelist and size limits, never `eval`.
- Tool arguments are untrusted model output in general.

## Licenses

| Component | License |
| --- | --- |
| TRACE dataset | CC-BY-4.0 (cite Alam 2026) |
| Vendored TRACE scoring code (`src/trace_tool/vendor/trace_rlvr`, `LICENSE` included) | Apache-2.0 |
| Qwen-Agent-derived zoom tool and test reference; Qwen2.5-VL / Qwen3-VL chat template fixtures | Apache-2.0 |
| RapidOCR | Apache-2.0 |
| onnxruntime | MIT |
| opencv-python / opencv-python-headless | Apache-2.0. The wheels bundle LGPL-2.1 FFmpeg, and the non-headless Linux wheel also bundles LGPL Qt. These are dynamically linked and fine for research use, but check them before redistributing an environment. |
| transformers (optional `check` extra) | Apache-2.0 |
| torch / torchvision (optional `check` extra) | BSD-3 |
| shapely | BSD-3 |
| pyclipper | MIT |
| omegaconf | BSD-3 |
| openai | Apache-2.0 |
| vLLM / verl | Apache-2.0 |
| json5 (optional) | Apache-2.0 |
| pytesseract and Tesseract (optional) | Apache-2.0 |

Apart from the LGPL components bundled in the OpenCV wheels, all of these are permissive. This repository does not yet have a license file of its own.
