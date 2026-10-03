# trace-tool

Tests whether giving Qwen3-VL tools improves it on [TRACE](https://huggingface.co/datasets/maveryn/trace):
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
src/trace_tool/prompting.py      messages that render exactly like Qwen3-VL's native tool template
src/trace_tool/agent.py          multi-turn episode loop against an OpenAI-compatible endpoint
src/trace_tool/run_eval.py       CLI: generate (resumable)
src/trace_tool/score_runs.py     CLI: score with TRACE's own code, per-run summaries, report
src/trace_tool/judge.py          CLI: TRACE's Qwen3-32B fallback extraction
src/trace_tool/compare.py        CLI: paired deltas, bootstrap CI, McNemar + Holm
src/trace_tool/rl/               Phase-2 scaffold for verl (reward, data prep, tool adapters)
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

## Run Phase 1

All scripts take the checkpoint as `MODEL` and derive the served name from it: `Qwen/Qwen3-VL-8B-Instruct` is served as `qwen3-vl-8b-instruct`. The served name is also used for output directories, so 8B and 4B runs never mix.

### 1. Serve the model

Each model and port pair runs in its own shell. Pass `DP=<gpus>` for data parallelism.

```bash
MODEL=Qwen/Qwen3-VL-8B-Instruct PORT=18100 scripts/serve_vllm.sh
```

### 2. Sanity-check template rendering

Run this in the vLLM environment, which has transformers and torch; otherwise `pip install -e ".[check]"`.

```bash
python scripts/check_template.py --model Qwen/Qwen3-VL-8B-Instruct --endpoint http://127.0.0.1:18100 --served qwen3-vl-8b-instruct
```

### 3. Smoke test (20 rows × every condition)

```bash
MODEL=Qwen/Qwen3-VL-8B-Instruct scripts/smoke_test.sh
```

### 4. Pilot (1000 rows, seed 42, all conditions), then the full run (2000 rows × 3 seeds, core conditions)

```bash
STAGE=pilot MODEL=Qwen/Qwen3-VL-8B-Instruct scripts/run_phase1.sh
```

```bash
STAGE=full MODEL=Qwen/Qwen3-VL-8B-Instruct scripts/run_phase1.sh
```

### 5. Judge pass (TRACE protocol; only missing/ambiguous extractions)

TRACE used an 8k judge context for 2k-token responses. 16k fits the 8k-token responses of `no_tool_budget`. A response that still doesn't fit is recorded as `rejected` and scores 0.

```bash
MODEL=Qwen/Qwen3-32B SERVED=qwen3-32b-judge PORT=18200 MAX_IMAGES=0 MAX_LEN=16384 scripts/serve_vllm.sh --revision 9216db5781bf21249d130ec9da846c4624c16137
```

```bash
trace-tool-judge --pending runs/full/qwen3-vl-8b-instruct/judge_pending.jsonl --out runs/full/qwen3-vl-8b-instruct/judge_results.jsonl
```

```bash
trace-tool-score runs/full/qwen3-vl-8b-instruct --judge-results runs/full/qwen3-vl-8b-instruct/judge_results.jsonl --report runs/full/qwen3-vl-8b-instruct/report.md
```

### 6. Paired comparison against the TRACE-protocol baseline

```bash
trace-tool-compare runs/full/qwen3-vl-8b-instruct --baseline no_tool --out runs/full/qwen3-vl-8b-instruct/compare.md
```

Repeat with `MODEL=Qwen/Qwen3-VL-4B-Instruct` (serve, smoke, pilot, full). Useful knobs:
- `trace-tool-eval --model x --conditions all --limit 1 --dry-run` prints the exact first prompt per condition (no endpoint or `--out` needed).
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
| `test_prompting.py` | Byte-identical rendering vs. the native `tools=` / `role: tool` path, through the Qwen3-VL Jinja template |
| `test_calculator.py` | Injection and blow-up inputs are rejected |
| `test_agent.py`, `test_cli_e2e.py` | Multi-turn loop, budgets, resume and the whole CLI chain, against a fake vLLM endpoint |
| `test_rl.py` | RL reward and data prep |

`TRACE_VALIDATION_PARQUET=/path/to/validation.parquet pytest` also checks that the gold answers of all 2,000 real rows score 1.0.

**Not tested here:**
- anything that needs a GPU, meaning real Qwen3-VL behaviour and vLLM's own rendering (`scripts/check_template.py` and the smoke test cover these on the server);
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
| Qwen-Agent-derived zoom tool and test reference; Qwen3-VL chat template fixture | Apache-2.0 |
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
