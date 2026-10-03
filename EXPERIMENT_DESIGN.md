# Tool calling for Qwen2.5-VL on TRACE: experiment design

## 1. Questions

| # | Question | Phase |
| --- | --- | --- |
| Q1 | Does inference-time access to tools (zoom-in, other image ops, calculator, OCR) improve Qwen2.5-VL-3B/7B (TRACE's base models) on the TRACE validation set, without any training? | 1 |
| Q2 | Is a gain caused by the tool outputs, or by the longer token budget and the changed prompt that come with tool use? | 1 |
| Q3 | Which tools help, on which TRACE domains and answer types, and how often does the model actually call them? | 1 |
| Q4 | If Q1 is positive: does GRPO with tools beat GRPO without tools on TRACE, at equal steps and data? | 2 |
| Q5 | Do tool-trained policies transfer better to external benchmarks, with and without tools at test time? | 2 |

## 2. Facts this design rests on

Verified against the sources on 2026-10-02.

**Zooming without Zooming (ZwZ), arXiv:2602.11858.**
- The paper's own method uses no tools at inference. It distils zoomed-in crops into single-pass training data.
- Its tool-use baseline is "Qwen3-VL with official tool use", built from the Qwen3-VL GitHub `think_with_images` cookbook. That cookbook defines one tool, `image_zoom_in_tool`, implemented in Qwen-Agent: `bbox_2d` in relative [0, 1000] coordinates, a 32 px minimum box side, and crops upscaled to at least 256×32×32 px.
- Table 4 (V*, HR-Bench 4K/8K, MME-RealWorld-en, average):

  | Model | Average |
  | --- | ---: |
  | Qwen3-VL-8B | 76.5 |
  | Qwen3-VL-8B + tool | 79.1 |
  | Qwen3-VL-4B | 73.7 |
  | Qwen3-VL-4B + tool | 77.3 |

- Table 9 lists the actions used by "thinking-with-images" agents: zoom in, zoom out, flip, rotate, denoise, 2D grounding, 3D grounding, drawing, and web search. It gives no API for any of them except zoom in.
- Footnote 1: zooming adds information mainly when the original image had to be downsampled.

**TRACE (arXiv:2607.19790, `maveryn/trace@dataset-v1`).**
- Validation set: 2,000 rows, two per task across 1,000 tasks, 11 domains. Rows by domain: charts 360, games 340, geometry 340, pages 160, three_d 120, illustrations 120, puzzles 120, symbolic 120, graph 120, physics 100, icons 100.
- Answer types: integer 1094, option_letter 420, string 384, number 102.
- Official validation protocol:
  - the TRACE system prompt;
  - temperature 0.6, top_p 0.95, max_tokens 2048;
  - image pixels bounded to 262,144–4,194,304;
  - deterministic answer extraction, with a Qwen3-32B extraction pass for missing or ambiguous answers;
  - exact-match scoring, with unresolved rows scored 0.
- Published results:

  | Model | Base | After TRACE GRPO |
  | --- | ---: | ---: |
  | Qwen2.5-VL-3B | 24.45 | 41.05 |
  | Qwen2.5-VL-7B | 34.25 | 51.55 |

- RL recipe: EasyR1 GRPO for 500 steps, rollout batch 128, n=8, lr 1e-6, no KL, clip 0.2/0.3. Reward = 0.95 × exact-JSON answer + 0.05 × format.
- Image size: median 0.76 MP, maximum 1.28 MP. Every image is below the 4.19 MP cap, so **nothing is downsampled at TRACE's resolution.**

**Qwen2.5-VL (verified against the checkpoints at TRACE's pinned revisions).**
- Its chat template has no tool support at all: no `tools` block and no `role: tool`. Qwen2.5-VL agents, including Qwen-Agent's `NousFnCallPrompt` and DeepEyes, put the Hermes tool block in the system prompt and return results as `<tool_response>` user turns. `<tool_call>` and `</tool_call>` are single tokens in its vocabulary.
- It grounds in absolute pixel coordinates of the image as it sees it (after resizing to its 28-pixel patch grid: patch 14 × merge 2), not Qwen3-VL's relative [0, 1000].
- The agentic baselines in ZwZ Table 4 (DeepEyes, Mini-o3, Thyme, Pixel-Reasoner) are all Qwen2.5-VL-7B. DeepEyes zooms with absolute pixel boxes and a 28-px minimum side, matching Qwen2.5-VL's native grounding, so we follow it. Mini-o3 and Pixel-Reasoner use relative / normalised boxes, and Thyme crops through generated code. The "official tool use" baseline is Qwen3-VL-specific.

**What follows for zoom.** At native resolution, zoom cannot add pixels. It can only change token density (a small crop is re-encoded with at least 256 visual tokens) and what the model attends to. ZwZ's gains came from high-resolution benchmarks where downsampling hides detail. **Prediction:** zoom gains on TRACE at native resolution will be small. They should grow in the `*_lowres` conditions, where the model sees a ~0.4 MP copy but the tools crop from the full-resolution original.

## 3. Tools

Coordinates follow each model family's native grounding frame (`src/trace_tool/models.py`):

| Family | Coordinates | Patch grid | Zoom grows boxes to |
| --- | --- | --- | --- |
| Qwen2.5-VL (primary) | absolute pixels of the image as the model sees it (DeepEyes' convention) | 28 px | 28 px per side |
| Qwen3-VL | relative [0, 1000] (cookbook) | 32 px | 32 px per side |

The tools convert model coordinates to the full-resolution source, so the low-res conditions still crop real detail. Tool outputs are shown at their own size, which is the frame for later calls on them. `img_idx` indexes every image in the conversation in order of appearance: 0 is the input, and each image a tool returns is appended. This is Qwen-Agent's convention. Tools always read the full-resolution input.

| Tool | Source | Arguments | Returns |
| --- | --- | --- | --- |
| `image_zoom_in_tool` | Exact port of Qwen-Agent / Qwen3-VL cookbook (ZwZ baseline); for Qwen2.5-VL the same tool in DeepEyes' absolute-pixel convention. Pixel parity is tested against the upstream code, and arguments are validated like Qwen-Agent's jsonschema check. | `bbox_2d`, `label`, `img_idx` | Crop, upscaled per Qwen-Agent |
| `image_rotate_tool` | ZwZ Table 9 "rotate"; schema ours | `angle` (counter-clockwise, degrees), `img_idx` | Rotated image, canvas expanded |
| `image_flip_tool` | ZwZ Table 9 "flip"; schema ours | `direction` (horizontal / vertical), `img_idx` | Mirrored image |
| `image_draw_tool` | ZwZ Table 9 "draw auxiliary lines"; schema ours | `lines`, `boxes`, `points`, `color`, `img_idx` | Annotated copy |
| `calculator` | Ours | `expression` | Exact integers; floats shown to 12 significant digits (15 above 1e10). Comparisons, `floor`/`ceil`, `//` and `%` use the value as shown, cancellation noise in `+`/`-` is removed, and degree trig is exact at special angles. `round` is Python's, like TRACE's ground truth. Uses a character/identifier/AST whitelist with no `eval`, and a per-call budget that bounds every big-integer operation before it runs. Common notation slips (`2pi`, `5!`, `15%` as percent, `1,200`, `math.sqrt`) return a corrective error. |
| `ocr` | Ours; RapidOCR 3.x (PP-OCR, onnxruntime, CPU) | `img_idx`, optional `bbox_2d` | Text lines with boxes in the family's frame (pixels of the image as shown for Qwen2.5-VL; [0, 1000] for Qwen3-VL) |

Table 9 actions **not** implemented, and why:
- **Zoom out, denoise:** TRACE images are clean renders shown at native size, so neither changes what the model sees.
- **2D grounding:** both families ground natively in the frame the tools use.
- **3D grounding / depth:** needs an extra perception model, which would turn the comparison into a test of that model.
- **Web search:** nothing external is relevant to synthetic scenes.

**Calls within a turn** run in order, each seeing the images earlier calls returned, as in Qwen-Agent. `img_idx` follows Python list indexing (−1 is the latest image).

**Prompting.** The tool block is Qwen-Agent's Nous text, which is also the Qwen3-VL template's own. It is appended to the TRACE system prompt, and tool results return as `<tool_response>` user turns. For Qwen2.5-VL, whose template has no tool path, this is the format Qwen2.5-VL agents are prompted with; `tests/test_qwen25.py` checks it through the Qwen2.5-VL template. `tests/test_prompting.py` renders our messages and the native `tools=` / `role: tool` path through the checkpoint's Jinja template and checks they are byte-identical. `scripts/check_template.py` repeats this with the real processor, and vLLM's `/tokenize`, on the server.

## 4. Phase 1: inference-time tools (Q1–Q3)

### Models
- **Primary:** `Qwen/Qwen2.5-VL-7B-Instruct`. **Secondary:** `Qwen/Qwen2.5-VL-3B-Instruct`. These are TRACE's base models, at TRACE's pinned revisions (`scripts/_model_env.sh`).
- **Built-in sanity check:** `no_tool` uses TRACE's validation protocol, so seed 42 should land near TRACE's published base scores, 34.25 (7B) and 24.45 (3B). A large gap means the setup is off.
- **Optional:** TRACE's GRPO checkpoints `maveryn/trace-qwen2.5-vl-7b` / `-3b` (51.55 / 41.05). These answer a related question: do tools still help after RL without tools? They are also the published "RL without tools" point for Phase 2.
- **Supported but secondary:** Qwen3-VL (`--family qwen3-vl`, auto-detected), for comparison with ZwZ's Qwen3-VL numbers.
- **Expect weaker zero-shot tool use than Qwen3-VL,** especially at 3B. That would be a finding about these models, not about the tools, and it is what Phase 2's RL would address.
- **Thinking variants are excluded from the main grid.** Their long reasoning would be cut off by TRACE's 2,048-token budget, and their templates handle earlier-turn reasoning differently. They can be added as a separate study.

### Conditions (`configs/conditions.yaml`)

Every condition uses the TRACE system prompt and TRACE decoding (temperature 0.6, top_p 0.95, top_k −1, seed per run). Unless noted, a condition allows 6 assistant turns, 2,048 tokens per turn, 8,192 generated tokens in total, and 2 tool calls per turn.

| Condition | Tools | Budget | Purpose |
| --- | --- | --- | --- |
| `no_tool` | — | 1 turn, 2,048 | TRACE protocol settings (prompt, decoding, pixel bounds, scoring). Primary baseline. |
| `no_tool_budget` | — | 1 turn, 8,192 | Controls for the larger token budget of tool runs (Q2) |
| `zoom` | zoom | default | ZwZ "official tool use" baseline |
| `visual_ops` | zoom, rotate, flip, draw | default | The ZwZ Table 9 operations we implement |
| `calculator` | calculator | default | Arithmetic: geometry, charts, physics |
| `ocr` | ocr | default | Text-heavy: pages, charts, symbolic |
| `all` | all six | default | Full toolset |
| `zoom_cookbook` | zoom | default | Prompt ablation: cookbook "research assistant" prompt before the TRACE prompt |
| `oracle_crop` | — | 1 turn | Upper bound for perfect localisation: adds a crop of the ground-truth annotation region. Diagnostic only, because it can leak the answer. |
| `no_tool_lowres` / `zoom_lowres` | — / zoom | as `no_tool` / `zoom` | Model sees ≤ 0.4 MP; tools read full resolution. Tests ZwZ footnote 1. |

**Q2 control.** A tool condition differs from `no_tool` in two ways: the appended tool block, and a multi-turn budget (6 turns, 8,192 generated tokens).
- `no_tool_budget` gives a single turn the same 8,192 tokens, which isolates the budget part.
- No condition isolates the prompt part (the tool block without tools to call). `zoom_cookbook` shows how much prompt wording alone moves accuracy; read Q2 conclusions with that limit in mind.

**Images.** Images are resized client-side (PIL bicubic) to the condition's pixel bounds, on the family's patch grid (28 px for Qwen2.5-VL, 32 for Qwen3-VL), before they are sent. TRACE instead sends the original PNG and lets the server resize it. The target sizes are the same, but the interpolation can differ slightly from vLLM's processor. Every condition shares this, so it does not bias comparisons between conditions. It is the one known difference from TRACE's published request.

### Metrics
- **Primary:** official accuracy, scored with TRACE's own code (vendored) and the judge fallback, on the final assistant message.
- **Secondary:**
  - strict exact-JSON accuracy, which is the RL reward term;
  - format rate (final JSON present and well-formed);
  - accuracy by domain and by answer type.
- **Tool behaviour:**
  - `tool_use_rate`: the share of episodes with at least one successful call to a tool the condition offers;
  - `tool_attempt_rate`: any request, including malformed, unknown-tool and limit-skipped calls;
  - requested and executed calls per episode;
  - tool error rate (failed / executed);
  - accuracy when a tool was used vs. not used (correlational: the model chooses when to call);
  - stop reasons.
- **Cost:** generated tokens, turns, and mean / median wall-clock latency per episode.

### Statistics
- All comparisons are paired on the same rows.
- **Per seed:** paired bootstrap 95% CI of Δaccuracy, plus an exact McNemar test (`trace-tool-compare`).
- **Seeds pooled:** each row's correctness is averaged over seeds before bootstrapping.
- **Multiple comparisons:** Holm correction across the compared conditions vs. `no_tool`. For the full run that is the 5 tool conditions plus `no_tool_budget`: `trace-tool-compare <out> --conditions no_tool_budget,zoom,visual_ops,calculator,ocr,all`. Without `--conditions`, every condition in the directory joins the family (in the pilot that includes the diagnostics).
- **Rough minimum detectable effect.** Assume 15–25% of rows flip between two conditions. Then the standard error of Δ is about √(0.2/n): ~1.0 pp at n = 2000 and ~1.4 pp at n = 1000. So, at α = 0.05 and 80% power:
  - a single seed resolves effects of about **≥ 3 pp** on the full set, and about 4 pp in the 1,000-row pilot;
  - three seeds reduce decoding noise, but not row-sampling noise.
  - Treat per-domain deltas on 100–360 rows as exploratory.

### Run plan
1. **Smoke** (`scripts/smoke_test.sh`, 20 rows × all conditions), plus `scripts/check_template.py`. Read about 20 tool transcripts by hand: valid JSON calls, sensible boxes, a final JSON answer.
2. **Pilot** (`STAGE=pilot`): 1,000 rows (one per task), seed 42, all 11 conditions, 7B. This is 11k episodes. Drop conditions that are clearly null or broken.
3. **Full** (`STAGE=full`): 2,000 rows × seeds 42, 43, 44, with the core conditions and any surviving diagnostics, on both 7B and 3B. This is up to 42k episodes per model.
4. **Judge pass** for missing/ambiguous extractions, then score, report and compare.

### Decision rule for Phase 2 (proposed; adjust if you prefer)
Go to Phase 2 if some tool condition T meets all three:
- (a) Seeds-pooled official accuracy beats **both** `no_tool` and `no_tool_budget` by ≥ 2 pp, with Holm-adjusted McNemar p < 0.05 on the per-seed tests.
- (b) The tool is actually used: `tool_use_rate` (episodes with a successful call) ≥ 10%.
- (c) The sign holds for the other model size (3B), or there is a clear reason it does not (e.g. 3B rarely calls tools).

A weaker signal, such as gains confined to one or two domains, still justifies a small Phase 2 pilot. Zero-shot tool competence is a lower bound on what RL can teach. Phase 2's tool set should be the best T, plus `all` if it is not worse.

### Threats to validity
- **Prompt sensitivity:** the tool block lengthens the system prompt. `zoom_cookbook` shows how much prompt wording alone moves results.
- **Instruct models may under-call or mis-call tools.** The tool error rate and stop reasons make this visible; it is a finding, not a bug.
- **OCR backend quality bounds the `ocr` condition.** `TRACE_TOOL_OCR_BACKEND=tesseract` is available as a check.
- **`oracle_crop` uses ground-truth annotations.** It is an upper bound on localisation, never a method.

## 5. What the harness records

For every condition, model and seed, the harness writes `responses.jsonl`, containing:
- the full transcript (images as index and size);
- every requested tool call with its arguments, result text, executed/ok flags and latency (calls the turn limit prevented are recorded as not executed);
- stop reason, turns and token counts.

Runs are resumable, and a crash mid-write is repaired on the next run. A rerun refuses to mix models, settings, dataset selections or OCR backends into an existing directory, and a lock stops two writers. Episodes that end in an endpoint error are retried on the next run and are excluded from scores until then; the report shows how many.

`run_config.json` records:
- model, model id, condition and seed;
- every setting, including the OCR backend and package versions;
- the system-prompt SHA-256 and exact tool schemas;
- the dataset revision, the git commit and whether the tree was dirty.

## 6. Phase 2: RL with vs. without tools (Q4–Q5), only if Phase 1 says so

### Framework
TRACE's EasyR1 backend is single-turn, so both arms run on **verl's `ToolAgentLoop`**. Running both arms in the same framework keeps the framework from confounding the comparison. verl's agent loop:
- supports tools that return images;
- masks tool tokens out of the loss (`response_mask`);
- uses Hermes tool parsing.

**Chat template.** verl renders tool schemas (`tools=`) and results (`role: tool`) through the policy's chat template. Qwen2.5-VL's template has neither, so its policy would never see the tools. `python -m trace_tool.rl.prepare_policy` copies the checkpoint and installs `configs/verl/qwen2_5_vl_tool_chat_template.jinja`, which renders tool turns byte-identically to the eval prompts (`tests/test_rl_template.py`). Use the prepared copy for every arm. If a multi-turn rollout ever lacks a tool-response turn boundary, the reward raises rather than silently crediting earlier turns.

Scaffold in this repo (marked UNTESTED on GPU):
- `trace_tool/rl/verl_tools.py`: BaseTool adapters. They pass verl our exact schemas.
- `trace_tool/rl/reward.py`: TRACE reward on the final assistant segment. Tested on CPU, including verl's special-token-stripped decoding.
- `trace_tool/rl/prepare_verl_data.py`: tested.
- `configs/verl/tools_*.yaml`.
- `scripts/rl/run_grpo_template.sh`.

### Arms
All arms train on TRACE train (64k rows, 64 per task), the same data order and the same hyperparameters.

| Arm | Agent loop | Tools at train time |
| --- | --- | --- |
| A0 | single-turn | none (TRACE recipe re-run in verl) |
| A1 | tool agent | best Phase-1 tool set (e.g. zoom) |
| A2 | tool agent | `all` (if different from A1) |
| A3 (optional) | tool agent | as A1, plus a DeepEyes-style bonus for correct answers that used a tool (tests whether tool use must be encouraged) |

Base models are Qwen2.5-VL-7B (and 3B), the same bases and data as TRACE's own runs. That gives A0 an external reference: TRACE's released EasyR1 checkpoints (51.55 / 41.05 on validation; +4.06 / +3.51 macro on 24 external benchmarks). If A0 in verl lands close to those, the framework switch did not change the recipe's effect, and A1/A2 vs. A0 isolates tools.

### Hyperparameters (TRACE EasyR1 → verl)
- GRPO, n = 8, 128 prompts per step, mini-batch 128 (one update per step), 500 steps.
- lr 1e-6 constant, weight decay 0.01, grad clip 1.0.
- clip_ratio_low 0.2 / clip_ratio_high 0.3, dual-clip c = 3.0.
- Token-mean loss; KL off; rollout temperature 1.0.
- Validation at temperature 0.6 / top_p 0.95.
- Reward 0.95 × answer + 0.05 × format, computed on the final assistant message.
- Tool arms: 6 assistant turns, 2 calls per turn, tool-response text up to 6,000 characters (verl's default of 256 would truncate OCR output).
- **Not portable:** EasyR1's `tau_positive` / `tau_negative`. The A0 arm checks the effect of dropping them, by comparing its TRACE-val gain with TRACE's published Qwen2.5-VL gains.

### Fairness
- verl's `response_length` counts tool-response tokens, images included. So the tool arms get 10,240 tokens against 2,048, while the per-turn generation cap stays the same.
- Compare arms at equal steps (primary) and report GPU-hours (secondary).
- **OCR in rollouts is CPU-bound,** about 5 s per full-image call on the CPU I tested. Results are cached process-wide by image content and region, and concurrent identical requests run the engine once. So the 8 rollouts of one prompt share one OCR run per region within a rollout worker. Check CPU headroom on the training node anyway.
- **Calls within a turn:** verl runs them in parallel, against the images from before the turn. The eval harness runs them in order, as Qwen-Agent does. This only matters when a single turn holds two dependent calls.

### Evaluation: 2 × 2 (train with/without tools × test with/without tools)
- **In-distribution:** TRACE validation, using this harness (`no_tool` and the tool condition).
- **External, without tools:** TRACE's 24-benchmark `trace_eval_v1` suite, through their VLMEvalKit harness. It is directly comparable to the published TRACE numbers.
- **External, with tools:**
  - the perception / high-resolution set where zoom is known to matter: V*, HR-Bench 4K/8K, MME-RealWorld, ZoomBench;
  - chart and document benchmarks for OCR: ChartQAPro, CharXiv.
  - This needs dataset adapters for `run_eval` (the agent loop and tools are dataset-agnostic). That is the main engineering item for Phase 2.
- **Training dynamics:** tool-call rate over training (do policies learn or abandon tools?), reward and accuracy curves, response length, tool error rate.

## 7. Open choices for you
1. GPU budget, and whether to include TRACE's RL checkpoints and Qwen3-VL. The default is Qwen2.5-VL-7B primary with 3B as the replication.
2. The Phase-2 decision rule above.
3. Whether to include Thinking variants as a separate study.
4. The external benchmark list for Phase 2 (default: TRACE's 24 plus the ZwZ perception set).
