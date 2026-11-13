# Final evaluation: MMLU-Pro for Vero and TRACE; TRACE validation for four new models

> **Sources.** The tables come from [`tables.md`](tables.md), [`mmlu_pro_summary.json`](mmlu_pro_summary.json) and
> [`trace_val_summary.json`](trace_val_summary.json).
> - [`final_eval/summarize.py`](../../final_eval/summarize.py) writes all three from the per-row run files. It refuses
>   to run while any TRACE row still waits for the judge.
> - Each per-model TRACE `report.md` is under [`trace_val/`](trace_val/).
> - Numbers marked † are not in those files. They are counts taken from the row-level outputs or the server logs
>   during the two verification passes. Each † number was reproduced by a second, independent reviewer, except where
>   the text says otherwise.
> - Row-level outputs stay local, as in the earlier studies.
> - Run on 2026-11-12, on held CHPC allocations, with vLLM 0.30.0 (torch 2.13 cu130, transformers 5.18).

## Takeaway

**1. MMLU-Pro: under the official protocol both RL-trained checkpoints score about 3–4 pp above their shared base,
Qwen2.5-VL-7B-Instruct. That advantage depends on the prompt format: in a chat-format run of MMLU-Pro's other official
prompt it is about 1 pp or less. RL training on visual reasoning did not hurt MMLU-Pro in either format.**

| model | official protocol (primary) | Δ vs base, strict [95% CI] | chat format (secondary), strict | Δ vs base [95% CI] |
| --- | ---: | --- | ---: | --- |
| Vero-Qwen25-7B | **52.67** | +4.01 [+3.22, +4.81] | 51.73 | +0.87 [+0.06, +1.66] |
| TRACE-Qwen2.5-VL-7B | **51.53** | +2.85 [+2.15, +3.55] | 51.61 | +0.76 [+0.00, +1.51] |
| Qwen2.5-VL-7B-Instruct (base of both) | 48.75 | — | 50.86 | — |

The official protocol is raw-text 5-shot chain of thought, greedy. There, every difference is significant (exact
McNemar, Holm over 3 pairs):

| pair | Δ | p (Holm) |
| --- | ---: | ---: |
| Vero − base | +4.01 | 1.1e-22 |
| TRACE − base | +2.85 | 1.0e-14 |
| Vero − TRACE | +1.16 [+0.41, +1.91] | 0.0027 |

Vero beats the base in all 14 categories and TRACE in 12 of 14 (TRACE trails in engineering −0.3 and health −1.1).

- Vero's largest gains: chemistry +8.0, math +6.1, history +5.8, biology +5.2, physics +4.8, business +3.9.
- TRACE's largest gains: business +6.7, math +5.8, chemistry +5.7, law +2.9, physics +2.8, economics +2.8.

**Why the chat-format numbers differ.** In chat format, MMLU-Pro's `evaluate_from_api.py` prompt goes in as one user
message and the models stop at their end-of-turn token. The gap to the base shrinks by roughly three quarters, mainly
because the base does better in chat:

| model | chat − raw-text, strict, same questions [95% CI] |
| --- | --- |
| base | +2.20 [+1.42, +2.98] |
| TRACE | +0.11 [−0.63, +0.86] |
| Vero | −0.94 [−1.59, −0.31] |

- The base's raw-text deficit is not a looping artifact. It is the same on the raw outputs that ran to the 2,048-token
  limit (+2.25 pp in chat) and on those that stopped (+2.14).
- The two runs change several things at once: chat template, prompt wording, number of exemplars (310 questions get
  fewer than 5 in raw text), token budget (2,048 vs 4,000), and the stop rule. So which of these costs the base in raw text is not identified.
- In chat format the gains over the base are not significant after Holm (p 0.11 for both). The bootstrap CIs in the
  table above are unadjusted.
- In chat format, the order of the two RL models is not robust. Unextractable chat answers are mostly math answers
  given as a boxed value instead of a letter: TRACE 172 of 181, base 123 of 213 (52 more of the base's are
  4,000-token cut-offs), Vero 1 of 1. A sensitivity check
  (not a protocol) credits a value that matches exactly one option. It gives Vero − base +0.22 [−0.59, +1.00],
  TRACE − base +1.09 [+0.35, +1.82] and Vero − TRACE −0.87 [−1.69, −0.09].

**2. TRACE validation, under TRACE's own protocol (2,000 rows × seeds 42, 43, 44): Qwen3.8-27B is best by a clear
margin.**

| model | mean ± SD over seeds | seeds-pooled [95% CI] | 2,048-token cut-offs |
| --- | ---: | --- | ---: |
| Qwen3.8-27B | **72.55 ± 0.93** | 72.55 [70.85, 74.22] | 16.3% |
| Qwen3.5-9B | **64.93 ± 0.67** | 64.93 [63.15, 66.75] | 17.3% |
| Gemma-4-12B-it | **62.27 ± 0.10** | 62.27 [60.40, 64.17] | 8.2% |
| LLaVA-OneVision-2-8B | **27.28 ± 0.16** | 27.28 [25.67, 28.92] | 0.0% |

Reference points:

- In this repo, with the same protocol and vLLM version (earlier commits): Qwen2.5-VL-7B-Instruct 33.80 and
  Qwen3.5-4B 61.65.
- Published TRACE validation numbers (TRACE paper, quoted in [EXPERIMENT_DESIGN.md](../../EXPERIMENT_DESIGN.md)):
  Qwen2.5-VL-7B 34.25, TRACE's GRPO-trained 7B 51.55. TRACE-7B was not re-run here.

How the models compare:

- Qwen3.8-27B leads Qwen3.5-9B by +7.62 pp [+6.13, +9.07] and Gemma by +10.28 [+8.48, +12.10].
- Qwen3.5-9B leads Gemma by +2.67 [+0.80, +4.53].
- Zero-shot, these three score 10.7–21.0 pp above the published TRACE-trained 7B.
- Qwen3.8's lead over Qwen3.5-9B is largest on icons (+17.7), games (+12.8), graph (+10.6) and three_d (+10.0). Pages
  is the only domain where it does not lead: 87.1 vs 88.3.

Two behaviours shape these numbers:

- **LLaVA-OneVision-2-8B answers without reasoning.** TRACE's prompt says "Reason carefully … end with a JSON
  object". LLaVA replies with bare JSON in 98.9%† of rows: 8–9 tokens on average, no cut-offs. It scores below the
  early-2025 Qwen2.5-VL-7B (27.28 vs 33.80). The images reach it correctly (exact visual-token accounting) and its
  rendered template carries TRACE's system prompt, so image handling and prompting are ruled out as harness faults.
- **The Qwen3.x models hit the 2,048-token cap most often.** Qwen3.8 and Qwen3.5 still write long chains of thought
  with thinking off, so 16–17% of their episodes are cut off, against 8.2% for Gemma. Cut-off rows score 10.7–16.5%†
  per seed, almost all of it through the judge. The cap's cost was measured only for Qwen3.5, in the October study:
  8,192 tokens (`no_tool_budget`) beat `no_tool` by +5.1 / +5.3 pp for 4B/9B. This file reports TRACE's protocol as is.

**3. Two measurement results.**

- **Gemma 4's image budget is decisive under vLLM 0.30.** The main run uses the checkpoint's own budget, 280 soft
  tokens. The first run tried the maximum, 1120, and Gemma collapsed to 19.52%: −42.75 pp [−44.87, −40.60]. The drop
  is 29–54 pp depending on the domain, leaving 15–25% in every domain and answer type. That run is kept only as a
  diagnostic.
  - The vision-token counts match the processor's formula on every row, so a processor mismatch is ruled out.
  - The cause is not identified. The leading hypothesis is unverified: at 1120 every image span (1,056–1,120 soft
    tokens†) exceeds Gemma 4's 1,024-token sliding window. That sends it through a Gemma-4-specific attention path
    in vLLM that 280-token images never reach.
- **The Qwen3.5-9B rerun reproduces the October run.**
  - Accuracy: 64.93 vs 65.25, a pooled difference of −0.32 pp [−1.20, +0.62]. Same checkpoint, protocol and vLLM
    version.
  - Rows agree 87–88% of the time with the same seed. Between different seeds within either run they agree about
    82%†. So seeded sampling is partly, not fully, reproducible across runs.

## 1. What was run

| model | checkpoint @ revision | evaluated on |
| --- | --- | --- |
| Vero-Qwen25-7B | `zlab-princeton/Vero-Qwen25-7B` @ `180e84be` | MMLU-Pro |
| TRACE-Qwen2.5-VL-7B | `maveryn/trace-qwen2.5-vl-7b` @ `4d0f1ae8` | MMLU-Pro |
| Qwen2.5-VL-7B-Instruct (reference: base of both) | `Qwen/Qwen2.5-VL-7B-Instruct` @ `cc594898` | MMLU-Pro |
| Qwen3.8-27B | `Qwen/Qwen3.8-27B` @ `1d4bf0f2` | TRACE validation |
| Qwen3.5-9B | `Qwen/Qwen3.5-9B` @ `c2022362` (same snapshot as the October study) | TRACE validation |
| Gemma-4-12B-it | `google/gemma-4-12B-it` @ `707f0a3b` | TRACE validation |
| LLaVA-OneVision-2-8B | `lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct` @ `878cde7e` | TRACE validation |

The base model was added to MMLU-Pro (it was not in the request) because, without it, "did RL training on visual
reasoning change text reasoning?" has no answer.

### MMLU-Pro protocol

**Primary: the official local protocol**, TIGER-AI-Lab/MMLU-Pro `evaluate_from_local.py` @ `f418b116`. The file and
its prompt file are vendored byte for byte and sha256-checked; the prompt builder, extractor and scorer are imported
from them unchanged.

- **Shots:** up to 5 chain-of-thought exemplars from the validation split of the same category, dropped one at a time
  while the prompt is ≥ 2,048 tokens. 11,722 of the 12,032 questions get 5.
- **Decoding:** raw-text prompts (no chat template), greedy, 2,048 new tokens, 4,096-token context, stop at
  `Question:`.
- **Scoring:** first-match regex chain. An unextractable answer gets the official seeded random letter. Strict
  accuracy (unextractable = wrong) is reported as well.
- **Data:** `TIGER-Lab/MMLU-Pro` @ `b189ec76`, 12,032 test questions.
- **Hardware:** one greedy engine per GPU on a question shard, all on 24 GB RTX PRO 4000 Blackwell cards.
  - Vero: 8 shards on grn085; shards 0, 6 and 7 were regenerated on grn084 (see Incidents).
  - TRACE: 8 shards on grn082.
  - Base: 4 shards on grn084.

**Secondary: the official API evaluator**, `evaluate_from_api.py` (same commit, vendored and sha256-checked).

- **Prompt:** its own `single_request` builds every prompt, with the API call swapped for a capture, so prompts are
  the official ones byte for byte. All 5 exemplars and the question go into one user message, rendered with the
  checkpoint's chat template.
- **Decoding:** greedy, 4,000 tokens, as in the script's OpenAI branch.
- **Scoring:** `**` is stripped, then the official extractor applied. Only strict accuracy is reported, because the
  API script's random fallback depends on its resume loop; it would add about +0.2 pp for TRACE and the base.
- **Hardware:** Vero on A100 80 GB (notch366) and RTX 6000 Ada (notch456) cards; TRACE and the base on 24 GB RTX PRO
  4000 cards. Under greedy decoding this changes only numerics.

### TRACE validation protocol

This repo's `no_tool` condition, i.e. TRACE's validation protocol:

- the TRACE system prompt (sha256 `d6105d53…`);
- temperature 0.6, top_p 0.95, top_k −1, one turn, 2,048 tokens, request seed = run seed;
- image pixels bounded to 262,144–4,194,304;
- deterministic extraction, then TRACE's Qwen3-32B judge (@ `9216db57`) for missing or ambiguous answers;
- exact-match scoring, with unresolved rows scored 0;
- dataset `maveryn/trace@dataset-v1` (sha `051d1441…`), 2,000 rows.

Per-model settings (`src/trace_tool/models.py`, `scripts/_model_env.sh`):

| model | family | thinking | images | server |
| --- | --- | --- | --- | --- |
| Qwen3.8-27B | `qwen3.8` | off (`enable_thinking: false`; the template then adds no reasoning-effort text) | TRACE bounds, 32-px grid (= its processor) | `max_num_seqs` 256 (the default 1,024 exceeds the 629 Mamba state blocks that fit beside the weights on a 96 GB card) |
| Qwen3.5-9B | `qwen3.5` | off | as above | as in October |
| Gemma-4-12B-it | `gemma4` | off (template default, also sent explicitly) | TRACE bounds only; Gemma's processor then resizes to its 280-token budget | audio inputs off |
| LLaVA-OV-2-8B | `llava-onevision-2` | none in its template | TRACE bounds, 28-px grid (= its processor) | `--trust-remote-code` (processor) |

The three new families accept only tool-free conditions; a tool condition refuses to start for them.

The judge ran at TP1 on a ≥ 90 GB card, as the TRACE-VERO judge-fault lesson requires: Qwen3.5-9B and LLaVA on an
RTX PRO 6000, Qwen3.8 and Gemma on an H200.

Hardware for the models themselves:

- Qwen3.8: the first 634, 637 and 33† rows of seeds 42, 43 and 44 ran on an RTX PRO 6000 (grn075); the rest ran on
  an H200 (grn029).
- Qwen3.5-9B and LLaVA: A100 80 GB (notch366).
- Gemma: seeds 42/43 on 2× RTX PRO 6000 (grn075), seed 44 wholly on the H200.

## MMLU-Pro

### 2.1 Main result (official protocol)

| model | official acc | strict acc | no extraction | truncated (2,048 tok) | mean output tokens | strict, last match (sensitivity) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vero-Qwen25-7B | 52.67 | 52.67 | 0 | 9,225 | 1,630 | 49.64 |
| TRACE-Qwen2.5-VL-7B | 51.53 | 51.50 | 12 | 10,859 | 1,868 | 47.56 |
| Qwen2.5-VL-7B-Instruct | 48.75 | 48.65 | 80 | 6,805 | 1,212 | 48.41 |

Paired, per question, strict correctness (95% bootstrap CI; exact McNemar, Holm over 3 pairs):

| pair | Δ (pp) | 95% CI | a right only | b right only | p (Holm) |
| --- | ---: | --- | ---: | ---: | ---: |
| Vero − base | +4.01 | [+3.22, +4.81] | 1,431 | 948 | 1.1e-22 |
| TRACE − base | +2.85 | [+2.15, +3.55] | 1,132 | 789 | 1.0e-14 |
| Vero − TRACE | +1.16 | [+0.41, +1.91] | 1,147 | 1,007 | 0.0027 |

Official accuracy (%) by category:

| category | n | Vero | TRACE | base |
| --- | ---: | ---: | ---: | ---: |
| biology | 717 | 75.03 | 71.41 | 69.87 |
| business | 789 | 58.05 | 60.84 | 54.12 |
| chemistry | 1,132 | 51.24 | 48.85 | 43.20 |
| computer science | 410 | 51.71 | 51.22 | 49.51 |
| economics | 844 | 63.27 | 63.98 | 61.14 |
| engineering | 969 | 40.97 | 36.74 | 37.05 |
| health | 818 | 52.57 | 51.22 | 52.32 |
| history | 381 | 48.03 | 44.88 | 42.26 |
| law | 1,101 | 28.07 | 27.79 | 24.89 |
| math | 1,351 | 63.80 | 63.58 | 57.74 |
| other | 924 | 46.65 | 48.05 | 45.56 |
| philosophy | 499 | 44.09 | 41.88 | 41.68 |
| physics | 1,299 | 51.58 | 49.58 | 46.73 |
| psychology | 798 | 64.29 | 62.28 | 61.65 |

### 2.2 What the models write after answering (official protocol)

In raw-text completion all three models keep writing after their answer until the 2,048-token limit, in 77% (Vero),
90% (TRACE) and 56% (base) of outputs. The official extractor scores the first "the answer is (X)" when the output
contains one, so this text is never scored.

- **Where the answer sits.** In truncated outputs, the first match is at a median 12% (Vero) and 5% (TRACE, base) of
  the characters†.
- **What comes after it.**
  - TRACE and the base mostly repeat ". the answer is (X).".
  - TRACE also often re-answers the prompt's own few-shot exemplar questions: 252† tails carry a "Question N" header,
    and roughly 380–480† contain verbatim exemplar text (depending on how long a match counts as verbatim).
  - Vero re-checks its own question ("To revalidate…"), and its letter sometimes changes.
- **Unextractable rows.** Every unextractable answer (0 / 12 / 80) ran to the limit without reaching an answer. Most
  are repetitive loops; a few are long iterative calculations that were cut off.
- **TRACE's checkpoint cannot stop on `<|endoftext|>`.** Its `generation_config.json` lists only `<|im_end|>` as EOS,
  while Vero and the base also list `<|endoftext|>`. Those two stop on it in 2,489† and 3,916† outputs; TRACE never
  does. So TRACE's truncation count and mean length are not a like-for-like behavioural comparison. The official
  script behaves the same way.
- **Last-match column (a scoring sensitivity, not a protocol).** It scores the last "answer is" instead of the first.
  The RL-tuned models' later text drifts more than the base's: the first and last letters differ in 973† (Vero),
  1,048† (TRACE) and 68† (base) truncated outputs. It does not explain the chat-format results below: under the first
  match, which the official numbers use, the post-answer text is never scored.

### 2.3 Secondary protocol: official API-style prompt as one chat message

| model | strict acc | no extraction | truncated (4,000 tok) | mean output tokens |
| --- | ---: | ---: | ---: | ---: |
| Vero-Qwen25-7B | 51.73 | 1 | 1 | 460 |
| TRACE-Qwen2.5-VL-7B | 51.61 | 181 | 16 | 327 |
| Qwen2.5-VL-7B-Instruct | 50.86 | 213 | 116 | 274 |

Paired, strict (95% bootstrap CI; exact McNemar, Holm over 3 pairs):

| pair | Δ (pp) | 95% CI | p (Holm) |
| --- | ---: | --- | ---: |
| Vero − base | +0.87 | [+0.06, +1.66] | 0.11 |
| TRACE − base | +0.76 | [+0.00, +1.51] | 0.11 |
| Vero − TRACE | +0.12 | [−0.71, +0.91] | 0.79 |

Format effect: chat strict minus raw-text strict, same questions (unadjusted exact McNemar):

| model | chat − raw (pp) | 95% CI | questions up / down | p |
| --- | ---: | --- | ---: | ---: |
| Vero | −0.94 | [−1.59, −0.31] | 687 / 800 | 0.0037 |
| TRACE | +0.11 | [−0.63, +0.86] | 1,086 / 1,073 | 0.80 |
| base | +2.20 | [+1.42, +2.98] | 1,282 / 1,017 | 3.6e-8 |

The base gains about the same in chat on questions whose raw-text output ran to 2,048 tokens (47.01 → 49.26, n 6,805)
and on those that stopped (50.79 → 52.94, n 5,227).

Notes on the chat run:

- **Templates differ by model.**
  - Vero's chat template replaces any system message with its own fixed 494-token† prompt, an image-QA instruction
    asking for `<think>…</think><answer>\boxed{}</answer>`, and opens the assistant turn with `<think>`.
  - Vero answers "The answer is \boxed{X}", so 87%† of its letters are found only by the extractor's last fallback
    (the last standalone letter). Scoring its last boxed letter instead gives 51.50† (51.52 with the official extractor as fallback).
  - TRACE and the base get Qwen's default "You are a helpful assistant." (TRACE does not get its RL-training prompt.)
- **Boxed-value answers.** Most unextractable chat answers are math answers given as a value instead of a letter.
  The value-mapped sensitivity in [`tables.md`](tables.md) credits a value that matches exactly one option:
  - TRACE: +120 rows (119 in math), 52.61;
  - base: +80 rows, 51.52;
  - Vero: +1 row, 51.74.
  It is also why TRACE's math looks different across formats. In chat, TRACE's strict math is 54.92 against 63.51 in raw
  text, almost entirely because of these unextractable answers.
- **Per-category differences are mixed.** In chat, TRACE leads the base on physics (+4.4), engineering (+3.1),
  business (+2.4), chemistry (+2.2) and computer science (+2.2), and trails on law (−3.0) and math (−1.8). The per-category table is in [`tables.md`](tables.md).

## TRACE validation

### 3.1 Main result

Official accuracy (%), TRACE protocol (`no_tool`), 2,000 rows per seed, all judge items resolved.

| model | seed 42 | seed 43 | seed 44 | mean ± SD | seeds-pooled [95% CI] | truncated | judge-resolved | unresolved | mean tokens |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Qwen3.8-27B | 72.40 | 71.70 | 73.55 | 72.55 ± 0.93 | 72.55 [70.85, 74.22] | 16.33% | 3.77% | 12.33% | 772 |
| Qwen3.5-9B | 64.20 | 65.50 | 65.10 | 64.93 ± 0.67 | 64.93 [63.15, 66.75] | 17.32% | 3.63% | 13.13% | 703 |
| Gemma-4-12B-it | 62.35 | 62.30 | 62.15 | 62.27 ± 0.10 | 62.27 [60.40, 64.17] | 8.20% | 1.30% | 6.92% | 549 |
| LLaVA-OneVision-2-8B | 27.10 | 27.40 | 27.35 | 27.28 ± 0.16 | 27.28 [25.67, 28.92] | 0.00% | 0.15% | 0.02% | 9 |
| *Qwen3.5-9B, 2026-10-06 run* | 65.40 | 64.60 | 65.75 | 65.25 ± 0.59 | 65.25 [63.50, 67.03] | 17.20% | 3.47% | 13.17% | 699 |
| *Qwen3.5-4B, 2026-10-06 run* | 62.60 | 61.55 | 60.80 | 61.65 ± 0.90 | 61.65 [59.82, 63.48] | 19.30% | 3.82% | 14.93% | 763 |
| *Qwen2.5-VL-7B-Instruct, 2026-10-05 run* | 33.85 | 32.80 | 34.75 | 33.80 ± 0.98 | 33.80 [32.20, 35.42] | 0.33% | 0.05% | 1.58% | 193 |
| *Qwen2.5-VL-3B-Instruct, 2026-10-05 run* | 24.50 | 24.15 | 24.10 | 24.25 ± 0.22 | 24.25 [22.93, 25.63] | 0.52% | 0.40% | 3.03% | 194 |

- The rows in italics are earlier runs of this repo under the same protocol, for reference.
- "Unresolved" rows have no extractable answer even after the judge, and score 0.
  - For the Qwen3.x models they are ≥ 99% 2,048-token cut-offs†.
  - For Qwen2.5-VL they are mostly extraction failures on outputs that were not cut off†.

Paired differences between the new models (seeds-pooled, same rows; 95% bootstrap CI over rows):

| pair | Δ (pp) | 95% CI |
| --- | ---: | --- |
| Qwen3.8-27B − Qwen3.5-9B | +7.62 | [+6.13, +9.07] |
| Qwen3.8-27B − Gemma-4-12B | +10.28 | [+8.48, +12.10] |
| Qwen3.8-27B − LLaVA-OV-2-8B | +45.27 | [+43.05, +47.45] |
| Qwen3.5-9B − Gemma-4-12B | +2.67 | [+0.80, +4.53] |
| Qwen3.5-9B − LLaVA-OV-2-8B | +37.65 | [+35.43, +39.87] |
| Gemma-4-12B − LLaVA-OV-2-8B | +34.98 | [+32.70, +37.27] |

### 3.2 By domain and answer type (seeds-pooled %)

| domain | n | Qwen3.8-27B | Qwen3.5-9B | Gemma-4-12B | LLaVA-OV-2-8B | Qwen2.5-VL-7B (Oct) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| charts | 360 | 81.57 | 75.74 | 69.72 | 25.09 | 42.87 |
| games | 340 | 61.47 | 48.63 | 50.59 | 26.67 | 23.24 |
| geometry | 340 | 81.57 | 75.49 | 73.53 | 20.20 | 40.69 |
| graph | 120 | 76.67 | 66.11 | 71.11 | 32.78 | 32.22 |
| icons | 100 | 76.00 | 58.33 | 62.00 | 29.33 | 27.67 |
| illustrations | 120 | 50.28 | 45.28 | 47.22 | 30.28 | 21.39 |
| pages | 160 | 87.08 | 88.33 | 71.67 | 34.17 | 42.92 |
| physics | 100 | 74.00 | 66.00 | 64.67 | 23.33 | 33.67 |
| puzzles | 120 | 55.00 | 48.89 | 46.67 | 27.78 | 25.56 |
| symbolic | 120 | 67.22 | 62.78 | 44.17 | 28.33 | 35.83 |
| three_d | 120 | 68.89 | 58.89 | 66.67 | 38.06 | 30.28 |

| answer type | n | Qwen3.8-27B | Qwen3.5-9B | Gemma-4-12B | LLaVA-OV-2-8B | Qwen2.5-VL-7B (Oct) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| integer | 1,094 | 74.19 | 65.02 | 63.80 | 24.80 | 31.66 |
| number | 102 | 85.29 | 83.33 | 74.84 | 5.23 | 43.14 |
| option_letter | 420 | 62.06 | 53.10 | 51.67 | 32.38 | 28.57 |
| string | 384 | 75.95 | 72.74 | 66.15 | 34.64 | 43.14 |

- **Gemma vs Qwen3.5-9B by domain.** Gemma is close overall, but leads on graph, icons, three_d, games and
  illustrations, and trails well behind on symbolic (44.2 vs 62.8) and pages (71.7 vs 88.3).
- **LLaVA-OV-2 vs the October Qwen2.5-VL-7B.** It is ahead only on three_d, illustrations, games, puzzles, icons and
  graph. It is far behind on geometry (−20.5), charts (−17.8), physics (−10.3), pages (−8.8) and symbolic (−7.5).

### 3.3 Gemma 4 diagnostics

Both diagnostic runs are complete and judged (3 seeds). Each is compared on the same rows with the main Gemma run:
checkpoint budget 280, images bounded only to TRACE's pixel range.

| run | acc | Δ vs main | 95% CI |
| --- | ---: | ---: | --- |
| budget 280, images also rounded to 48-px multiples on the client | 61.80 | −0.47 | [−1.48, +0.58] |
| budget 1120 (max), images rounded to 48 px | 19.52 | −42.75 | [−44.87, −40.60] |

**How the main setting was chosen:**

1. The first run used budget 1120, because 1120 tokens cover every TRACE image's area; the intent was to match the
   Qwen models' "no downscaling". In fact Gemma's processor resizes every image to fill its budget, so at 1120 every
   TRACE image was upscaled (median 1.8×†). It scored 19.5%.
2. A run of seed 42 at the default 280 scored 61.3% before the judge. That replaced 1120.
3. The protocol audit then found the client-side 48-px rounding redundant. It had distorted the most extreme aspect
   ratio by about 14–17%† depending on direction (≤ 8.5% for every other image). So the main run was redone with TRACE's pixel bounds only. The
   rounding turned out not to matter: −0.47 pp, not significant.

At 280, Gemma's processor downscales about 1,280† of the 2,000 TRACE images (to as little as 0.65× per side†),
upscales about 690†, and leaves about 30† unchanged. Gemma therefore does not see TRACE images at native size. Its
result is its score at its default image budget.

**About the 1120 collapse:**

- *Ruled out:* the HF processor ignoring the budget, and placeholder/feature count mismatches. The prompt-token
  counts equal the processor's formula on every row†.
- *Ruled out:* a single bad server. It appears on all three servers that ran it.
- *Open:* plain upsampling is possible but a weak explanation for a 43 pp drop.
- *Leading hypothesis (unverified):* the sliding-window path described in takeaway 3. A 560-budget run would separate
  the two; it was not run.

## Measurement checks

**Independent MMLU-Pro re-derivation** (separate agents, not using our scorer):

- Official protocol:
  - Completeness: each model has exactly 12,032 question ids, no duplicates, correct shard membership. The vendored
    files match upstream byte for byte.
  - Re-score: a re-implemented regex chain and `save_res` flow reproduces every count (correct, wrong, strict,
    no-extraction, truncated, tokens), per category and in total, for all 3 models.
  - Prompt fidelity, checked on all 12,032 rows per model: 0 mismatches in shot count or prompt length.
  - Settings: temperature 0, 2,048 tokens and stop `Question:` are confirmed in code and in every engine log. Prompt
    token counts equal plain tokenization (no chat-template tokens).
  - Paired statistics are exact; bootstrap CIs agree within 0.03 pp.
- Chat protocol:
  - Every captured prompt equals an independent rebuild of the official `single_request` prompt on all 12,032 rows.
  - Rendered chat-template token counts equal the recorded prompt lengths on all rows, for all 3 models.
  - Strict accuracy, per-category values, no-extraction and truncation counts, and the paired statistics all
    reproduce exactly.

**TRACE protocol audit and recomputation** (separate agents):

- Templates, rendered for a TRACE row:
  - Qwen3.8/3.5 get the standard empty think block and no reasoning-effort text (without the kwarg, Qwen3.8's
    template adds "Reasoning effort is set to xhigh" and opens a `<think>`).
  - Gemma has no `<|think|>`.
  - LLaVA uses TRACE's system prompt.
- Thinking markers: none in any main run.
- Rows: 2,000 unique rows per seed, every `error` null, no judge item pending.
- Decoding: identical parameters in every `run_config.json`.
- Images: for Qwen and LLaVA, every shown image equals the expected client-side fit, and the prompt-token counts show
  the server applied no further resize. For the final Gemma run, `server.json` records the settings next to each seed.
- Recomputation: every number in the TRACE tables (per seed, pooled, CIs, breakdowns, pairwise, diagnostics,
  reproduction) was recomputed from the per-row files without `summarize.py`, and matches. CIs agree within 0.05 pp.
- Judge: TP1, all requests 200, nothing rejected. A spot-check of 20 judge-resolved rows per model (audit only, not
  re-reviewed) found:
  - Qwen3.8 15/20 clear agreement plus 2 borderline;
  - Qwen3.5 16/20 plus 2;
  - LLaVA 9/9.
  The misreads come from 2,048-token cut-offs and go both ways. Among judge-resolved rows, the net effect is under
  about 0.6 pp. Rows the judge left unresolved (scored 0) were not audited for missed answers.
- Hardware: the Qwen3.8 rows that moved from RTX PRO 6000 to H200 show no detectable effect. The
  difference-in-differences against rows served on the H200 in every seed is −1.3 to −1.6 pp† (two estimates), with a
  95% CI of about ±3 pp spanning 0. The resumed rows also ran under a newer commit, whose only change on this path was
  the family-detection regex (same result for this model).
- Gemma's default-budget run: every end-of-turn stop is clean (the JSON, optionally in a code fence, then nothing),
  and about 1%† of rows are repetition loops.

## Incidents and deviations (all resolved; none changes a reported number)

- **Qwen3.8 first start.** It failed because vLLM's default `max_num_seqs` (1,024) exceeds the 629 Mamba state blocks
  that fit. It was fixed with 256; requests beyond the cap queue, which changes no output.
- **Vero MMLU shards 0, 6 and 7** failed at engine start with FlashInfer ninja build errors. Twenty engines were
  building the same JIT kernel into one NFS cache at once, a likely race. The three shards were regenerated from
  scratch on grn084.
  - Shard 0 ran with `VLLM_USE_FLASHINFER_SAMPLER=0`. Under greedy decoding vLLM returns the argmax before any
    top-k/top-p step, so this should not change a token. That follows from the code; the shard was not generated both
    ways.
- **Steps reporting rc=2.** The TRACE and base MMLU steps exited 2 because `run_mmlu_pro.sh` was edited while they ran.
  That happened after scoring, and the outputs re-score identically.
- **Preemption.** A preempted guest allocation (notch371) killed one step of the Gemma-1120 run. It resumed elsewhere;
  that run is diagnostic only.
- **Single-card serving.** `trace_val_gen.sh` did not pass DP to the server, so the night's first 2-GPU TRACE
  launches served from one card. That cost time only and is fixed; the final Gemma run used DP2 for seeds 42/43.
- **Qwen3.8 rows moved between cards.** They were moved from grn075 to the H200 mid-run to finish sooner. Rows are
  resumable and keyed by row; see the hardware check above.
- **Gemma prompt rendering.** vLLM sends Gemma's system prompt as a content list, after which Gemma's template appears
  to add one trailing space token (audit only, not re-verified). This applies to every Gemma request.

## Files

| file | contents |
| --- | --- |
| `tables.md` | every table above (except the hand-picked domain columns), generated |
| `mmlu_pro_summary.json` | per-model totals, per-category counts, paired tests; chat protocol, format effect, value-mapped sensitivity |
| `trace_val_summary.json` | per-seed summaries, pooled CIs, domain and answer-type breakdowns, run settings, pairwise, diagnostics, reproducibility |
| `trace_val/<model>/report.md` | trace-tool's per-model report (per-seed and by-domain tables) |
| [`../../final_eval/`](../../final_eval/) | harnesses and the vendored official MMLU-Pro code (with `NOTICE.md`) |

These stay on the cluster checkout, untracked, as in the earlier studies:

- row-level outputs (`runs/final_eval/…`);
- per-run logs and the vLLM server logs (`logs/final_eval/`);
- the CHPC-specific placement scripts (GPU claims, `srun` steps, claim heartbeat), which `final_eval/README.md`
  describes.
