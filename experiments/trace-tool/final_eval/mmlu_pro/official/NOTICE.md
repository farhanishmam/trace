# Vendored from TIGER-AI-Lab/MMLU-Pro

Files copied byte for byte from https://github.com/TIGER-AI-Lab/MMLU-Pro at commit
`f418b116db00b065c2aea046518d8fcf74d39872` (2026-03-18), under the Apache License 2.0 (`LICENSE`).

| file | sha256 |
| --- | --- |
| `evaluate_from_local.py` | `d0a4ae51b80efa8f99713a21253ad5542913ef2ac9915e95d40c8a62174e9401` |
| `cot_prompt_lib/initial_prompt.txt` | `d7337806b4d84eef7a86bcb20bf3d64b65e72736c1603e16d08cadebac436f0a` |
| `evaluate_from_api.py` | `6e00c0d57ed307b9e707b712ea796fbd2286c6d21889cd63bb58090fa72c5a99` |

`../mmlu_pro_api_eval.py` (secondary, chat-format protocol) calls `single_request` from `evaluate_from_api.py` with its
API call replaced by a function that captures the prompt, so each chat prompt is the official one byte for byte.

`../mmlu_pro_eval.py` imports `preprocess`, `select_by_category`, `generate_cot_prompt`, `extract_answer` and
`save_res` from `evaluate_from_local.py` unchanged, and checks both hashes before it runs. Nothing here is modified.
