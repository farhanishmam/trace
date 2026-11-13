| run | model | condition | seed | n | official acc | det. acc | strict acc | format | tool use | calls/ep | tool err | turns | gen tok | median s | pending judge | endpoint errors |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| runs/final_eval/trace_val/qwen3.5-9b | qwen3.5-9b | no_tool | 42 | 2000 | 64.20 | 62.35 | 62.35 | 81.70 | 0.00 | 0.00 | 0.00 | 1.00 | 718 | 35.8 | 0 | 0 |
| runs/final_eval/trace_val/qwen3.5-9b | qwen3.5-9b | no_tool | 43 | 2000 | 65.50 | 63.60 | 63.55 | 83.50 | 0.00 | 0.00 | 0.00 | 1.00 | 695 | 36.4 | 0 | 0 |
| runs/final_eval/trace_val/qwen3.5-9b | qwen3.5-9b | no_tool | 44 | 2000 | 65.10 | 63.15 | 63.05 | 82.75 | 0.00 | 0.00 | 0.00 | 1.00 | 696 | 35.6 | 0 | 0 |

Official accuracy (%) by domain

| model | condition | seed | charts | games | geometry | graph | icons | illustrations | pages | physics | puzzles | symbolic | three_d |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3.5-9b | no_tool | 42 | 74.44 | 48.53 | 75.88 | 65.00 | 60.00 | 50.00 | 88.12 | 60.00 | 46.67 | 60.83 | 54.17 |
| qwen3.5-9b | no_tool | 43 | 77.78 | 50.29 | 75.29 | 66.67 | 59.00 | 38.33 | 87.50 | 69.00 | 49.17 | 60.00 | 65.00 |
| qwen3.5-9b | no_tool | 44 | 75.00 | 47.06 | 75.29 | 66.67 | 56.00 | 47.50 | 89.38 | 69.00 | 50.83 | 67.50 | 57.50 |
