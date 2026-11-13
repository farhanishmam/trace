| run | model | condition | seed | n | official acc | det. acc | strict acc | format | tool use | calls/ep | tool err | turns | gen tok | median s | pending judge | endpoint errors |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| runs/final_eval/trace_val/qwen3.8-27b | qwen3.8-27b | no_tool | 42 | 2000 | 72.40 | 70.65 | 70.55 | 84.10 | 0.00 | 0.00 | 0.00 | 1.00 | 764 | 76.1 | 0 | 0 |
| runs/final_eval/trace_val/qwen3.8-27b | qwen3.8-27b | no_tool | 43 | 2000 | 71.70 | 69.45 | 69.50 | 83.85 | 0.00 | 0.00 | 0.00 | 1.00 | 769 | 74.1 | 0 | 0 |
| runs/final_eval/trace_val/qwen3.8-27b | qwen3.8-27b | no_tool | 44 | 2000 | 73.55 | 70.80 | 70.85 | 83.05 | 0.00 | 0.00 | 0.00 | 1.00 | 782 | 38.0 | 0 | 0 |

Official accuracy (%) by domain

| model | condition | seed | charts | games | geometry | graph | icons | illustrations | pages | physics | puzzles | symbolic | three_d |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3.8-27b | no_tool | 42 | 83.06 | 62.06 | 80.29 | 76.67 | 77.00 | 48.33 | 86.25 | 72.00 | 50.83 | 65.83 | 73.33 |
| qwen3.8-27b | no_tool | 43 | 81.94 | 60.88 | 80.00 | 74.17 | 74.00 | 46.67 | 86.25 | 76.00 | 56.67 | 70.00 | 62.50 |
| qwen3.8-27b | no_tool | 44 | 79.72 | 61.47 | 84.41 | 79.17 | 77.00 | 55.83 | 88.75 | 74.00 | 57.50 | 65.83 | 70.83 |
