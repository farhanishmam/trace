---
title: Trace Task Explorer
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.20.0
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Generate and inspect 1,000 grounded visual-reasoning tasks.
suggested_hardware: cpu-basic
models:
  - maveryn/trace-qwen2.5-vl-3b
  - maveryn/trace-qwen2.5-vl-7b
datasets:
  - maveryn/trace
tags:
  - visual-reasoning
  - synthetic-data
  - reinforcement-learning
  - rlvr
  - multimodal
---

# Trace Task Explorer

This Space generates any of Trace's 1,000 deterministic, grounded
visual-reasoning tasks. Select a domain, scene, task, and seed to inspect the
rendered image, answer-mode prompt, typed supervision, public annotation
overlay, exact reward contract, and sanitized public execution trace.

The Space is CPU-only and may need a short cold start. It accepts no uploads and
stores no generated examples. The package dependency is pinned to the exact
source revision shown in the **Reproduce** tab.

- [Source repository](https://github.com/maveryn/trace)
- [Documentation](https://maveryn.github.io/trace/)
- [Paper](https://arxiv.org/abs/2607.19790)
- [Dataset](https://huggingface.co/datasets/maveryn/trace)
- [Hugging Face collection](https://huggingface.co/collections/maveryn/trace-6a604291b4be4ed6399b9f24)
- [3B checkpoint](https://huggingface.co/maveryn/trace-qwen2.5-vl-3b)
- [7B checkpoint](https://huggingface.co/maveryn/trace-qwen2.5-vl-7b)

## Citation

```bibtex
@misc{alam2026trace,
  title         = {Trace: A Taxonomy-Guided Environment for Multidomain Visual Reasoning},
  author        = {Alam, Md Tanvirul},
  year          = {2026},
  eprint        = {2607.19790},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2607.19790}
}
```
