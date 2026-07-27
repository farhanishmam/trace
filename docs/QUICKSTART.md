# Quickstart

## Install

For a standard installation from a clone:

```bash
git clone https://github.com/maveryn/trace.git
cd trace
python -m pip install -e .
```

For development and reproducible generation on Python 3.10-3.12, create and
activate a virtual environment, then install the release constraints:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints/release.txt -e ".[test]"
```

Use the compatibility constraints for package and CLI development on Python
3.14:

```bash
python -m pip install -c constraints/compat-py314.txt -e ".[test]"
```

Install Parquet and Hugging Face export support with:

```bash
python -m pip install -e ".[export]"
```

## Inspect The Registry

```bash
trace-list --domain puzzles
```

## Build A Small Dataset

```bash
trace-generate --config examples/configs/minimal_build.yaml
```

The builder stages candidate instances, applies the configured acceptance and
validation contracts, then atomically finalizes the dataset directory. The
command reports the finalized dataset path as `dataset_root`.

## Export JSONL For RLVR

```bash
trace-export out/datasets/<dataset-id> \
  --output trace-train.jsonl \
  --format jsonl \
  --prompt-variant answer
```

Parquet export requires the optional `export` dependencies.

## Run Public Smoke Tests

```bash
pytest -q tests/test_public_smoke.py
```
