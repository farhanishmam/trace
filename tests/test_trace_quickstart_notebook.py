from __future__ import annotations

import json
from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    REPO_ROOT / "examples" / "notebooks" / "trace_quickstart.ipynb"
)
PINNED_REVISION = "bb7fdd1fc8a0f8a2e3db7efe910a14e81d58feb7"
SECRET_RE = re.compile(
    r"(?:hf_[A-Za-z0-9]{20,}|"
    r"gh[pousr]_[A-Za-z0-9]{30,}|"
    r"sk-[A-Za-z0-9_-]{20,})"
)


def _notebook() -> dict[str, object]:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_quickstart_notebook_is_valid_and_output_free() -> None:
    notebook = _notebook()

    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] >= 5
    cells = notebook["cells"]
    assert isinstance(cells, list)
    assert cells
    for cell in cells:
        assert isinstance(cell, dict)
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        source = "".join(cell["source"])
        if source.startswith("%"):
            continue
        compile(source, str(NOTEBOOK_PATH), "exec")


def test_quickstart_notebook_pins_code_and_covers_the_public_workflow() -> None:
    text = NOTEBOOK_PATH.read_text(encoding="utf-8")

    assert PINNED_REVISION in text
    assert "task_geometry__graph_paper__polygon_area_value" in text
    assert "generate_task" in text
    assert "sanitize_trace_payload_for_public_annotation" in text
    assert "resolve_reward_contract" in text
    assert "score_trace_response" in text
    assert 'scores[\\\"overall\\\"] == 1.0' in text
    assert "list_task_ids" in text


def test_quickstart_notebook_contains_no_credentials() -> None:
    text = NOTEBOOK_PATH.read_text(encoding="utf-8")

    assert SECRET_RE.search(text) is None
    assert "hf-token" not in text.lower()
