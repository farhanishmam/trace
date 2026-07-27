from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCREENSHOT = (
    REPO_ROOT / "docs" / "assets" / "examples" / "trace-space-quickstart.png"
)
SPACE_URL = "https://huggingface.co/spaces/maveryn/trace"
COLAB_URL = (
    "https://colab.research.google.com/github/maveryn/trace/blob/main/"
    "examples/notebooks/trace_quickstart.ipynb"
)
PINNED_REVISION = "bb7fdd1fc8a0f8a2e3db7efe910a14e81d58feb7"


def test_landing_quickstarts_follow_the_domain_montage() -> None:
    pages = {
        "README.md": (
            "docs/assets/paper-domain-montage/trace-paper-domain-montage.png",
            "## How Trace Works",
        ),
        "docs/README.md": (
            "assets/paper-domain-montage/trace-paper-domain-montage.png",
            "## How Trace works",
        ),
    }

    for relative_path, (montage, architecture_heading) in pages.items():
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert text.index(montage) < text.index("## Try Trace")
        assert text.index("## Try Trace") < text.index(architecture_heading)


def test_live_demo_link_is_above_the_domain_montage() -> None:
    pages = {
        "README.md": (
            "docs/assets/paper-domain-montage/trace-paper-domain-montage.png"
        ),
        "docs/README.md": (
            "assets/paper-domain-montage/trace-paper-domain-montage.png"
        ),
    }

    for relative_path, montage in pages.items():
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert text.index(SPACE_URL) < text.index(montage)


def test_landing_quickstarts_link_all_runnable_surfaces() -> None:
    for relative_path in ("README.md", "docs/README.md"):
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert SPACE_URL in text
        assert COLAB_URL in text
        assert PINNED_REVISION in text
        assert "task_geometry__graph_paper__polygon_area_value" in text
        assert "seed=42" in text
        assert "trace-space-quickstart.png" in text
        assert "researchers and engineers" in text


def test_landing_quickstart_screenshot_is_a_png() -> None:
    assert SCREENSHOT.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
