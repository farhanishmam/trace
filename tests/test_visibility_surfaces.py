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
TRANSFER_CLAIM = (
    "macro-average across 24 external benchmarks by **+3.51** and **+4.06 "
    "percentage points**"
)
BENCHMARK_EXPANDER = "View all 24 benchmark results"
TRACE_VALIDATION_SUMMARY = (
    "On 2,000 unseen TRACE instances, accuracy improves from 24.45 to 41.05 "
    "at 3B and from 34.25 to 51.55 at 7B."
)


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


def test_readme_leads_with_the_verified_transfer_result() -> None:
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    assert TRANSFER_CLAIM in normalized
    assert normalized.index(TRANSFER_CLAIM) < normalized.index(
        "docs/assets/paper-domain-montage/trace-paper-domain-montage.png"
    )


def test_landing_pages_collapse_the_full_benchmark_table() -> None:
    for relative_path in ("README.md", "docs/README.md"):
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert text.count(BENCHMARK_EXPANDER) == 1
        assert text.index(BENCHMARK_EXPANDER) < text.index(
            "<!-- trace-eval-v1-base-trace-table:start -->"
        )
        assert text.index("<!-- trace-eval-v1-base-trace-table:end -->") < text.index(
            "</details>",
            text.index("<!-- trace-eval-v1-base-trace-table:end -->"),
        )


def test_primary_resource_links_stay_focused() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    badge_block = readme.split('<p align="center">', 1)[1].split("</p>", 1)[0]
    assert badge_block.count("<a href=") == 4
    for label in ("Live Demo", "Paper", "Hugging Face Collection", "Dataset"):
        assert f'alt="{label}"' in badge_block
    for label in ("Project Page", "Reproduce", "3B Model", "7B Model"):
        assert f'alt="{label}"' not in badge_block

    docs = (REPO_ROOT / "docs/README.md").read_text(encoding="utf-8")
    hero_links = docs.split('<div class="trace-resource-links">', 1)[1].split(
        "</div>",
        1,
    )[0]
    assert hero_links.count('class="md-button') == 5
    for label in ("Live Demo", "Code", "Paper", "Collection", "Dataset"):
        assert f">{label}</a>" in hero_links
    assert ">Reproduce</a>" not in hero_links


def test_landing_pages_keep_external_transfer_as_the_single_results_section() -> None:
    pages = {
        "README.md": "## External Benchmark Transfer",
        "docs/README.md": "## External benchmark transfer",
    }
    for relative_path, transfer_heading in pages.items():
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        normalized = " ".join(text.split())
        assert "## TRACE Validation" not in text
        assert "## TRACE validation" not in text
        assert normalized.count(TRACE_VALIDATION_SUMMARY) == 1
        assert text.index(transfer_heading) < text.index(
            "<!-- trace-eval-v1-base-trace-table:start -->"
        )
        assert "Improvement: +16.60" not in text


def test_readme_keeps_installation_brief_and_links_the_full_guide() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split("## Installation", 1)[1].split("## Generate", 1)[0]
    assert section.count("```bash") == 1
    assert "python -m pip install -e ." in section
    assert "docs/QUICKSTART.md#install" in section
    for detail in (".venv", "constraints/release.txt", "compat-py314.txt", "[export]"):
        assert detail not in section

    guide = (REPO_ROOT / "docs/QUICKSTART.md").read_text(encoding="utf-8")
    for detail in (
        "constraints/release.txt",
        "constraints/compat-py314.txt",
        '".[export]"',
    ):
        assert detail in guide


def test_public_contribution_surfaces_are_linked() -> None:
    for relative_path in ("README.md", "CONTRIBUTING.md", "docs/DEVELOPMENT.md"):
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert "https://github.com/maveryn/trace/issues" in text
        assert "https://github.com/maveryn/trace/discussions" in text
