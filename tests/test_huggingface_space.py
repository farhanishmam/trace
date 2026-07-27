from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import sys

from PIL import Image
import pytest
import yaml

from trace_tasks.core.annotation_sanitization import PUBLIC_IMAGE_ANNOTATION_TYPES

REPO_ROOT = Path(__file__).resolve().parents[1]
SPACE_ROOT = REPO_ROOT / "examples" / "huggingface_space"
if str(SPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(SPACE_ROOT))

from overlay import PUBLIC_ANNOTATION_TYPES, render_annotation_overlay  # noqa: E402
from trace_demo import (  # noqa: E402
    DEFAULT_TASK_ID,
    MAX_SEED,
    PINNED_REVISION,
    build_catalog,
    generate_demo,
    load_presets,
    sample_random_selection,
    validate_seed,
)


def test_space_catalog_exposes_the_complete_public_taxonomy() -> None:
    catalog = build_catalog()

    assert len(catalog.task_ids) == 1_000
    assert len(catalog.domains) == 11
    assert sum(len(scenes) for scenes in catalog.scenes_by_domain.values()) == 277
    assert DEFAULT_TASK_ID in catalog.tasks("geometry", "graph_paper")


def test_space_random_selection_samples_the_complete_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = build_catalog()
    chosen_task_id = catalog.task_ids[-1]
    calls: dict[str, object] = {}

    def choose(options: tuple[str, ...]) -> str:
        calls["options"] = options
        return chosen_task_id

    def randbelow(upper_bound: int) -> int:
        calls["upper_bound"] = upper_bound
        return upper_bound - 1

    monkeypatch.setattr("trace_demo.secrets.choice", choose)
    monkeypatch.setattr("trace_demo.secrets.randbelow", randbelow)

    selection = sample_random_selection(catalog)

    assert calls["options"] == catalog.task_ids
    assert calls["upper_bound"] == MAX_SEED + 1
    assert selection.task_id == chosen_task_id
    assert selection.task_id in catalog.tasks(
        selection.domain,
        selection.scene_id,
    )
    assert selection.seed == MAX_SEED


def test_space_presets_match_the_release_gallery() -> None:
    presets = load_presets()
    gallery = json.loads(
        (REPO_ROOT / "docs" / "gallery" / "manifest.v1.json").read_text(
            encoding="utf-8"
        )
    )
    expected = [
        {
            "domain": item["domain"],
            "scene_id": item["scene_id"],
            "task_id": item["task_id"],
            "seed": item["seed"],
        }
        for item in gallery["examples"]
    ]
    actual = [
        {
            "domain": item.domain,
            "scene_id": item.scene_id,
            "task_id": item.task_id,
            "seed": item.seed,
        }
        for item in presets
    ]

    assert actual == expected
    assert Counter(item.domain for item in presets) == {
        domain: 2 for domain in build_catalog().domains
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, 0), (42, 42), (42.0, 42), ("42", 42), (MAX_SEED, MAX_SEED)],
)
def test_space_accepts_browser_safe_integer_seeds(value: object, expected: int) -> None:
    assert validate_seed(value) == expected


@pytest.mark.parametrize(
    "value",
    [None, True, -1, MAX_SEED + 1, 1.5, "1.5", "", object()],
)
def test_space_rejects_invalid_seeds(value: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        validate_seed(value)


_ANNOTATION_VALUES = {
    "bbox": [10, 10, 50, 50],
    "bbox_sequence": [[10, 10, 50, 50], [60, 60, 90, 90]],
    "bbox_set": [[10, 10, 50, 50], [60, 60, 90, 90]],
    "bbox_map": {"source": [10, 10, 50, 50]},
    "bbox_set_map": {"targets": [[10, 10, 50, 50], [60, 60, 90, 90]]},
    "point": [25, 25],
    "point_sequence": [[25, 25], [75, 75]],
    "point_set": [[25, 25], [75, 75]],
    "point_map": {"source": [25, 25]},
    "point_set_map": {"targets": [[25, 25], [75, 75]]},
    "segment": [[10, 10], [90, 90]],
    "segment_set": [[[10, 10], [90, 90]], [[90, 10], [10, 90]]],
}


@pytest.mark.parametrize("annotation_type", sorted(_ANNOTATION_VALUES))
def test_space_overlay_renders_every_public_annotation_type(
    annotation_type: str,
) -> None:
    source = Image.new("RGB", (100, 100), "white")
    overlay = render_annotation_overlay(
        source,
        {"type": annotation_type, "value": _ANNOTATION_VALUES[annotation_type]},
    )

    assert overlay.mode == "RGB"
    assert overlay.size == source.size
    assert overlay.tobytes() != source.tobytes()


def test_space_overlay_type_set_stays_synced_with_public_abi() -> None:
    assert PUBLIC_ANNOTATION_TYPES == PUBLIC_IMAGE_ANNOTATION_TYPES


def test_space_default_generation_exposes_public_contracts() -> None:
    result = generate_demo(DEFAULT_TASK_ID, 42, catalog=build_catalog())

    assert result.prompt
    assert result.original_image.size == result.annotation_overlay.size
    assert result.ground_truth["answer_gt"]["type"]
    assert result.ground_truth["annotation_gt"]["type"]
    assert result.reward_contract["reward_contract_version"] == "v0"
    assert result.trace_summary["instance_seed"] == 42
    assert result.public_trace["witness_symbolic"] == {
        "type": result.ground_truth["annotation_gt"]["type"],
        "count": len(result.ground_truth["annotation_gt"]["value"]),
    }
    assert "**Typed result**" in result.links_markdown
    assert result.reward_contract["answer"]["id"] in result.links_markdown
    assert result.reward_contract["annotation"]["id"] in result.links_markdown
    assert PINNED_REVISION in result.reproduction
    assert "hf_" not in json.dumps(result.public_trace)


def test_space_card_and_runtime_are_pinned_for_public_cpu_deployment() -> None:
    readme = (SPACE_ROOT / "README.md").read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n", readme, flags=re.DOTALL)
    assert match is not None
    metadata = yaml.safe_load(match.group(1))

    assert metadata["sdk"] == "gradio"
    assert metadata["sdk_version"] == "6.20.0"
    assert metadata["python_version"] == "3.12"
    assert metadata["app_file"] == "app.py"
    assert metadata["suggested_hardware"] == "cpu-basic"
    assert metadata["colorFrom"] in {
        "red",
        "yellow",
        "green",
        "blue",
        "indigo",
        "purple",
        "pink",
        "gray",
    }
    assert metadata["colorTo"] in {
        "red",
        "yellow",
        "green",
        "blue",
        "indigo",
        "purple",
        "pink",
        "gray",
    }
    assert len(metadata["short_description"]) <= 60
    assert metadata["datasets"] == ["maveryn/trace"]
    assert metadata["models"] == [
        "maveryn/trace-qwen2.5-vl-3b",
        "maveryn/trace-qwen2.5-vl-7b",
    ]

    requirements = (SPACE_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert requirements.strip().endswith(f"@{PINNED_REVISION}")
    release_constraints = {
        line.split("==", 1)[0].lower(): line
        for line in (
            REPO_ROOT / "constraints" / "release.txt"
        ).read_text(encoding="utf-8").splitlines()
        if "==" in line
    }
    space_pins = [
        line
        for line in requirements.splitlines()
        if "==" in line
    ]
    assert len(space_pins) == 17
    for requirement in space_pins:
        package = requirement.split("==", 1)[0].lower()
        assert release_constraints[package] == requirement
    assert (SPACE_ROOT / "packages.txt").read_text(encoding="utf-8").strip() == (
        "libcairo2"
    )


def test_space_has_no_public_prediction_api_or_upload_input() -> None:
    source = (SPACE_ROOT / "app.py").read_text(encoding="utf-8")

    assert "max_size=32" in source
    assert "default_concurrency_limit=1" in source
    assert "footer_links=[]" in source
    assert source.count("api_name=False") == 5
    assert "gr.File(" not in source
    assert "sources=" not in source
    assert 'gr.Button("Random question"' in source
    assert "_random_question" in source
    assert "Random seed" not in source
    assert "domain.input(" in source
    assert "scene_id.input(" in source
