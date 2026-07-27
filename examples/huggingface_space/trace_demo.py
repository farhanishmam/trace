"""Framework-independent logic for the public Trace task explorer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from numbers import Integral, Real
from pathlib import Path
import secrets
from typing import Any

from PIL import Image

from trace_tasks import generate_task, list_task_ids
from trace_tasks.core.annotation_sanitization import (
    sanitize_trace_payload_for_public_annotation,
)
from trace_tasks.core.reward_contracts import resolve_reward_contract
from trace_tasks.core.source_layout_policy import parse_public_task_id
from trace_tasks.core.taxonomy import ACTIVE_DOMAINS

from overlay import render_annotation_overlay

REPOSITORY_URL = "https://github.com/maveryn/trace"
DOCUMENTATION_URL = "https://maveryn.github.io/trace/"
DATASET_URL = "https://huggingface.co/datasets/maveryn/trace"
SPACE_URL = "https://huggingface.co/spaces/maveryn/trace"
COLAB_URL = (
    "https://colab.research.google.com/github/maveryn/trace/blob/main/"
    "examples/notebooks/trace_quickstart.ipynb"
)
PINNED_REVISION = "bb7fdd1fc8a0f8a2e3db7efe910a14e81d58feb7"
DEFAULT_TASK_ID = "task_geometry__graph_paper__polygon_area_value"
DEFAULT_DOMAIN = "geometry"
DEFAULT_SCENE_ID = "graph_paper"
DEFAULT_SEED = 42
MAX_SEED = (1 << 53) - 1
MAX_ATTEMPTS = 100


@dataclass(frozen=True)
class Preset:
    """One deterministic curated example."""

    domain: str
    scene_id: str
    task_id: str
    seed: int

    @property
    def label(self) -> str:
        objective = parse_public_task_id(self.task_id).objective_contract
        return f"{self.domain} · {self.scene_id} · {objective} · seed {self.seed}"


@dataclass(frozen=True)
class TaskCatalog:
    """Cascading domain, scene, and task choices."""

    task_ids: tuple[str, ...]
    domains: tuple[str, ...]
    scenes_by_domain: dict[str, tuple[str, ...]]
    tasks_by_scene: dict[tuple[str, str], tuple[str, ...]]

    def scenes(self, domain: str) -> tuple[str, ...]:
        if domain not in self.scenes_by_domain:
            raise ValueError(f"unknown domain: {domain!r}")
        return self.scenes_by_domain[domain]

    def tasks(self, domain: str, scene_id: str) -> tuple[str, ...]:
        key = (domain, scene_id)
        if key not in self.tasks_by_scene:
            raise ValueError(f"unknown scene: {domain}/{scene_id}")
        return self.tasks_by_scene[key]


@dataclass(frozen=True)
class RandomSelection:
    """One uniformly sampled registered task and browser-safe seed."""

    domain: str
    scene_id: str
    task_id: str
    seed: int


@dataclass(frozen=True)
class DemoResult:
    """Serializable outputs shown by the Gradio wrapper."""

    original_image: Image.Image
    annotation_overlay: Image.Image
    prompt: str
    ground_truth: dict[str, Any]
    reward_contract: dict[str, Any]
    trace_summary: dict[str, Any]
    public_trace: dict[str, Any]
    reproduction: str
    links_markdown: str


def build_catalog(task_ids: Sequence[str] | None = None) -> TaskCatalog:
    """Build deterministic cascading choices from the installed registry."""

    resolved_task_ids = tuple(task_ids if task_ids is not None else list_task_ids())
    if not resolved_task_ids:
        raise ValueError("Trace registry is empty")
    if len(set(resolved_task_ids)) != len(resolved_task_ids):
        raise ValueError("Trace registry contains duplicate task ids")

    mutable_scenes: dict[str, set[str]] = {}
    mutable_tasks: dict[tuple[str, str], list[str]] = {}
    for task_id in sorted(resolved_task_ids):
        parts = parse_public_task_id(task_id)
        mutable_scenes.setdefault(parts.domain, set()).add(parts.scene_id)
        mutable_tasks.setdefault((parts.domain, parts.scene_id), []).append(task_id)

    active = [domain for domain in ACTIVE_DOMAINS if domain in mutable_scenes]
    extras = sorted(set(mutable_scenes).difference(active))
    domains = tuple([*active, *extras])
    scenes = {
        domain: tuple(sorted(mutable_scenes[domain]))
        for domain in domains
    }
    tasks = {
        key: tuple(sorted(values))
        for key, values in sorted(mutable_tasks.items())
    }
    return TaskCatalog(
        task_ids=tuple(sorted(resolved_task_ids)),
        domains=domains,
        scenes_by_domain=scenes,
        tasks_by_scene=tasks,
    )


def sample_random_selection(
    catalog: TaskCatalog | None = None,
) -> RandomSelection:
    """Sample uniformly from every registered task and choose a fresh seed."""

    resolved_catalog = catalog or build_catalog()
    task_id = secrets.choice(resolved_catalog.task_ids)
    parts = parse_public_task_id(task_id)
    return RandomSelection(
        domain=parts.domain,
        scene_id=parts.scene_id,
        task_id=task_id,
        seed=secrets.randbelow(MAX_SEED + 1),
    )


def load_presets(path: Path | None = None) -> tuple[Preset, ...]:
    """Load curated deterministic examples bundled with the Space."""

    preset_path = path or Path(__file__).with_name("presets.json")
    payload = json.loads(preset_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "trace_space_presets_v1":
        raise ValueError("unsupported Trace Space preset schema")

    presets: list[Preset] = []
    for raw in payload.get("presets", []):
        preset = Preset(
            domain=str(raw["domain"]),
            scene_id=str(raw["scene_id"]),
            task_id=str(raw["task_id"]),
            seed=validate_seed(raw["seed"]),
        )
        parts = parse_public_task_id(preset.task_id)
        if (parts.domain, parts.scene_id) != (preset.domain, preset.scene_id):
            raise ValueError(f"preset taxonomy mismatch: {preset.task_id}")
        presets.append(preset)
    if len(presets) != 22:
        raise ValueError(f"expected 22 curated presets, found {len(presets)}")
    return tuple(presets)


def validate_seed(value: Any) -> int:
    """Return a browser-safe integer seed."""

    if isinstance(value, bool) or value is None:
        raise ValueError("seed must be an integer")
    if isinstance(value, Integral):
        seed = int(value)
    elif isinstance(value, Real) and math.isfinite(float(value)):
        if not float(value).is_integer():
            raise ValueError("seed must be an integer")
        seed = int(value)
    elif isinstance(value, str):
        normalized = value.strip()
        if not normalized or not normalized.isdecimal():
            raise ValueError("seed must be an integer")
        seed = int(normalized)
    else:
        raise ValueError("seed must be an integer")
    if seed < 0 or seed > MAX_SEED:
        raise ValueError(f"seed must be between 0 and {MAX_SEED}")
    return seed


def generate_demo(
    task_id: str,
    seed: Any,
    *,
    catalog: TaskCatalog | None = None,
) -> DemoResult:
    """Generate one deterministic task and its public inspection payloads."""

    resolved_catalog = catalog or build_catalog()
    normalized_task_id = str(task_id).strip()
    if normalized_task_id not in set(resolved_catalog.task_ids):
        raise ValueError("choose a registered Trace task")
    normalized_seed = validate_seed(seed)

    output = generate_task(
        normalized_task_id,
        seed=normalized_seed,
        params={},
        max_attempts=MAX_ATTEMPTS,
    )
    answer_gt = json_safe(output.answer_gt.to_dict())
    annotation_gt = json_safe(output.annotation_gt.to_dict())
    reward_contract = resolve_reward_contract(
        answer_type=output.answer_gt.type,
        annotation_type=output.annotation_gt.type,
    ).to_dict()
    public_trace = sanitize_trace_payload_for_public_annotation(
        output.trace_payload,
        annotation_gt=output.annotation_gt,
    )
    public_trace = json_safe(public_trace)
    overlay = render_annotation_overlay(output.image, annotation_gt)
    parts = parse_public_task_id(normalized_task_id)

    query_spec = public_trace.get("query_spec", {})
    prompt_trace = query_spec if isinstance(query_spec, Mapping) else {}
    trace_summary = {
        "task_id": normalized_task_id,
        "taxonomy": {
            "domain": parts.domain,
            "scene_id": parts.scene_id,
            "objective_contract": parts.objective_contract,
        },
        "instance_seed": normalized_seed,
        "resolved_scene_id": output.scene_id,
        "query_id": output.query_id,
        "image": {
            "image_id": output.image_id,
            "width": output.image.width,
            "height": output.image.height,
        },
        "answer_type": output.answer_gt.type,
        "annotation_type": output.annotation_gt.type,
        "prompt_selection": {
            key: json_safe(prompt_trace[key])
            for key in (
                "template_id",
                "prompt_variant",
                "prompt_variant_active_key",
            )
            if key in prompt_trace
        },
        "task_versions": json_safe(output.task_versions),
        "trace_sections": sorted(public_trace),
    }

    source_path = (
        f"src/trace_tasks/tasks/{parts.domain}/{parts.scene_id}/"
        f"{parts.objective_contract}.py"
    )
    doc_path = f"docs/tasks/{parts.domain}/{parts.scene_id}/{normalized_task_id}.md"
    source_url = f"{REPOSITORY_URL}/blob/{PINNED_REVISION}/{source_path}"
    task_doc_url = f"{REPOSITORY_URL}/blob/{PINNED_REVISION}/{doc_path}"
    answer_preview = json.dumps(
        answer_gt["value"],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).replace("`", "'")
    if len(answer_preview) > 120:
        answer_preview = f"{answer_preview[:117]}..."
    links = (
        f"**Typed result** · answer `{answer_gt['type']}` = `{answer_preview}` · "
        f"annotation `{annotation_gt['type']}`\n\n"
        f"**Verifier** · `{reward_contract['answer']['id']}` + "
        f"`{reward_contract['annotation']['id']}`\n\n"
        f"Generated from [`{normalized_task_id}`]({task_doc_url}) at "
        f"[revision `{PINNED_REVISION[:7]}`]({source_url}). "
        f"[Documentation]({DOCUMENTATION_URL}) · "
        f"[Dataset]({DATASET_URL}) · [Colab]({COLAB_URL})"
    )

    reproduction = "\n".join(
        [
            "python -m pip install \\",
            '  "trace-tasks @ git+https://github.com/maveryn/trace.git'
            f'@{PINNED_REVISION}"',
            "",
            "python - <<'PY'",
            "from trace_tasks import generate_task",
            "",
            f'task_id = "{normalized_task_id}"',
            f"sample = generate_task(task_id, seed={normalized_seed}, max_attempts=100)",
            "sample.image.save('trace-example.png')",
            "print(sample.prompt)",
            "print(sample.answer_gt.to_dict())",
            "print(sample.annotation_gt.to_dict())",
            "PY",
        ]
    )

    return DemoResult(
        original_image=output.image.convert("RGB"),
        annotation_overlay=overlay,
        prompt=output.prompt,
        ground_truth={
            "answer_gt": answer_gt,
            "annotation_gt": annotation_gt,
        },
        reward_contract=json_safe(reward_contract),
        trace_summary=json_safe(trace_summary),
        public_trace=public_trace,
        reproduction=reproduction,
        links_markdown=links,
    )


def json_safe(value: Any) -> Any:
    """Convert Trace payload values to strict JSON-compatible objects."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if math.isfinite(number):
            return number
        return str(number)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [json_safe(item) for item in sorted(value, key=str)]
    if hasattr(value, "to_dict"):
        return json_safe(value.to_dict())
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        return json_safe(value.item())
    return str(value)


__all__ = [
    "COLAB_URL",
    "DEFAULT_DOMAIN",
    "DEFAULT_SCENE_ID",
    "DEFAULT_SEED",
    "DEFAULT_TASK_ID",
    "DemoResult",
    "MAX_ATTEMPTS",
    "MAX_SEED",
    "PINNED_REVISION",
    "Preset",
    "SPACE_URL",
    "TaskCatalog",
    "build_catalog",
    "generate_demo",
    "json_safe",
    "load_presets",
    "validate_seed",
]
