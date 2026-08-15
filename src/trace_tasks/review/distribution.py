"""Answer-distribution checks over materialized contributor-review artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from trace_tasks.core.json_io import write_json_file
from trace_tasks.core.source_layout_policy import parse_public_task_id

from .materialize import iter_materialized_records
from .models import MATERIALIZATION_SCHEMA_VERSION, RECIPE_ID


ANSWER_DISTRIBUTION_SCHEMA_VERSION = "trace-answer-distribution-v1"


def canonicalize_answer_value(*, answer_type: str, answer_value: Any) -> str:
    """Return one deterministic answer label for frequency counting."""

    kind = str(answer_type)
    if kind == "integer":
        return str(int(answer_value))
    if kind == "number":
        numeric = float(answer_value)
        if not math.isfinite(numeric):
            raise ValueError("number answers must be finite")
        return format(numeric, ".12g")
    return str(answer_value)


def parse_numeric_answer_value(
    *, answer_type: str, answer_value: Any
) -> float | None:
    """Return the answer scalar when the public answer type is numeric."""

    kind = str(answer_type)
    if kind not in {"integer", "number"}:
        return None
    numeric = float(answer_value)
    if not math.isfinite(numeric):
        raise ValueError("number answers must be finite")
    return numeric


def _format_bound(value: float) -> str:
    text = format(float(value), ".6f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text if text else "0"


def _five_bin_report(values: Sequence[float]) -> dict[str, Any]:
    """Build a fixed five-bin equal-width summary for numeric answers."""

    if not values:
        return {
            "bin_count": 5,
            "range_min": None,
            "range_max": None,
            "max_bin_count": 0,
            "max_bin_frequency": 0.0,
            "bins": [],
        }
    v_min = float(min(values))
    v_max = float(max(values))
    total = len(values)
    if abs(v_max - v_min) <= 1e-12:
        return {
            "bin_count": 5,
            "range_min": v_min,
            "range_max": v_max,
            "max_bin_count": total,
            "max_bin_frequency": 1.0,
            "bins": [
                {
                    "bin_id": 0,
                    "label": f"[{_format_bound(v_min)}, {_format_bound(v_max)}]",
                    "count": total,
                }
            ],
        }

    bin_count = 5
    width = (v_max - v_min) / bin_count
    counts = [0 for _ in range(bin_count)]
    for raw in values:
        value = float(raw)
        if abs(value - v_max) <= 1e-12:
            index = bin_count - 1
        else:
            index = int((value - v_min) / width)
            index = max(0, min(bin_count - 1, index))
        counts[index] += 1

    bins: list[dict[str, Any]] = []
    for index, count in enumerate(counts):
        start = v_min + (index * width)
        end = v_min + ((index + 1) * width)
        label = f"[{_format_bound(start)}, {_format_bound(end)})"
        if index == bin_count - 1:
            label = f"[{_format_bound(start)}, {_format_bound(end)}]"
        bins.append({"bin_id": index, "label": label, "count": count})
    max_count = max(counts)
    return {
        "bin_count": bin_count,
        "range_min": v_min,
        "range_max": v_max,
        "max_bin_count": max_count,
        "max_bin_frequency": max_count / total,
        "bins": bins,
    }


def evaluate_answer_distribution(
    answers: Sequence[Mapping[str, Any]],
    *,
    min_unique_answers: int = 4,
    max_answer_frequency: float = 1.0 / 3.0,
) -> dict[str, Any]:
    """Evaluate lightweight anti-degeneracy checks over one answer sample."""

    if not answers:
        raise ValueError("answers must be non-empty")
    if int(min_unique_answers) <= 0:
        raise ValueError("min_unique_answers must be greater than zero")
    if not 0.0 < float(max_answer_frequency) <= 1.0:
        raise ValueError(
            "max_answer_frequency must be greater than zero and at most one"
        )

    labels: Counter[str] = Counter()
    answer_type_counts: Counter[str] = Counter()
    numeric_values: list[float] = []
    numeric_only = True
    for item in answers:
        answer_type = str(item.get("answer_type", "")).strip()
        answer_value = item.get("answer_value")
        label = canonicalize_answer_value(
            answer_type=answer_type, answer_value=answer_value
        )
        labels[label] += 1
        answer_type_counts[answer_type] += 1
        numeric = parse_numeric_answer_value(
            answer_type=answer_type, answer_value=answer_value
        )
        if numeric is None:
            numeric_only = False
        else:
            numeric_values.append(numeric)

    sample_count = sum(labels.values())
    unique_answers = len(labels)
    max_answer_count = max(labels.values())
    max_answer_frequency_observed = max_answer_count / sample_count
    min_unique_target = min(int(min_unique_answers), sample_count)
    check_unique = unique_answers >= min_unique_target
    check_max_frequency = max_answer_frequency_observed < float(
        max_answer_frequency
    )

    numeric_bins = _five_bin_report(numeric_values) if numeric_only else None
    max_bin_frequency_observed = (
        None
        if numeric_bins is None
        else float(numeric_bins["max_bin_frequency"])
    )
    top_answers = [
        {"answer_label": label, "count": count}
        for label, count in sorted(
            labels.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    return {
        "sample_count": sample_count,
        "answer_type_counts": dict(sorted(answer_type_counts.items())),
        "unique_answers": unique_answers,
        "max_answer_count": max_answer_count,
        "max_answer_frequency": max_answer_frequency_observed,
        "top_answers": top_answers,
        "five_bin_numeric": numeric_bins,
        "checks": {
            "min_unique_answers": {
                "threshold": min_unique_target,
                "observed": unique_answers,
                "pass": check_unique,
            },
            "max_answer_frequency": {
                "threshold": float(max_answer_frequency),
                "observed": max_answer_frequency_observed,
                "pass": check_max_frequency,
            },
            "max_five_bin_frequency": (
                {
                    "observed": max_bin_frequency_observed,
                    "pass": None,
                    "status": "reported_not_gated_numeric_answers",
                }
                if numeric_bins is not None
                else {
                    "observed": None,
                    "pass": None,
                    "status": "not_applicable_non_numeric_answers",
                }
            ),
        },
        "pass": bool(check_unique and check_max_frequency),
    }


def _scene_filters(
    values: Sequence[str] | None,
) -> tuple[set[str], set[tuple[str, str]]]:
    bare: set[str] = set()
    qualified: set[tuple[str, str]] = set()
    for value in values or ():
        text = str(value)
        if "/" not in text:
            bare.add(text)
            continue
        domain, separator, scene_id = text.partition("/")
        if not separator or not domain or not scene_id or "/" in scene_id:
            raise ValueError("scene filters must be SCENE_ID or DOMAIN/SCENE_ID")
        qualified.add((domain, scene_id))
    return bare, qualified


def _required_text(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"materialized record {field} must be a non-empty string")
    return value


def _load_task_manifest(
    root: Path, *, domain: str, scene_id: str, task_id: str
) -> tuple[Path, dict[str, Any]]:
    """Load one contained materialization manifest for a public task."""

    try:
        resolved_root = root.resolve(strict=True)
        task_dir = (root / domain / scene_id / task_id).resolve(strict=True)
        task_dir.relative_to(resolved_root)
        manifest_path = (task_dir / "manifest.json").resolve(strict=True)
        manifest_path.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid materialized task path: {task_id}") from exc
    if not manifest_path.is_file():
        raise ValueError(f"materialized manifest is not a regular file: {task_id}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot load materialized manifest: {manifest_path}"
        ) from exc
    if not isinstance(manifest, dict):
        raise ValueError(
            f"materialized manifest must be an object: {manifest_path}"
        )
    if manifest.get("schema_version") != MATERIALIZATION_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported materialized manifest schema_version: {task_id}"
        )
    if manifest.get("recipe_id") != RECIPE_ID:
        raise ValueError(f"unsupported materialized manifest recipe_id: {task_id}")
    if (
        manifest.get("task_id") != task_id
        or manifest.get("domain") != domain
        or manifest.get("scene_id") != scene_id
    ):
        raise ValueError(f"materialized manifest taxonomy mismatch: {task_id}")
    query_ids = manifest.get("query_ids")
    if (
        not isinstance(query_ids, list)
        or any(not isinstance(value, str) or not value for value in query_ids)
        or query_ids != sorted(set(query_ids))
    ):
        raise ValueError(
            f"materialized manifest query_ids must be sorted unique strings: {task_id}"
        )
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"materialized manifest entries must be a list: {task_id}")
    entry_query_ids: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"materialized manifest entries must be objects: {task_id}"
            )
        query_id = entry.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            raise ValueError(
                "materialized manifest entry query_id must be a non-empty "
                f"string: {task_id}"
            )
        entry_query_ids.append(query_id)
    if sorted(set(entry_query_ids)) != query_ids:
        raise ValueError(
            f"materialized manifest entries do not cover query_ids: {task_id}"
        )
    _required_text(manifest, "recipe_digest")
    return task_dir, manifest


def audit_materialized_answer_distributions(
    output_root: Path | str,
    *,
    task_ids: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
    scene_ids: Sequence[str] | None = None,
    query_ids: Sequence[str] | None = None,
    min_unique_answers: int = 4,
    max_answer_frequency: float = 1.0 / 3.0,
) -> dict[str, Any]:
    """Audit answer diversity per task/query in a materialized review slice."""

    if int(min_unique_answers) <= 0:
        raise ValueError("min_unique_answers must be greater than zero")
    if not 0.0 < float(max_answer_frequency) <= 1.0:
        raise ValueError(
            "max_answer_frequency must be greater than zero and at most one"
        )

    selected_tasks = {str(value) for value in task_ids or ()}
    selected_domains = {str(value) for value in domains or ()}
    selected_queries = {str(value) for value in query_ids or ()}
    bare_scenes, qualified_scenes = _scene_filters(scene_ids)

    answers_by_task_query: dict[
        tuple[str, str, str], dict[str, list[dict[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
    identity_by_task: dict[tuple[str, str, str], str] = {}
    checked_records = 0
    for record in iter_materialized_records(output_root):
        task_id = str(record.get("task_id", ""))
        domain = str(record.get("domain", ""))
        scene_id = str(record.get("scene_id", ""))
        query_id = str(record.get("query_id", ""))
        if selected_tasks and task_id not in selected_tasks:
            continue
        if selected_domains and domain not in selected_domains:
            continue
        if (
            (bare_scenes or qualified_scenes)
            and scene_id not in bare_scenes
            and (domain, scene_id) not in qualified_scenes
        ):
            continue
        if selected_queries and query_id not in selected_queries:
            continue

        task_id = _required_text(record, "task_id")
        domain = _required_text(record, "domain")
        scene_id = _required_text(record, "scene_id")
        query_id = _required_text(record, "query_id")
        recipe_digest = _required_text(record, "recipe_digest")
        task_parts = parse_public_task_id(task_id)
        if (domain, scene_id) != (task_parts.domain, task_parts.scene_id):
            raise ValueError(
                f"materialized record taxonomy does not match task_id {task_id}"
            )
        answer = record.get("answer_gt")
        if not isinstance(answer, Mapping) or set(answer) != {"type", "value"}:
            raise ValueError(
                "materialized record answer_gt must contain exactly type and "
                f"value: {task_id}/{query_id}"
            )
        answer_type = answer.get("type")
        if not isinstance(answer_type, str) or not answer_type:
            raise ValueError(
                "materialized record answer_gt.type must be a non-empty string: "
                f"{task_id}/{query_id}"
            )

        task_key = (domain, scene_id, task_id)
        previous_digest = identity_by_task.setdefault(task_key, recipe_digest)
        if previous_digest != recipe_digest:
            raise ValueError(
                f"materialized task mixes recipe digests: {task_id}"
            )
        answers_by_task_query[task_key][query_id].append(
            {"answer_type": answer_type, "answer_value": answer.get("value")}
        )
        checked_records += 1

    if checked_records == 0:
        raise ValueError("answer-distribution filters selected no materialized records")

    task_reports: list[dict[str, Any]] = []
    failed_task_ids: list[str] = []
    for task_key in sorted(answers_by_task_query):
        domain, scene_id, task_id = task_key
        _, manifest = _load_task_manifest(
            Path(output_root),
            domain=domain,
            scene_id=scene_id,
            task_id=task_id,
        )
        if manifest["recipe_digest"] != identity_by_task[task_key]:
            raise ValueError(
                f"materialized records do not match manifest recipe_digest: {task_id}"
            )
        observed_counts = {
            query_id: len(answers)
            for query_id, answers in answers_by_task_query[task_key].items()
        }
        expected_counts = Counter(
            str(entry["query_id"])
            for entry in manifest["entries"]
            if not selected_queries or str(entry["query_id"]) in selected_queries
        )
        if observed_counts != dict(expected_counts):
            raise ValueError(
                f"materialized record counts do not match manifest entries: {task_id}"
            )
        observed_query_ids = sorted(observed_counts)
        per_query = {
            query_id: evaluate_answer_distribution(
                answers,
                min_unique_answers=min_unique_answers,
                max_answer_frequency=max_answer_frequency,
            )
            for query_id, answers in sorted(answers_by_task_query[task_key].items())
        }
        combined_answers = [
            answer
            for query_id in sorted(answers_by_task_query[task_key])
            for answer in answers_by_task_query[task_key][query_id]
        ]
        overall = evaluate_answer_distribution(
            combined_answers,
            min_unique_answers=min_unique_answers,
            max_answer_frequency=max_answer_frequency,
        )
        failed_query_ids = [
            query_id
            for query_id, query_report in per_query.items()
            if not bool(query_report["pass"])
        ]
        task_pass = not failed_query_ids
        if not task_pass:
            failed_task_ids.append(task_id)
        task_reports.append(
            {
                "schema_version": ANSWER_DISTRIBUTION_SCHEMA_VERSION,
                "recipe_digest": identity_by_task[task_key],
                "task_id": task_id,
                "domain": domain,
                "scene_id": scene_id,
                "sample_count": len(combined_answers),
                "query_ids": observed_query_ids,
                "failed_query_ids": failed_query_ids,
                "overall": overall,
                "per_query_id": per_query,
                "pass": task_pass,
            }
        )

    task_count = len(task_reports)
    return {
        "schema_version": ANSWER_DISTRIBUTION_SCHEMA_VERSION,
        "config": {
            "gating_scope": "per_query_id",
            "min_unique_answers": int(min_unique_answers),
            "max_answer_frequency": float(max_answer_frequency),
            "numeric_bin_summary": "reported_only_five_equal_width_bins",
        },
        "checked_records": checked_records,
        "task_count": task_count,
        "passed": not failed_task_ids,
        "summary": {
            "total_tasks": task_count,
            "passed_tasks": task_count - len(failed_task_ids),
            "failed_tasks": len(failed_task_ids),
            "failed_task_ids": failed_task_ids,
        },
        "tasks": task_reports,
    }


def write_task_distribution_reports(
    output_root: Path | str, report: Mapping[str, Any]
) -> list[str]:
    """Write recipe-bound task diagnostics beside materialized artifacts."""

    root = Path(output_root)
    if report.get("schema_version") != ANSWER_DISTRIBUTION_SCHEMA_VERSION:
        raise ValueError("unsupported answer-distribution report schema_version")
    tasks = report.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("answer-distribution report tasks must be a list")
    pending: list[tuple[Path, dict[str, Any]]] = []
    for task_report in tasks:
        if not isinstance(task_report, Mapping):
            raise ValueError("answer-distribution task report must be an object")
        if (
            task_report.get("schema_version")
            != ANSWER_DISTRIBUTION_SCHEMA_VERSION
        ):
            raise ValueError(
                "unsupported answer-distribution task report schema_version"
            )
        task_id = _required_text(task_report, "task_id")
        domain = _required_text(task_report, "domain")
        scene_id = _required_text(task_report, "scene_id")
        recipe_digest = _required_text(task_report, "recipe_digest")
        parts = parse_public_task_id(task_id)
        if (domain, scene_id) != (parts.domain, parts.scene_id):
            raise ValueError(f"answer-distribution taxonomy mismatch: {task_id}")
        task_dir, manifest = _load_task_manifest(
            root, domain=domain, scene_id=scene_id, task_id=task_id
        )
        if (
            manifest.get("recipe_digest") != recipe_digest
        ):
            raise ValueError(
                "answer-distribution report does not match materialized "
                f"manifest: {task_id}"
            )
        output_path = task_dir / "answer_distribution.json"
        if output_path.exists() and not output_path.is_file():
            raise ValueError(
                f"answer-distribution output is not a regular file: {task_id}"
            )
        pending.append((output_path, dict(task_report)))

    resolved_root = root.resolve(strict=True)
    for output_path, payload in pending:
        write_json_file(output_path, payload)
    return [
        output_path.relative_to(resolved_root).as_posix()
        for output_path, _ in pending
    ]


__all__ = [
    "ANSWER_DISTRIBUTION_SCHEMA_VERSION",
    "audit_materialized_answer_distributions",
    "canonicalize_answer_value",
    "evaluate_answer_distribution",
    "parse_numeric_answer_value",
    "write_task_distribution_reports",
]
