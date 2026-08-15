"""Answer-distribution checks over public contributor-review artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trace_tasks.review.distribution import (
    ANSWER_DISTRIBUTION_SCHEMA_VERSION,
    audit_materialized_answer_distributions,
    evaluate_answer_distribution,
    write_task_distribution_reports,
)
from trace_tasks.review.models import MATERIALIZATION_SCHEMA_VERSION, RECIPE_ID


def test_distribution_checks_pass_balanced_categorical_answers() -> None:
    rows = []
    for label in ("A", "B", "C", "D", "E"):
        rows.extend(
            {"answer_type": "option_letter", "answer_value": label}
            for _ in range(20)
        )
    report = evaluate_answer_distribution(rows)
    assert report["checks"]["min_unique_answers"]["pass"] is True
    assert report["checks"]["max_answer_frequency"]["pass"] is True
    assert report["checks"]["max_five_bin_frequency"]["pass"] is None
    assert report["pass"] is True


def test_distribution_checks_fail_on_max_answer_frequency() -> None:
    rows = [
        *(
            {"answer_type": "integer", "answer_value": 1}
            for _ in range(36)
        ),
        *(
            {"answer_type": "integer", "answer_value": value}
            for value in range(2, 10)
            for _ in range(8)
        ),
    ]
    report = evaluate_answer_distribution(rows)
    assert report["sample_count"] == 100
    assert report["checks"]["max_answer_frequency"]["pass"] is False
    assert report["pass"] is False


def test_distribution_reports_numeric_five_bin_summary_without_gating() -> None:
    values = [0.0] * 60 + [4.0] * 10 + [-4.0] * 10 + [2.0] * 10 + [-2.0] * 10
    report = evaluate_answer_distribution(
        [{"answer_type": "number", "answer_value": value} for value in values]
    )
    assert report["sample_count"] == 100
    assert report["checks"]["min_unique_answers"]["pass"] is True
    assert report["checks"]["max_answer_frequency"]["pass"] is False
    assert report["checks"]["max_five_bin_frequency"] == {
        "observed": 0.6,
        "pass": None,
        "status": "reported_not_gated_numeric_answers",
    }
    assert report["pass"] is False


def test_distribution_pass_ignores_numeric_bin_skew_when_core_checks_pass() -> None:
    values = (
        [0.0] * 20
        + [1.0] * 18
        + [2.0] * 18
        + [9.0] * 16
        + [10.0] * 14
        + [11.0] * 14
    )
    report = evaluate_answer_distribution(
        [{"answer_type": "number", "answer_value": value} for value in values]
    )
    assert report["checks"]["min_unique_answers"]["pass"] is True
    assert report["checks"]["max_answer_frequency"]["pass"] is True
    assert report["checks"]["max_five_bin_frequency"]["pass"] is None
    assert report["checks"]["max_five_bin_frequency"]["observed"] is not None
    assert report["pass"] is True


def _materialized_distribution_fixture(root: Path) -> tuple[str, Path]:
    task_id = "task_symbolic__abacus__distribution_test"
    task_root = root / "symbolic" / "abacus" / task_id
    digest = f"sha256:{'a' * 64}"
    ordinal = 0
    entries = []
    for query_id, answer_type, values in (
        ("letters", "option_letter", ("A", "B", "C", "D")),
        ("values", "integer", (1, 2, 3, 4)),
    ):
        data_root = task_root / "data" / query_id
        data_root.mkdir(parents=True, exist_ok=True)
        for value in values:
            payload = {
                "task_id": task_id,
                "domain": "symbolic",
                "scene_id": "abacus",
                "query_id": query_id,
                "recipe_digest": digest,
                "answer_gt": {"type": answer_type, "value": value},
            }
            (data_root / f"{ordinal:04d}.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            entries.append({"query_id": query_id})
            ordinal += 1
    (task_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": MATERIALIZATION_SCHEMA_VERSION,
                "recipe_id": RECIPE_ID,
                "task_id": task_id,
                "domain": "symbolic",
                "scene_id": "abacus",
                "recipe_digest": digest,
                "query_ids": ["letters", "values"],
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )
    return task_id, task_root


def test_materialized_distribution_groups_queries_and_writes_bound_task_report(
    tmp_path: Path,
) -> None:
    task_id, task_root = _materialized_distribution_fixture(tmp_path)
    report = audit_materialized_answer_distributions(
        tmp_path, task_ids=[task_id]
    )
    assert report["schema_version"] == ANSWER_DISTRIBUTION_SCHEMA_VERSION
    assert report["checked_records"] == 8
    assert report["passed"] is True
    assert report["tasks"][0]["query_ids"] == ["letters", "values"]
    assert report["tasks"][0]["overall"]["sample_count"] == 8
    assert all(
        query_report["pass"]
        for query_report in report["tasks"][0]["per_query_id"].values()
    )

    written = write_task_distribution_reports(tmp_path, report)
    assert written == [
        f"symbolic/abacus/{task_id}/answer_distribution.json"
    ]
    task_payload = json.loads(
        (task_root / "answer_distribution.json").read_text(encoding="utf-8")
    )
    assert task_payload["recipe_digest"] == f"sha256:{'a' * 64}"
    assert task_payload["pass"] is True


def test_materialized_distribution_rejects_empty_selection_and_manifest_drift(
    tmp_path: Path,
) -> None:
    _, task_root = _materialized_distribution_fixture(tmp_path)
    with pytest.raises(ValueError, match="selected no materialized records"):
        audit_materialized_answer_distributions(tmp_path, query_ids=["missing"])

    report = audit_materialized_answer_distributions(tmp_path)
    manifest = json.loads((task_root / "manifest.json").read_text(encoding="utf-8"))
    manifest["recipe_digest"] = f"sha256:{'b' * 64}"
    (task_root / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="does not match materialized manifest"):
        write_task_distribution_reports(tmp_path, report)


def test_materialized_distribution_rejects_partial_record_sets(
    tmp_path: Path,
) -> None:
    _, task_root = _materialized_distribution_fixture(tmp_path)
    next((task_root / "data" / "letters").glob("*.json")).unlink()
    with pytest.raises(ValueError, match="record counts do not match"):
        audit_materialized_answer_distributions(tmp_path)
