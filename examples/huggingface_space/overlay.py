"""Render review-only overlays from Trace's public image annotations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from PIL import Image, ImageDraw, ImageFont

PUBLIC_ANNOTATION_TYPES = frozenset(
    {
        "bbox",
        "bbox_map",
        "bbox_sequence",
        "bbox_set",
        "bbox_set_map",
        "point",
        "point_map",
        "point_sequence",
        "point_set",
        "point_set_map",
        "segment",
        "segment_set",
    }
)

_COLORS = (
    (229, 57, 53, 235),
    (30, 136, 229, 235),
    (0, 137, 123, 235),
    (251, 140, 0, 235),
    (142, 36, 170, 235),
)


def render_annotation_overlay(
    source_image: Image.Image,
    annotation_gt: Mapping[str, Any],
) -> Image.Image:
    """Return an RGB image with a public annotation drawn over the source."""

    image = source_image.convert("RGBA").copy()
    draw = ImageDraw.Draw(image, "RGBA")
    annotation_type = str(annotation_gt.get("type", "")).strip()
    if annotation_type not in PUBLIC_ANNOTATION_TYPES:
        raise ValueError(f"unsupported public annotation type: {annotation_type!r}")

    items = list(_annotation_items(annotation_type, annotation_gt.get("value")))
    for index, (label, geometry_kind, geometry) in enumerate(items):
        color = _COLORS[index % len(_COLORS)]
        if geometry_kind == "bbox":
            _draw_bbox(draw, geometry, color=color, label=label)
        elif geometry_kind == "point":
            _draw_point(draw, geometry, color=color, label=label)
        elif geometry_kind == "segment":
            _draw_segment(draw, geometry, color=color, label=label)

    if not items:
        _draw_empty_annotation_badge(draw, annotation_type)
    return image.convert("RGB")


def _annotation_items(annotation_type: str, value: Any):
    if annotation_type == "bbox":
        yield "answer", "bbox", value
    elif annotation_type in {"bbox_set", "bbox_sequence"} and _sequence(value):
        for index, bbox in enumerate(value):
            yield str(index + 1), "bbox", bbox
    elif annotation_type == "bbox_map" and isinstance(value, Mapping):
        for key, bbox in sorted(value.items(), key=lambda item: str(item[0])):
            yield str(key), "bbox", bbox
    elif annotation_type == "bbox_set_map" and isinstance(value, Mapping):
        for key, boxes in sorted(value.items(), key=lambda item: str(item[0])):
            if not _sequence(boxes):
                continue
            for index, bbox in enumerate(boxes):
                yield f"{key}:{index + 1}", "bbox", bbox
    elif annotation_type == "point":
        yield "answer", "point", value
    elif annotation_type in {"point_set", "point_sequence"} and _sequence(value):
        for index, point in enumerate(value):
            yield str(index + 1), "point", point
    elif annotation_type == "point_map" and isinstance(value, Mapping):
        for key, point in sorted(value.items(), key=lambda item: str(item[0])):
            yield str(key), "point", point
    elif annotation_type == "point_set_map" and isinstance(value, Mapping):
        for key, points in sorted(value.items(), key=lambda item: str(item[0])):
            if not _sequence(points):
                continue
            for index, point in enumerate(points):
                yield f"{key}:{index + 1}", "point", point
    elif annotation_type == "segment":
        yield "answer", "segment", value
    elif annotation_type == "segment_set" and _sequence(value):
        for index, segment in enumerate(value):
            yield str(index + 1), "segment", segment


def _draw_bbox(
    draw: ImageDraw.ImageDraw,
    value: Any,
    *,
    color: tuple[int, ...],
    label: str,
) -> None:
    if not _numeric_sequence(value, 4):
        return
    x0, y0, x1, y1 = (float(item) for item in value)
    draw.rectangle((x0, y0, x1, y1), outline=color, width=4)
    _draw_label(draw, (x0, y0), label, color)


def _draw_point(
    draw: ImageDraw.ImageDraw,
    value: Any,
    *,
    color: tuple[int, ...],
    label: str,
) -> None:
    if not _numeric_sequence(value, 2):
        return
    x, y = (float(item) for item in value)
    radius = 7.0
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    _draw_label(draw, (x + radius, y - radius), label, color)


def _draw_segment(
    draw: ImageDraw.ImageDraw,
    value: Any,
    *,
    color: tuple[int, ...],
    label: str,
) -> None:
    if not _sequence(value) or len(value) != 2:
        return
    if not _numeric_sequence(value[0], 2) or not _numeric_sequence(value[1], 2):
        return
    points = [(float(point[0]), float(point[1])) for point in value]
    draw.line(points, fill=color, width=5)
    _draw_label(draw, points[0], label, color)


def _draw_label(
    draw: ImageDraw.ImageDraw,
    point: tuple[float, float],
    label: str,
    color: tuple[int, ...],
) -> None:
    text = str(label)[:48]
    if not text:
        return
    font = ImageFont.load_default()
    left, top = max(0.0, float(point[0])), max(0.0, float(point[1]) - 16.0)
    right = left + max(18.0, float(draw.textlength(text, font=font)) + 8.0)
    draw.rectangle((left, top, right, top + 16.0), fill=color)
    draw.text((left + 4.0, top + 2.0), text, fill=(255, 255, 255, 255), font=font)


def _draw_empty_annotation_badge(
    draw: ImageDraw.ImageDraw,
    annotation_type: str,
) -> None:
    _draw_label(
        draw,
        (8.0, 22.0),
        f"{annotation_type}: empty witness",
        (69, 90, 100, 235),
    )


def _sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _numeric_sequence(value: Any, length: int) -> bool:
    if not _sequence(value) or len(value) != length:
        return False
    return all(
        isinstance(item, (int, float)) and not isinstance(item, bool)
        for item in value
    )


__all__ = ["PUBLIC_ANNOTATION_TYPES", "render_annotation_overlay"]
