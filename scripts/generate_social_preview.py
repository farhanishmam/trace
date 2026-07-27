#!/usr/bin/env python3
"""Generate the branded Trace social-preview image from committed assets."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
MONTAGE_PATH = (
    REPO_ROOT
    / "docs"
    / "assets"
    / "paper-domain-montage"
    / "trace-paper-domain-montage.png"
)
MARK_PATH = REPO_ROOT / "docs" / "assets" / "brand" / "trace-mark-light.png"
FONT_PATH = (
    REPO_ROOT
    / "src"
    / "trace_tasks"
    / "resources"
    / "assets"
    / "fonts"
    / "google_fonts"
    / "noto_sans"
    / "NotoSans[wdth,wght].ttf"
)
OUTPUT_PATH = (
    REPO_ROOT / "docs" / "assets" / "social" / "trace-social-preview.png"
)

CANVAS_SIZE = (1280, 640)
BACKGROUND = "#102a3b"
SURFACE = "#f7f9f9"
INK = "#17242d"
MUTED = "#667983"
WHITE = "#ffffff"
BLUE = "#3974d7"
TEAL = "#159b91"
CORAL = "#e85d4a"
GOLD = "#e5a62b"


def _font(size: int, variation: str) -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(FONT_PATH, size=size)
    font.set_variation_by_name(variation)
    return font


def _rounded_image(
    image: Image.Image,
    size: tuple[int, int],
    radius: int,
) -> tuple[Image.Image, Image.Image]:
    fitted = ImageOps.fit(
        image.convert("RGB"),
        size,
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, size[0] - 1, size[1] - 1),
        radius=radius,
        fill=255,
    )
    return fitted, mask


def generate(output_path: Path = OUTPUT_PATH) -> Path:
    """Render and save the social preview."""

    canvas = Image.new("RGB", CANVAS_SIZE, BACKGROUND)
    draw = ImageDraw.Draw(canvas)

    draw.ellipse((-180, -240, 340, 280), fill="#163c54")
    draw.ellipse((1030, 430, 1390, 790), fill="#123f49")
    draw.rounded_rectangle((610, 44, 1224, 596), radius=30, fill=SURFACE)

    with Image.open(MARK_PATH) as mark_source:
        mark = mark_source.convert("RGBA").resize(
            (70, 70),
            resample=Image.Resampling.LANCZOS,
        )
    draw.rounded_rectangle((58, 48, 142, 132), radius=18, fill=WHITE)
    canvas.paste(mark, (65, 55), mark)
    draw.text((162, 54), "Trace", font=_font(54, "Bold"), fill=WHITE)

    draw.text(
        (62, 164),
        "VERIFIABLE VISUAL REASONING",
        font=_font(19, "SemiBold"),
        fill="#7fc6ed",
        spacing=4,
    )
    draw.multiline_text(
        (58, 204),
        "Grounded tasks\nfor post-training",
        font=_font(55, "Bold"),
        fill=WHITE,
        spacing=0,
    )
    draw.multiline_text(
        (62, 354),
        "1,000 deterministic tasks across\n277 scenes and 11 visual domains",
        font=_font(25, "Regular"),
        fill="#d9e6eb",
        spacing=8,
    )

    draw.rounded_rectangle((58, 458, 548, 572), radius=20, fill="#183c50")
    draw.text(
        (82, 474),
        "+3.51 / +4.06 pp",
        font=_font(34, "Bold"),
        fill="#63d4c8",
    )
    draw.text(
        (82, 521),
        "macro-average transfer across 24 external benchmarks",
        font=_font(16, "Medium"),
        fill="#d9e6eb",
    )

    with Image.open(MONTAGE_PATH) as montage_source:
        montage, montage_mask = _rounded_image(
            montage_source,
            size=(566, 292),
            radius=18,
        )
    canvas.paste(montage, (634, 68), montage_mask)

    draw.text(
        (638, 390),
        "EVERY INSTANCE INCLUDES",
        font=_font(16, "SemiBold"),
        fill=MUTED,
    )
    features = (
        (BLUE, "typed answer"),
        (TEAL, "verifier metadata"),
        (CORAL, "replayable execution trace"),
    )
    y = 431
    for color, label in features:
        draw.rounded_rectangle((638, y + 3, 658, y + 23), radius=5, fill=color)
        draw.text((674, y - 2), label, font=_font(21, "Medium"), fill=INK)
        y += 45

    draw.rounded_rectangle((1116, 550, 1174, 556), radius=3, fill=GOLD)
    draw.text(
        (638, 548),
        "trace · deterministic · metadata-grounded",
        font=_font(15, "Medium"),
        fill=MUTED,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(
        output_path,
        format="PNG",
        optimize=True,
        compress_level=9,
    )
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Output PNG path (default: repository social asset)",
    )
    args = parser.parse_args()

    output_path = generate(args.output.resolve())
    payload = output_path.read_bytes()
    print(
        f"{output_path}: {CANVAS_SIZE[0]}x{CANVAS_SIZE[1]}, "
        f"{len(payload)} bytes, sha256={hashlib.sha256(payload).hexdigest()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
