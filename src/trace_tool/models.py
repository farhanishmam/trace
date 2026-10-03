"""Model families: the conventions tools must follow to match how each model sees images.

| family | vision patch x merge | grounding coordinates | zoom tool convention from |
| --- | --- | --- | --- |
| qwen2.5-vl | 14 x 2 = 28 | absolute pixels of the image as the model sees it (after resizing) | DeepEyes (Qwen2.5-VL-7B): absolute pixel boxes, 28-px minimum side |
| qwen3-vl | 16 x 2 = 32 | relative [0, 1000] | Qwen3-VL cookbook ``image_zoom_in_tool`` (ZwZ "official tool use") |

Other Qwen2.5-VL agents differ: Mini-o3 and Pixel-Reasoner zoom with relative /
normalised boxes and Thyme crops through generated code. We follow DeepEyes
because it matches Qwen2.5-VL's native grounding frame.

Tools take coordinates in the family's native grounding frame, so the model can
use the same boxes it would output for grounding. The zoom tool grows tiny
boxes to one patch-merge unit per side: 28 px for Qwen2.5-VL (as DeepEyes
does), 32 px for Qwen3-VL (as Qwen-Agent does).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# TRACE's pinned checkpoints (rlvr/evaluation/trace_validation/suite.v1.json).
PINNED_REVISIONS = {
    "Qwen/Qwen2.5-VL-3B-Instruct": "66285546d2b821cf421d4f5eb2576359d3770cd3",
    "Qwen/Qwen2.5-VL-7B-Instruct": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
    "maveryn/trace-qwen2.5-vl-3b": "2ec2374d5c219e6b12e26bda93d3b3adeb1e30c5",
    "maveryn/trace-qwen2.5-vl-7b": "4d0f1ae8ee25022058090dbdbff61957ece7331d",
}


@dataclass(frozen=True)
class ModelFamily:
    name: str
    patch_factor: int  # images are resized to multiples of this
    coords: str  # "abs": pixels of the image as shown; "rel1000": relative [0, 1000]
    zoom_min_side: int  # zoom boxes are grown to at least this many pixels per side


QWEN25_VL = ModelFamily("qwen2.5-vl", patch_factor=28, coords="abs", zoom_min_side=28)
QWEN3_VL = ModelFamily("qwen3-vl", patch_factor=32, coords="rel1000", zoom_min_side=32)
FAMILIES = {f.name: f for f in (QWEN25_VL, QWEN3_VL)}


def detect_family(model_id: str) -> ModelFamily:
    """Family from an HF id, served name or checkpoint path, e.g. Qwen/Qwen2.5-VL-7B-Instruct,
    maveryn/trace-qwen2.5-vl-3b, /ckpt/qwen2_5_vl_7b_grpo."""
    key = re.sub(r"[^a-z0-9]", "", model_id.lower())
    if "qwen25vl" in key:
        return QWEN25_VL
    if "qwen3vl" in key:
        return QWEN3_VL
    raise ValueError(f"cannot tell the model family of {model_id!r}; pass --family {'/'.join(FAMILIES)}")


def get_family(name: str) -> ModelFamily:
    if name not in FAMILIES:
        raise ValueError(f"unknown model family {name!r}; choose from {sorted(FAMILIES)}")
    return FAMILIES[name]
