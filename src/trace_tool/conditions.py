"""Load evaluation conditions from configs/conditions.yaml."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from .agent import DECODING_PRESETS, EpisodeConfig
from .tools import TOOL_GROUPS, build_tools

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "conditions.yaml"

_EPISODE_KEYS = {
    "max_turns",
    "max_tokens_per_turn",
    "max_total_tokens",
    "max_calls_per_turn",
    "decoding",
    "min_pixels",
    "max_pixels",
    "oracle_crop",
}


def load_conditions(path: str | Path = DEFAULT_CONFIG) -> dict[str, dict[str, Any]]:
    raw = yaml.safe_load(Path(path).read_text())
    defaults = raw.get("defaults", {})
    out = {}
    for name, overrides in (raw.get("conditions") or {}).items():
        merged = {**defaults, **(overrides or {})}
        unknown = set(merged) - _EPISODE_KEYS - {"tools", "system_prompt_files"}
        if unknown:
            raise ValueError(f"condition {name!r}: unknown keys {sorted(unknown)}")
        if merged["decoding"] not in DECODING_PRESETS:
            raise ValueError(f"condition {name!r}: unknown decoding preset {merged['decoding']!r}")
        out[name] = merged
    return out


TRACE_PROMPT_FILE = "prompts/trace_system_prompt.txt"
TRACE_PROMPT_FILE_SHA256 = "f394927d9abcfb7a1e43ef48a30c29b8c70e6facdbda314d7b27c59d8c3ae900"


def read_system_prompt(files: list[str], root: Path = REPO_ROOT) -> str:
    """Each file is stripped, as TRACE's generator and EasyR1 dataset do."""
    parts = []
    for name in files:
        raw = (root / name).read_bytes()
        if name == TRACE_PROMPT_FILE and hashlib.sha256(raw).hexdigest() != TRACE_PROMPT_FILE_SHA256:
            raise RuntimeError(f"{name} differs from TRACE's pinned system prompt")
        parts.append(raw.decode("utf-8").strip())
    return "\n\n".join(parts)


def build_episode_config(spec: dict[str, Any], *, seed: int, root: Path = REPO_ROOT) -> EpisodeConfig:
    tools = build_tools(list(spec.get("tools") or []))
    return EpisodeConfig(
        system_prompt=read_system_prompt(list(spec["system_prompt_files"]), root),
        tools=tools,
        seed=seed,
        **{k: spec[k] for k in _EPISODE_KEYS},
    )


def describe(spec: dict[str, Any], cfg: EpisodeConfig) -> dict[str, Any]:
    """Settings recorded in run_config.json (and compared on resume)."""
    return {
        **{k: spec[k] for k in sorted(_EPISODE_KEYS)},
        "tools": sorted(cfg.tools),
        "tool_groups_requested": list(spec.get("tools") or []),
        "system_prompt_files": list(spec["system_prompt_files"]),
        "system_prompt_sha256": hashlib.sha256(cfg.system_prompt.encode()).hexdigest(),
        "tool_schemas": [tool.schema() for tool in cfg.tools.values()],
        "decoding_params": DECODING_PRESETS[spec["decoding"]],
        **_ocr_settings(cfg),
    }


def _ocr_settings(cfg: EpisodeConfig) -> dict[str, Any]:
    """OCR output depends on the backend and its package versions, so they are resume keys."""
    tool = cfg.tools.get("ocr")
    if tool is None:
        return {}
    from .tools.ocr import backend_versions

    return {"ocr_backend": tool.backend_name, "ocr_min_score": tool.min_score, "ocr_versions": backend_versions(tool.backend_name)}


__all__ = ["DEFAULT_CONFIG", "TOOL_GROUPS", "build_episode_config", "describe", "load_conditions", "read_system_prompt"]
