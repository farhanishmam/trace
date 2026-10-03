"""Parse Hermes-style ``<tool_call>{...}</tool_call>`` blocks from model output."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

try:  # Qwen-Agent parses tool calls with json5; optional here
    import json5  # type: ignore
except ImportError:  # pragma: no cover
    json5 = None

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
HALLUCINATED_RESPONSE_MARKER = "<tool_response>"


@dataclass
class ParsedCall:
    name: str | None
    arguments: dict[str, Any] | None
    raw: str
    error: str | None = None


def _loads(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if json5 is None:
            raise
        return json5.loads(text)


def _parse_one(raw: str) -> ParsedCall:
    body = raw.strip()
    try:
        payload = _loads(body)
    except Exception as exc:
        return ParsedCall(None, None, raw, f"Error: tool call is not valid JSON ({exc})")
    if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
        return ParsedCall(None, None, raw, 'Error: tool call must be {"name": <function-name>, "arguments": {...}}')
    arguments = payload.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = _loads(arguments) if arguments.strip() else {}
        except Exception as exc:
            return ParsedCall(payload["name"], None, raw, f"Error: arguments are not valid JSON ({exc})")
    if not isinstance(arguments, dict):
        return ParsedCall(payload["name"], None, raw, "Error: arguments must be a JSON object")
    return ParsedCall(payload["name"], arguments, raw)


def parse_tool_calls(text: str) -> list[ParsedCall]:
    calls = [_parse_one(m.group(1)) for m in _TOOL_CALL_RE.finditer(text)]
    tail = text[text.rfind("</tool_call>") + len("</tool_call>") :] if calls else text
    if "<tool_call>" in tail:  # opened but never closed (usually a length cut-off)
        raw = tail[tail.index("<tool_call>") + len("<tool_call>") :]
        calls.append(ParsedCall(None, None, raw, "Error: unterminated <tool_call> block"))
    return calls


def strip_hallucinated_responses(text: str) -> tuple[str, bool]:
    """Cut the text where the model starts writing its own ``<tool_response>``."""
    cut = text.find(HALLUCINATED_RESPONSE_MARKER)
    if cut < 0:
        return text, False
    return text[:cut].rstrip("\n"), True
