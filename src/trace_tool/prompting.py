"""Tool prompts in Qwen-Agent's Nous format, for Qwen2.5-VL and Qwen3-VL.

We do not send ``tools=`` to the server (that would need vLLM's tool parser and
image support in ``role: tool`` messages). Instead, like Qwen-Agent's
``NousFnCallPrompt``, the tool block is appended to the system message and tool
results are sent as user messages wrapped in ``<tool_response>`` tags. With the
Qwen3-VL chat template this renders to the same string as the native
``tools=`` + ``role: tool`` path (tests/test_prompting.py). Qwen2.5-VL's
template has no tool path; this is the layout Qwen-Agent prompts it with
(tests/test_qwen25.py, checked against the real tokenizers).
"""

from __future__ import annotations

import base64
import io
import json
from typing import Any

from PIL import Image

# Verbatim from the Qwen3-VL chat template (Qwen/Qwen3-VL-8B-Instruct), and
# identical to Qwen-Agent's FN_CALL_TEMPLATE.
TOOLS_HEADER = (
    "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
)
TOOLS_FOOTER = (
    "\n</tools>\n\nFor each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
)

TOOL_RESPONSE_OPEN = "<tool_response>\n"
TOOL_RESPONSE_CLOSE = "\n</tool_response>"

IMAGE_MARKER = "<image>"


def hf_tojson(value: Any) -> str:
    """transformers' chat-template ``tojson`` filter with default arguments."""
    return json.dumps(value, ensure_ascii=False)


def render_tools_block(tool_schemas: list[dict[str, Any]]) -> str:
    return TOOLS_HEADER + "".join("\n" + hf_tojson(schema) for schema in tool_schemas) + TOOLS_FOOTER


def system_message(system_prompt: str | None, tool_schemas: list[dict[str, Any]]) -> dict[str, Any] | None:
    if tool_schemas:
        block = render_tools_block(tool_schemas)
        content = f"{system_prompt}\n\n{block}" if system_prompt else block
        return {"role": "system", "content": content}
    if system_prompt:
        return {"role": "system", "content": system_prompt}
    return None


# Internal message format: content is a str or a list of parts, where a part is
# {"type": "text", "text": str} or {"type": "image", "image": PIL.Image, "index": int}.
# `index` is the image's position in ToolContext.images.


def text_part(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def image_part(image: Image.Image, index: int) -> dict[str, Any]:
    return {"type": "image", "image": image, "index": index}


def merge_text_parts(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for part in parts:
        if part["type"] == "text":
            if not part["text"]:
                continue
            if merged and merged[-1]["type"] == "text":
                merged[-1] = text_part(merged[-1]["text"] + part["text"])
                continue
        merged.append(part)
    return merged


def user_prompt_content(prompt: str, images: list[tuple[Image.Image, int]]) -> list[dict[str, Any]]:
    """Split a TRACE prompt on its ``<image>`` markers, as TRACE's generator does."""
    segments = prompt.split(IMAGE_MARKER)
    if len(segments) - 1 != len(images):
        raise ValueError(f"prompt has {len(segments) - 1} {IMAGE_MARKER} markers for {len(images)} images")
    parts: list[dict[str, Any]] = []
    for i, segment in enumerate(segments):
        if segment:
            parts.append(text_part(segment))
        if i < len(images):
            parts.append(image_part(*images[i]))
    return parts


def tool_response_content(responses: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """One user turn holding consecutive tool responses, as the template renders ``role: tool`` runs."""
    parts: list[dict[str, Any]] = []
    for i, body in enumerate(responses):
        parts.append(text_part(("\n" if i else "") + TOOL_RESPONSE_OPEN))
        parts.extend(body)
        parts.append(text_part(TOOL_RESPONSE_CLOSE))
    return merge_text_parts(parts)


def encode_png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class ImageURLCache:
    """Encodes each image once per episode; images are re-sent on every turn."""

    def __init__(self) -> None:
        self._cache: dict[int, tuple[Image.Image, str]] = {}

    def url(self, image: Image.Image) -> str:
        hit = self._cache.get(id(image))
        if hit is None or hit[0] is not image:
            hit = (image, encode_png_data_url(image))
            self._cache[id(image)] = hit
        return hit[1]


def to_openai_messages(messages: list[dict[str, Any]], cache: ImageURLCache) -> list[dict[str, Any]]:
    out = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            out.append({"role": message["role"], "content": content})
            continue
        parts = []
        for part in content:
            if part["type"] == "text":
                parts.append({"type": "text", "text": part["text"]})
            else:
                parts.append({"type": "image_url", "image_url": {"url": cache.url(part["image"])}})
        out.append({"role": message["role"], "content": parts})
    return out


def to_transcript(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """JSON-serialisable copy with images replaced by their index and size."""
    out = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            out.append({"role": message["role"], "content": content})
            continue
        parts = []
        for part in content:
            if part["type"] == "text":
                parts.append({"type": "text", "text": part["text"]})
            else:
                parts.append({"type": "image", "index": part["index"], "size": list(part["image"].size)})
        out.append({"role": message["role"], "content": parts})
    return out


def render_text(messages: list[dict[str, Any]]) -> str:
    """Plain-text rendering for debugging; images become ``<image k>``."""
    chunks = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            body = content
        else:
            body = "".join(p["text"] if p["type"] == "text" else f"<image {p['index']}>" for p in content)
        chunks.append(f"<|{message['role']}|>\n{body}")
    return "\n".join(chunks)
