#!/usr/bin/env python
"""Check on the server that our flattened tool messages render like the native tool path.

1. Renders a two-turn tool conversation through the checkpoint's own processor
   (transformers ``apply_chat_template``) both ways: native ``tools=`` + ``role: tool``
   and our flattened form. They must be identical.
2. Optionally (``--endpoint``) sends the flattened form to vLLM's ``/tokenize`` +
   ``/detokenize`` and compares with (1) after collapsing ``<|image_pad|>`` runs.
   This step is best-effort: some vLLM versions do not accept images in /tokenize.

    python scripts/check_template.py --model Qwen/Qwen3-VL-8B-Instruct
    python scripts/check_template.py --model Qwen/Qwen3-VL-8B-Instruct \
        --endpoint http://127.0.0.1:18100 --served qwen3-vl-8b-instruct
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request

from PIL import Image

from trace_tool.prompting import encode_png_data_url, system_message, text_part, tool_response_content, user_prompt_content
from trace_tool.tools import build_tools


def _parts(parts, url):
    return [{"type": "text", "text": p["text"]} if p["type"] == "text" else {"type": "image_url", "image_url": {"url": url}} for p in parts]


def _post(url: str, payload: dict) -> dict:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF id or local path of the checkpoint")
    p.add_argument("--endpoint", help="vLLM server root, e.g. http://127.0.0.1:18100")
    p.add_argument("--served", help="served model name (for --endpoint)")
    args = p.parse_args()

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    image = Image.new("RGB", (320, 320), (200, 30, 30))
    url = encode_png_data_url(image)
    schemas = [t.schema() for t in build_tools(["zoom", "calculator"]).values()]
    sys_text = "You are a helpful assistant."
    user = _parts(user_prompt_content('<image>Count the boxes.\n{"answer":2}', [(image, 0)]), url)
    a1 = '<tool_call>\n{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [0, 0, 500, 500], "label": "box", "img_idx": 0}}\n</tool_call>'
    bodies = [[{"type": "image", "image": image, "index": 1}], [text_part("42")]]

    native = [{"role": "system", "content": sys_text}, {"role": "user", "content": user}, {"role": "assistant", "content": a1}]
    native += [{"role": "tool", "content": _parts(b, url)} for b in bodies]
    ours = [system_message(sys_text, schemas), {"role": "user", "content": user}, {"role": "assistant", "content": a1},
            {"role": "user", "content": _parts(tool_response_content(bodies), url)}]

    native_text = processor.apply_chat_template(native, tools=schemas, tokenize=False, add_generation_prompt=True)
    ours_text = processor.apply_chat_template(ours, tokenize=False, add_generation_prompt=True)
    if native_text != ours_text:
        print("MISMATCH between native and flattened rendering\n--- native\n" + native_text + "\n--- ours\n" + ours_text)
        return 1
    print("OK: processor renders the flattened messages identically to the native tool path")

    if args.endpoint:
        root = args.endpoint.rstrip("/").removesuffix("/v1")
        try:
            tokens = _post(f"{root}/tokenize", {"model": args.served, "messages": ours, "add_generation_prompt": True})["tokens"]
            server_text = _post(f"{root}/detokenize", {"model": args.served, "tokens": tokens})["prompt"]
        except Exception as exc:
            print(f"SKIPPED server check ({type(exc).__name__}: {exc}); compare a logged prompt manually if needed")
            return 0
        collapse = lambda s: re.sub(r"(<\|image_pad\|>)+", "<|image_pad|>", s)
        if collapse(server_text) != collapse(ours_text):
            print("MISMATCH between vLLM rendering and processor rendering\n--- vllm\n" + server_text + "\n--- processor\n" + ours_text)
            return 1
        print("OK: vLLM renders the flattened messages identically")
    return 0


if __name__ == "__main__":
    sys.exit(main())
