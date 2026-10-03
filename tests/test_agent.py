import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

from trace_tool.agent import Completion, EpisodeConfig, annotation_region, oracle_crop_image, run_episode
from trace_tool.conditions import build_episode_config, load_conditions, read_system_prompt
from trace_tool.scoring import score_row
from trace_tool.tools import build_tools
from trace_tool.models import QWEN3_VL, QWEN25_VL

from .conftest import make_image, make_row

ZOOM_CALL = '<tool_call>\n{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [0, 0, 400, 400], "label": "box", "img_idx": 0}}\n</tool_call>'
CALC_CALL = '<tool_call>\n{"name": "calculator", "arguments": {"expression": "1 + 2"}}\n</tool_call>'


class ScriptedBackend:
    def __init__(self, replies, finish="stop"):
        self.replies = list(replies)
        self.requests = []
        self.finish = finish

    async def complete(self, messages, *, max_tokens, sampling, seed):
        self.requests.append({"messages": messages, "max_tokens": max_tokens, "sampling": sampling, "seed": seed})
        text = self.replies.pop(0)
        return Completion(text=text, finish_reason=self.finish, prompt_tokens=100, completion_tokens=50)


def _run(row, cfg, backend):
    with ThreadPoolExecutor(2) as pool:
        return asyncio.run(run_episode(row, cfg, backend, pool))


def test_zoom_then_calculator_then_answer(row):
    cfg = EpisodeConfig(family=QWEN3_VL, system_prompt=read_system_prompt(["prompts/trace_system_prompt.txt"]), tools=build_tools(["zoom", "calculator"], QWEN3_VL))
    backend = ScriptedBackend([ZOOM_CALL, "Two shapes.\n" + CALC_CALL, 'So 3.\n{"answer": 3}'])
    record = _run(row, cfg, backend)

    assert record["stop_reason"] == "answer"
    assert record["turns"] == 3 and record["n_tool_calls"] == 2
    assert [c["name"] for c in record["tool_calls"]] == ["image_zoom_in_tool", "calculator"]
    assert all(c["ok"] for c in record["tool_calls"])
    assert record["n_images"] == 2  # original + zoom crop
    assert score_row(record["final_response"], row.answer_gt, row.annotation_gt, row.reward_contract)["strict_correct"] == 1.0

    # request 2 replays the assistant turn and sends the crop inside <tool_response>
    msgs = backend.requests[1]["messages"]
    assert msgs[0]["role"] == "system" and "# Tools" in msgs[0]["content"]
    assert msgs[0]["content"].startswith("You are a helpful, conversational assistant")
    assert msgs[2] == {"role": "assistant", "content": ZOOM_CALL}
    tool_turn = msgs[3]
    assert tool_turn["role"] == "user"
    assert [p["type"] for p in tool_turn["content"]] == ["text", "image_url", "text"]
    assert tool_turn["content"][0]["text"] == "<tool_response>\n"
    assert tool_turn["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert backend.requests[2]["messages"][5]["content"] == [{"type": "text", "text": "<tool_response>\n3\n</tool_response>"}]
    # the transcript is JSON-serialisable
    json.dumps(record)


def test_no_tool_condition_is_single_turn(row):
    spec = load_conditions()["no_tool"]
    cfg = build_episode_config(spec, seed=42, family=QWEN3_VL)
    backend = ScriptedBackend([ZOOM_CALL])  # tool-looking text is not executed without tools
    record = _run(row, cfg, backend)
    assert record["turns"] == 1 and record["n_tool_calls"] == 0
    req = backend.requests[0]
    assert req["max_tokens"] == 2048 and req["seed"] == 42
    assert req["sampling"] == {"temperature": 0.6, "top_p": 0.95, "top_k": -1, "repetition_penalty": 1.0, "presence_penalty": 0.0}
    assert "# Tools" not in req["messages"][0]["content"]


def test_turn_limit_and_budget(row):
    cfg = EpisodeConfig(family=QWEN3_VL, system_prompt="s", tools=build_tools(["calculator"], QWEN3_VL), max_turns=3, max_total_tokens=1000)
    backend = ScriptedBackend([CALC_CALL] * 3)
    record = _run(row, cfg, backend)
    assert record["stop_reason"] == "max_turns" and record["turns"] == 3 and record["n_tool_calls"] == 3
    # the last turn's call is recorded but was never executed
    assert [c["executed"] for c in record["tool_calls"]] == [True, True, False]
    assert record["tool_calls"][-1]["skipped"] == "max_turns"

    cfg = EpisodeConfig(family=QWEN3_VL, system_prompt="s", tools=build_tools(["calculator"], QWEN3_VL), max_turns=10, max_total_tokens=120, max_tokens_per_turn=100)
    backend = ScriptedBackend([CALC_CALL] * 10)
    record = _run(row, cfg, backend)
    assert record["stop_reason"] == "token_budget"
    assert [r["max_tokens"] for r in backend.requests] == [100, 70, 20]


def test_errors_are_returned_to_model(row):
    cfg = EpisodeConfig(family=QWEN3_VL, system_prompt="s", tools=build_tools(["calculator"], QWEN3_VL), max_calls_per_turn=1)
    bad = '<tool_call>\n{"name": "nope", "arguments": {}}\n</tool_call>\n' + CALC_CALL
    backend = ScriptedBackend([bad, '{"answer": 3}'])
    record = _run(row, cfg, backend)
    tool_text = backend.requests[1]["messages"][-1]["content"][0]["text"]
    assert "unknown function 'nope'" in tool_text and "at most 1 tool calls" in tool_text
    assert [c["ok"] for c in record["tool_calls"]] == [False, False]


def test_endpoint_error_is_recorded(row):
    class Failing:
        async def complete(self, *a, **k):
            raise RuntimeError("boom")

    record = _run(row, EpisodeConfig(family=QWEN3_VL, system_prompt="s"), Failing())
    assert record["stop_reason"] == "error" and "boom" in record["error"]


def test_oracle_crop_and_lowres(row):
    assert annotation_region(row.annotation_gt) == [50, 50, 460, 310]
    assert annotation_region({"type": "point_map", "value": {"a": [10, 20], "b": [30, 5]}}) == [10, 5, 30, 20]
    assert oracle_crop_image(make_image(), {"type": "point", "value": [300, 200]}, QWEN3_VL) is not None

    spec = load_conditions()["oracle_crop"]
    backend = ScriptedBackend(['{"answer": 3}'])
    record = _run(row, build_episode_config(spec, seed=1, family=QWEN3_VL), backend)
    content = backend.requests[0]["messages"][1]["content"]
    assert [p["type"] for p in content] == ["image_url", "image_url", "text"]
    assert record["n_images"] == 2

    big = make_row(image=make_image(1600, 1200))
    spec = load_conditions()["no_tool_lowres"]
    record = _run(big, build_episode_config(spec, seed=1, family=QWEN3_VL), ScriptedBackend(['{"answer": 3}']))
    w, h = record["shown_image_size"]
    assert w * h <= 401408 and w % 32 == 0 and h % 32 == 0


def test_all_conditions_build():
    for name, spec in load_conditions().items():
        cfg = build_episode_config(spec, seed=0, family=QWEN3_VL)
        assert cfg.max_turns >= 1, name


def test_rejected_request_is_terminal(row):
    from trace_tool.agent import RequestRejected

    class Rejecting:
        async def complete(self, *a, **k):
            raise RequestRejected("maximum context length exceeded")

    record = _run(row, EpisodeConfig(family=QWEN3_VL, system_prompt="s"), Rejecting())
    assert record["stop_reason"] == "rejected"


def test_rgba_composited_on_white():
    from PIL import Image

    from trace_tool.data import to_rgb

    rgba = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    assert to_rgb(rgba).getpixel((0, 0)) == (255, 255, 255)


def test_tool_timeout_excludes_queue_time():
    """With one worker thread, the second call waits for a slot; its timeout must not run meanwhile."""
    import time

    from trace_tool.agent import _execute
    from trace_tool.parsing import ParsedCall
    from trace_tool.tools.base import Tool, ToolContext, ToolResult

    class Slow(Tool):
        name = "slow"

        def run(self, args, ctx):
            time.sleep(0.6)
            return ToolResult(text="done")

    async def go():
        with ThreadPoolExecutor(1) as pool:
            calls = [ParsedCall("slow", {}, "") for _ in range(2)]
            return await asyncio.gather(*(_execute(c, {"slow": Slow()}, ToolContext([]), pool, 1.0) for c in calls))

    results = asyncio.run(go())
    assert [r.ok for r, _ in results] == [True, True]
    assert all(0.5 < latency < 0.9 for _, latency in results)


def test_elongated_zoom_becomes_tool_error():
    tool = build_tools(["zoom"], QWEN3_VL)["image_zoom_in_tool"]
    from trace_tool.tools.base import ToolContext

    result = tool({"bbox_2d": [0, 0, 1, 1000], "label": "x", "img_idx": 0}, ToolContext([make_image(800, 600)]))
    assert not result.ok and "elongated" in result.text


def test_calls_in_a_turn_run_in_order_and_see_earlier_images(row):
    two_zooms = (
        '<tool_call>\n{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [0, 0, 500, 500], "label": "a", "img_idx": 0}}\n</tool_call>\n'
        '<tool_call>\n{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [0, 0, 500, 500], "label": "b", "img_idx": 1}}\n</tool_call>'
    )
    cfg = EpisodeConfig(family=QWEN3_VL, system_prompt="s", tools=build_tools(["zoom"], QWEN3_VL))
    record = _run(row, cfg, ScriptedBackend([two_zooms, '{"answer": 3}']))
    assert [c["ok"] for c in record["tool_calls"]] == [True, True]  # call 2 zooms into call 1's output
    assert record["n_images"] == 3


def test_zoom_validates_like_qwen_agent():
    from trace_tool.tools.base import ToolContext

    tool = build_tools(["zoom"], QWEN3_VL)["image_zoom_in_tool"]
    ctx = ToolContext([make_image(800, 600), make_image(400, 300)])
    assert not tool({"bbox_2d": [0, 0, 500, 500], "label": "x"}, ctx).ok  # img_idx required
    assert not tool({"bbox_2d": [0, 0, 500, 500], "img_idx": 0}, ctx).ok  # label required
    assert not tool({"bbox_2d": ["0", 0, 500, 500], "label": "x", "img_idx": 0}, ctx).ok  # numbers only
    latest = tool({"bbox_2d": [0, 0, 1000, 1000], "label": "x", "img_idx": -1}, ctx)
    assert latest.ok and latest.meta["abs_box"] == [0, 0, 400, 300]  # -1 = latest image, as in Python


def test_no_tool_text_is_scored_verbatim(row):
    text = 'Counting.\n<tool_response>\nnone\n</tool_response>\nSo three.\n{"answer": 3}'
    record = _run(row, build_episode_config(load_conditions()["no_tool"], seed=1, family=QWEN3_VL), ScriptedBackend([text]))
    assert record["final_response"] == text and record["hallucinated_tool_responses"] == 0


def test_timeout_race_keeps_finished_result():
    """A call that finished while the loop was blocked past its deadline is not reported as timed out."""
    import time

    from trace_tool.agent import _execute
    from trace_tool.parsing import ParsedCall
    from trace_tool.tools.base import Tool, ToolContext, ToolResult

    class Quick(Tool):
        name = "quick"

        def run(self, args, ctx):
            time.sleep(0.05)
            return ToolResult(text="done")

    async def go():
        with ThreadPoolExecutor(1) as pool:
            task = asyncio.ensure_future(_execute(ParsedCall("quick", {}, ""), {"quick": Quick()}, ToolContext([]), pool, 0.1))
            await asyncio.sleep(0)
            time.sleep(0.3)  # block the loop past the deadline; the tool finishes meanwhile
            return await task

    result, _ = asyncio.run(go())
    assert result.ok and result.text == "done"
