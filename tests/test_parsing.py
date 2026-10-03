from trace_tool.parsing import parse_tool_calls, strip_hallucinated_responses


def test_single_call():
    text = 'Let me zoom.\n<tool_call>\n{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [1, 2, 3, 4], "label": "a", "img_idx": 0}}\n</tool_call>'
    (call,) = parse_tool_calls(text)
    assert call.error is None and call.name == "image_zoom_in_tool"
    assert call.arguments["bbox_2d"] == [1, 2, 3, 4]


def test_multiple_and_string_arguments():
    text = (
        '<tool_call>\n{"name": "calculator", "arguments": "{\\"expression\\": \\"1+1\\"}"}\n</tool_call>\n'
        '<tool_call>\n{"name": "ocr", "arguments": {"img_idx": 0}}\n</tool_call>'
    )
    calls = parse_tool_calls(text)
    assert [c.name for c in calls] == ["calculator", "ocr"]
    assert calls[0].arguments == {"expression": "1+1"}


def test_malformed_and_unterminated():
    calls = parse_tool_calls("<tool_call>\n{not json}\n</tool_call>\nthen <tool_call>\n{\"name\": \"ocr\"")
    assert len(calls) == 2
    assert calls[0].error and "valid JSON" in calls[0].error
    assert calls[1].error == "Error: unterminated <tool_call> block"


def test_no_calls():
    assert parse_tool_calls('The answer is 3.\n{"answer": 3}') == []


def test_strip_hallucinated_response():
    text = '<tool_call>\n{"name": "ocr", "arguments": {}}\n</tool_call>\n<tool_response>\nfake\n</tool_response>'
    cut, flagged = strip_hallucinated_responses(text)
    assert flagged and cut.endswith("</tool_call>")
