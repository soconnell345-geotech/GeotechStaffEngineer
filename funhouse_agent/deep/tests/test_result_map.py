"""TA-30: a large JSON result opens with a map of its keys, so the preview
deepagents shows of a result moved to scratch says where each key is (field
session 2026-10-15: the model saw the start of a 277,000-character line,
guessed a key and gave up)."""

import json

from langchain_core.messages import ToolMessage

from funhouse_agent.deep.result_map import (MIN_CHARS, LargeResultKeyMap,
                                            mapped_text)


def _kriging_like():
    return {"summary": {"n": 2500, "mean": 12.3},
            "krige_field": [i / 7.0 for i in range(2500)],
            "krige_variance": [i / 13.0 for i in range(2500)],
            "grid_x": list(range(2500)), "notes": "fitted " * 400}


def test_the_map_names_each_key_at_the_line_it_starts_on():
    text = mapped_text(json.dumps(_kriging_like()))
    lines = text.splitlines()
    assert lines[0].startswith("KEY MAP")
    import re
    for key in _kriging_like():
        m = re.search(re.escape(key) + r" \(line ([\d,]+)", text)
        n = int(m.group(1).replace(",", ""))
        assert lines[n - 1].startswith(f' "{key}":'), key
    assert "a list of 2,500 numbers" in text
    # the body is the same JSON
    first = next(i for i, ln in enumerate(lines) if ln == "{")
    assert json.loads("\n".join(lines[first:])) == _kriging_like()


def test_the_map_is_in_deepagents_own_preview():
    from deepagents.middleware._message_eviction import \
        _create_content_preview
    preview = _create_content_preview(mapped_text(json.dumps(
        _kriging_like())))
    # A str up to deepagents 0.7.13; a ContentPreview with .text by 0.7.24.
    preview = getattr(preview, "text", preview)
    assert "krige_variance (line" in preview
    assert "read_file" in preview


def test_small_or_non_json_results_pass_unchanged():
    small = json.dumps({"a": 1})
    assert mapped_text(small) is small
    big_text = "x" * (MIN_CHARS + 10)
    assert mapped_text(big_text) is big_text
    big_list = json.dumps(list(range(20000)))
    assert mapped_text(big_list) is big_list


def test_the_middleware_maps_a_tool_message():
    msg = ToolMessage(content=json.dumps(_kriging_like()), tool_call_id="t1",
                      name="call_agent")
    out = LargeResultKeyMap().wrap_tool_call(None, lambda req: msg)
    assert out.content.startswith("KEY MAP")
    assert out.tool_call_id == "t1" and out.name == "call_agent"


def test_every_agent_carries_it():
    """The primary and every helper (inside deepagents' own filesystem
    middleware, so it maps a result before it is moved)."""
    from funhouse_agent.deep import agent as agent_mod
    import inspect
    src = inspect.getsource(agent_mod.build_deep_agent)
    assert src.count("LargeResultKeyMap()") >= 3
