"""Figures the user sees must be the figures the agent checked
(field session 2026-10-15, TRACE_REVIEW section 6).

- The ``calc`` helper draws the figures but had no way to look at one: it
  tried ``read_file`` on its PNG, was refused, and the figure went out
  unseen. It now has ``analyze_image``.
- The agent hand-wrote a chart file (``*.plotly.json``) over plot_data's,
  then checked plot_data's PNG, a different figure, and told the owner the
  chart had passed. ``save_file`` now says so when it writes a chart file.
"""

import json
import os

from funhouse_agent.vision_tools import (HAND_CHART_NOTE,
                                         dispatch_extended_tool)


def _save(path, content):
    return json.loads(dispatch_extended_tool(
        "save_file", {"path": str(path), "content": content}, None, {}))


def test_a_hand_written_chart_is_flagged(tmp_path):
    out = _save(tmp_path / "map.plotly.json", '{"data": [], "layout": {}}')
    assert "error" not in out
    assert out["chart_note"] == HAND_CHART_NOTE
    assert "plot_data" in out["chart_note"]


def test_the_png_beside_it_is_named(tmp_path):
    (tmp_path / "map.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    out = _save(tmp_path / "map.plotly.json", '{"data": [], "layout": {}}')
    assert "map.png beside it is now a different figure" in out["chart_note"]


def test_other_files_carry_no_chart_note(tmp_path):
    out = _save(tmp_path / "notes.json", '{"a": 1}')
    assert "error" not in out and "chart_note" not in out


def test_the_calc_helper_can_look_at_its_figures():
    from funhouse_agent.deep.agent import build_calc_subagent
    spec = build_calc_subagent()
    names = {t.name for t in spec["tools"]}
    assert "analyze_image" in names
    assert "save_file" in names
