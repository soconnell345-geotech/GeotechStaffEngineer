"""A file rewritten in a later turn is carded in both turns; the page must
still draw.

Field session 2026-10-15: the agent wrote a README in one turn and rewrote it
in the next. Both turns carry a card for the same path, the download button's
key was ``dlcard_<path>``, and Streamlit stopped the page with
``StreamlitDuplicateElementKey`` -- the conversation could not be answered
again. A chart shown twice has the same trouble with ``st.plotly_chart``.
"""

import json
import os

import pytest

pytest.importorskip("streamlit")

import webapp.core as core
from webapp.tests.test_app_smoke import _mk_at

_TURN = {"n": 0}


def _stream_rewrites(_agent, _messages, thread_id, **_kw):
    _TURN["n"] += 1
    n = _TURN["n"]
    files = core.conversation_files_dir(thread_id)
    os.makedirs(files, exist_ok=True)
    with open(os.path.join(files, "README_build.md"), "w",
              encoding="utf-8") as fh:
        fh.write(f"# Build notes, revision {n}\n")
    fig = {"data": [{"type": "scatter", "x": [0, n], "y": [0, 1]}],
           "layout": {"title": {"text": "same chart"}}}
    with open(os.path.join(files, "plan.plotly.json"), "w",
              encoding="utf-8") as fh:
        json.dump(fig, fh)
    yield {"kind": "token", "text": f"revision {n}"}
    yield {"kind": "turn_done", "answer": f"revision {n}", "turn_tokens": 2}


def test_a_file_carded_in_two_turns_does_not_stop_the_page(monkeypatch,
                                                            tmp_path):
    _TURN["n"] = 0
    at = _mk_at(monkeypatch, tmp_path, _stream_rewrites).run()
    at.chat_input[0].set_value("write the build").run()
    assert not at.exception
    at.chat_input[0].set_value("revise the build").run()
    assert not at.exception, [e.value for e in at.exception]

    carded = [os.path.basename(p)
              for e in at.session_state["transcript"]
              for p in e.get("artifacts", [])]
    # the same README is carded in both turns: the case that crashed
    assert carded.count("README_build.md") == 2
    # and a plain rerun (what every later message does) still draws
    at.run()
    assert not at.exception
    keys = [b.key for b in at.get("download_button")]
    assert len(keys) == len(set(keys))


def test_charts_are_drawn_in_their_own_colours(monkeypatch, tmp_path):
    """Streamlit's theme recoloured chart text for the dark page, so titles,
    axis labels and legends vanished on a chart's light paper (field session
    2026-10-15). The card draws the figure as written, like its PNG."""
    _TURN["n"] = 0
    at = _mk_at(monkeypatch, tmp_path, _stream_rewrites).run()
    at.chat_input[0].set_value("write the build").run()
    assert not at.exception
    charts = at.get("plotly_chart")
    assert charts, "the chart card was not drawn"
    assert all(c.proto.theme == "" for c in charts)
