"""The locked box after live wave w9 (BACKLOG GM-107).

``module_work/live_smoke/runs/w9-figures/TRACE_REVIEW.md``: a plot_data
draft's chart stood in for the figure redrawn in code (fix 1); a draft the
helper called superseded kept its card (fix 2); looks were asked about the
lettering only (fix 3); a helper drew six times against a cap of three
(fix 7); lettering printed at 4.6-5.6 pt and nothing measured it (fix 12);
the look's 2,048 px cap also capped the report copy (fix 15). Every run here
is a real child process.
"""

import json
import os
import time

import pytest

pytest.importorskip("matplotlib")

from funhouse_agent import plot_box  # noqa: E402
from funhouse_agent.deep import keep_work  # noqa: E402
from funhouse_agent.plot_box import retire_drafts, run_plot_code  # noqa: E402

DRAW = ("import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\n"
        "ax.plot([0, 1], [0, 1])\nax.set_xlabel('x (m)')\n"
        "save_figure(fig, {name!r})\n")


def _draw(root, name="fig", **kw):
    return run_plot_code(DRAW.format(name=name), root=str(root), **kw)


def _old(path, seconds=3600):
    """Date a file ``seconds`` back."""
    t = time.time() - seconds
    os.utime(path, (t, t))


# ---------------------------------------------------------------------------
# fix 1: a redraw in code is not hidden behind an older chart
# ---------------------------------------------------------------------------

def test_a_redraw_removes_the_chart_an_earlier_draw_left(tmp_path):
    with keep_work.turn_scope():
        time.sleep(0.05)
        (tmp_path / "gsd.png").write_bytes(b"draft png")
        (tmp_path / "gsd.plotly.json").write_text('{"data": []}')
        r = _draw(tmp_path, "gsd")
    assert r["status"] == "success", r
    assert not (tmp_path / "gsd.plotly.json").exists()
    assert any("removed gsd.plotly.json" in n for n in r["notes"])


def test_a_chart_from_an_earlier_turn_stays_with_that_turn(tmp_path):
    (tmp_path / "gsd.plotly.json").write_text('{"data": []}')
    _old(tmp_path / "gsd.plotly.json")
    with keep_work.turn_scope():
        r = _draw(tmp_path, "gsd")
    assert r["status"] == "success", r
    assert (tmp_path / "gsd.plotly.json").exists()


def test_a_chart_saved_with_its_png_in_one_run_is_the_one_shown(tmp_path):
    """The code chose to save a chart: it stays, dated with its picture, so
    the chat (which shows the newer of the two) still shows the chart."""
    code = ("import plotly.graph_objects as go\nimport matplotlib.pyplot as plt\n"
            "save_chart(go.Figure(go.Scatter(x=[1, 2], y=[3, 4])), 'c')\n"
            "f, ax = plt.subplots(); ax.plot([1, 2], [3, 4], 'o')\n"
            "save_figure(f, 'c')\n")
    r = run_plot_code(code, root=str(tmp_path))
    assert r["status"] == "success", r
    chart, png = tmp_path / "c.plotly.json", tmp_path / "c.png"
    assert chart.exists()
    assert os.stat(chart).st_mtime_ns >= os.stat(png).st_mtime_ns


# ---------------------------------------------------------------------------
# fix 2: replaces= retires a draft drawn under another name
# ---------------------------------------------------------------------------

def test_replaces_retires_this_turns_draft_and_its_chart(tmp_path):
    with keep_work.turn_scope():
        time.sleep(0.05)
        (tmp_path / "draft.png").write_bytes(b"x")
        (tmp_path / "draft.plotly.json").write_text('{"data": []}')
        r = _draw(tmp_path, "final", replaces=["draft.png"])
    assert r["status"] == "success", r
    assert sorted(r["retired"]) == ["draft.plotly.json", "draft.png"]
    assert not (tmp_path / "draft.png").exists()
    assert not (tmp_path / "draft.plotly.json").exists()
    assert (tmp_path / "final.png").exists()


def test_replaces_refuses_what_is_not_a_draft_of_this_turn(tmp_path):
    mine = tmp_path / "mine"
    other = tmp_path / "other"
    mine.mkdir()
    other.mkdir()
    (other / "x.png").write_bytes(b"theirs")
    (mine / "spt.csv").write_text("depth,N\n1,5\n")
    (mine / "upload.png").write_bytes(b"the user's")
    _old(mine / "upload.png")
    with keep_work.turn_scope():
        r = _draw(mine, "final", replaces=[
            str(other / "x.png"), "../other/x.png", "spt.csv",
            "upload.png", "final.png", "nothing_here.png"])
    assert r["status"] == "success", r
    why = {d["name"]: d["why"] for d in r["not_retired"]}
    assert "outside the working folder" in why[str(other / "x.png")]
    assert "outside the working folder" in why["../other/x.png"]
    assert "not a picture or chart" in why["spt.csv"]
    assert "before this turn" in why["upload.png"]
    assert "the figure this call drew" in why["final.png"]
    assert "no such figure" in why["nothing_here.png"]
    assert "retired" not in r
    assert (other / "x.png").exists() and (mine / "spt.csv").exists()
    assert (mine / "upload.png").exists() and (mine / "final.png").exists()


def test_a_failed_draw_retires_nothing(tmp_path):
    (tmp_path / "draft.png").write_bytes(b"x")
    r = run_plot_code("raise ValueError('no')", root=str(tmp_path),
                      replaces=["draft.png"])
    assert r["status"] == "error"
    assert (tmp_path / "draft.png").exists()
    assert any("nothing was retired" in n for n in r["notes"])


def test_retire_drafts_takes_a_name_without_its_extension(tmp_path):
    since = time.time() - 60
    (tmp_path / "plan_v1.png").write_bytes(b"x")
    (tmp_path / "plan_v1.plotly.json").write_text("{}")
    (tmp_path / "plan_v1.svg").write_text("<svg/>")
    out = retire_drafts(["plan_v1"], root=str(tmp_path), since=since)
    assert sorted(out["retired"]) == ["plan_v1.plotly.json", "plan_v1.png",
                                      "plan_v1.svg"]
    assert retire_drafts([], root=str(tmp_path)) == {}
    assert "no working folder" in json.dumps(retire_drafts(["a.png"],
                                                           root=""))


def test_with_no_turn_the_helper_run_dates_the_drafts(tmp_path):
    """A host that opens no turn scope (the eval harness): the running
    helper's start tells drafts from older files; with neither, nothing is
    removed, uploads included."""
    (tmp_path / "upload.png").write_bytes(b"the user's")
    _old(tmp_path / "upload.png")
    (tmp_path / "chart.plotly.json").write_text("{}")
    out = retire_drafts(["upload.png"], root=str(tmp_path))
    assert "nothing was removed" in out["not_retired"][0]["why"]
    assert (tmp_path / "upload.png").exists()
    # plot_code outside any turn or helper keeps a same-name chart
    r = _draw(tmp_path, "chart")
    assert r["status"] == "success" and (tmp_path / "chart.plotly.json").exists()
    rec = keep_work.HelperRecord("t1", "figure", "draw it")
    token = keep_work._HELPER.set(rec)
    try:
        time.sleep(0.05)
        (tmp_path / "draft.png").write_bytes(b"x")
        out = retire_drafts(["upload.png", "draft.png"], root=str(tmp_path))
    finally:
        keep_work._HELPER.reset(token)
    assert out["retired"] == ["draft.png"]
    assert "before this turn" in out["not_retired"][0]["why"]
    assert (tmp_path / "upload.png").exists()


# ---------------------------------------------------------------------------
# fixes 3 and 7: a look about content, and one count of draws per helper
# ---------------------------------------------------------------------------

def test_next_asks_what_the_figure_shows_against_the_data(tmp_path):
    r = _draw(tmp_path)
    nxt = r["next"]
    for words in ("two or three plotted values", "compare them here with "
                  "the data", "a stair or a smoothed curve between readings",
                  "source's own words", "legend repeat", "layout_check"):
        assert words in nxt, words
    assert "redraw" in nxt.lower() and "at most twice" in nxt


def _helper_with_draws(n, helper="figure"):
    rec = keep_work.HelperRecord("t1", helper, "draw it")
    for k in range(n):
        tool = "plot_data" if k % 2 else "plot_code"
        rec.add_result(tool, f"c{k}", json.dumps({"status": "success"}))
    rec.add_result("plot_code", "bad", json.dumps({"status": "error"}))
    return rec


def test_the_third_draw_in_a_helper_asks_for_the_best_one(tmp_path,
                                                          monkeypatch):
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path))
    tool = plot_box.make_plot_code_tool()
    token = keep_work._HELPER.set(_helper_with_draws(2))
    try:
        out = json.loads(tool.invoke({"code": DRAW.format(name="v3")}))
    finally:
        keep_work._HELPER.reset(token)
    assert out["draw"] == 3
    assert "hand back the best version" in out["next"]
    assert "Do not draw again" in out["next"]


def test_a_first_draw_may_be_redrawn_and_says_how_many(tmp_path,
                                                       monkeypatch):
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path))
    tool = plot_box.make_plot_code_tool()
    token = keep_work._HELPER.set(_helper_with_draws(0))
    try:
        out = json.loads(tool.invoke({"code": DRAW.format(name="v1")}))
    finally:
        keep_work._HELPER.reset(token)
    assert out["draw"] == 1 and "draw 1 of at most 3" in out["next"]
    assert "Do not draw again" not in out["next"]
    # outside a helper nothing is counted
    plain = _draw(tmp_path, "v1b")
    assert "draw" not in plain


def test_only_the_figure_helper_is_capped(tmp_path, monkeypatch):
    """The calc helper draws a package's several figures: its fourth
    separate figure is not told to stop drawing."""
    from funhouse_agent.deep.figure_helper import FIGURE_HELPER_NAME
    assert plot_box.CAPPED_HELPER == FIGURE_HELPER_NAME
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path))
    tool = plot_box.make_plot_code_tool()
    for helper in ("calc", "general-purpose"):
        token = keep_work._HELPER.set(_helper_with_draws(3, helper))
        try:
            out = json.loads(tool.invoke(
                {"code": DRAW.format(name=f"{helper}_fig4")}))
        finally:
            keep_work._HELPER.reset(token)
        assert out["status"] == "success", out
        assert "draw" not in out, helper
        assert "Do not draw again" not in out["next"], helper


def test_the_description_no_longer_says_redraw_until_it_is_right():
    desc = plot_box.plot_code_description()
    assert "redraw until it is right" not in desc
    assert "its drafts by plot_code and plot_data, under any name, count " \
        "together" in desc
    for words in ("replaces", "print_width_in", "look_file",
                  "never a stair"):
        assert words in desc, words
    tool = plot_box.make_plot_code_tool()
    assert {"replaces", "print_width_in"} <= set(
        tool.tool_call_schema.model_json_schema()["properties"])


# ---------------------------------------------------------------------------
# fix 12: the lettering's printed size
# ---------------------------------------------------------------------------

def test_lettering_is_measured_at_the_width_it_is_printed(tmp_path):
    code = ("import matplotlib.pyplot as plt\n"
            "fig, ax = plt.subplots(figsize=(13, 6))\n"
            "ax.plot([0, 1], [0, 1])\n"
            "ax.set_xlabel('x (m)', fontsize=9); ax.tick_params(labelsize=9)\n"
            "save_figure(fig, 'wide')\n")
    r = run_plot_code(code, root=str(tmp_path))
    assert r["status"] == "success", r
    (let,) = r["lettering"]
    assert let["file"] == "wide.png" and let["print_width_in"] == 6.5
    # 9 pt on a figure saved about 10.6 in wide (the tight crop), printed
    # 6.5 in wide
    assert let["smallest_pt"] < 6.0 and let["texts_under_7_pt"] > 0
    assert 9.5 < let["saved_width_in"] < 13.0
    assert " pt at 6.5 in" in let["says"]
    assert "under 7 pt" in r["next"]
    # printed at its own width it reads at its own size
    same = run_plot_code(code, root=str(tmp_path), print_width_in=13)
    assert same["lettering"][0]["smallest_pt"] > 8.0
    assert "under 7 pt" not in same["next"]


def test_a_figure_drawn_at_report_width_passes(tmp_path):
    code = ("import matplotlib.pyplot as plt\n"
            "fig, ax = plt.subplots(figsize=(6.5, 4))\n"
            "ax.plot([0, 1], [0, 1]); ax.tick_params(labelsize=8)\n"
            "ax.set_xlabel('x (m)', fontsize=9)\n"
            "save_figure(fig, 'narrow')\n")
    r = run_plot_code(code, root=str(tmp_path))
    (let,) = r["lettering"]
    assert let["texts_under_7_pt"] == 0 and let["smallest_pt"] >= 7.0


# ---------------------------------------------------------------------------
# fix 15: the report copy at full resolution, a look copy beside it
# ---------------------------------------------------------------------------

def test_the_report_png_keeps_the_dpi_asked(tmp_path):
    code = ("import matplotlib.pyplot as plt\n"
            "fig, ax = plt.subplots(figsize=(14, 6)); ax.plot([1, 2])\n"
            "save_figure(fig, 'profile', dpi=200)\n")
    r = run_plot_code(code, root=str(tmp_path))
    assert r["status"] == "success", r
    (img,) = r["images"]
    assert img["file"] == "profile.png"
    assert img["width_px"] > 2048            # about 13 in at 200 dpi
    assert img["look_file"] == ".scratch/profile_look.png"
    assert max(img["look_px"]) <= 2048
    assert (tmp_path / ".scratch" / "profile_look.png").is_file()
    # the look copy is not a second figure
    assert [i["file"] for i in r["images"]] == ["profile.png"]


def test_a_small_figure_has_no_look_copy(tmp_path):
    r = _draw(tmp_path, "small")
    (img,) = r["images"]
    assert "look_file" not in img
    assert not (tmp_path / ".scratch").exists()
