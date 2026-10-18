"""The locked box for plotting code (BACKLOG GM-98).

The owner, 2026-10-15: "I've been frustrated by the calc packages' lack of
creative figures... Please try to build it as securely as possible, don't
do things that would trip the package up"; "Don't limit the run time too
much, I want it to cook". Every run here is a real child process.
"""

import json
import os

import pytest

pytest.importorskip("matplotlib")

from funhouse_agent import plot_box, progress_notes, review_flags  # noqa: E402
from funhouse_agent.plot_box import child as box_child  # noqa: E402
from funhouse_agent.plot_box import run_plot_code  # noqa: E402


@pytest.fixture
def folders(tmp_path):
    """A conversation folder with an upload, and a neighbour's folder with
    something private in it."""
    mine = tmp_path / "mine"
    other = tmp_path / "other"
    mine.mkdir()
    other.mkdir()
    (mine / "spt.csv").write_text("depth,N\n1,5\n2,8\n3,12\n")
    (other / "secret.txt").write_text("TOP SECRET")
    return mine, other


def _run(code, root, **kw):
    return run_plot_code(code, root=str(root), **kw)


# ---------------------------------------------------------------------------
# the static check
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("code, words", [
    ("import os", "imports 'os'"),
    ("import subprocess", "imports 'subprocess'"),
    ("from matplotlib.cbook import os", "imports 'os' from"),
    ("from . import x", "imports from '.'"),
    ("print(getattr(1, 'real'))", "uses 'getattr'"),
    ("eval('1+1')", "uses 'eval'"),
    ("print(().__class__.__bases__)", "'__class__'"),
    ("print('{0.__class__}'.format(1))", "dunder"),
    ("import pandas as pd\npd.DataFrame().query('a>0')", "'.query'"),
    ("import numpy as np\nnp.ctypeslib", "'.ctypeslib'"),
    ("from string import Formatter", "imports from 'string'"),
    ("import typing", "imports 'typing'"),
])
def test_the_static_check_refuses_the_ways_out(code, words):
    problems = box_child.check_code(code)
    assert problems and any(words in p for p in problems), problems


def test_the_static_check_passes_ordinary_plotting_code():
    code = ("import numpy as np\nimport pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from matplotlib.patches import Polygon\n"
            "from scipy.interpolate import griddata\n"
            "from mpl_toolkits.axes_grid1 import make_axes_locatable\n"
            "import plotly.graph_objects as go\nimport math, statistics\n"
            "class Wall:\n    def __init__(self, h):\n        self.h = h\n"
            "fig, ax = plt.subplots()\nax.set_title(f'{math.pi:.2f}')\n")
    assert box_child.check_code(code) == []


def test_a_refused_script_costs_no_process(folders, monkeypatch):
    mine, _ = folders

    def no_popen(*a, **k):
        raise AssertionError("a process was started")
    monkeypatch.setattr(plot_box.subprocess, "Popen", no_popen)
    r = _run("import os\nos.system('x')", mine)
    assert r["status"] == "error" and "was not run" in r["error"]


# ---------------------------------------------------------------------------
# what the box makes
# ---------------------------------------------------------------------------

def test_a_figure_is_drawn_from_a_file_and_the_data(folders):
    mine, _ = folders
    code = (
        "import pandas as pd\nimport matplotlib.pyplot as plt\n"
        "df = pd.read_csv('spt.csv')\n"
        "fig, (a, b) = plt.subplots(1, 2, sharey=True)\n"
        "a.plot(df.N, df.depth, 'o-'); a.invert_yaxis()\n"
        "b.plot(data['su'], df.depth, 's')\n"
        "print('rows', len(df))\n"
        "save_figure(fig, 'profile')\n")
    r = _run(code, mine, data={"su": [20, 30, 45]})
    assert r["status"] == "success", r
    (img,) = r["images"]
    assert img["file"] == "profile.png" and os.path.isfile(img["output_path"])
    assert img["width_px"] > 100 and img["height_px"] > 100
    assert img["html_img_tag"].startswith("<img src=")
    assert "rows 3" in r["printed"]
    assert "analyze_image" in r["next"]


def test_a_figure_left_open_is_saved_for_the_code(folders):
    mine, _ = folders
    r = _run("import matplotlib.pyplot as plt\nplt.plot([1, 2], [3, 4])",
             mine, name="left open")
    assert r["status"] == "success", r
    assert [i["file"] for i in r["images"]] == ["left open.png"]
    assert any("left open" in n for n in r["notes"])


def test_a_plotly_chart_is_saved_beside_its_png(folders):
    mine, _ = folders
    code = ("import plotly.graph_objects as go\n"
            "import matplotlib.pyplot as plt\n"
            "save_chart(go.Figure(go.Scatter(x=[1, 2], y=[3, 4])), 'cases')\n"
            "f, ax = plt.subplots(); ax.plot([1, 2], [3, 4], 'o')\n"
            "save_figure(f, 'cases')\n")
    r = _run(code, mine)
    assert r["status"] == "success", r
    assert [c["file"] for c in r["charts"]] == ["cases.plotly.json"]
    with open(r["charts"][0]["output_path"], encoding="utf-8") as fh:
        assert "data" in json.load(fh)


def test_the_plan_furniture_is_in_the_box(folders):
    mine, _ = folders
    code = ("import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\n"
            "ax.plot([0, 40], [0, 20]); ax.set_aspect('equal')\n"
            "ax.set_axis_off()\n"
            "scale_bar(ax, 2, -3, nice_length(40), 0.6, 'm')\n"
            "north_arrow(ax, (36, 16), 25)\n"
            "save_figure(fig, 'plan')\n")
    r = _run(code, mine)
    assert r["status"] == "success", r


def test_a_redraw_overwrites_its_own_png(folders):
    mine, _ = folders
    code = ("import matplotlib.pyplot as plt\nplt.plot([1, 2])\n"
            "save_figure(name='same')\n")
    assert _run(code, mine)["status"] == "success"
    again = _run(code, mine)
    assert again["status"] == "success", again
    assert [i["file"] for i in again["images"]] == ["same.png"]


def test_code_that_saves_nothing_is_told_how(folders):
    mine, _ = folders
    r = _run("x = 1 + 1", mine)
    assert r["status"] == "error" and "save_figure" in r["error"]


def test_the_codes_own_error_names_its_line(folders):
    mine, _ = folders
    r = _run("import numpy as np\nx = 1\ny = np.zeros(3) + [1, 2]\n", mine)
    assert r["status"] == "error"
    assert "your code, line 3" in r["error"]
    assert "child.py" not in r["error"]


# ---------------------------------------------------------------------------
# the walls
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("make_code", [
    lambda s: f"print(open({s!r}).read())",
    lambda s: f"import pandas as pd\nprint(pd.read_csv({s!r}))",
    lambda s: f"import numpy as np\nprint(np.loadtxt({s!r}, dtype=str))",
    lambda s: "print(open('../other/secret.txt').read())",
])
def test_nothing_outside_the_conversation_folder_is_read(folders, make_code):
    mine, other = folders
    r = _run(make_code(str(other / "secret.txt")), mine)
    assert r["status"] == "error"
    assert "outside the conversation folder" in r["error"]
    assert "TOP SECRET" not in json.dumps(r)


def test_no_folder_outside_is_listed(folders):
    mine, other = folders
    code = ("import matplotlib.pyplot as plt\n"
            "from matplotlib import font_manager\n"
            f"print(font_manager.list_fonts({str(other)!r}, ['txt']))\n")
    r = _run(code, mine)
    assert r["status"] == "error"
    assert "secret" not in r.get("printed", "")


@pytest.mark.parametrize("code, words", [
    ("import matplotlib.pyplot as plt\nplt.plot([1])\n"
     "plt.savefig('../other/x.png')", "outside the conversation folder"),
    ("open('evil.py', 'w').write('x = 1')", "figure, chart and data files"),
    ("import numpy as np\nnp.save('x.npy', np.ones(3))",
     "figure, chart and data files"),
    ("open('spt.csv', 'w').write('gone')", "there before this run"),
])
def test_writes_stay_in_the_folder_and_to_figures(folders, code, words):
    mine, other = folders
    r = _run(code, mine)
    assert r["status"] == "error" and words in r["error"], r
    assert sorted(os.listdir(other)) == ["secret.txt"]
    assert (mine / "spt.csv").read_text().startswith("depth,N")
    assert not (mine / "evil.py").exists()


def test_no_network(folders):
    mine, _ = folders
    r = _run("import pandas as pd\npd.read_csv('http://127.0.0.1:9/x.csv')",
             mine)
    assert r["status"] == "error"
    assert "refused in the plotting box" in r["error"]


def test_the_child_gets_no_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("PROMPTER_API_KEY", "sk-not-for-the-box")
    monkeypatch.setenv("GRAPH_CLIENT_SECRET", "nope")
    env = plot_box._child_env(str(tmp_path))
    assert "PROMPTER_API_KEY" not in env and "GRAPH_CLIENT_SECRET" not in env
    assert env["MPLBACKEND"] == "Agg" and env["TMP"] == str(tmp_path)
    assert not any("KEY" in k or "SECRET" in k or "TOKEN" in k for k in env)


def test_the_time_limit_stops_a_loop(folders, monkeypatch):
    mine, _ = folders
    monkeypatch.setattr(plot_box, "MIN_SECONDS", 3.0)
    r = _run("while True:\n    pass", mine, time_limit_s=3)
    assert r["status"] == "error" and "time limit" in r["error"]
    assert r["time_limit_s"] == 3


def test_the_memory_ceiling_holds(folders, monkeypatch):
    mine, _ = folders
    monkeypatch.setenv(plot_box.MEMORY_ENV, "1024")
    r = _run("import numpy as np\nx = np.ones((16000, 16000))\n"
             "print(x.sum())", mine)
    assert r["status"] == "error", r
    assert "memory" in r["error"].lower()


def test_a_long_drawing_leaves_a_note_for_the_label(folders, monkeypatch):
    mine, _ = folders
    seen = []
    monkeypatch.setattr(plot_box, "NOTE_EVERY_S", 0.05)
    monkeypatch.setattr(progress_notes, "set_note",
                        lambda text, key=None: seen.append(text))
    r = _run("import matplotlib.pyplot as plt\nplt.plot([1, 2])", mine)
    assert r["status"] == "success"
    assert seen and all(t.startswith("drawing with code: ") for t in seen)


# ---------------------------------------------------------------------------
# settings, the tool, the switch
# ---------------------------------------------------------------------------

def test_settings_and_the_ceiling(monkeypatch, folders):
    mine, _ = folders
    for env in plot_box.SETTINGS_ENVS:
        monkeypatch.delenv(env, raising=False)
    assert plot_box.settings() == {"seconds": 600, "max_seconds": 3600,
                                   "memory_mb": 2048, "at_once": 2}
    monkeypatch.setenv(plot_box.MAX_SECONDS_ENV, "120")
    r = _run("import matplotlib.pyplot as plt\nplt.plot([1])", mine,
             time_limit_s=5000)
    assert r["time_limit_s"] == 120
    assert any("ceiling" in n for n in r["notes"])
    # the settings are cleared with the switches, so an arm is exact
    for env in plot_box.SETTINGS_ENVS:
        assert env in review_flags.SETTINGS_ENVS


def test_the_tool_describes_the_box_and_refuses_unknown_arguments():
    tool = plot_box.make_plot_code_tool()
    assert tool.name == "plot_code"
    for words in ("LOCKED BOX", "save_figure", "save_chart", "scale_bar",
                  "time_limit_s", "no network",
                  # the owner's conventions (GM-97, 2026-10-15)
                  "panels sharing one elevation axis",
                  "a profile against elevation rising upward",
                  "stresses in kPa, forces in kN",
                  "a wall or pile drawn to scale with its diagrams",
                  "test results as points"):
        assert words in tool.description, words
    assert "depth increasing downward" not in tool.description
    out = tool.invoke({"code": "x = 1", "colour": "red"})
    assert "colour" in str(out)


def test_a_png_fits_one_look(folders):
    """Live wave w8: 200 dpi saved 2538 x 1546 px and every look tiled. Since
    wave w9 the report copy keeps the dpi asked and the look reads a copy
    that one look takes whole."""
    mine, _ = folders
    r = _run("import matplotlib.pyplot as plt\n"
             "fig, ax = plt.subplots(figsize=(14, 9)); ax.plot([1, 2])\n"
             "save_figure(fig, 'wide')", mine)
    assert r["status"] == "success", r
    (img,) = r["images"]
    assert max(img["look_px"]) <= 2100
    assert (mine / img["look_file"]).is_file()
    assert "its look_file" in r["next"]


def test_the_layout_check_measures_what_a_look_can_miss(folders):
    """Live wave w8: a tile cleared a label the water-table line ran
    through. The box measures it from the drawing."""
    mine, _ = folders
    code = ("import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\n"
            "ax.plot([0, 10], [0, 10])\n"
            "ax.axhline(5, label='water table')\n"
            "ax.text(5, 5, 'on the line', ha='center', va='center')\n"
            "ax.text(2, 8, 'first label'); ax.text(2.1, 8.05, 'second label')\n"
            "save_figure(fig, 'crowded')")
    r = _run(code, mine)
    assert r["status"] == "success", r
    (check,) = r["layout_check"]
    assert check["file"] == "crowded.png"
    assert ["first label", "second label"] in check["text_overlaps"]
    assert any(t == "on the line" for t, _ in check["text_on_a_line"])
    assert "layout_check" in r["next"]
    # the check's time per saved figure (GM-108)
    (seconds,) = r["box"]["layout_check_seconds"]
    assert 0 <= seconds < 30


def _layout(mine, body, name="lay"):
    r = _run("import matplotlib.pyplot as plt\n"
             "from matplotlib.patches import Rectangle\n" + body
             + f"\nsave_figure(fig, {name!r})", mine)
    assert r["status"] == "success", r
    return (r.get("layout_check") or [{}])[0]


def test_layout_ignores_tick_labels_that_are_never_drawn(folders):
    """Live wave w8b: a "100" beyond a panel's limits was reported against
    the next panel's "0"; matplotlib keeps that label but never draws it."""
    mine, _ = folders
    chk = _layout(mine, (
        "fig, (a, b) = plt.subplots(1, 2, figsize=(4, 3))\n"
        "fig.subplots_adjust(wspace=0)\n"
        "a.set_xticks([0, 50, 100]); a.set_xlim(0, 60)\n"
        "b.set_xlim(0, 60); b.set_yticks([])\n"))
    pairs = chk.get("text_overlaps", [])
    assert not any("100" in p for p in pairs), pairs


def test_layout_measures_an_annotation_by_its_text(folders):
    """Live wave w8b: an annotation's box took in its leader line."""
    mine, _ = folders
    chk = _layout(mine, (
        "fig, ax = plt.subplots(); ax.set_xlim(0, 2); ax.set_ylim(0, 1)\n"
        "ax.annotate('stem 0.35', xy=(1, 0.1), xytext=(1, 0.85), ha='center',"
        " arrowprops={'arrowstyle': '-'})\n"
        "ax.text(1.05, 0.4, 'toe 0.70')\n"))
    assert not chk.get("text_overlaps"), chk


def test_layout_counts_labels_that_touch(folders):
    """Live wave w8b: "200" against "0" with no gap reads "2000"."""
    mine, _ = folders
    chk = _layout(mine, (
        "fig, ax = plt.subplots(); ax.set_xlim(0, 10); ax.set_ylim(0, 10)\n"
        "ax.text(5, 5, '200', ha='right'); ax.text(5, 5, '0', ha='left')\n"))
    assert ["200", "0"] in chk.get("text_overlaps", []), chk


def test_layout_finds_an_edge_or_frame_through_lettering(folders):
    """Live wave w8b: a pressure diagram's edge and the axes frame ran
    through labels the check passed. (The frame label sits between ticks:
    on a tick the check names the tick mark, the line that shows most.)"""
    mine, _ = folders
    chk = _layout(mine, (
        "fig, ax = plt.subplots(); ax.set_xlim(0, 10); ax.set_ylim(0, 10)\n"
        "ax.add_patch(Rectangle((1, 1), 4, 4, facecolor='0.9',"
        " edgecolor='k'))\n"
        "ax.text(5, 3, 'across the edge', ha='center')\n"
        "ax.text(3, 2, 'inside', ha='center', fontsize=6)\n"
        "ax.text(0, 7, 'on the frame', ha='center')\n"))
    crossed = {t: what for t, what in chk.get("text_on_a_line", [])}
    assert "across the edge" in crossed and "inside" not in crossed, chk
    assert "frame" in crossed.get("on the frame", ""), chk


def test_layout_finds_a_leader_through_lettering(folders):
    """Review of GM-109: an annotation draws its leader apart from its text,
    and no leader was measured, so one across another label passed."""
    mine, _ = folders
    chk = _layout(mine, (
        "fig, ax = plt.subplots(); ax.set_xlim(0, 10); ax.set_ylim(0, 10)\n"
        "ax.text(5, 5, 'Other label', ha='center', va='center')\n"
        "ax.annotate('Note', xy=(9.5, 5), xytext=(0.5, 5), va='center',"
        " arrowprops={'arrowstyle': '-|>', 'lw': 1.2})\n"))
    crossed = {t: what for t, what in chk.get("text_on_a_line", [])}
    assert crossed.get("Other label") == "the leader of 'Note'", chk
    assert "Note" not in crossed, chk


# ---------------------------------------------------------------------------
# the layout check judged by the pixels drawn (GM-108)
# ---------------------------------------------------------------------------
# Live wave w9: 22 of the check's 36 reports were false while it judged
# lettering by boxes, and it never tested lettering outside the axes (its
# trace review, §2.4 and §9). Each class is drawn here in the test process,
# without the box, through the box's own check.

def _figure(w=5, h=4):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(w, h))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    return fig, ax


def _found(fig):
    out = box_child._layout_check(fig, "x.png")
    assert "unchecked" not in out, out
    return ({tuple(p) for p in out.get("text_overlaps", [])},
            dict(out.get("text_on_a_line", [])),
            dict(out.get("text_cut_off", [])))


def test_layout_judges_rotated_lettering_by_its_letters():
    """w9: "A-line" (33 degrees) and "U-line" (45) in four runs, and
    "envelope" (29), were reported because their upright boxes took in the
    line they ran beside; no line showed on their letters."""
    import math
    fig, ax = _figure()
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.plot([0, 10], [0, 6.5], "k-")
    slope = math.degrees(math.atan(0.65))
    ax.text(4, 2.85, "beside the line", rotation=slope,
            rotation_mode="anchor", ha="center", va="bottom")
    ax.text(7, 4.55, "on the line", rotation=slope, rotation_mode="anchor",
            ha="center", va="center")
    _pairs, crossed, _cut = _found(fig)
    assert "beside the line" not in crossed, crossed
    assert crossed.get("on the line") == "a line", crossed


def test_layout_lets_a_legend_frame_hide_the_lines_behind_it():
    """w9: all three legend entries of a grain-size chart, in both runs,
    were reported "on a line": dividers the legend's opaque frame hid."""
    fig, ax = _figure()
    for x in (1, 2, 3):
        ax.axvline(x, color="0.55", ls="--")
    ax.plot([0, 10], [5, 5], label="first curve")
    ax.plot([0, 10], [6, 6], label="second curve")
    ax.legend(loc="lower left", framealpha=1)
    pairs, crossed, _cut = _found(fig)
    assert not crossed and not pairs, (crossed, pairs)


def test_legend_entries_are_lettering_too():
    """Without a frame a line shows through a legend entry; lettering under
    an opaque legend meets "the legend"."""
    fig, ax = _figure()
    ax.plot([0, 10], [2, 2], label="first curve")
    ax.axhline(5, color="k")
    ax.legend(loc="center", bbox_to_anchor=(0.5, 0.5), frameon=False)
    _pairs, crossed, _cut = _found(fig)
    assert crossed.get("first curve") == "a line", crossed
    fig, ax = _figure()
    ax.plot([0, 10], [5, 5], label="first curve")
    ax.text(1.0, 0.9, "hidden note")
    ax.legend(loc="lower left", framealpha=1)
    pairs, _crossed, _cut = _found(fig)
    assert pairs == {("hidden note", "the legend")}, pairs


def test_layout_lets_a_box_hide_the_line_beneath():
    """w9: the N60 values "4" and "16", drawn on white boxes above the
    contact lines, were reported in all four runs. A line drawn over the box
    still runs through, and so does one showing through a box 85 % white
    ("El. 98.2", which the look confirmed)."""
    fig, ax = _figure()
    ax.axhline(8, color="k", zorder=2)
    ax.text(5, 8, "boxed over", ha="center", va="center", zorder=3,
            bbox={"fc": "white", "ec": "none", "pad": 2})
    ax.axhline(6, color="k", zorder=5)
    ax.text(5, 6, "line over box", ha="center", va="center", zorder=3,
            bbox={"fc": "white", "ec": "none", "pad": 2})
    ax.axhline(4, color="#1f4fbf", zorder=2)
    ax.text(5, 4, "El. 98.2", ha="center", va="center", zorder=8,
            bbox={"fc": "white", "ec": "none", "alpha": 0.85, "pad": 0.8})
    _pairs, crossed, _cut = _found(fig)
    assert "boxed over" not in crossed, crossed
    assert "line over box" in crossed and "El. 98.2" in crossed, crossed


def test_a_box_is_its_lettering_s_footprint():
    """w9: a white-boxed "24" against "Medium": 5 px between the letters,
    but the box ran up to the M and the two read as one word."""
    fig, ax = _figure()
    ax.text(5, 2, "24", ha="right", va="center", fontsize=7.5,
            bbox={"fc": "white", "ec": "none", "alpha": 0.8, "pad": 0.6})
    ax.text(5.05, 2, "Medium", ha="left", va="center", style="italic",
            fontsize=9)
    ax.text(5, 7, "on its own", ha="right", va="center",
            bbox={"fc": "white", "ec": "none", "pad": 0.6})
    ax.text(6.5, 7, "apart", ha="left", va="center")
    pairs, _crossed, _cut = _found(fig)
    assert pairs == {("24", "Medium")}, pairs


def test_layout_tests_lettering_outside_the_axes():
    """w9 (the miss): "Cobbles", in a strip above the axes, crossed by a
    divider drawn with clip_on=False; text outside the axes was never
    tested. Titles outside the axes overlap like any lettering."""
    fig, ax = _figure()
    strip = ax.get_xaxis_transform()
    for x, word in ((2, "Cobbles"), (6, "Gravel")):
        ax.text(x, 1.08, word, transform=strip, ha="center", va="center",
                clip_on=False)
    for x in (2.1, 4):
        ax.plot([x, x], [1.0, 1.12], transform=strip, color="0.55",
                clip_on=False)
    _pairs, crossed, _cut = _found(fig)
    assert crossed == {"Cobbles": "a line"}, crossed
    fig, ax = _figure()
    ax.set_title("Panel title here")
    fig.suptitle("Figure title", y=0.93)
    pairs, _crossed, _cut = _found(fig)
    assert pairs == {("Panel title here", "Figure title")}, pairs


def test_layout_finds_lettering_cut_off():
    """Lettering its axes clips, and lettering outside the saved figure."""
    fig, ax = _figure()
    ax.text(9.6, 5, "clipped words", clip_on=True)
    ax.text(2, 5, "whole words", clip_on=True)
    beyond = fig.text(0.98, 0.5, "beyond the edge")
    beyond.set_in_layout(False)
    _pairs, _crossed, cut = _found(fig)
    assert cut == {"clipped words": "the edge of its axes",
                   "beyond the edge": "the figure edge"}, cut


def test_layout_still_finds_true_collisions():
    """What the w9 review confirmed true stays found: a line through
    lettering, letters overlapping, labels touching on one line, an edge
    and a frame through lettering; markers on a label are data ink too."""
    from matplotlib.patches import Rectangle
    fig, ax = _figure()
    ax.axhline(5, label="water table")
    ax.text(5, 5, "on the line", ha="center", va="center")
    ax.text(2, 8, "first label")
    ax.text(2.1, 8.05, "second label")
    ax.text(5, 3, "200", ha="right")
    ax.text(5, 3, "0", ha="left")
    ax.scatter([8.5], [7.15], s=60, zorder=3)
    ax.text(8, 7, "on a point")
    ax.add_patch(Rectangle((6, 0.5), 2, 1.5, facecolor="0.9",
                           edgecolor="k"))
    ax.text(8, 1.2, "across the edge", ha="center")
    ax.text(0, 9.3, "on the frame", ha="center")
    pairs, crossed, _cut = _found(fig)
    assert {("first label", "second label"), ("200", "0")} <= pairs, pairs
    assert crossed.get("on the line") == "water table", crossed
    assert crossed.get("on a point") == "markers", crossed
    assert crossed.get("across the edge") == "an outline", crossed
    assert crossed.get("on the frame") == "the left frame line", crossed


def test_the_layout_check_leaves_matplotlib_as_it_was():
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.text import Text
    before = [vars(k)["draw"] for k in (Text, Line2D, Patch)]
    fig, ax = _figure()
    ax.plot([0, 10], [0, 10])
    ax.text(5, 5, "label", bbox={"fc": "white"})
    _found(fig)
    assert [vars(k)["draw"] for k in (Text, Line2D, Patch)] == before
    assert not fig.artists
    assert fig.texts == [] and ax.texts[0].get_bbox_patch().get_visible()


def _legend_over_lines(n_lines=40, entries=6):
    import numpy as np
    fig, ax = _figure()
    x = np.linspace(0, 10, 40)
    for k in range(n_lines):
        ax.plot(x, 0.25 * k + 0.3 * np.sin(x + k), color="0.6", lw=0.5)
    handles = [ax.plot([], [], label=f"entry {k}")[0]
               for k in range(entries)]
    ax.legend(handles=handles, loc="center", framealpha=1)
    return fig


def test_an_opaque_legend_keeps_only_its_own_drawing(monkeypatch):
    """Review of GM-108: every line was kept again for the legend's frame,
    where only the frame's own drawing is read (an 18-entry legend over
    1,500 lines held 4.9 GB, past the box's 2 GB); and past a memory budget
    the check says it went unchecked."""
    from matplotlib.text import Text
    seen = {}
    found = box_child._Ink.found

    def keep(self):
        seen["ink"] = self
        return found(self)
    monkeypatch.setattr(box_child._Ink, "found", keep)
    _found(_legend_over_lines())
    ink = seen["ink"]
    owners = {k: i for i, k in ink.region.items()}
    frames = [k for k, i in owners.items()
              if not isinstance(ink.leaves[i][0], Text)]
    assert frames, "no legend frame region"
    for k in frames:
        assert {leaf for _o, _kd, leaf, _c in ink.layers[k]} == {owners[k]}
    monkeypatch.setattr(box_child, "LAYOUT_MB", 0)
    out = box_child._layout_check(_legend_over_lines(), "x.png")
    assert "in memory" in out["unchecked"], out
    assert set(out) == {"file", "unchecked"}


def test_the_layout_check_gives_up_in_time(monkeypatch):
    """Review of GM-108: 3,000 texts took 11 s a save. Past its time the
    check says so instead of slowing the save, and leaves matplotlib as it
    was."""
    from matplotlib.collections import Collection
    from matplotlib.text import Text
    before = [vars(k)["draw"] for k in (Text, Collection)]
    monkeypatch.setattr(box_child, "LAYOUT_SECONDS", 0.0)
    fig, ax = _figure()
    for k in range(30):
        ax.text(k % 10, k // 3, f"P{k}")
    out = box_child._layout_check(fig, "x.png")
    assert "too many texts to check in time" in out["unchecked"], out
    assert [vars(k)["draw"] for k in (Text, Collection)] == before
    assert not fig.artists


def test_3d_lettering_is_measured_where_it_is_drawn():
    """Review of GM-108: 3D text is placed only while it draws, so it was
    measured at its raw coordinates: called cut off at the figure edge,
    with the line through it missed."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(5, 4))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(projection="3d")
    ax.plot([0, 1], [0, 1], [0, 1], color="k")
    mid = ax.text(0.5, 0.5, 0.5, "mid label", ha="center", va="center")
    ax.text(0.1, 0.9, 0.9, "clear label")
    _pairs, crossed, cut = _found(fig)
    assert crossed.get("mid label") == "a line", crossed
    assert "clear label" not in crossed, crossed
    assert not cut, cut
    assert (mid._x, mid._y) == (0.5, 0.5)       # left as it was


def test_a_mesh_is_drawing_too():
    """Review of GM-108: pcolormesh (and a colorbar's fill) drew outside the
    check: a mesh over lettering, and a white line through lettering on a
    mesh, went unseen."""
    import numpy as np
    fig, ax = _figure()
    ax.text(2.5, 2, "under the mesh", ha="center", va="center", zorder=1)
    X, Y = np.meshgrid(np.linspace(1.5, 3.5, 8), np.linspace(1.5, 2.5, 8))
    ax.pcolormesh(X, Y, X, zorder=5, shading="auto")
    X, Y = np.meshgrid(np.linspace(0, 10, 30), np.linspace(5, 10, 30))
    ax.pcolormesh(X, Y, X + Y, cmap="viridis", shading="auto")
    ax.axhline(7.5, color="white", lw=1.5)
    ax.text(5, 7.5, "white line through", ha="center", va="center")
    _pairs, crossed, _cut = _found(fig)
    assert crossed.get("under the mesh") == ("an image or filled area "
                                             "drawn over it"), crossed
    assert crossed.get("white line through") == "a line", crossed


def test_a_collection_s_edges_run_through_lettering():
    """Review of GM-108: fill_between's (and a PatchCollection's) edges
    counted as fill, so an edge through a label was missed. The fill under
    lettering still hides nothing; the colours are left as they were."""
    import numpy as np
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    fig, ax = _figure()
    band = ax.fill_between(np.linspace(0, 10, 30), 1, 3, facecolor="#f2dc8a",
                           edgecolor="k", lw=1.5)
    ax.text(3, 3, "on the edge", ha="center", va="center")
    ax.text(3, 2, "inside the fill", ha="center", va="center")
    boxes = PatchCollection([Rectangle((5, 5), 4, 3)], edgecolor="k",
                            lw=1.5, cmap="viridis")
    boxes.set_array([1.0])                       # a mapped collection
    ax.add_collection(boxes)
    ax.text(7, 8, "on the box edge", ha="center", va="center")
    ax.text(7, 6.5, "inside the box", ha="center", va="center")
    faces = np.array(band.get_facecolor())
    _pairs, crossed, _cut = _found(fig)
    assert crossed == {"on the edge": "an outline",
                       "on the box edge": "an outline"}, crossed
    assert np.array_equal(band.get_facecolor(), faces)
    assert "update_scalarmappable" not in vars(boxes)


def test_a_failed_run_saves_nothing_and_asks_for_no_look(folders):
    """Live wave w8b: a failed run said LOOK and saved its half-drawn
    figure under the name the good one would carry."""
    mine, _ = folders
    r = _run("import matplotlib.pyplot as plt\nplt.plot([1, 2])\n"
             "raise ValueError('x and y must be the same length')",
             mine, name="half")
    assert r["status"] == "error" and "same length" in r["error"]
    assert "next" not in r and "layout_check" not in r
    assert not (mine / "half.png").exists()


def test_a_clean_figure_has_no_layout_notes(folders):
    mine, _ = folders
    r = _run("import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\n"
             "ax.plot([0, 1], [0, 1])\nax.set_xlabel('x (m)')\n"
             "save_figure(fig, 'clean')", mine)
    assert r["status"] == "success" and "layout_check" not in r


def test_the_result_keeps_the_boxs_own_record(folders):
    mine, _ = folders
    r = _run("import matplotlib.pyplot as plt\nplt.plot([1])", mine)
    box = r["box"]
    assert box["limits_applied"], "no OS limit was applied on this host"
    assert box["run_seconds"] is not None and box["start_seconds"] >= 0


def test_a_chart_is_said_to_go_unchecked(folders):
    mine, _ = folders
    r = _run("import plotly.graph_objects as go\n"
             "save_chart(go.Figure(go.Scatter(x=[1], y=[2])), 'c')", mine)
    assert r["status"] == "success"
    assert any("does not check the chart" in n for n in r["notes"])


def test_a_look_waits_for_a_drawing_in_its_folder(tmp_path):
    import threading
    import time
    from funhouse_agent import drawing_wait
    started = threading.Event()

    def draw():
        with drawing_wait.drawing(str(tmp_path)):
            started.set()
            time.sleep(0.4)

    t = threading.Thread(target=draw)
    t.start()
    started.wait(2)
    assert drawing_wait.drawing_in(str(tmp_path))
    assert not drawing_wait.drawing_in(str(tmp_path / "other"))
    waited = drawing_wait.wait_until_drawn(str(tmp_path), timeout=5)
    t.join()
    assert waited >= 0.2 and not drawing_wait.drawing_in(str(tmp_path))


def test_plot_code_marks_its_folder_while_it_draws(folders, monkeypatch):
    from funhouse_agent import drawing_wait
    mine, _ = folders
    seen = []
    real = drawing_wait.drawing

    def spy(folder):
        seen.append(folder)
        return real(folder)
    monkeypatch.setattr(drawing_wait, "drawing", spy)
    _run("import matplotlib.pyplot as plt\nplt.plot([1])", mine)
    assert seen and os.path.samefile(seen[0], mine)


def test_plot_data_says_when_one_figure_is_redrawn_again(tmp_path):
    """Live wave w8 (F55): five versions of one figure (wall.png,
    wall2.png, ...) and then the turn ran out of steps."""
    from funhouse_agent._fileio import working_dir_bound
    from funhouse_agent.deep.tools import make_vision_tools
    (tool,) = make_vision_tools(include={"plot_data"})
    series = [{"x": [0, 1], "y": [0, 1], "label": "a"}]
    with working_dir_bound(str(tmp_path)):
        outs = [json.loads(tool.invoke({"series": series,
                                        "output_path": name}))
                for name in ("wall_figure.png", "wall_figure2.png",
                             "wall_figure_v3.png")]
    assert "redraw_note" not in outs[0] and "redraw_note" not in outs[1]
    assert "drawn 3 times" in outs[2]["redraw_note"]
    assert "plot_code" in outs[2]["redraw_note"]


def test_off_by_default_then_offered_to_both_helpers(monkeypatch):
    from funhouse_agent.deep.agent import build_calc_subagent
    from funhouse_agent.deep.figure_helper import (PLOT_CODE_STEP,
                                                   build_figure_subagent)

    def names(spec):
        return {t.name for t in spec["tools"]}

    monkeypatch.delenv(review_flags.PLOT_CODE_ENV, raising=False)
    assert "plot_code" not in names(build_calc_subagent())
    fig = build_figure_subagent()
    assert fig is None or "plot_code" not in names(fig)

    monkeypatch.setenv(review_flags.PLOT_CODE_ENV, "1")
    calc = build_calc_subagent()
    assert "plot_code" in names(calc)
    assert "plot_code" in calc["system_prompt"]
    fig = build_figure_subagent()
    if fig is not None:
        assert "plot_code" in names(fig)
        assert fig["system_prompt"].endswith(PLOT_CODE_STEP)
