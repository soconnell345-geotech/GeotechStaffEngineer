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
                  "time_limit_s", "no network"):
        assert words in tool.description
    out = tool.invoke({"code": "x = 1", "colour": "red"})
    assert "colour" in str(out)


def test_a_png_fits_one_look(folders):
    """Live wave w8: 200 dpi saved 2538 x 1546 px and every look tiled."""
    mine, _ = folders
    r = _run("import matplotlib.pyplot as plt\n"
             "fig, ax = plt.subplots(figsize=(14, 9)); ax.plot([1, 2])\n"
             "save_figure(fig, 'wide')", mine)
    assert r["status"] == "success", r
    assert max(r["images"][0]["width_px"], r["images"][0]["height_px"]) \
        <= 2100


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
