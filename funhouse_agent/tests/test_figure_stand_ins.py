"""A figure that only holds a place never reaches the user (BACKLOG GM-190).

Live wave w14 (``module_work/live_smoke/runs/w14-judgment/TRACE_REVIEW.md``
§7.5, §8 fix 9): in a parallel batch the calc helper sent ``plot_data`` a
series named as a stand-in with axes named by bare letters; the PNG and its
chart were written into the conversation folder and both reached the user.
Every figure a tool saves is delivered, so the tools refuse such a figure
before anything is written: ``plot_data`` on every route, and the plotting
box's ``save_figure`` / ``save_chart`` and its saving of figures left open.
The rule is :func:`funhouse_agent.plot_box.figures.stand_in`.
"""

import os

import pytest

from funhouse_agent.plot_box.figures import STAND_IN_NOTE, stand_in


# ---------------------------------------------------------------------------
# the rule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("words", [
    {"series": ["placeholder"], "axes": ["x", "y"]},
    {"series": ["Place-holder series"], "axes": ["Depth (m)"]},
    {"titles": ["Dummy data"], "axes": ["q (kPa)"]},
    {"series": ["Lorem ipsum"]},
    {"drawn": False},
    {"drawn": False, "titles": ["Skin friction"],
     "notes": ["No layer data available"]},
    {"drawn": False, "notes": ["N/A"]},
    {"drawn": False, "titles": ["No data"], "axes": ["x (m)", "y (m)"]},
])
def test_stand_ins_are_named(words):
    assert stand_in(**words)


@pytest.mark.parametrize("words", [
    # a real figure, named in any one way
    {"series": ["Vesic q_ult (kPa)"]},
    {"titles": ["Sensitivity of q_ult to friction angle"], "axes": ["x"]},
    {"axes": ["Friction angle (deg)", "y"]},
    {"axes": ["x", "y"], "notes": ["B-1"]},
    # names left out, or bare letters, are not a stand-in: a plan, a
    # section or a coordinate plot is named so; the look judges them
    {},
    {"axes": ["x", "y"]},
    {"axes": ["x", "z"]},
    {"axes": ["X axis"]},
    # "dummy" alone names real things
    {"series": ["Dummy pile load test"]},
    {"titles": ["Dummy joint spacing"], "axes": ["Slab length (m)"]},
    # a real group or a note about missing readings, over data
    {"series": ["SPT N60", "No data"], "axes": ["N60", "Depth (m)"]},
    {"titles": ["CPT tip resistance"], "notes": ["no data below 12 m"]},
    # a diagram or a table drawn in words, one cell of it "N/A"
    {"drawn": False, "notes": ["Site", "Wall"]},
    {"drawn": False, "titles": ["Boring summary"],
     "notes": ["Boring", "B-1", "Water", "N/A", "Refusal", "None"]},
])
def test_real_figures_pass(words):
    assert stand_in(**words) is None


# ---------------------------------------------------------------------------
# plot_data, through dispatch (the calc helper's route in the trace)
# ---------------------------------------------------------------------------

@pytest.fixture
def folder(tmp_path, monkeypatch):
    from funhouse_agent._fileio import DEFAULT_OUTPUT_DIR_ENV
    monkeypatch.setenv(DEFAULT_OUTPUT_DIR_ENV, str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _figures(folder):
    return sorted(f for f in os.listdir(folder)
                  if f.endswith((".png", ".json")))


def test_plot_data_refuses_the_trace_call_and_writes_nothing(folder):
    from funhouse_agent.dispatch import call_agent
    res = call_agent("profile_figure", "plot_data", {
        "series": [{"x": [410.0, 520.0, 380.0, 450.0, 495.0],
                    "y": [1, 2, 3, 4, 5], "label": "placeholder"}],
        "xlabel": "x", "ylabel": "y"})
    assert "status" not in res
    assert res["error"].startswith("ValueError: plot_data drew no figure: ")
    assert "'placeholder'" in res["error"]
    assert STAND_IN_NOTE in res["error"]
    assert _figures(folder) == []


@pytest.mark.parametrize("params", [
    # coordinates named by bare letters, with nothing else named
    {"series": [{"x": [0, 12.5, 30], "y": [0, 8, 22], "style": "markers"}],
     "xlabel": "x", "ylabel": "y"},
    {"series": [{"x": [0, 10, 20], "y": [5, 3, 0]}],
     "xlabel": "x", "ylabel": "z"},
    # a plan drawn as a drawing, its coordinate system named 'x'
    {"series": [{"x": [0, 40, 40, 0, 0], "y": [0, 0, 25, 25, 0],
                 "line_dash": "dashdot"}], "plan_view": True, "xlabel": "x"},
])
def test_plot_data_draws_real_data_named_by_bare_letters(folder, params):
    pytest.importorskip("matplotlib")
    from funhouse_agent.dispatch import call_agent
    res = call_agent("profile_figure", "plot_data", params)
    assert res.get("status") == "success", res


def test_plot_data_draws_a_named_figure(folder):
    pytest.importorskip("matplotlib")
    from funhouse_agent.dispatch import call_agent
    res = call_agent("profile_figure", "plot_data", {
        "series": [{"x": [30, 32, 34], "y": [410.0, 520.0, 660.0],
                    "label": "Vesic q_ult (kPa)"}],
        "xlabel": "Friction angle phi (deg)",
        "ylabel": "Ultimate bearing capacity q_ult (kPa)",
        "output_path": "sensitivity.png"})
    assert res.get("status") == "success", res
    assert "sensitivity.png" in _figures(folder)


def test_plot_data_counts_point_names_and_notes_as_names(folder):
    pytest.importorskip("matplotlib")
    from funhouse_agent.dispatch import call_agent
    res = call_agent("profile_figure", "plot_data", {
        "series": [{"x": [0, 10], "y": [0, 5], "labels": ["B-1", "B-2"]}],
        "xlabel": "x", "ylabel": "y", "output_path": "borings.png"})
    assert res.get("status") == "success", res


# ---------------------------------------------------------------------------
# the plotting box: save_figure, save_chart and figures left open
# ---------------------------------------------------------------------------

def _box(code, root):
    pytest.importorskip("matplotlib")
    from funhouse_agent.plot_box import run_plot_code
    return run_plot_code(code, root=str(root))


def test_box_saves_the_figures_and_not_the_stand_ins(tmp_path):
    res = _box(
        "import matplotlib.pyplot as plt\n"
        "f1, a1 = plt.subplots()\n"
        "a1.plot([30, 32, 34], [410, 520, 660])\n"
        "a1.set_xlabel('Friction angle (deg)'); a1.set_ylabel('q_ult (kPa)')\n"
        "print(save_figure(f1, 'sensitivity'))\n"
        "f2, a2 = plt.subplots()\n"
        "a2.plot([1, 2], [3, 4], label='placeholder'); a2.legend()\n"
        "print(save_figure(f2, 'slot'))\n"
        "f3, a3 = plt.subplots()\n"
        "a3.set_title('Skin friction')\n"
        "a3.text(0.5, 0.5, 'No layer data available', "
        "transform=a3.transAxes)\n"
        "print(save_figure(f3, 'empty_panel'))\n", tmp_path)
    assert res["status"] == "success", res
    assert [i["file"] for i in res["images"]] == ["sensitivity.png"]
    assert not (tmp_path / "slot.png").exists()
    assert not (tmp_path / "empty_panel.png").exists()
    note = next(n for n in res["notes"] if n.startswith("Not saved"))
    assert "slot.png: 'placeholder'" in note
    assert "empty_panel.png: it draws no data" in note
    assert "None" in res["printed"]        # save_figure says it saved none


def test_box_does_not_save_a_blank_figure_left_open(tmp_path):
    res = _box("import matplotlib.pyplot as plt\n"
               "fig, ax = plt.subplots()\n", tmp_path)
    assert res["status"] == "error"
    assert res["error"].startswith("No figure was saved, each a stand-in: "
                                   "figure.png: nothing is drawn on it")
    assert not any(f.endswith(".png") for f in os.listdir(tmp_path))


def test_box_saves_a_diagram_drawn_in_words_and_refuses_a_stand_in_chart(
        tmp_path):
    pytest.importorskip("plotly")
    res = _box(
        "import matplotlib.pyplot as plt\n"
        "import plotly.graph_objects as go\n"
        "fig, ax = plt.subplots()\n"
        "ax.axis('off')\n"
        "ax.text(0.2, 0.5, 'Retained soil', bbox=dict(fc='w'))\n"
        "ax.annotate('Wall', xy=(0.2, 0.5), xytext=(0.7, 0.5), "
        "arrowprops=dict(arrowstyle='->'))\n"
        "save_figure(fig, 'sketch')\n"
        "save_chart(go.Figure(go.Scatter(x=[1, 2], y=[3, 4], "
        "name='placeholder')), 'sketch_chart')\n", tmp_path)
    assert res["status"] == "success", res
    assert (tmp_path / "sketch.png").exists()
    assert not (tmp_path / "sketch_chart.plotly.json").exists()
    assert "charts" not in res
    assert any("sketch_chart.plotly.json: 'placeholder'" in n
               for n in res["notes"])


def test_box_saves_real_figures_the_rule_could_mistake(tmp_path):
    """Real figures that look sparse to the rule: coordinates named by bare
    letters, arrows alone (a quiver has no paths until it is drawn), a
    table drawn in text with an "N/A" cell, and a Plotly drawing made of
    layout shapes."""
    pytest.importorskip("plotly")
    res = _box(
        "import matplotlib.pyplot as plt\n"
        "import plotly.graph_objects as go\n"
        "f1, a1 = plt.subplots()\n"
        "a1.scatter([0, 12, 30], [0, 8, 22])\n"
        "a1.set_xlabel('x'); a1.set_ylabel('y')\n"
        "save_figure(f1, 'coords')\n"
        "f2, a2 = plt.subplots()\n"
        "a2.quiver([0, 1], [0, 1], [1, 1], [1, 0])\n"
        "save_figure(f2, 'arrows')\n"
        "f3, a3 = plt.subplots(); a3.axis('off')\n"
        "for i, (k, v) in enumerate([('Boring', 'B-1'), ('Water', 'N/A'), "
        "('Refusal', 'None')]):\n"
        "    a3.text(0.1, 0.8 - 0.2 * i, k); a3.text(0.6, 0.8 - 0.2 * i, v)\n"
        "save_figure(f3, 'table')\n"
        "fig = go.Figure()\n"
        "fig.add_shape(type='rect', x0=0, x1=4, y0=0, y1=2)\n"
        "save_chart(fig, 'section')\n", tmp_path)
    assert res["status"] == "success", res
    assert sorted(i["file"] for i in res["images"]) == [
        "arrows.png", "coords.png", "table.png"]
    assert (tmp_path / "section.plotly.json").exists()
    assert not any(n.startswith("Not saved") for n in res.get("notes", []))


def test_box_names_a_refused_figure_when_the_code_then_fails(tmp_path):
    """save_figure returns None for a stand-in; code that uses the name
    then fails, and the result still says why nothing was saved."""
    res = _box(
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots()\n"
        "ax.plot([1, 2], [3, 4], label='placeholder'); ax.legend()\n"
        "name = save_figure(fig, 'slot')\n"
        "print(name.upper())\n", tmp_path)
    assert res["status"] == "error"
    assert any(n.startswith("Not saved, as a stand-in: slot.png")
               for n in res["notes"]), res
