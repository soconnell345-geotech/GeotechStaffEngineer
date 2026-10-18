"""Standard chart lines from the references (BACKLOG GM-153).

Live waves w9 and w12 (flow F57) drew the plasticity chart's A-line from
memory: down to PI 0 at LL 20, where the chart has it level at PI 4 up to
LL 25.5, and "PI 2.9 at LL 24" where the chart gives 4. The lines now come
from ``geotech_references.chart_lines`` (UFS 3-220-10 Figure 1-2, p. 14),
reached through plot_data's ``reference_chart`` and the plotting box's
``chart_lines`` helper, each known only through its own description.
"""

import importlib
import json
import sys

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("geotech_references.chart_lines")

from funhouse_agent.adapters import profile_figure_adapter as pfa  # noqa: E402
from funhouse_agent import plot_box  # noqa: E402

LIMITS = {"x": [24, 32, 45, 38], "y": [4, 12, 22, 9],
          "label": "Atterberg limits", "style": "markers"}


def _trace(fig, name):
    return next(t for t in fig["data"] if t.get("name") == name)


def test_the_overlay_draws_the_printed_a_line(tmp_path):
    out = pfa._run_plot_data({
        "series": [LIMITS], "reference_chart": "uscs_plasticity",
        "output_path": str(tmp_path / "pc.png")})
    assert out["status"] == "success" and out["file_exists"]
    assert not out.get("warnings"), out.get("warnings")
    # the caller's series alone in the result; the chart said apart
    assert [s["label"] for s in out["series"]] == ["Atterberg limits"]
    chart = out["reference_chart"]
    assert chart["source"]["where"].startswith("Figure 1-2")
    assert chart["source"]["printed_page"] == "14"
    eq = {l["name"]: l["equation"] for l in chart["lines"]}
    assert eq["A-line"] == ("Horizontal at PI = 4 to LL = 25.5, then "
                            "PI = 0.73(LL - 20)")
    assert {"A-line", "U-line", "LL = 50", "PI = LL"} <= set(eq)
    assert {"CL-ML", "CL or OL", "CH or OH", "ML or OL",
            "MH or OH"} == set(chart["zones_named"])
    with open(out["plotly_json_path"], encoding="utf-8") as fh:
        fig = json.load(fh)
    a = _trace(fig, "A-line")
    pts = list(zip(a["x"], a["y"]))
    assert pts[0] == (4.0, 4.0) and (25.5, 4.0) in pts
    assert min(a["y"]) == 4.0                     # never down to PI 0
    for x, y in pts:
        if x > 25.5:
            assert y == pytest.approx(0.73 * (x - 20.0), abs=1e-6)
    assert a["showlegend"] is False
    assert "PI = 0.73(LL - 20)" in a["hovertext"][0]
    # the chart is drawn under the data, and the data keep their colour
    names = [t.get("name") for t in fig["data"]]
    assert names.index("A-line") < names.index("Atterberg limits")
    assert names.index("CL-ML") < names.index("Atterberg limits")
    from profile_figure.data_plot import _PALETTE
    assert _trace(fig, "Atterberg limits")["marker"]["color"] == _PALETTE[0]
    u = _trace(fig, "U-line")
    assert list(zip(u["x"], u["y"]))[:2] == [(16.0, 0.0), (16.0, 7.0)]
    hatched = _trace(fig, "CL-ML")
    assert min(hatched["y"]) == 4.0 and max(hatched["y"]) == 7.0
    # the chart's own axis names and frame, when the caller gives none
    assert fig["layout"]["xaxis"]["title"]["text"] == "Liquid Limit (LL)"
    assert fig["layout"]["yaxis"]["title"]["text"] == "Plasticity Index (PI)"


def _names(params):
    out, n_added, said = pfa._reference_overlay(params)
    notes = {n["text"]: n for n in out["annotations"]
             if n.get("color") == pfa._CHART_INK}
    return out, n_added, said, notes


def _outlines():
    from geotech_references.chart_lines import chart_lines
    return {z["name"]: [tuple(p) for p in z["outline"]]
            for z in chart_lines("uscs_plasticity")["zones"]}


def _clear_of(note, points, scale=(110.0, 60.0)):
    """No point under the name's footprint (with a marker's room)."""
    box = pfa._footprint(note["text"], (note["x"], note["y"]), scale,
                         pfa._letter({}), pad=0.012)
    return not any(pfa._covers(box, p) for p in points)


@pytest.mark.parametrize("point", [(22.0, 5.5), (24.0, 4.0)])
def test_a_zone_name_never_sits_on_the_data(point):
    # (24, 4) is the w12 case: the white "CL-ML" box hid it
    out, n_added, said, notes = _names({
        "series": [{"x": [point[0]], "y": [point[1]], "label": "S"}],
        "reference_chart": "uscs_plasticity"})
    cl_ml = notes["CL-ML"]
    assert not any("box" in n for n in notes.values())
    # moved to a clear spot inside its own zone, with no leader
    assert "point" not in cl_ml
    assert pfa._edge_gap((cl_ml["x"], cl_ml["y"]), _outlines()["CL-ML"],
                         (110.0, 60.0))[0]
    assert _clear_of(cl_ml, [point])
    assert (cl_ml["x"], cl_ml["y"]) != (21.0, 5.5)
    assert "CL-ML" in said["zone_names_moved"]
    # a zone with no point under its name keeps its printed place
    for name, at in (("CH or OH", (70.0, 46.15)), ("CL or OL", (40.0, 21.7))):
        assert (notes[name]["x"], notes[name]["y"]) == at
        assert "point" not in notes[name]
    # the chart's series come first, the caller's last, coloured as before
    assert out["series"][n_added:][0]["color"] == "#1f4e79"
    assert all(s["in_legend"] is False for s in out["series"][:n_added])


#: Points over all five zones, several where the names are printed.
SPREAD = {"x": [22, 30, 45, 35, 45, 65, 85, 70, 90, 40, 40, 70, 76],
          "y": [5.5, 14, 25, 6, 12, 40, 50, 25, 30, 21, 7, 46, 20],
          "label": "Atterberg limits", "style": "markers"}


def test_spread_data_names_stay_in_their_zones_with_no_leaders(tmp_path):
    out = pfa._run_plot_data({
        "series": [SPREAD], "reference_chart": "uscs_plasticity",
        "output_path": str(tmp_path / "spread.png")})
    assert out["status"] == "success" and out["file_exists"]
    assert not out.get("warnings"), out.get("warnings")
    points = list(zip(SPREAD["x"], SPREAD["y"]))
    outlines = _outlines()
    names = {a["text"]: a for a in out["annotations"]}
    assert set(names) == set(outlines)
    for name, a in names.items():
        assert "point" not in a, name           # no leader: no new ink
        note = {"text": name, "x": a["at"][0], "y": a["at"][1]}
        assert pfa._edge_gap(tuple(a["at"]), outlines[name],
                             (110.0, 60.0))[0], name
        assert _clear_of(note, points), name


def test_a_zone_with_no_clear_spot_is_named_above_pi_ll(tmp_path):
    # points all along the CL-ML band: its name cannot sit in it
    xs = [6.5 + 1.5 * k for k in range(15)]
    out = pfa._run_plot_data({
        "series": [{"x": xs, "y": [5.5] * len(xs), "label": "S",
                    "style": "markers"}],
        "reference_chart": "uscs_plasticity",
        "output_path": str(tmp_path / "band.png")})
    assert out["status"] == "success"
    cl_ml = next(a for a in out["annotations"] if a["text"] == "CL-ML")
    assert cl_ml["at"][1] > cl_ml["at"][0]      # above the PI = LL line
    assert pfa._edge_gap(tuple(cl_ml["point"]), _outlines()["CL-ML"],
                         (110.0, 60.0))[0]
    assert "above the PI = LL line" in out["reference_chart"][
        "zone_names_moved"]


def test_the_callers_series_are_numbered_as_given(tmp_path):
    out = pfa._run_plot_data({
        "series": [{"x": [30, 40], "y": [10, float("nan")], "label": "S"}],
        "reference_chart": "uscs_plasticity",
        "output_path": str(tmp_path / "nan.png")})
    assert any(w.startswith("series[0]: dropped 1")
               for w in out["warnings"]), out["warnings"]
    with pytest.raises(ValueError, match=r"series\[0\] style"):
        pfa._run_plot_data({
            "series": [{"x": [30], "y": [10], "style": "zigzag"}],
            "reference_chart": "uscs_plasticity",
            "output_path": str(tmp_path / "bad.png")})


def test_the_callers_axes_win_and_the_lines_keep_inside_them(tmp_path):
    out = pfa._run_plot_data({
        "series": [LIMITS], "reference_chart": "plasticity_chart",
        "xlabel": "LL (%)", "ylabel": "PI (%)", "xlim": [0, 60],
        "ylim": [0, 30], "output_path": str(tmp_path / "pc2.png")})
    assert out["status"] == "success"
    assert out["reference_chart"]["drawn_over"] == {"x": [0.0, 60.0],
                                                   "y": [0.0, 30.0]}
    with open(out["plotly_json_path"], encoding="utf-8") as fh:
        fig = json.load(fh)
    assert fig["layout"]["xaxis"]["title"]["text"] == "LL (%)"
    for t in fig["data"]:
        if t.get("name") in ("A-line", "U-line", "LL = 50", "PI = LL"):
            assert max(t["x"]) <= 60.0 + 1e-9 and max(t["y"]) <= 30.0 + 1e-9


def test_the_grain_size_chart_sets_its_log_axis(tmp_path):
    out = pfa._run_plot_data({
        "series": [{"x": [19, 4.75, 0.425, 0.075, 0.01],
                    "y": [100, 88, 55, 22, 8], "label": "S-3"}],
        "reference_chart": "uscs_particle_sizes", "xlim": [100, 0.001],
        "output_path": str(tmp_path / "gs.png")})
    assert out["status"] == "success"
    assert not out.get("warnings"), out.get("warnings")
    assert "logarithmic" in out["reference_chart"]["axis_note"]
    names = [l["name"] for l in out["reference_chart"]["lines"]]
    assert names[:4] == ["No. 200", "No. 40", "No. 10", "No. 4"]


def test_an_unknown_chart_is_refused_with_the_charts_there_are(tmp_path):
    with pytest.raises(ValueError, match="uscs_plasticity"):
        pfa._run_plot_data({"series": [LIMITS],
                            "reference_chart": "casagrande_1948",
                            "output_path": str(tmp_path / "x.png")})


def test_without_the_library_module_plot_data_does_not_offer_it(
        tmp_path, monkeypatch):
    """With a reference library that has no chart_lines (1.4.1, on the
    cluster today), the argument is in no schema or description, and is
    refused by name."""
    from funhouse_agent.deep.figure_helper import build_figure_subagent
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "geotech_references.chart_lines", None)
    try:
        importlib.reload(pfa)
        assert "reference_chart" not in pfa._VALID_PLOT
        assert "reference_chart" not in pfa.METHOD_INFO["plot_data"][
            "parameters"]
        spec = build_figure_subagent()
        plot = next(t for t in spec["tools"] if t.name == "plot_data")
        assert "reference_chart" not in plot.description
        schema = plot.tool_call_schema.model_json_schema()
        assert "reference_chart" not in schema["properties"]
        assert "chart_lines" not in plot_box.plot_code_description()
        out = plot.invoke({"series": [LIMITS],
                           "reference_chart": "uscs_plasticity",
                           "output_path": "no_chart"})
        assert "reference_chart" in str(out) and not (
            tmp_path / "no_chart.png").exists()
    finally:
        monkeypatch.undo()
        importlib.reload(pfa)
    assert "reference_chart" in pfa._VALID_PLOT


def test_without_the_library_module_it_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(pfa, "_chart_lines", lambda: None)
    with pytest.raises(ValueError, match="chart_lines"):
        pfa._run_plot_data({"series": [LIMITS],
                            "reference_chart": "uscs_plasticity",
                            "output_path": str(tmp_path / "x.png")})
    monkeypatch.setitem(sys.modules, "geotech_references.chart_lines", None)
    assert "chart_lines" not in plot_box.plot_code_description()


def test_the_descriptions_name_it_and_no_prompt_does(tmp_path, monkeypatch):
    info = pfa.METHOD_INFO["plot_data"]["parameters"]["reference_chart"]
    assert info["required"] is False
    for name in ("uscs_plasticity", "aashto_plasticity",
                 "uscs_particle_sizes", "PI = 0.73(LL - 20)"):
        assert name in info["description"]
    from funhouse_agent.deep.tools import _plot_description
    assert "reference_chart: A standard chart's lines" in _plot_description()
    box = " ".join(plot_box.plot_code_description().split())
    assert "chart_lines(name, x_range, y_range)" in box
    assert "chart_line_value('uscs_plasticity', 'A-line', 24) is 4.0" in box
    # the figure helper's own plot_data tool takes it and says so; its
    # prompt does not (special-purpose tools are known by their own
    # description only)
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path))
    from funhouse_agent.deep.figure_helper import build_figure_subagent
    spec = build_figure_subagent()
    plot = next(t for t in spec["tools"] if t.name == "plot_data")
    assert "reference_chart" in plot.description
    assert "reference_chart" in plot.tool_call_schema.model_json_schema()[
        "properties"]
    for words in ("reference_chart", "chart_lines", "A-line"):
        assert words not in spec["system_prompt"]
    out = json.loads(plot.invoke({"series": [LIMITS],
                                  "reference_chart": "uscs_plasticity",
                                  "output_path": "helper_pc"}))
    assert out["status"] == "success", out
    assert out["reference_chart"]["name"] == "uscs_plasticity"


def test_the_box_draws_from_chart_lines(tmp_path):
    code = (
        "import matplotlib.pyplot as plt\n"
        "c = chart_lines('uscs_plasticity', (0, 60), (0, 40))\n"
        "fig, ax = plt.subplots()\n"
        "for line in c['lines']:\n"
        "    xs = [p[0] for p in line['points']]\n"
        "    ys = [p[1] for p in line['points']]\n"
        "    ax.plot(xs, ys, color='0.3', lw=1)\n"
        "print(chart_line_value('uscs_plasticity', 'A-line', 24),\n"
        "      c['source']['printed_page'],\n"
        "      c['lines'][0]['points'][:2])\n"
        "save_figure(fig, 'pc_box')\n")
    r = plot_box.run_plot_code(code, root=str(tmp_path))
    assert r["status"] == "success", r
    assert "4.0 14 [[4.0, 4.0], [25.5, 4.0]]" in r["printed"]


def test_zone_names_carry_a_halo_not_a_box(tmp_path):
    """The hatching stops short of the CL-ML name (a ring of the
    background round the letters), and nothing is boxed over the data."""
    out, n_added, said, notes = _names({
        "series": [SPREAD], "reference_chart": "uscs_plasticity"})
    assert all(n.get("halo") is True and "box" not in n
               for n in notes.values())
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from profile_figure.data_plot import _clean_notes, _draw_notes
    fig, ax = plt.subplots()
    drawn = _draw_notes(ax, _clean_notes(
        [{"text": "CL-ML", "x": 0.5, "y": 0.5, "halo": True},
         {"text": "plain", "x": 0.2, "y": 0.2}], None, False, False))
    assert drawn[0].get_path_effects() and not drawn[1].get_path_effects()
    plt.close(fig)
    with pytest.raises(ValueError, match="halo"):
        _clean_notes([{"text": "x", "x": 0, "y": 0, "halo": "not-a-colour"}],
                     None, False, False)
