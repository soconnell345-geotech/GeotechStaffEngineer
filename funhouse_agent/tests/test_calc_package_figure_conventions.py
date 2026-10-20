"""Calc-package figures drawn by the owner's figure rules (backlog GM-200,
GM-206): pressures in kPa, the ground where the analysis has it, a depth
axis that says it is depth where no elevation is given, and a pile group
drawn as a plan (no axes, a scale bar, a north arrow, piles to scale) with
its names clear of the piles.

Each figure is built from a package's own inputs, drawn for the width it
prints at and checked by the GM-194 layout check.
"""

import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from matplotlib.text import Text  # noqa: E402

import funhouse_agent.adapters.calc_package as cp  # noqa: E402
from geotech_common.layout_ink import draw_for_print, layout_check  # noqa: E402


@pytest.fixture
def built(monkeypatch):
    """Run a package builder and return what it handed to the package
    renderer: (result, analysis)."""
    seen = {}

    def _capture(module, result, analysis, params, analysis_type,
                 extra=None, run=None):
        seen.update(result=result, analysis=analysis)
        return {"status": "success"}

    monkeypatch.setattr(cp, "_build_response", _capture)

    def run(method, params):
        assert cp.METHOD_REGISTRY[method](dict(params))["status"] == "success"
        return seen["result"], seen["analysis"]
    return run


def _texts(fig):
    return [t.get_text() for t in fig.findobj(Text) if t.get_text()]


def _layout_hits(fig):
    """The layout check's findings on the figure as it prints."""
    draw_for_print(fig, 5.0)
    found = layout_check(fig, "x", dpi=150)
    plt.close(fig)
    return {k: v for k, v in found.items() if k != "file"}


# --- M-O pressure figure ---------------------------------------------------

def test_mo_pressure_figure_is_in_kpa(built):
    from seismic_geotech.calc_steps import _plot_pressure_diagram
    r, analysis = built("seismic_package", {
        "analysis_type": "seismic_earth_pressure", "phi": 32, "kh": 0.2,
        "gamma": 19.0, "H": 5.0, "kv": 0.05})
    fig = _plot_pressure_diagram(r, analysis)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Lateral earth pressure (kPa)"
    texts = _texts(fig)
    assert not any("normalized" in t.lower() for t in texts)
    # the static triangle's base is K_A gamma H, in kPa
    assert f"K_AγH = {2 * r.PA_static / 5.0:.1f} kPa" in texts
    assert _layout_hits(fig) == {}


# --- cantilever wall section -----------------------------------------------

def _green_lines(ax):
    return [ln for ln in ax.get_lines()
            if matplotlib.colors.same_color(ln.get_color(), "g")]


@pytest.mark.parametrize("params", [
    {"wall_height": 4.0, "gamma_backfill": 18.0, "phi_backfill": 30.0},
    {"wall_height": 7.0, "gamma_backfill": 19.0, "phi_backfill": 34.0,
     "backfill_slope": 18.0, "base_width": 4.5, "toe_length": 1.2,
     "base_thickness": 0.8, "include_passive": True},
])
def test_cantilever_toe_ground_is_at_the_top_of_the_base(built, params):
    # The analysis takes the ground in front at the top of the footing
    # (passive on the base's embedment); the figure drew it at the wall's
    # full height, as though the wall retained nothing (GM-200)
    from retaining_walls.calc_steps import _get, _plot_cantilever_section
    r, analysis = built("retaining_wall_package", params)
    geom = _get(analysis, "geom")
    fig = _plot_cantilever_section(r, analysis)
    toe_side = [ln for ln in _green_lines(fig.axes[0])
                if max(ln.get_xdata()) <= 0.0]
    assert len(toe_side) == 1
    assert set(toe_side[0].get_ydata()) == {geom.base_thickness}
    assert _layout_hits(fig) == {}


# --- downdrag profiles -----------------------------------------------------

_DD_LAYERS = [
    {"soil_type": "cohesive", "thickness": 8.0, "unit_weight": 17.0,
     "cu": 30.0, "Cc": 0.3, "e0": 1.0, "settling": True},
    {"soil_type": "cohesionless", "thickness": 15.0, "unit_weight": 19.0,
     "phi": 35.0, "settling": False}]


@pytest.mark.parametrize("fill, ground", [
    (0.0, "ground"), (2.0, "original ground")])
def test_downdrag_profiles_say_depth_with_no_elevation(built, fill, ground):
    from downdrag.calc_steps import (
        _plot_axial_load, _plot_settlement_profile, _plot_skin_friction,
    )
    r, analysis = built("downdrag_package", {
        "pile_length": 20.0, "pile_diameter": 0.3, "Q_dead": 500.0,
        "fill_thickness": fill, "layers": _DD_LAYERS})
    for plot in (_plot_axial_load, _plot_skin_friction,
                 _plot_settlement_profile):
        fig = plot(r, analysis)
        assert fig.axes[0].get_ylabel() == (
            f"Depth below {ground} (m), no elevation given")
        assert _layout_hits(fig) == {}


# --- pile group plan view --------------------------------------------------

def _plan(built, params):
    from pile_group.calc_steps import _get, _plot_plan_view
    r, analysis = built("pile_group_package", params)
    return _plot_plan_view(_get(r, "pile_forces"), r, analysis), analysis


def test_pile_plan_is_a_drawing_with_piles_to_scale(built):
    fig, _ = _plan(built, {"n_rows": 2, "n_cols": 3, "spacing": 1.8,
                           "Vz": 2400.0, "My": 900.0, "pile_diameter": 0.6})
    ax = fig.axes[0]
    assert not ax.axison                         # no axes, ticks or frame
    texts = _texts(fig)
    assert "N" in texts                          # the north arrow
    assert any(t.endswith(" m") for t in texts)  # the scale bar's length
    joined = "\n".join(texts)
    assert "North drawn up: none was given." in joined
    assert "Piles to scale, D = 0.60 m." in joined
    # every pile drawn at its diameter, not sized by its force
    radii = {round(p.radius, 9) for p in ax.patches if isinstance(p, Circle)}
    assert radii == {0.3}
    # each pile named with its force, as the table gives it
    assert sum(bool(re.match(r"P\d+-\d+\n[+-][\d,]+\.\d kN$", t))
               for t in texts) == 6
    assert _layout_hits(fig) == {}


def test_pile_plan_north_from_the_plan(built):
    fig, analysis = _plan(built, {
        "n_rows": 3, "n_cols": 4, "spacing": 1.5, "Vz": 2400.0,
        "Mx": 300.0, "pile_diameter": 0.45, "north_deg": 35.0})
    assert analysis["north_deg"] == 35.0
    assert "North drawn up" not in "\n".join(_texts(fig))
    assert _layout_hits(fig) == {}


def test_pile_plan_without_a_diameter_draws_symbols_and_says_so(built):
    fig, _ = _plan(built, {"n_rows": 3, "n_cols": 3, "spacing": 1.5,
                           "Vz": 3000.0, "Mx": 400.0, "My": 200.0})
    assert "Piles drawn as symbols, not to scale: no diameter given." in (
        "\n".join(_texts(fig)))
    assert _layout_hits(fig) == {}


def test_large_pile_group_leaves_the_forces_to_the_table(built):
    # 64 piles at 1.2 m: the forces cannot be written clear of the piles at
    # the printed size, so the names stand alone and the key says where the
    # forces are; nothing overlaps (GM-206)
    from pile_group.calc_steps import _get, _plot_pile_load_bar
    params = {"n_rows": 8, "n_cols": 8, "spacing": 1.2, "Vz": 12000.0,
              "Mx": 6000.0, "My": 9000.0, "pile_diameter": 0.4}
    fig, _ = _plan(built, params)
    texts = _texts(fig)
    assert "Too many piles to write each force at this scale:" in (
        "\n".join(texts))
    assert not any(re.match(r"P\d+-\d+\n", t) for t in texts)
    assert sum(bool(re.fullmatch(r"P\d+-\d+", t)) for t in texts) == 64
    assert _layout_hits(fig) == {}
    r, _ = built("pile_group_package", params)
    assert _layout_hits(_plot_pile_load_bar(_get(r, "pile_forces"), r)) == {}
