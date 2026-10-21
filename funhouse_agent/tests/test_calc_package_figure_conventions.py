"""Calc-package figures drawn by the owner's figure rules (backlog GM-200,
GM-206, GM-216, GM-218): pressures in kPa, the ground where the analysis
has it, the wall's earth pressure where the analysis applies it, a depth
axis that says it is depth where no elevation is given and an elevation
axis where one is, a pile group drawn as a plan (no axes, a scale bar, a
north arrow, piles to scale, the cap and property line where given) with
its names clear of the piles, and no geometry drawn that was not given
without the figure saying so.

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
                 extra=None, run=None, package=None):
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
    # no elevation is given: the axes say what they measure from
    assert fig.axes[0].get_ylabel() == (
        "Height above the underside of the base (m)")
    assert fig.axes[0].get_xlabel() == "Distance from the toe (m)"
    assert _layout_hits(fig) == {}


def _coloured(ax, colour):
    """The patches of ``ax`` filled with ``colour`` (any alpha)."""
    from matplotlib.colors import to_rgb
    return [p for p in ax.patches
            if all(abs(a - b) < 1e-6
                   for a, b in zip(p.get_facecolor()[:3], to_rgb(colour)))]


@pytest.mark.parametrize("params", [
    {"wall_height": 5.0, "gamma_backfill": 19.0, "phi_backfill": 32.0,
     "surcharge": 12.0, "dead_load_surcharge": 8.0},
    {"wall_height": 7.0, "gamma_backfill": 19.0, "phi_backfill": 34.0,
     "backfill_slope": 18.0, "base_width": 4.5, "toe_length": 1.2,
     "base_thickness": 0.8},
])
def test_cantilever_pressure_on_the_plane_through_the_heel(built, params):
    # The analysis applies K_a over H_active on the vertical plane through
    # the back of the heel, the surcharge's K_a q with it; the figure drew
    # one triangle on the stem's face over the stem only (GM-216)
    from retaining_walls.calc_steps import (
        _cantilever_terms, _get, _plot_cantilever_section)
    r, analysis = built("retaining_wall_package", params)
    ep = _cantilever_terms(r, analysis)[0]
    B, H_a = _get(analysis, "geom").base_width, ep["H_active"]
    fig = _plot_cantilever_section(r, analysis)
    ax = fig.axes[0]
    (earth,) = _coloured(ax, "#fca5a5")
    xs, ys = earth.get_xy()[:, 0], earth.get_xy()[:, 1]
    assert (ys.min(), ys.max()) == pytest.approx((0.0, H_a))
    q = ep["q_live"] + ep["q_dead"]
    surcharge = _coloured(ax, "#fdba74")
    if q:
        (band,) = surcharge
        assert (band.get_x(), band.get_y()) == pytest.approx((B, 0.0))
        assert band.get_height() == pytest.approx(H_a)
        assert xs.min() == pytest.approx(B + band.get_width())
        # K_a q against K_a gamma H_a, to the same scale
        assert band.get_width() / (xs.max() - xs.min()) == pytest.approx(
            q / (_get(analysis, "gamma_backfill") * H_a))
    else:
        assert not surcharge
        assert xs.min() == pytest.approx(B)
    joined = "\n".join(_texts(fig))
    assert (f"Active pressure on the vertical plane through the back of "
            f"the heel, {H_a:.2f} m high") in joined
    assert f"P_a = {ep['Pa']:.0f} kN/m, {ep['z_Pa']:.2f} m above" in joined
    assert ("Surcharge, K_a q" in joined) == bool(q)
    assert _layout_hits(fig) == {}


_KEYED = {"wall_height": 6.0, "gamma_backfill": 19.0, "phi_backfill": 30.0,
          "has_shear_key": True, "key_depth": 0.6, "include_passive": True}


def _key(ax, depth):
    (key,) = [p for p in ax.patches
              if getattr(p, "get_y", None) and p.get_y()
              == pytest.approx(-depth) and p.get_height()
              == pytest.approx(depth)]
    return key


@pytest.mark.parametrize("given, said", [
    ({}, "no key position or width given."),
    ({"key_distance_from_toe": 2.4, "key_width": 0.5}, None),
    ({"key_width": 0.5}, "no key position given."),
])
def test_shear_key_drawn_where_given_else_under_the_stem_and_said(
        built, given, said):
    # The analysis has no place for the key (its passive is on the plane
    # below the toe over the key's depth): drawn where given, else under
    # the stem as UFC 3-220-20 Fig. 4-29 draws it, and the figure says so
    from retaining_walls.calc_steps import _get, _plot_cantilever_section
    r, analysis = built("retaining_wall_package", {**_KEYED, **given})
    geom = _get(analysis, "geom")
    fig = _plot_cantilever_section(r, analysis)
    key = _key(fig.axes[0], 0.6)
    assert key.get_x() == pytest.approx(
        given.get("key_distance_from_toe", geom.toe_length))
    assert key.get_width() == pytest.approx(
        given.get("key_width", geom.stem_thickness_base))
    joined = "\n".join(_texts(fig))
    assert ("Shear key drawn under the stem" in joined) == bool(said)
    if said:
        assert said in joined
    assert "the key is not counted" not in joined
    assert _layout_hits(fig) == {}


def test_shear_key_without_passive_says_it_is_not_counted(built):
    from retaining_walls.calc_steps import _plot_cantilever_section
    r, analysis = built("retaining_wall_package",
                        {**_KEYED, "include_passive": False})
    fig = _plot_cantilever_section(r, analysis)
    assert ("No passive resistance in the sliding check: the key is not "
            "counted.") in "\n".join(_texts(fig))
    assert _layout_hits(fig) == {}


@pytest.mark.parametrize("given, words", [
    ({"key_distance_from_toe": 3.4, "key_width": 0.5},
     "runs past the back of the base"),
    ({"key_width": 0}, "key_width must be a length in m, above 0"),
    ({"key_distance_from_toe": -1.0}, "key_distance_from_toe must be"),
    ({"key_distance_from_toe": "nan"}, "key_distance_from_toe must be"),
    ({"key_distance_from_toe": True}, "key_distance_from_toe must be"),
])
def test_shear_key_place_off_the_base_is_refused_by_name(
        given, words, tmp_path, monkeypatch):
    import funhouse_agent.dispatch as D
    monkeypatch.chdir(tmp_path)
    r = D.call_agent("calc_package", "retaining_wall_package",
                     {**_KEYED, "base_width": 3.6, **given})
    assert words in r.get("error", ""), r


def test_shear_key_place_with_no_key_is_refused(tmp_path, monkeypatch):
    import funhouse_agent.dispatch as D
    monkeypatch.chdir(tmp_path)
    r = D.call_agent("calc_package", "retaining_wall_package", {
        "wall_height": 4.0, "gamma_backfill": 18.0, "phi_backfill": 30.0,
        "key_width": 0.4})
    assert "the wall has no shear key for key_width" in r.get("error", ""), r


def test_mse_section_says_what_its_axes_measure_from():
    # No elevation is given: "Elevation" with 0 at the base of the
    # reinforced zone said nothing of where it was measured from (GM-218)
    from retaining_walls import MSEWallGeometry, analyze_mse_wall
    from retaining_walls.calc_steps import (
        _plot_mse_internal, _plot_mse_section)
    import dataclasses
    from retaining_walls.reinforcement import RIBBED_STEEL_STRIP_75x4
    # GM-256: the strip at its horizontal spacing (Rc = 0.1)
    strip = dataclasses.replace(RIBBED_STEEL_STRIP_75x4,
                                horizontal_spacing=0.75)
    geom = MSEWallGeometry(wall_height=6.0, reinforcement_spacing=0.6)
    r = analyze_mse_wall(geom, 18.0, 34.0, strip)
    analysis = {"wall_type": "mse", "geom": geom, "gamma_backfill": 18.0,
                "phi_backfill": 34.0,
                "reinforcement": strip}
    fig = _plot_mse_section(r, analysis)
    assert fig.axes[0].get_ylabel() == (
        "Height above the base of the reinforced zone (m)")
    assert fig.axes[0].get_xlabel() == "Distance from the wall face (m)"
    assert _layout_hits(fig) == {}
    fig = _plot_mse_internal(r, analysis)
    assert fig.axes[0].get_ylabel() == (
        "Depth below the top of the wall, z (m)")
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


@pytest.mark.parametrize("fill, ground", [
    (0.0, "Ground at El. 102.50 m (z = 0)"),
    (2.0, "Original ground, under the fill, at El. 102.50 m (z = 0)")])
def test_downdrag_profiles_on_elevation_when_given(built, fill, ground):
    # A ground elevation draws the profiles against elevation, never depth
    # and elevation together; the results stay in depth and no number
    # changes (GM-216)
    import numpy as np
    from downdrag.calc_steps import (
        _plot_axial_load, _plot_settlement_profile, _plot_skin_friction,
    )
    params = {"pile_length": 20.0, "pile_diameter": 0.3, "Q_dead": 500.0,
              "fill_thickness": fill, "layers": _DD_LAYERS}
    plain, plain_analysis = built("downdrag_package", params)
    r, analysis = built("downdrag_package",
                        {**params, "ground_elevation": 102.5})
    assert r.neutral_plane_depth == plain.neutral_plane_depth
    assert np.array_equal(r.axial_load, plain.axial_load)
    assert r.to_dict()["ground_elevation_m"] == 102.5
    assert "ground_elevation_m" not in plain.to_dict()
    np_el = 102.5 - r.neutral_plane_depth
    for plot in (_plot_axial_load, _plot_skin_friction,
                 _plot_settlement_profile):
        fig = plot(r, analysis)
        ax = fig.axes[0]
        assert ax.get_ylabel() == "Elevation (m)"
        assert not ax.yaxis_inverted()           # elevation runs up
        lo, hi = ax.get_ylim()
        assert lo < 102.5 - 20.0 and 102.5 < hi
        texts = "\n".join(_texts(fig))
        assert ground in texts
        assert f"El. {np_el:.2f} m" in texts
        assert "Depth" not in texts
        assert _layout_hits(fig) == {}
    # and the captions name the plane as the axes do (GM-216 review: they
    # read "z = 7.99 m" under an elevation axis)
    from downdrag.calc_steps import get_figures
    for caps, named in ((get_figures(r, analysis), f"El. {np_el:.2f} m"),
                        (get_figures(plain, plain_analysis),
                         f"z = {plain.neutral_plane_depth:.2f} m")):
        caps = [f.caption for f in caps]
        assert named in caps[0] and named in caps[1], caps


@pytest.mark.parametrize("elevation, label", [
    (None, "Depth below ground (m), no elevation given"),
    (50.0, "Elevation (m)")])
def test_downdrag_result_plots_say_their_axis(elevation, label):
    # The result's own plots read "Depth (m)" (GM-216)
    from downdrag import DowndragAnalysis, DowndragSoilLayer, \
        DowndragSoilProfile
    soil = DowndragSoilProfile(layers=[
        DowndragSoilLayer(thickness=8.0, soil_type="cohesive",
                          unit_weight=17.0, cu=30.0, Cc=0.3, e0=1.0,
                          settling=True),
        DowndragSoilLayer(thickness=15.0, soil_type="cohesionless",
                          unit_weight=19.0, phi=35.0)], gwt_depth=0.0)
    r = DowndragAnalysis(soil=soil, pile_length=20.0, pile_diameter=0.3,
                         Q_dead=500.0, fill_thickness=1.0,
                         ground_elevation=elevation).compute()
    if elevation is None:
        label = label.replace("ground", "original ground")
    for plot in (r.plot_axial_load, r.plot_settlement):
        ax = plot(show=False)
        assert ax.get_ylabel() == label
        assert ax.yaxis_inverted() == (elevation is None)
        plt.close(ax.figure)


@pytest.mark.parametrize("value", ["high", float("inf"), True])
def test_downdrag_ground_elevation_not_a_number_is_refused(value):
    from downdrag import DowndragAnalysis, DowndragSoilLayer, \
        DowndragSoilProfile
    soil = DowndragSoilProfile(layers=[DowndragSoilLayer(
        thickness=10.0, soil_type="cohesionless", unit_weight=19.0,
        phi=32.0)])
    with pytest.raises(ValueError, match="ground_elevation must be"):
        DowndragAnalysis(soil=soil, pile_length=8.0, pile_diameter=0.3,
                         ground_elevation=value)


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


def test_tension_piles_are_hatched_for_grey_print(built):
    # Blue and red print as the same grey, and a large group leaves the
    # signed forces to the table: tension piles and their key are hatched
    # as well as red (GM-200 review)
    from matplotlib.patches import Patch
    fig, _ = _plan(built, {"n_rows": 6, "n_cols": 6, "spacing": 1.2,
                           "Vz": 2000.0, "My": 9000.0,
                           "pile_diameter": 0.4})
    ax = fig.axes[0]
    circles = [p for p in ax.patches if isinstance(p, Circle)]
    hatched = [p for p in circles if p.get_hatch()]
    assert 0 < len(hatched) < len(circles)
    assert all(p.get_facecolor()[0] > 0.5 for p in hatched)    # the red
    keyed = [h for lg in fig.legends + ([ax.get_legend()]
                                        if ax.get_legend() else [])
             for h in lg.legend_handles if isinstance(h, Patch)]
    assert any(h.get_hatch() for h in keyed)
    assert "− tension (hatched)" in "\n".join(_texts(fig))
    assert _layout_hits(fig) == {}


_CAP = [[-2.4, -1.5], [2.4, -1.5], [2.4, 1.5], [-2.4, 1.5]]
_PROPERTY = [[-6.0, 3.5], [6.0, 3.5]]


@pytest.mark.parametrize("given", [
    {}, {"cap_outline": _CAP}, {"property_line": _PROPERTY},
    {"cap_outline": _CAP, "property_line": _PROPERTY}])
def test_pile_plan_draws_the_cap_and_property_line_only_when_given(
        built, given):
    # Never made up: drawn as given, and the plan says what was not given
    # (GM-216)
    from matplotlib.patches import Polygon
    fig, analysis = _plan(built, {
        "n_rows": 2, "n_cols": 3, "spacing": 1.8, "Vz": 2400.0,
        "My": 900.0, "pile_diameter": 0.6, **given})
    ax = fig.axes[0]
    caps = [p for p in ax.patches if isinstance(p, Polygon)]
    dashdot = [ln for ln in ax.get_lines() if ln.get_linestyle() == "-."]
    joined = "\n".join(_texts(fig))
    if "cap_outline" in given:
        (cap,) = caps
        assert cap.get_xy()[:4].tolist() == _CAP
        assert "Pile cap" in joined
    else:
        assert not caps
    if "property_line" in given:
        (line,) = dashdot
        assert list(zip(line.get_xdata(), line.get_ydata())) == [
            tuple(p) for p in _PROPERTY]
        assert "Property line" in joined
    else:
        assert not dashdot
    missing = [w for w, k in (("cap outline", "cap_outline"),
                              ("property line", "property_line"))
               if k not in given]
    if missing:
        assert f"No {' or '.join(missing)} given: not drawn." in joined
    else:
        assert "given: not drawn" not in joined
    assert _layout_hits(fig) == {}


@pytest.mark.parametrize("name, value", [
    ("cap_outline", [[0, 0], [1, 0]]),
    ("cap_outline", [[0, 0], [1, 0], [1, "a"]]),
    ("cap_outline", [[0, 0], [1, 0], [1, float("nan")]]),
    ("property_line", [[0, 0]]),
    ("property_line", "north side"),
    ("property_line", [[0, 0], [1, 2, 3]]),
])
def test_plan_lines_not_points_are_refused_by_name(name, value, tmp_path,
                                                   monkeypatch):
    import funhouse_agent.dispatch as D
    monkeypatch.chdir(tmp_path)
    r = D.call_agent("calc_package", "pile_group_package", {
        "n_rows": 2, "n_cols": 2, "spacing": 1.5, "Vz": 1000.0,
        name: value})
    assert f"{name} must be [[x, y], ...]" in r.get("error", ""), r


# --- a package's own inputs left out (GM-218) ------------------------------

def test_a_packages_own_inputs_left_out_are_listed_and_assumed(
        tmp_path, monkeypatch):
    # north_deg and pile_diameter skipped defaults_used and the Assumed
    # Inputs: a package's own inputs now go the tool's way
    import html as htmlmod

    import funhouse_agent.dispatch as D
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "pg.html"
    r = D.call_agent("calc_package", "pile_group_package", {
        "n_rows": 2, "n_cols": 2, "spacing": 1.5, "Vz": 1000.0,
        "format": "html", "output_path": str(out)})
    assert "error" not in r, r
    used = r["defaults_used"]
    for name in ("north_deg", "pile_diameter", "cap_outline",
                 "property_line"):
        assert name in used, used
    assert used["north_deg"].startswith("north drawn up")
    page = htmlmod.unescape(out.read_text(encoding="utf-8"))
    assert "not given: no north given" in page
    assert "not given: no cap outline given" in page
    # given, an own input is not listed as a default
    r = D.call_agent("calc_package", "pile_group_package", {
        "n_rows": 2, "n_cols": 2, "spacing": 1.5, "Vz": 1000.0,
        "north_deg": 20.0, "pile_diameter": 0.4,
        "format": "html", "output_path": str(out)})
    assert "north_deg" not in r["defaults_used"]
    assert "pile_diameter" not in r["defaults_used"]


def test_shear_key_place_listed_only_for_a_wall_with_a_key(tmp_path,
                                                           monkeypatch):
    import funhouse_agent.dispatch as D
    monkeypatch.chdir(tmp_path)
    r = D.call_agent("calc_package", "retaining_wall_package",
                     {**_KEYED, "format": "html",
                      "output_path": str(tmp_path / "w.html")})
    assert "error" not in r, r
    assert r["defaults_used"]["key_distance_from_toe"].startswith(
        "under the stem")
    assert "key_width" in r["defaults_used"]
    r = D.call_agent("calc_package", "retaining_wall_package", {
        "wall_height": 4.0, "gamma_backfill": 18.0, "phi_backfill": 30.0,
        "format": "html", "output_path": str(tmp_path / "w.html")})
    assert "key_distance_from_toe" not in r.get("defaults_used", {})
    assert "key_width" not in r.get("defaults_used", {})


@pytest.mark.parametrize("north", [400.0, -361.0, "nan", "inf", "north",
                                   True])
def test_north_deg_out_of_range_is_refused_by_name(north, tmp_path,
                                                   monkeypatch):
    import funhouse_agent.dispatch as D
    monkeypatch.chdir(tmp_path)
    r = D.call_agent("calc_package", "pile_group_package", {
        "n_rows": 2, "n_cols": 2, "spacing": 1.5, "Vz": 1000.0,
        "north_deg": north})
    assert "north_deg" in r.get("error", ""), r
    assert "-360 to 360" in r["error"]


def test_the_cap_edge_stops_short_of_the_names(built):
    # The edge ran behind the names' halos and showed in slivers between
    # the letters (GM-224): it is cut clear of each name's box
    from matplotlib.text import Annotation
    cap = [[-2.4, -1.2], [2.4, -1.2], [2.4, 1.5], [-2.4, 1.5]]
    fig, _ = _plan(built, {
        "n_rows": 2, "n_cols": 3, "spacing": 1.8, "Vz": 2400.0,
        "My": 900.0, "pile_diameter": 0.6, "cap_outline": cap})
    ax = fig.axes[0]
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    names = [t.get_window_extent(rend) for t in ax.texts
             if isinstance(t, Annotation) and "kN" in t.get_text()]
    edges = [ln for ln in ax.get_lines() if ln.get_zorder() == 5
             and ln.get_color() == "#444444"]
    assert names and edges
    for ln in edges:
        pts = ax.transData.transform(list(zip(ln.get_xdata(),
                                              ln.get_ydata())))
        for bb in names:
            # sample along the segment: no point inside a name's box
            for k in range(21):
                x = pts[0][0] + (pts[1][0] - pts[0][0]) * k / 20
                y = pts[0][1] + (pts[1][1] - pts[0][1]) * k / 20
                assert not (bb.x0 < x < bb.x1 and bb.y0 < y < bb.y1)
