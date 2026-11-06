"""GM-307: the MSE calc package, what GM-114 left.

Made-up walls only. Each new number is pinned to a value worked by hand
from FHWA GEC-11 (FHWA-NHI-10-024) Vol I as printed, and the package's
numbers are the mse_wall tool's own (one code path, GM-141).

Part 1, a sloping or broken backslope (GEC-11 Vol I Sec. 4.4.4): the
package prints Kab (Eqs. 4-2, 4-3, p. 4-15), h and F_T (Eq. 4-7, p. 4-20),
F_T's components (Eq. 4-10, p. 4-21), V2 (Figs. 4-3, 4-4) and, inside,
S_eq and sigma_2 (Fig. 4-11, p. 4-38).

Part 2, a battered face (10 deg or more): Kab with theta (Eqs. 4-2, 4-3,
p. 4-15; Fig. 4-5), the internal Ka of Eq. 4-26 (p. 4-36) and the
extensible failure surface of Fig. 4-9(b) (p. 4-34).
"""

import math
import re

import pytest

from calc_package.data_model import CalcStep
from funhouse_agent import dispatch as D


#: A made-up wall under a slope that continues at 18 deg, geogrids.
SLOPE = {"wall_type": "mse", "wall_height": 6.0, "gamma_backfill": 19.0,
         "phi_backfill": 34.0, "reinforcement_length": 5.0,
         "reinforcement_spacing": 0.6, "phi_foundation": 30.0,
         "q_allowable": 400.0, "phi_retained": 30.0, "gamma_retained": 18.0,
         "backfill_slope": 18.0, "reinforcement_name": "geogrid_gec11_e1_gg2"}

#: A made-up wall under a broken backslope (crest 2 m up), traffic behind
#: the crest, steel strips.
BROKEN = {**SLOPE, "backfill_slope": 26.57, "broken_backslope_height": 2.0,
          "surcharge": 12.0, "reinforcement_name": "ribbed_steel_strip_50x4",
          "reinforcement_horizontal_spacing": 0.9}


def _package(params, tmp_path, name="mse.html"):
    out = D.call_agent("calc_package", "retaining_wall_package", {
        **params, "output_path": str(tmp_path / name)})
    assert out.get("status") == "success", out
    return out, (tmp_path / name).read_text(encoding="utf-8")


def _tool(params):
    return D.call_agent("retaining_walls", "mse_wall", {
        k: v for k, v in params.items() if k != "wall_type"})


def _steps(params):
    from funhouse_agent.adapters.retaining_walls import run_mse_wall
    from retaining_walls.calc_steps import get_calc_steps
    result, used = run_mse_wall(
        {k: v for k, v in params.items() if k != "wall_type"})
    return [item for section in get_calc_steps(
        result, {**used, "wall_type": "mse"}) for item in section.items
        if isinstance(item, CalcStep)]


# --- Part 1: a sloping or broken backslope ---------------------------------

def test_kab_and_the_thrust_under_a_slope_by_hand(tmp_path):
    # GEC-11 Eqs. 4-2, 4-3 with theta 90, delta = beta = 18, phi_b 30:
    # Gamma = [1 + sqrt(sin 48 sin 12 / (sin 72 sin 108))]^2 = 1.9974,
    # Kab = sin^2 120 / (1.9974 sin 72) = 0.3948;
    # h = 6 + 5 tan 18 = 7.6246 m (Eq. 4-7);
    # F_T = 0.5 x 0.3948 x 18 x 7.6246^2 = 206.6 kN/m; F_H = F_T cos 18
    # = 196.5 (Eq. 4-10); V2 = 19 x 5 x 5 tan 18 / 2 = 77.2 kN/m;
    # S_eq = 0.5 x 0.7 x 6 tan 18 = 0.682 m, sigma_2 = 19 x 0.682 = 12.96
    gamma = (1 + math.sqrt(math.sin(math.radians(48))
                           * math.sin(math.radians(12))
                           / (math.sin(math.radians(72))
                              * math.sin(math.radians(108))))) ** 2
    assert gamma == pytest.approx(1.9974, abs=1e-4)
    kab = math.sin(math.radians(120)) ** 2 / (gamma
                                              * math.sin(math.radians(72)))
    assert kab == pytest.approx(0.3948, abs=1e-4)
    steps = {s.title: s for s in _steps(SLOPE)}
    assert steps["Coulomb Active Earth Pressure Coefficient (retained "
                 "fill)"].result_value == "0.3948"
    assert steps["Backslope Geometry (slope that continues)"
                 ].result_value == "7.625"
    assert steps["Retained Backfill Thrust"].result_value == "206.6"
    assert steps["Components of F_T (inclined at δ = β)"
                 ].result_value == "196.5"
    assert steps["Slope Fill over the Reinforced Zone"
                 ].result_value == "77.2"
    assert steps["Sloping Backfill Surcharge on the Reinforcement"
                 ].result_value == "12.96"
    _, html = _package(SLOPE, tmp_path)
    for cite in ("Eqs. 4-2, 4-3 (p. 4-15)", "Eq. 4-7 (p. 4-20)",
                 "Eq. 4-10 (p. 4-21)", "Fig. 4-11 (p. 4-38)",
                 "Eq. 4-21 (p. 4-28)"):
        assert cite in html, cite


def test_a_broken_backslope_by_hand():
    # Fig. 4-4: I = atan(2 / (2 x 6)) = 9.46 deg; h = min(6 + 5 tan 26.57,
    # 6 + 2) = 8.000 m; Eqs. 4-2, 4-3 with delta = beta = I:
    # Kab = 0.3477; F2 = Kab q h = 0.3477 x 12 x 8 = 33.4 kN/m
    steps = {s.title: s for s in _steps(BROKEN)}
    geo = steps["Backslope Geometry (broken backslope)"]
    assert geo.result_value == "8.000"
    assert "atan(2.000 / 12.00) = 9.46°" in geo.substitution
    I = math.radians(math.degrees(math.atan(2.0 / 12.0)))
    p = math.radians(30.0)
    gamma = (1 + math.sqrt(math.sin(p + I) * math.sin(p - I)
                           / (math.sin(math.pi / 2 - I)
                              * math.sin(math.pi / 2 + I)))) ** 2
    kab = math.sin(math.pi / 2 + p) ** 2 / (gamma * math.sin(math.pi / 2 - I))
    assert steps["Coulomb Active Earth Pressure Coefficient (retained "
                 "fill)"].result_value == f"{kab:.4f}" == "0.3477"
    assert steps["Traffic Thrust behind the Crest"].result_value == \
        f"{kab * 12.0 * 8.0:.1f}" == "33.4"
    assert "H1" in [s.result_name for s in steps.values()]


@pytest.mark.parametrize("params", [SLOPE, BROKEN], ids=["slope", "broken"])
def test_the_package_and_the_tool_agree(params, tmp_path):
    out, html = _package(params, tmp_path)
    tool = _tool(params)
    assert out["FOS_sliding"] == round(tool["FOS_sliding"], 2)
    assert out["FOS_overturning"] == round(tool["FOS_overturning"], 2)
    assert out["FOS_bearing"] == round(tool["FOS_bearing"], 2)
    assert out["sigma_v_kPa"] == round(tool["sigma_v_kPa"], 1)
    bs = tool["backslope"]
    assert f"{bs['Kab']:.4f}" in html
    assert f"{bs['h_m']:.3f}" in html
    assert f"{bs['FT_kN_per_m']:.1f}" in html
    assert f"{bs['V2_kN_per_m']:.1f}" in html
    assert f"{bs['sigma_2_kPa']:.2f}" in html


@pytest.mark.parametrize("params", [SLOPE, BROKEN], ids=["slope", "broken"])
def test_every_backslope_step_cites_a_printed_page(params):
    steps = _steps(params)
    for step in steps:
        assert re.search(r"\bpp?\. (?:[A-Z]\d*|\d+)-\d+",
                         step.reference or ""), (step.title, step.reference)
    assert not any("Rankine Active" in s.title for s in steps)


# --- Part 2: a battered face -------------------------------------------------

#: A made-up wall with its face battered 12 deg, level backfill.
BATTER = {"wall_type": "mse", "wall_height": 6.0, "gamma_backfill": 19.0,
          "phi_backfill": 34.0, "reinforcement_length": 4.5,
          "reinforcement_spacing": 0.6, "phi_foundation": 30.0,
          "q_allowable": 400.0, "phi_retained": 30.0, "gamma_retained": 18.0,
          "face_batter": 12.0, "reinforcement_name": "ribbed_steel_strip_50x4",
          "reinforcement_horizontal_spacing": 0.9}


def test_a_battered_face_by_hand(tmp_path):
    # theta = 102: Kab = sin^2 132 / [(1 + sin 30 / sin 102)^2 sin^3 102]
    # = 0.2584; F_T = 0.5 x 0.2584 x 18 x 36 = 83.7; at delta - omega =
    # -12 deg: F_H = 81.9, F_V = -17.4 (points up), at 4.5 + 2 tan 12 =
    # 4.925 m; FOS_s = (513 - 17.41) tan 30 / 81.89 = 3.494; FOS_ot =
    # (513 x 2.888 - 17.41 x 4.925) / (81.89 x 2) = 8.521; e < 0, sigma_v
    # = 495.6 / 4.5 = 110.1 kPa; internal Ka (Eq. 4-26) = 0.2087
    steps = {s.title: s for s in _steps(BATTER)}
    assert steps["Battered Face"].result_value == "102"
    assert steps["Coulomb Active Earth Pressure Coefficient (retained "
                 "fill)"].result_value == "0.2584"
    assert steps["Retained Backfill Thrust"].result_value == "83.7"
    comp = steps["Components of F_T (at δ to the normal of the back)"]
    assert comp.result_value == "81.9"
    assert "-17.4" in comp.substitution and "4.925 m" in comp.substitution
    assert steps["Internal Ka for a Battered Face"].result_value == "0.2087"
    assert "0.3281" in steps["Internal Ka for a Battered Face"].notes
    out, html = _package(BATTER, tmp_path)
    assert out["FOS_sliding"] == 3.49
    assert out["FOS_overturning"] == 8.52
    assert out["sigma_v_kPa"] == 110.1
    for cite in ("Fig. 4-5 (p. 4-16)", "Eq. 4-26 (p. 4-36)",
                 "Fig. 4-9(a) note, p. 4-34"):
        assert cite in html, cite


def test_a_battered_geogrid_wall_prints_its_failure_plane():
    # Fig. 4-9(b), phi 34, theta 102, beta 0: psi = 56.00 deg
    steps = {s.title: s for s in _steps({
        **BATTER, "reinforcement_name": "geogrid_gec11_e1_gg2",
        "reinforcement_horizontal_spacing": None})}
    assert steps["Failure Surface for a Battered Face (extensible)"
                 ].result_value == "56.00"


@pytest.mark.parametrize("extra", [
    {}, {"surcharge": 10.0}, {"backfill_slope": 15.0},
    {"reinforcement_name": "geogrid_gec11_e1_gg2",
     "reinforcement_horizontal_spacing": None}],
    ids=["level", "surcharge", "slope", "geogrid"])
def test_the_battered_package_and_the_tool_agree(extra, tmp_path):
    params = {k: v for k, v in {**BATTER, **extra}.items() if v is not None}
    out, html = _package(params, tmp_path)
    tool = _tool(params)
    assert out["FOS_sliding"] == round(tool["FOS_sliding"], 2)
    assert out["FOS_overturning"] == round(tool["FOS_overturning"], 2)
    assert out["sigma_v_kPa"] == round(tool["sigma_v_kPa"], 1)
    assert out["FOS_pullout_min"] == round(min(
        lv["FOS_pullout"] for lv in tool["internal_results"]), 2)
    assert f"{tool['face_batter']['Ka']:.4f}" in html
    for step in _steps(params):
        assert re.search(r"\bpp?\. (?:[A-Z]\d*|\d+)-\d+",
                         step.reference or ""), (step.title, step.reference)


def test_a_battered_face_under_a_broken_backslope_is_refused():
    out = D.call_agent("calc_package", "retaining_wall_package", {
        **BATTER, "backfill_slope": 20.0, "broken_backslope_height": 1.0})
    assert "broken backslope" in out["error"]
    assert "beta_cs" in out["error"]


def test_a_near_vertical_face_is_printed_as_vertical(tmp_path):
    near, _ = _package({**BATTER, "face_batter": 6.0}, tmp_path, "n.html")
    flat, _ = _package({**BATTER, "face_batter": 0.0}, tmp_path, "f.html")
    assert near["FOS_sliding"] == flat["FOS_sliding"]
    assert near["sigma_v_kPa"] == flat["sigma_v_kPa"]
