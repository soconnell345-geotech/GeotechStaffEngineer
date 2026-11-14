"""GM-233: the base in rock against the concrete's bearing strength.

GEC-10 (FHWA-NHI-18-024) p. 10-25: the base resistance in rock may be
governed by the structural strength of the concrete in bearing, the
factored resistances compared case by case (phi 0.70 structural against
0.50 geotechnical). Eq. 12-7 (p. 12-8): P_r = 0.70 [0.85 f'c A1 m], A1 the
socket's cross-sectional area, m = 2.0 on competent rock. Example 10-4
(p. 10-28): 0.70 [0.85 (576 ksf) pi/4 (6 ft)^2 x 2] = 19,380 kips against
0.50 x 25,447 = 12,723 kips, the geotechnical base governing.

Values other than the printed example's are made up.
"""

import math

import pytest

from drilled_shaft import (DrillShaft, DrillShaftAnalysis, ShaftSoilLayer,
                           ShaftSoilProfile)
from drilled_shaft.lrfd import apply_lrfd

FT = 0.3048
KSF = 47.880259  # kPa
KIP = 4.4482216  # kN


def _run(fc, qu=360 * KSF, socket=None, diameter=6 * FT, layers=None):
    shaft = DrillShaft(diameter=diameter, length=10 * FT, concrete_fc=fc,
                       socket_diameter=socket)
    soil = ShaftSoilProfile(layers=layers or [
        ShaftSoilLayer(5.0, "rock", 24.0, qu=qu, RQD=100.0)])
    return DrillShaftAnalysis(shaft=shaft, soil=soil).compute()


def _choice(r):
    return [c for c in r.choices
            if c["what"] == ("base resistance in rock, against the "
                             "concrete in bearing")]


def test_example_10_4_geotechnical_governs():
    """Example 10-4 (p. 10-28): 19,380 kips > 12,723 kips."""
    r = _run(576 * KSF)
    cb = r.base_steps["concrete_bearing"]
    assert cb["equation"] == "12-7" and cb["m"] == 2.0
    assert cb["phi_Pn_kN"] / KIP == pytest.approx(19380, rel=1e-3)
    assert cb["phi_Qt_geotechnical_kN"] / KIP == pytest.approx(12723,
                                                               rel=1e-3)
    assert cb["governs"] == "geotechnical"
    lrfd = apply_lrfd(r)
    assert lrfd["phi_Qt_kN"] == pytest.approx(0.50 * r.Q_tip, abs=0.1)
    assert lrfd["phi_Qt_governed_by"] == "geotechnical"
    (c,) = _choice(r)
    assert "geotechnical base resistance governs" in c["chosen"]


def test_concrete_governs():
    """Made-up: f'c 10 MPa under 2.5 qu of a 17 MPa rock; the factored base
    is 0.70 x 0.85 x 10,000 x A1 x 2, below 0.50 Q_tip."""
    r = _run(10000.0, qu=17000.0)
    A1 = math.pi * (6 * FT) ** 2 / 4
    expected = 0.70 * 0.85 * 10000.0 * A1 * 2.0
    cb = r.base_steps["concrete_bearing"]
    assert cb["phi_Pn_kN"] == pytest.approx(expected, rel=1e-4)
    assert cb["governs"] == "concrete"
    lrfd = apply_lrfd(r)
    assert lrfd["phi_Qt_kN"] == pytest.approx(expected, abs=0.1)
    assert lrfd["phi_Qt_geotechnical_kN"] == pytest.approx(0.50 * r.Q_tip,
                                                           abs=0.1)
    assert lrfd["phi_Qt_governed_by"].startswith("concrete")
    # the nominal base stays the geotechnical one
    assert r.Q_tip == pytest.approx(2.5 * 17000.0 * A1, rel=1e-6)
    (c,) = _choice(r)
    assert "concrete governs" in c["chosen"]


def test_socket_area_is_A1():
    """A1 is the socket's cross-sectional area (p. 12-8)."""
    r = _run(20000.0, socket=1.2, diameter=1.5)
    assert r.base_steps["concrete_bearing"]["A1_m2"] == pytest.approx(
        math.pi * 1.2 ** 2 / 4, abs=1e-3)


def test_not_checked_without_fc():
    r = _run(None)
    assert "concrete_bearing" not in r.base_steps
    assert "phi_Qt_governed_by" not in apply_lrfd(r)
    (c,) = _choice(r)
    assert c["chosen"].startswith("not checked")


def test_base_in_soil_has_no_check():
    """p. 10-25 and p. 12-8 speak of a base on rock only."""
    layers = [ShaftSoilLayer(2.0, "rock", 24.0, qu=5000.0, RQD=100.0),
              ShaftSoilLayer(10.0, "cohesive", 19.0, cu=150.0)]
    shaft = DrillShaft(diameter=1.0, length=6.0, concrete_fc=28000.0)
    r = DrillShaftAnalysis(shaft=shaft,
                           soil=ShaftSoilProfile(layers=layers)).compute()
    assert "concrete_bearing" not in r.base_steps
    assert not _choice(r)
