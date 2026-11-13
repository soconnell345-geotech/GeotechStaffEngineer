"""Phase E / v5.4 E8 validation — multiple surcharge zones.

Slide2 Verification #9 = ACADS 4 (Giam & Donald 1989): weak-seam noncircular slope
with a piezometric surface AND two external distributed loads (Table 9.2): a bench
load ~20 kPa over x=23-43 and a crest load ~20->40 kPa over x=70-80. V-028
validated the seam-governed FOS but OMITTED the loads -- the module supported only
ONE surcharge zone.

E8 adds ``SlopeGeometry.surcharges`` (a list of (pressure, x_start, x_end) zones,
summed on top of the single ``surcharge``/``surcharge_x_range`` pair). No change to
slices/search -- both go through ``surcharge_at``.

VERDICT: **capability built; V-028 itself does NOT validate** (GM-140 step 1,
2026-11-13: see test_published_v028_slope.py). With each point weighed once,
Spencer and GLE have no solution on the V-028 seam surface, so the loads are
measured with Janbu's simplified method (corrected), which solves: the two zones
lower it from 1.073 to 1.034 (3.7 %); the crest zone alone gives 1.023, and the
bench zone alone, over the toe, raises it to 1.085. (While the soil was weighed
three times the loads looked small beside it: Spencer 0.808 -> 0.806.) Default (no
``surcharges``) is identical to V-028. The crest zone's published 20->40 kPa ramp
is represented as a uniform 30 kPa (the mean); NOT tuned. The piezometric line is
Table 9.3 as printed.

See slope_stability/geometry.py (surcharges / surcharge_at), VALIDATION.md B11,
and test_published_v028_slope.py (the no-surcharge V-028 record).
"""

import pytest

from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.slip_surface import PolylineSlipSurface
from slope_stability.slices import build_slices
from slope_stability.gle import janbu_fos

_SURF = [(20.0, 28.0), (43.0, 28.0), (68.0, 40.0), (84.0, 40.0)]
_SEAM_TOP = [(20.0, 19.0), (84.0, 37.0)]
_SEAM_BOT = [(20.0, 18.0), (84.0, 36.0)]
_PIEZO = [(20.0, 27.75), (43.0, 27.75), (49.0, 29.8), (60.0, 34.0),
          (66.0, 35.8), (74.0, 37.6), (80.0, 38.4), (84.0, 38.4)]
# Table 9.2 external loads: bench 20 kPa (x 23-43); crest 20->40 kPa (x 70-80),
# represented as a uniform 30 kPa (the mean of the published ramp).
_ZONES = [(20.0, 23.0, 43.0), (30.0, 70.0, 80.0)]
# V-028 weak-seam critical surface (pinned in test_published_v028_slope.py).
_SURFACE = [(41.987, 28.0), (49.095, 26.656), (56.203, 28.249),
            (63.312, 30.304), (70.42, 35.555), (77.528, 40.0)]


def _geom(surcharges=None):
    return SlopeGeometry(
        surface_points=_SURF,
        soil_layers=[
            SlopeSoilLayer(name="upper", top_elevation=40.0, bottom_elevation=15.0,
                           gamma=18.84, phi=20.0, c_prime=28.5,
                           bottom_boundary_points=_SEAM_TOP),
            SlopeSoilLayer(name="weak", top_elevation=40.0, bottom_elevation=15.0,
                           gamma=18.84, phi=10.0, c_prime=0.0,
                           bottom_boundary_points=_SEAM_BOT),
            SlopeSoilLayer(name="lower", top_elevation=40.0, bottom_elevation=15.0,
                           gamma=18.84, phi=20.0, c_prime=28.5),
        ],
        gwt_points=_PIEZO, surcharges=surcharges)


def _fos(surcharges=None, single=0.0):
    """Janbu's simplified method with the f0 correction (Spencer and GLE
    do not converge on this surface since GM-140 step 1)."""
    slip = PolylineSlipSurface(points=_SURFACE)
    g = _geom(surcharges)
    if single:
        g.surcharge = single
        g.surcharge_x_range = (70.0, 80.0)
    return janbu_fos(build_slices(g, slip, 50), slip)[0]


def test_v028e8_no_surcharge_matches_v028():
    """Default (no surcharges) reproduces the V-028 seam record."""
    assert _fos() == pytest.approx(1.073, abs=0.005)


def test_v028e8_two_zones_lower_fos():
    """The two ACADS-4 surcharge zones lower the FOS (the crest-side zone
    adds driving weight; the bench zone sits over the toe and resists)."""
    f0, f2 = _fos(), _fos(_ZONES)
    assert f2 == pytest.approx(1.034, abs=0.005)     # ours (a record)
    assert f2 < f0
    assert _fos([_ZONES[1]]) == pytest.approx(1.023, abs=0.005)
    assert _fos([_ZONES[0]]) > f0


def test_v028e8_zones_and_single_surcharge_coexist():
    """The zones sum on TOP of the legacy single surcharge; a zone the surface
    does not cross does not change the FOS (no slices under it)."""
    # a zone entirely upslope of the sliding mass (x 10-18) is inert
    assert _fos([(50.0, 10.0, 18.0)]) == pytest.approx(_fos(), abs=1e-9)
    # single surcharge + a zone both apply (sum): heavier load, lower FOS,
    # the same as one zone of the summed pressure
    both = _fos([(30.0, 70.0, 80.0)], single=15.0)
    assert both < _fos([(30.0, 70.0, 80.0)])
    assert both == pytest.approx(_fos([(45.0, 70.0, 80.0)]), abs=1e-9)


def test_v028e8_bad_zone_rejected():
    with pytest.raises(ValueError, match="x_start < x_end"):
        _geom([(20.0, 43.0, 23.0)])
    with pytest.raises(ValueError, match="non-negative"):
        _geom([(-5.0, 23.0, 43.0)])
