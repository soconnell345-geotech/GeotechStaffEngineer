"""BACKLOG GM-113: under beta_method 'depth' a layer's (N1)60 or OCR is
refused, not silently left unused. The O'Neill and Reese (1999) beta takes
the depth and N60 (FHWA NHI-06-089 Eqs. 9-41 to 9-43); (N1)60 and OCR are
inputs of the GEC-10 (2018) beta only (Eqs. 10-8 to 10-13). Made-up
profile."""

import pytest

from drilled_shaft import (DrillShaft, DrillShaftAnalysis, ShaftSoilLayer,
                           ShaftSoilProfile)


def _analysis(beta_method, **first):
    layers = [
        ShaftSoilLayer(thickness=6.0, soil_type="cohesionless",
                       unit_weight=19.0, phi=32.0, N60=20.0, **first),
        ShaftSoilLayer(thickness=10.0, soil_type="cohesionless",
                       unit_weight=19.0, phi=34.0, N60=30.0),
    ]
    return DrillShaftAnalysis(
        shaft=DrillShaft(diameter=1.0, length=12.0),
        soil=ShaftSoilProfile(layers=layers), beta_method=beta_method)


@pytest.mark.parametrize("given", [{"OCR": 2.0}, {"N1_60": 18.0},
                                   {"OCR": 2.0, "N1_60": 18.0}])
def test_depth_beta_refuses_n1_60_and_ocr(given):
    with pytest.raises(ValueError, match="beta_method 'depth' does not take"):
        _analysis("depth", **given).compute()
    with pytest.raises(ValueError, match="0-6 m"):
        _analysis("depth", **given).capacity_vs_depth(depth_min=4.0,
                                                      n_points=3)


def test_depth_beta_without_them_runs_as_before():
    r = _analysis("depth").compute()
    assert r.Q_ultimate > 0


def test_the_rational_beta_still_takes_them():
    plain = _analysis("rational").compute()
    with_ocr = _analysis("rational", OCR=2.0).compute()
    assert with_ocr.Q_ultimate != plain.Q_ultimate


def test_the_depth_alternative_under_rational_is_unchanged():
    # the rational run's choices compute the depth beta as an alternative;
    # the check sits in compute(), not in the alternative's _compute()
    d = _analysis("rational", OCR=2.0).compute().to_dict()
    alts = [a for c in d["choices"] for a in c.get("alternatives", [])
            if str(a).startswith("beta_method 'depth'")]
    assert alts and not any("refused" in a for a in alts)
