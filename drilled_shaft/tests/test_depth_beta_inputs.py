"""BACKLOG GM-113: under beta_method 'depth' a layer's (N1)60 and OCR are
taken, not refused. The depth beta (O'Neill and Reese 1999) uses neither,
but the result's beta_method 'rational' alternative (GEC-10 2018 Eqs. 10-8
to 10-13, in choices) is computed with them. Made-up profile."""

import pytest

from drilled_shaft import (DrillShaft, DrillShaftAnalysis, ShaftSoilLayer,
                           ShaftSoilProfile)


def _analysis(beta_method, **first):
    first = {"phi": 32.0, **first}
    layers = [
        ShaftSoilLayer(thickness=6.0, soil_type="cohesionless",
                       unit_weight=19.0, N60=20.0, **first),
        ShaftSoilLayer(thickness=10.0, soil_type="cohesionless",
                       unit_weight=19.0, phi=34.0, N60=30.0),
    ]
    return DrillShaftAnalysis(
        shaft=DrillShaft(diameter=1.0, length=12.0),
        soil=ShaftSoilProfile(layers=layers), beta_method=beta_method)


def _rational_alternative(result):
    alts = [a for c in result.to_dict()["choices"]
            for a in c.get("alternatives", [])
            if str(a).startswith("beta_method 'rational'")]
    assert len(alts) == 1, alts
    return alts[0]


@pytest.mark.parametrize("given", [{"OCR": 2.0},
                                   {"phi": 0.0, "N1_60": 18.0}])
def test_depth_beta_takes_them_for_the_rational_alternative(given):
    plain = _analysis("depth").compute()
    with_them = _analysis("depth", **given).compute()
    # the depth beta itself uses neither
    assert with_them.Q_ultimate == plain.Q_ultimate
    # the rational alternative in choices is computed with them
    assert (_rational_alternative(with_them)
            != _rational_alternative(plain))


def test_capacity_vs_depth_takes_them_under_the_depth_beta():
    plain = _analysis("depth").capacity_vs_depth(depth_min=4.0, n_points=3)
    with_ocr = _analysis("depth", OCR=2.0).capacity_vs_depth(depth_min=4.0,
                                                             n_points=3)
    assert with_ocr == plain and len(plain) == 3


def test_the_rational_beta_takes_them():
    plain = _analysis("rational").compute()
    with_ocr = _analysis("rational", OCR=2.0).compute()
    assert with_ocr.Q_ultimate != plain.Q_ultimate
