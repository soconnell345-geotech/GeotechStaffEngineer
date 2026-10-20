"""V-058: GEC-10 (2018) Illustrative Example 10-1, side resistance in
cohesionless soil (GM-241).

Source: GEC-10 (FHWA-NHI-18-024) Sec. 10.3.5.1, Example 10-1, pp. 10-12 to
10-14, read from ``geotech-references/docs/GEC 10.pdf``. A 5 ft shaft in two
silty sand layers: Layer 1, 0-15 ft, gamma 115 pcf, (N1)60 14, N60 9; Layer
2, 15-33 ft, gamma 125 pcf below the water table at 15 ft, (N1)60 28, N60
27. Printed (p. 10-14): phi' 38 and 41; sigma'p 5,768 and 13,890 psf (Eq.
10-11 with m = 0.8, silty sands); OCR 6.7 and 6.1; Ko 1.24 and 1.13; beta
0.97 and 0.98; fSN 837 and 2,248 psf; RSN 197.2 and 635.6 kips; sum 833
kips. Side resistance is taken from the ground surface.

The example rounds beta to two places before fSN = beta sigma'v (0.97 x
862.5 = 837 psf); the module carries it unrounded (0.967, 834 psf), so
fSN and RSN agree to 1 %. Its RSN2 line prints '698,659 lb' for 635.6 kips
(pi x 5 x 18 x 2,248 = 635.6 kips).

GEC-10 (2018) Eq. 10-13 is the default beta since GM-241; this runs the
tool's own path with the example's inputs.
"""

import pytest

FT, PCF, PSF, KIP = 0.3048, 0.157087, 0.0478803, 4.44822

_PARAMS = {
    "diameter": 5 * FT, "shaft_length": 33 * FT, "gwt_depth": 15 * FT,
    "top_exclusion": 0.0,
    "layers": [
        {"thickness": 15 * FT, "soil_type": "cohesionless",
         "unit_weight": 115 * PCF, "phi": 38, "N60": 9,
         "sigma_p_correlation": "silty_sand", "description": "Layer 1"},
        {"thickness": 18 * FT, "soil_type": "cohesionless",
         "unit_weight": 125 * PCF, "phi": 41, "N60": 27,
         "sigma_p_correlation": "silty_sand", "description": "Layer 2"},
    ],
}

#: p. 10-14, as printed
_PRINTED = {
    "Layer 1": {"sigma_p_psf": 5768, "OCR": 6.7, "Ko": 1.24, "beta": 0.97,
                "fSN_psf": 837, "RSN_kips": 197.2},
    "Layer 2": {"sigma_p_psf": 13890, "OCR": 6.1, "Ko": 1.13, "beta": 0.98,
                "fSN_psf": 2248, "RSN_kips": 635.6},
}


def _run(params):
    from funhouse_agent.dispatch import call_agent
    r = call_agent("drilled_shaft", "drilled_shaft_capacity", params)
    assert "error" not in r, r
    return r


def test_v058_each_layer_as_printed():
    r = _run(_PARAMS)
    layers = {lb["description"]: lb for lb in r["side_resistance_by_layer"]}
    for name, want in _PRINTED.items():
        lb, st = layers[name], layers[name]["beta_steps"]
        assert st["sigma_p_kPa"] / PSF == pytest.approx(
            want["sigma_p_psf"], rel=1e-3), name
        assert st["OCR"] == pytest.approx(want["OCR"], abs=0.05), name
        assert st["Ko"] == pytest.approx(want["Ko"], abs=0.01), name
        assert st["beta"] == pytest.approx(want["beta"], abs=0.01), name
        assert not st["shallow_limit"], name
        assert lb["fs_kPa"] / PSF == pytest.approx(want["fSN_psf"],
                                                   rel=0.01), name
        assert lb["side_resistance_kN"] / KIP == pytest.approx(
            want["RSN_kips"], rel=0.01), name


def test_v058_the_sum_as_printed():
    r = _run(_PARAMS)
    assert r["Q_side_sand_kN"] / KIP == pytest.approx(833, rel=0.01)


def test_v058_phi_from_the_blow_counts():
    """phi' by 27.5 + 9.2 log (N1)60 (Appendix A Eq. A-2) from the
    example's (N1)60 gives 38.0 and 40.8 degrees, which it rounds to 38
    and 41."""
    params = {**_PARAMS, "layers": [
        {**lay, "N1_60": n1} for lay, n1 in zip(_PARAMS["layers"], (14, 28))]}
    r = _run(params)
    phis = [lb["beta_steps"]["phi_deg"] for lb in r["side_resistance_by_layer"]]
    assert phis == pytest.approx([38.0, 40.8], abs=0.05)
    assert r["Q_side_sand_kN"] / KIP == pytest.approx(833, rel=0.02)
