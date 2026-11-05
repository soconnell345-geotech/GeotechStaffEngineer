"""GM-156: ``analyze_excavation`` states each layer's soil model and the K0
of its in-situ stresses in ``choices``. Made-up layers."""

import math

import pytest

from fem2d.analysis import _excavation_soil_choices


def _by_what(layers):
    return {c['what']: c for c in _excavation_soil_choices(layers)}


def test_mohr_coulomb_and_jaky_by_default():
    w = _by_what([{'name': 'sand', 'E': 3e4, 'nu': 0.3, 'c': 0,
                   'phi': 30, 'gamma': 18}])
    m = w["soil model, sand"]
    assert m['chosen'].startswith("Mohr-Coulomb") and m['how'] == 'default'
    assert any("Hardening Soil" in a for a in m['alternatives'])
    k = w["K0 for the in-situ stresses, sand"]
    assert k['how'] == 'default'
    assert k['chosen'].startswith(f"{1 - math.sin(math.radians(30)):.3f}")
    assert "Eq. 10-2, p. 10-10" in k['basis']


def test_given_k0_and_hs_model():
    w = _by_what([{'name': 'clay', 'E': 2e4, 'nu': 0.3, 'c': 5, 'phi': 25,
                   'gamma': 19, 'K0': 0.8, 'model': 'hs', 'E50_ref': 2e4,
                   'Eur_ref': 6e4, 'm': 0.5, 'p_ref': 100, 'R_f': 0.9}])
    assert w["soil model, clay"]['chosen'].startswith("Hardening Soil")
    assert w["soil model, clay"]['how'] == 'given'
    k = w["K0 for the in-situ stresses, clay"]
    assert k['chosen'] == pytest.approx(0.8) and k['how'] == 'given'


def test_undrained_layer_nu_ratio_and_elastic_layer():
    w = _by_what([{'E': 1e4, 'nu': 0.45, 'c': 40, 'phi': 0, 'gamma': 18},
                  {'E': 5e4, 'nu': 0.25, 'c': 0, 'phi': 0, 'gamma': 20}])
    k = w["K0 for the in-situ stresses, layer 1"]
    assert k['chosen'].startswith(f"{0.45 / 0.55:.3f}")
    assert "4-6.4.1" in k['basis']
    assert w["soil model, layer 1"]['chosen'].startswith("Mohr-Coulomb")
    assert w["soil model, layer 2"]['chosen'] == "linear elastic"
