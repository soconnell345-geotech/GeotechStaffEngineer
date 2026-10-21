"""
GM-83: analyze_site_response against the OpenSeesPy docs' effective stress
example, the recipe the module adapts.

The example: "Effective Stress Site Response Analysis of a Layered Soil
Column", OpenSeesPy documentation section 14.4.2 (freeFieldEffective.py,
https://openseespydoc.readthedocs.io/en/latest/src/freeFieldEffective.html),
converted to openseespy by Harsh Mistry (University of Manchester) from
the OpenSees wiki example by Christopher McGann, HyungSuk Shin, Pedro
Arduino and Peter Mackenzie-Helnwein (University of Washington),
https://opensees.berkeley.edu/wiki/index.php/Effective_Stress_Site_Response_Analysis_of_a_Layered_Soil_Column
-- 30 m of PressureDependMultiYield02 sand (a 2 m dry crust, 8 m of loose
sand, 20 m of dense sand), water 2 m down, a 2 % slope, nine-node u-p
elements, the same Gilroy No. 1 motion as the total stress example.

``opensees_docs_free_field_effective.py`` ports it (a test helper); it
reproduces the wiki's published pore pressure ratios: peak r_u 0.43 at
3 m, 0.97 at 7 m and 0.56 at 16 m against the wiki's figure for the
sloping column (about 0.42, 0.97 and 0.55). One run takes about 60 s, so
the tests use the port's stored numbers (EFFECTIVE_REFERENCE; openseespy
3.8.0, Python 3.12, WSL, 2026-10-15) and run it only through gravity and
one second of shaking.

What to expect, and why:

* Until GM-85 the module could not run this column: its elastoplastic
  gravity stage failed at the first step, because it ran gravity at the
  shaking permeability (1e-4 m/s). It now follows the example: gravity at
  1 m/s, then 1e-4 m/s set through element parameters. Sand columns with
  elements of about 2 m and less failed the same way.
* Against the port with the inputs the module cannot set as the example
  does set as the module sets them (bulk moduli, the fluid's bulk
  modulus, the dense layer's contraction and dilation constants, level
  ground, the Rayleigh band): spectral accelerations from 0.1 to 1.5 s
  within 8.7 % (measured -8.7 % at 0.5 s to +4.4 % at 0.1 s; -6.2 to
  +4.5 %, also worst at 0.5 s, with the fixed pressure stabilization
  1e-5 before GM-260; pinned at 10 %, the test's own allowance, because
  the elements differ: the module's 4-node SSPquadUP needs a pressure
  stabilization, the example's 9-node 9_4_QuadUP needs none, and the
  example gives no tolerance of its own), and peak r_u at 3, 7 and 16 m
  0.36 / 0.90 / 0.98 against 0.38 / 0.91 / 0.97 (pinned within 0.1).
  PGA is not compared: in liquefying sand it is a dilation spike
  (0.30 g against 0.55 g) that depends on the element.
* Against the port as written the dense layer is the difference: the
  module gives every sand the Dr = 40 % contraction constants, so its
  dense layer liquefies (r_u 0.98 at 16 m against 0.56); Sa from 0.1 to
  1.5 s -27 to +15 %.
"""

import numpy as np
import pytest

from opensees_agent.opensees_utils import has_opensees
from opensees_agent.tests import opensees_docs_free_field_effective as eff
from opensees_agent.tests import opensees_wiki_free_field as wiki

requires_opensees = pytest.mark.skipif(
    not has_opensees(), reason="openseespy not installed")

TEST_TIME_LIMIT_S = 300.0

# The port's numbers on the stored record (first 20 s of Gilroy No. 1).
# sa_g / amplification at wiki.PERIODS_S (0.1, 0.2, 0.3, 0.4, 0.5, 0.64,
# 1.0, 1.5 s); ru: peak r_u at 3, 7 and 16 m below the surface. The
# spectra re-computed 2026-10-21 from the same runs (PGA, its time and
# r_u reproduced) by Nigam-Jennings (GM-272); Newmark at the record's step
# had Sa up to 0.6 % off and amplification up to 1.8 % (0.1 s).
EFFECTIVE_REFERENCE = {
    "as_written": {
        "pga_g": 0.3564, "pga_time_s": 4.330,
        "sa_g": [0.8291, 0.5640, 0.6774, 0.7913, 0.7534, 0.6184, 0.3248,
                 0.3111],
        "ru": [0.43, 0.97, 0.56]},
    # level, module inputs, Rayleigh at the module's site frequency
    # (1.7746 Hz) and five times it at the example's ratio there (0.40 %)
    "module_inputs": {
        "pga_g": 0.5451, "pga_time_s": 3.840,
        "sa_g": [0.9169, 0.5093, 0.6540, 0.7339, 0.6054, 0.5792, 0.3227,
                 0.3031],
        "amplification": [0.8745, 0.5219, 0.4629, 0.3695, 0.5046, 1.4081,
                          1.0395, 1.3665],
        "ru": [0.38, 0.91, 0.97]},
}


class TestEffectiveExampleMapping:

    def test_closest_inputs(self):
        """Fifteen 2 m sublayers of four 0.5 m elements; each Vs puts the
        example's G = Gr (p/101)^0.5 at the module's reference pressure;
        the damping is the example's Rayleigh ratio at the module's site
        frequency."""
        from opensees_agent.site_response import (
            _sigma_v_eff, element_permeability)
        inputs, f1 = eff.closest_api_inputs()
        layers = inputs["layers"]
        assert len(layers) == 15 and inputs["n_elem_per_layer"] == 4
        # the element gets the example's 1e-4 (a k of 9.81e-4 m/s; GM-79)
        for L in layers:
            assert element_permeability(L["permeability"]) == pytest.approx(
                1.0e-4, rel=1e-12)
        assert f1 == pytest.approx(1.7746, abs=1e-4)
        assert inputs["damping"] == pytest.approx(0.003989, abs=1e-6)
        # the loose layer's top sublayer (2-4 m) and the deepest (28-30 m)
        for i, Gr, depth in ((1, 9.0e4, 3.0), (14, 1.3e5, 29.0)):
            L = layers[i]
            p = _sigma_v_eff(layers, depth, 2.0) * (1 + 2 * L["K0"]) / 3
            assert L["density"] * L["Vs"] ** 2 == pytest.approx(
                Gr * np.sqrt(p / 101.0), rel=1e-9)


class TestAgainstTheOpenSeesPyEffectiveExample:

    @requires_opensees
    def test_port_gravity_is_hydrostatic(self):
        """The port through gravity and 1 s of shaking: pore pressure at
        the base 28 m of water (274.7 kPa; the 2 % slope tilts gravity by
        0.02 %), zero in the dry crust."""
        accel, dt = wiki.load_record()
        r = eff.run_free_field_effective(
            wiki.velocity_from_acceleration(accel[:200], dt), motion_dt=dt)
        base = r["pp_nodes"][0][0]
        assert r["pp0"][base] == pytest.approx(9.81 * 28.0, rel=0.002)
        crust = [n for n, y in r["pp_nodes"] if y > 28.0 + 1e-9]
        assert crust and all(abs(r["pp0"][n]) < 1e-9 for n in crust)
        assert np.all(np.isfinite(r["surface_accel_g"]))

    @requires_opensees
    def test_module_matches_the_port_it_can_express(self):
        """Sa from 0.1 to 1.5 s within 10 % of the port with the module's
        inputs (measured -8.7 % at 0.5 s to +4.4 %, -8.6 to +4.4 % by the
        Newmark spectrum before GM-272; -6.2 to +4.5 % with
        the fixed alpha 1e-5 before GM-260: the elements differ, see the
        module docstring), and peak r_u at 3, 7 and 16 m within 0.1
        (measured 0.36 / 0.90 / 0.98 against 0.38 / 0.91 / 0.97). Until
        GM-85 the gravity stage raised."""
        from opensees_agent.site_response import analyze_site_response
        accel, dt = wiki.load_record()
        inputs, _ = eff.closest_api_inputs()
        r = analyze_site_response(accel_history=accel, dt=dt,
                                  time_limit_s=TEST_TIME_LIMIT_S, **inputs)
        got = wiki.summarize(r.surface_accel_g, accel, dt)
        ref = EFFECTIVE_REFERENCE["module_inputs"]
        for T, g, e in zip(wiki.PERIODS_S, got["amplification"],
                           ref["amplification"]):
            assert g == pytest.approx(e, rel=0.10), f"T = {T} s"
        d = np.asarray(r.depths)
        order = np.argsort(d)
        ru = np.interp([3.0, 7.0, 16.0], d[order],
                       np.asarray(r.max_pore_pressure_ratio)[order])
        np.testing.assert_allclose(ru, ref["ru"], atol=0.1)
