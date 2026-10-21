"""
GM-83: analyze_site_response against the OpenSees wiki's own example.

The example: "Site Response Analysis of a Layered Soil Column (Total Stress
Analysis)", prepared by Christopher McGann and Pedro Arduino, University of
Washington,
https://opensees.berkeley.edu/wiki/index.php/Site_Response_Analysis_of_a_Layered_Soil_Column_(Total_Stress_Analysis)
-- a 40 m layer of PressureIndependMultiYield soil (rho 1.7 Mg/m3, Vs
250 m/s, cohesion 95 kPa) on rock (760 m/s, 2.4 Mg/m3) through a
Lysmer-Kuhlemeyer dashpot, shaken by the Gilroy No. 1 east-west record
applied as a force c * v(t) (Joyner & Chen 1975). It is the total stress
sibling of the effective stress example our module's recipe comes from.

``opensees_wiki_free_field.py`` ports the example's freeFieldSingle.tcl
line for line (a test helper, with notes on what the Tcl does); the input
motion is the first 20 s of the example's GilroyNo1EW.out, stored in
``data/`` in micro-g (the file's header says how it was derived). The
port reproduces the example's published result: surface PGA 0.353 g at
4.43 s against the wiki's Figure 2 (about 0.35 g near 4.5 s), Sa(0.4 s)
1.38 g against its Figure 3 (a peak of about 1.4 g near 0.4 s).

REFERENCE holds the port's numbers (openseespy 3.8.0, Python 3.12, WSL,
2026-10-15; regenerate with ``opensees_wiki_free_field.reference_numbers``;
the spectra re-computed 2026-10-21 from the same runs, which reproduce
every PGA, by Nigam-Jennings (GM-272): Newmark at the record's step had
them up to 3.5 % off at 0.1 s, twenty record steps):
5 %-damped spectral accelerations at PERIODS_S and the amplification
spectrum (surface / input Sa), three ways: as written, linear elastic,
and with the three inputs analyze_site_response could not set as the Tcl
does (Poisson's ratio, the Rayleigh frequencies, the peak shear strain) set
as the module sets them; and (GM-79, which made the first two inputs) as
written but for the peak shear strain, which the module given the Tcl's own
Poisson's ratio and Rayleigh damping reproduces.

What to expect, and why (measured on the 20 s record):

* Linear elastic, with the Tcl's own Rayleigh damping (2 % at 0.2 and
  20 Hz, ``rayleigh_freqs_hz``): the same column, so within 3 % from 0.1
  to 1.5 s and PGA within 5 % (measured: both to 1e-7). With damping
  matched at the site frequency instead (0.41 %, Rayleigh at it and five
  times it), the module's column is less damped above about 8 Hz: Sa at
  0.1 s 8.0 % above the port's, PGA +2.7 % (GM-272).
* Nonlinear, against the port with the three inputs set as the module
  sets them: the same model (on this column the Tcl's quad, an SSPquad
  and the module's dry SSPquadUP give the same motion to 1e-12), so it
  should agree to a few per cent, and it does: amplification within
  0.1 % from 0.1 to 1.5 s, PGA -0.1 %. Until GM-84 it did not: the
  dynamic stage ran KrylovNewton at NormDispIncr 1e-3 m, which accepted
  unconverged steps when elements are small (here 0.25 m, where 1 mm is
  0.4 % strain), adding high-frequency noise and taking 9-35 % off Sa
  from 0.1 to 1.5 s. It now runs Newton, as the Tcl does.
* Nonlinear, against the port as written: within 15 % from 0.3 to 1.5 s
  (measured -2.8 to +11.6 %): Poisson's ratio 0 (the Tcl) against 0.495
  (the module's clay) moves Sa at 0.64-1.5 s by 7-12 %. PGA comes out
  higher in the module (+23 %): its column is less damped at high
  frequency, where a nonlinear column's peaks live.
"""

import functools

import numpy as np
import pytest

from opensees_agent.opensees_utils import has_opensees
from opensees_agent.tests import opensees_wiki_free_field as wiki

requires_opensees = pytest.mark.skipif(
    not has_opensees(), reason="openseespy not installed")

# Each run here takes 7-15 s; a stuck solver step fails instead of hanging
TEST_TIME_LIMIT_S = 300.0

# The port's numbers on the stored record (see the module docstring).
# sa_g / amplification are at wiki.PERIODS_S = 0.1, 0.2, 0.3, 0.4, 0.5,
# 0.64 (the site period), 1.0, 1.5 s.
REFERENCE = {
    "input_sa_g": [1.0485, 0.9758, 1.4128, 1.9864, 1.1999, 0.4113, 0.3104,
                   0.2218],
    "as_written": {
        "pga_g": 0.3528, "pga_time_s": 4.430,
        "sa_g": [0.9571, 0.7660, 1.0947, 1.3777, 0.8000, 0.3882, 0.4193,
                 0.3644],
        "amplification": [0.9128, 0.7850, 0.7749, 0.6936, 0.6668, 0.9438,
                          1.3508, 1.6431]},
    "linear": {
        "pga_g": 0.7895, "pga_time_s": 4.300,
        "sa_g": [1.0541, 1.8435, 1.7781, 2.5961, 2.1181, 0.7405, 0.4983,
                 0.2781],
        "amplification": [1.0053, 1.8891, 1.2586, 1.3070, 1.7652, 1.8002,
                          1.6055, 1.2541]},
    "api_expressible": {
        "pga_g": 0.4331, "pga_time_s": 4.525,
        "sa_g": [0.9563, 0.7500, 1.0697, 1.3629, 0.7781, 0.4296, 0.4681,
                 0.3885],
        "amplification": [0.9121, 0.7685, 0.7572, 0.6861, 0.6485, 1.0443,
                          1.5082, 1.7518]},
    # The port as written but for the peak shear strain (0.1), which is
    # all the module cannot set since GM-79 (openseespy 3.8.0, WSL,
    # 2026-10-20; wiki.OWN_INPUTS).
    "own_inputs": {
        "pga_g": 0.3500, "pga_time_s": 4.430,
        "sa_g": [0.9485, 0.7675, 1.0881, 1.3686, 0.7934, 0.3866, 0.4180,
                 0.3637],
        "amplification": [0.9046, 0.7865, 0.7702, 0.6890, 0.6612, 0.9399,
                          1.3468, 1.6399]},
}


@functools.lru_cache(maxsize=None)
def _ours(linear_elastic):
    """analyze_site_response on the example's column (one run per kind)."""
    from opensees_agent.site_response import analyze_site_response
    accel, dt = wiki.load_record()
    r = analyze_site_response(accel_history=accel, dt=dt,
                              linear_elastic=linear_elastic,
                              time_limit_s=TEST_TIME_LIMIT_S,
                              **wiki.closest_api_inputs())
    return wiki.summarize(r.surface_accel_g, accel, dt)


def _assert_spectrum(got, ref, rel, what, periods=wiki.PERIODS_S):
    for T, g, e in zip(wiki.PERIODS_S, got, ref):
        if T in periods:
            assert g == pytest.approx(e, rel=rel), f"{what} at T = {T} s"


# ===========================================================================
# Tier 1: the stored record and the damping mapping (no openseespy)
# ===========================================================================

class TestStoredRecord:

    def test_record_is_the_examples_first_20_s(self):
        """4,000 values at 0.005 s; the peak 0.473148 g at 4.225 s, as in
        the example's GilroyNo1EW.out."""
        accel, dt = wiki.load_record()
        assert dt == 0.005
        assert len(accel) == 4000
        k = int(np.argmax(np.abs(accel)))
        assert abs(accel[k]) == pytest.approx(0.473148, abs=1e-6)
        assert k * dt == pytest.approx(4.225)
        assert accel[0] == pytest.approx(-0.002252, abs=1e-6)

    def test_velocity_matches_the_examples_velocity_file(self):
        """processMotion.m's rule (a * 9.81, cumtrapz from 0, times dt)
        on the stored values: the first, the peak and the 20 s values of
        the example's velocityHistory.out."""
        accel, dt = wiki.load_record()
        v = wiki.velocity_from_acceleration(accel, dt)
        assert v[0] == 0.0
        assert v[1] == pytest.approx(-1.2591160e-04, abs=2e-9)
        assert np.max(np.abs(v)) == pytest.approx(0.3386837, abs=2e-6)

    def test_input_spectrum(self):
        """The input's 5 %-damped Sa at PERIODS_S, the denominators of
        REFERENCE's amplification spectra."""
        accel, dt = wiki.load_record()
        s = wiki.summarize(accel, accel, dt)
        _assert_spectrum(s["sa_g"], REFERENCE["input_sa_g"], 1e-3,
                         "input Sa")
        assert s["amplification"] == pytest.approx([1.0] * 8)

    def test_the_examples_damping_at_the_site_frequency(self):
        """2 % Rayleigh at 0.2 and 20 Hz is 0.41 % at the column's site
        frequency (1.5625 Hz), the ratio the module is given."""
        assert wiki.SITE_FREQUENCY_HZ == pytest.approx(1.5625)
        assert wiki.rayleigh_ratio(wiki.SITE_FREQUENCY_HZ) == pytest.approx(
            0.004082, abs=1e-6)
        assert wiki.rayleigh_ratio(20.0) == pytest.approx(0.02, rel=1e-6)
        inputs = wiki.closest_api_inputs()
        assert inputs["damping"] == pytest.approx(0.004082, abs=1e-6)
        # the module's cohesi = sqrt(3)/2 * su lands on the Tcl's 95 kPa
        su = inputs["layers"][0]["su"]
        assert np.sqrt(3.0) / 2.0 * su == pytest.approx(95.0)


# ===========================================================================
# Tier 2: the port and the module (openseespy)
# ===========================================================================

class TestAgainstTheOpenSeesWikiExample:

    @requires_opensees
    def test_port_reproduces_the_published_example(self):
        """The port, as written, on the stored record: the wiki's Figure 2
        (surface PGA about 0.35 g near 4.5 s) and Figure 3 (Sa peak about
        1.4 g near 0.4 s), and REFERENCE within 1 %."""
        accel, dt = wiki.load_record()
        r = wiki.run_free_field_single(
            wiki.velocity_from_acceleration(accel, dt), motion_dt=dt)
        assert r["numEleY"] == 160 and r["dT"] == 0.005
        s = wiki.summarize(r["surface_accel_g"], accel, dt)
        assert s["pga_g"] == pytest.approx(0.35, abs=0.02)
        assert s["pga_time_s"] == pytest.approx(4.45, abs=0.1)
        assert s["sa_g"][wiki.PERIODS_S.index(0.4)] == pytest.approx(
            1.4, abs=0.1)
        ref = REFERENCE["as_written"]
        assert s["pga_g"] == pytest.approx(ref["pga_g"], rel=0.01)
        _assert_spectrum(s["sa_g"], ref["sa_g"], 0.01, "Sa")

    @requires_opensees
    def test_linear_column_matches_the_port(self):
        """Linear elastic (no updateMaterialStage in the port,
        linear_elastic=True in the module) with the Tcl's own Rayleigh
        damping (2 % at 0.2 and 20 Hz, ``rayleigh_freqs_hz``; its Poisson's
        ratio 0 too, which changes nothing in a linear column):
        amplification within 3 % from 0.1 to 1.5 s and PGA within 5 %
        (measured: both to 1e-7, the same column).

        Until GM-272 the module ran here with damping matched at the site
        frequency (0.41 %, Rayleigh at it and five times it), which damps
        less than the Tcl's above about 8 Hz: Sa at 0.1 s is 8.0 % above
        the port's (lsim on both records agrees) and PGA 2.7 %. It passed
        at 3 % (measured +1.9 %) only because the Newmark spectrum at the
        record's step read the module's 0.1 s ordinate 2.5 % low and the
        port's 3.3 % high."""
        from opensees_agent.site_response import analyze_site_response
        accel, dt = wiki.load_record()
        r = analyze_site_response(accel_history=accel, dt=dt,
                                  linear_elastic=True,
                                  time_limit_s=TEST_TIME_LIMIT_S,
                                  **wiki.own_api_inputs())
        got = wiki.summarize(r.surface_accel_g, accel, dt)
        ref = REFERENCE["linear"]
        assert got["pga_g"] == pytest.approx(ref["pga_g"], rel=0.05)
        _assert_spectrum(got["amplification"], ref["amplification"], 0.03,
                         "amplification")

    @requires_opensees
    def test_nonlinear_column_against_the_port_as_written(self):
        """Amplification within 15 % from 0.3 to 1.5 s (measured -2.6 to
        +11.7 %, the positive side from Poisson's ratio); PGA higher, by
        10 to 35 % (measured +23 %: less damping above 8 Hz). Until GM-84
        the unconverged steps scattered it (-9.8 to -15.3 %, PGA +29 %;
        -17 % at 1.0 s and PGA +38 % on the unrounded full record), so
        this was pinned at 25 % and PGA under +50 %."""
        got = _ours(False)
        ref = REFERENCE["as_written"]
        _assert_spectrum(got["amplification"], ref["amplification"], 0.15,
                         "amplification", periods=(0.3, 0.4, 0.5, 0.64,
                                                   1.0, 1.5))
        assert 1.10 < got["pga_g"] / ref["pga_g"] < 1.35

    @requires_opensees
    def test_nonlinear_column_matches_the_port_it_can_express(self):
        """Against the port with Poisson's ratio 0.495, Rayleigh damping
        at the site frequency and five times it (0.41 %) and peak shear
        strain 0.1, as the module sets them: the same model, so
        amplification within 3 % from 0.1 to 1.5 s and PGA within 5 %.
        Measured (Newton, GM-84): amplification within 0.1 %, PGA -0.1 %.
        Both solved to a 100 times tighter test (Newton or KrylovNewton
        at 1e-5 m) move amplification by under 0.6 % and PGA by -4 to
        -5 % (a peak is one spike). Before GM-84 (KrylovNewton at 1e-3 m):
        PGA +5 %, amplification -35 to -9 %."""
        got = _ours(False)
        ref = REFERENCE["api_expressible"]
        assert got["pga_g"] == pytest.approx(ref["pga_g"], rel=0.05)
        _assert_spectrum(got["amplification"], ref["amplification"], 0.03,
                         "amplification")

    @requires_opensees
    def test_nonlinear_column_with_the_examples_own_inputs(self):
        """GM-79: with the Tcl's own Poisson's ratio (0) and Rayleigh
        damping (2 % at 0.2 and 20 Hz) given through ``layers[i]['nu']``
        and ``rayleigh_freqs_hz``, against the port with only the peak
        shear strain at the module's 0.1: the same model, so amplification
        within 3 % from 0.1 to 1.5 s and PGA within 5 %. Measured
        2026-10-20: amplification within 0.03 %, PGA -0.05 %; against the
        port as written (peak strain 0.05) PGA -0.9 %."""
        from opensees_agent.site_response import analyze_site_response
        accel, dt = wiki.load_record()
        r = analyze_site_response(accel_history=accel, dt=dt,
                                  time_limit_s=TEST_TIME_LIMIT_S,
                                  **wiki.own_api_inputs())
        got = wiki.summarize(r.surface_accel_g, accel, dt)
        ref = REFERENCE["own_inputs"]
        assert got["pga_g"] == pytest.approx(ref["pga_g"], rel=0.05)
        _assert_spectrum(got["amplification"], ref["amplification"], 0.03,
                         "amplification")
        assert r.rayleigh_freqs_hz == [0.2, 20.0]
        (nu,) = [c for c in r.choices if c["what"].endswith("Poisson's ratio")]
        assert nu["chosen"] == 0.0 and nu["how"] == "given"
