"""pystrata_agent.compat: the same soils and the same answers on pystrata
0.5 (Python 3.12) and 0.6 (Python 3.13+), BACKLOG INF-68.

The reference numbers are pystrata 0.5.4's (run 2026-11-06 on Python
3.12, `module_work/inf68/CHECK.md`); pystrata 0.6.0 on Python 3.13 gave
the same to 1e-15. INF-68 asks for the same amplification and compatible
Vs within 1 %, which is the tolerance pinned here.
"""

import os
import sys

import numpy as np
import pytest

from pystrata_agent.pystrata_utils import has_pystrata

requires_pystrata = pytest.mark.skipif(
    not has_pystrata(), reason="pystrata not installed")


@pytest.mark.skipif(not os.environ.get("CI") or sys.version_info < (3, 13),
                    reason="only CI's Python 3.13 job must have pystrata 0.6")
def test_ci_313_job_imports_pystrata():
    """CI's 3.13 job installs pystrata 0.6; if it failed to import, every
    0.6 test here would skip and the job would stay green."""
    assert has_pystrata(), "pystrata did not import on CI's Python 3.13 job"

KPA_TO_ATM = 1.0 / 101.325
STRAINS = np.logspace(-6, -1.5, 20)


def _modified_hyperbolic(strains, strain_ref, curvature, masing, dmin):
    """Darendeli (2001) modified hyperbolic G/Gmax and damping, written
    out here so the check does not lean on either pystrata."""
    g = 1.0 / (1.0 + (strains / strain_ref) ** curvature)
    sp, rp = strains * 100.0, strain_ref * 100.0
    a1 = (100.0 / np.pi) * (
        4.0 * (sp - rp * np.log((sp + rp) / rp)) / (sp ** 2 / (sp + rp))
        - 2.0)
    c1 = -1.1143 * curvature ** 2 + 1.8618 * curvature + 0.2523
    c2 = 0.0805 * curvature ** 2 - 0.0710 * curvature - 0.0095
    c3 = -0.0005 * curvature ** 2 + 0.0002 * curvature + 0.0003
    dm = c1 * a1 + c2 * a1 ** 2 + c3 * a1 ** 3
    d = np.maximum.accumulate(masing * dm * g ** 0.1 / 100.0) + dmin
    return g, d


@requires_pystrata
class TestSoilTypes:

    @pytest.mark.parametrize("sigma", [30.0, 101.3, 300.0])
    def test_darendeli(self, sigma):
        import pystrata
        from pystrata_agent import compat
        st = compat.darendeli_soil_type(pystrata, 18.0, plas_index=20,
                                        ocr=2.0, stress_mean=sigma)
        p = sigma * KPA_TO_ATM
        ref = (0.0352 + 0.0010 * 20 * 2.0 ** 0.3246) * p ** 0.3483 / 100
        dmin = (0.8005 + 0.0129 * 20 * 2.0 ** -0.1069) * p ** -0.2889 / 100
        g, d = _modified_hyperbolic(STRAINS, ref, 0.9190,
                                    0.6329 - 0.00566 * np.log(10), dmin)
        assert np.allclose(st.mod_reduc.strains, STRAINS, rtol=1e-12)
        assert np.allclose(st.mod_reduc.values, g, rtol=1e-12)
        assert np.allclose(st.damping.values, d, rtol=1e-12)
        assert st.unit_wt == pytest.approx(18.0)

    @pytest.mark.parametrize("sigma", [30.0, 101.3, 300.0])
    def test_menq(self, sigma):
        import pystrata
        from pystrata_agent import compat
        st = compat.menq_soil_type(pystrata, 20.0, uniformity_coeff=3.0,
                                   diam_mean=0.5, stress_mean=sigma)
        p = sigma * KPA_TO_ATM
        ref = 0.12 * 3.0 ** -0.6 * p ** (0.5 * 3.0 ** -0.15) / 100
        dmin = 0.55 * 3.0 ** 0.1 * 0.5 ** -0.3 * p ** -0.08 / 100
        curv = 0.86 + 0.1 * np.log10(p)
        g, d = _modified_hyperbolic(STRAINS, ref, curv,
                                    0.6329 - 0.00566 * np.log(10), dmin)
        assert np.allclose(st.mod_reduc.values, g, rtol=1e-12)
        assert np.allclose(st.damping.values, d, rtol=1e-12)

    def test_custom_keeps_its_points(self):
        import pystrata
        from pystrata_agent import compat
        st = compat.custom_soil_type(
            pystrata, 17.0, [1e-4, 1e-3, 1e-2, 1e-1],
            [1.0, 0.92, 0.55, 0.10], [0.01, 0.025, 0.07, 0.20])
        assert list(st.mod_reduc.values) == [1.0, 0.92, 0.55, 0.10]
        assert list(st.damping.values) == [0.01, 0.025, 0.07, 0.20]
        assert st.is_nonlinear

    def test_linear(self):
        import pystrata
        from pystrata_agent import compat
        st = compat.linear_soil_type(pystrata, 24.0, 0.01)
        assert st.mod_reduc is None
        assert st.damping_min == pytest.approx(0.01)
        assert not st.is_nonlinear

    def test_tolerance_unit_follows_the_api(self):
        """0.5 compares in percent, 0.6 in decimal."""
        import pystrata
        from pystrata_agent import compat
        tol = compat.eql_tolerance(pystrata, 0.01)
        if compat.has_factories(pystrata):
            assert tol == pytest.approx(0.01)
        else:
            assert tol == pytest.approx(1.0)


def _dar(t=10.0, vs=250, uw=18.0, pi=30):
    return {"thickness": t, "Vs": vs, "unit_wt": uw,
            "soil_model": "darendeli", "plas_index": pi}


def _menq(t=8.0, vs=350, uw=20.0):
    return {"thickness": t, "Vs": vs, "unit_wt": uw, "soil_model": "menq"}


def _custom():
    return {"thickness": 12.0, "Vs": 180, "unit_wt": 17.0,
            "soil_model": "custom",
            "strains": [0.0001, 0.001, 0.01, 0.1],
            "mod_reduc": [1.0, 0.92, 0.55, 0.10],
            "damping_values": [0.01, 0.025, 0.07, 0.20]}


_ROCK = {"thickness": 0, "Vs": 760, "unit_wt": 24.0,
         "soil_model": "linear", "damping": 0.01}

# (layers, kwargs, amplification, iterations, lowest compatible Vs m/s)
# from pystrata 0.5.4.
REFERENCE = {
    "darendeli": ([_dar(), _ROCK], {}, 1.44960, 6, 149.923),
    "menq": ([_menq(), _ROCK], {}, 1.07955, 6, 253.331),
    "custom": ([_custom(), _ROCK], {}, 1.85878, 4, 170.097),
    "darendeli_over_menq": (
        [_dar(6.0, 200, 17.5, 20), _menq(10.0, 320, 20.0), _ROCK],
        {"target_pga_g": 0.2}, 2.21588, 11, 89.985),
}


@requires_pystrata
@pytest.mark.parametrize("case", sorted(REFERENCE))
def test_eql_matches_pystrata_054(case):
    from pystrata_agent import analyze_eql_site_response
    layers, kw, amp, n_it, vs_min = REFERENCE[case]
    r = analyze_eql_site_response(layers=layers, motion="synthetic_pulse",
                                  **kw)
    assert r.converged is True
    assert r.n_iterations == n_it
    assert r.amplification_factor == pytest.approx(amp, rel=0.01)
    assert float(np.min(r.compatible_Vs)) == pytest.approx(vs_min, rel=0.01)
