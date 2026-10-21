"""
GM-80 (stress-strain curves for the site-response materials) and GM-79
(solver-step counts and the inputs that were fixed: Poisson's ratio, the
Rayleigh frequencies, the sand constants, the element count per layer).

Tier 1 (no openseespy): the curves, the backbone fit, the commands the
column sends (a recording stand-in for openseespy), the input checks.
Tier 2 (openseespy, and pystrata for the equivalent-linear runs): the
amplification of the GM-77 dry sand column at 0.30 g with each curve.
"""

import math

import numpy as np
import pytest

from opensees_agent import curves as C
from opensees_agent.site_response import (
    _layer_plans, _run_site_response_model, analyze_site_response,
    _validate_site_response_inputs)
from opensees_agent.tests.test_opensees_agent import (
    _FakeOps, PDMY02_ARG_NAMES, SAND33, CLAY60, requires_opensees,
    TEST_TIME_LIMIT_S)


def _has_pystrata():
    try:
        import pystrata  # noqa: F401
        return True
    except ImportError:
        return False


requires_pystrata = pytest.mark.skipif(not _has_pystrata(),
                                       reason="pystrata not installed")


# ===========================================================================
# The curves
# ===========================================================================

class TestCurves:

    @requires_pystrata
    @pytest.mark.parametrize("sigma_m", [30.0, 101.3, 300.0])
    def test_darendeli_and_menq_match_pystrata(self, sigma_m):
        """The same constants as the pystrata implementation that
        pystrata_agent runs: G/Gmax and damping to machine precision."""
        import pystrata
        strains = np.logspace(-6, -1.5, 20)
        d = pystrata.site.DarendeliSoilType(
            18.0, plas_index=15, ocr=2, stress_mean=sigma_m,
            strains=strains)
        g, dm = C.modified_hyperbolic(
            strains, *C.darendeli_params(sigma_m, 15, 2))
        assert np.allclose(g, d.mod_reduc.values, rtol=1e-12)
        assert np.allclose(dm, d.damping.values, rtol=1e-12)
        m = pystrata.site.MenqSoilType(
            18.0, uniformity_coeff=3, diam_mean=0.5, stress_mean=sigma_m,
            strains=strains)
        g, dm = C.modified_hyperbolic(
            strains, *C.menq_params(sigma_m, 3, 0.5))
        assert np.allclose(g, m.mod_reduc.values, rtol=1e-12)
        assert np.allclose(dm, m.damping.values, rtol=1e-12)

    def test_darendeli_reference_strain_at_one_atmosphere(self):
        """PI 0, OCR 1, 1 atm: gamma_r = 0.0352 %, D_min = 0.8005 %."""
        gr, a, dmin, b = C.darendeli_params(C.P_ATM_KPA)
        assert gr == pytest.approx(0.000352)
        assert a == 0.9190 and dmin == pytest.approx(0.008005)
        assert C.modified_hyperbolic([gr], gr, a, dmin, b)[0][0] \
            == pytest.approx(0.5)

    def test_backbone_ends_at_the_strength(self):
        g_max, tau_f = 76000.0, C.sand_strength(33.0, 60.0)
        pairs, info = C.fit_backbone(
            lambda g: 1.0 / (1.0 + (g / 2e-3) ** 0.92), g_max, tau_f)
        strains, ratios = np.array(pairs[0::2]), np.array(pairs[1::2])
        taus = g_max * strains * ratios
        assert taus[-1] == pytest.approx(tau_f)
        assert np.all(np.diff(strains) > 0) and np.all(np.diff(taus) > 0)
        assert info["how"] == "the curve reaches the strength"
        assert info["strength_strain_pct"] < 10.0
        # what the material re-derives from the last pair (OpenSees
        # source): M = sqrt(3) tau / p, sin(phi) = 3M / (6 + M)
        M = math.sqrt(3.0) * taus[-1] / 60.0
        assert math.degrees(math.asin(3 * M / (6 + M))) \
            == pytest.approx(33.0)

    def test_a_curve_below_the_strength_ends_at_it_at_10_percent(self):
        pairs, info = C.fit_backbone(lambda g: 1.0 / (1.0 + g / 1e-5),
                                     50000.0, 200.0)
        assert pairs[-2] == pytest.approx(0.1)
        assert 50000.0 * pairs[-2] * pairs[-1] == pytest.approx(200.0)
        assert "stays below" in info["how"]

    def test_masing_damping_of_a_hyperbolic_backbone(self):
        """Against the closed form for a hyperbolic backbone (Kramer
        1996, Sec. 6.4), within the 31-point discretisation."""
        g_max, gr = 50000.0, 1e-3
        pairs, _ = C.fit_backbone(lambda g: 1.0 / (1.0 + g / gr),
                                  g_max, 1e9)
        x = np.array([1e-4, 1e-3, 1e-2])
        closed = ((4 / np.pi) * (1 + gr / x)
                  * (1 - np.log(1 + x / gr) * gr / x) - 2 / np.pi)
        assert np.allclose(C.masing_damping(pairs, g_max, x), closed,
                           rtol=0.08, atol=0.002)

    @pytest.mark.parametrize("table, match", [
        ({"strain_pct": [0.1, 0.01], "G_Gmax": [0.5, 0.9]}, "rise"),
        ({"strain_pct": [0.01, 0.1], "G_Gmax": [0.9, 1.2]}, r"\(0, 1\]"),
        ({"strain_pct": [0.01, 0.1, 1.0], "G_Gmax": [0.9, 0.5, 0.01]},
         "shear stress"),
        ({"strain": [0.01, 0.1], "G_Gmax": [0.9, 0.5]}, "strain_pct"),
    ])
    def test_user_table_checks(self, table, match):
        with pytest.raises(ValueError, match=match):
            C.table_ggmax(table, "t")


# ===========================================================================
# The commands the column sends (no openseespy)
# ===========================================================================

def _run_fake(layers, rayleigh_freqs_hz=None, n_elem_per_layer=2,
              **fake_kw):
    ops = _FakeOps(**fake_kw)
    acc = 0.1 * np.sin(np.linspace(0.0, 6.0, 50))
    plans = _layer_plans(layers, 0.0, n_elem_per_layer)
    result = _run_site_response_model(
        ops, layers=layers, accel_g=acc, dt_motion=0.01, gwt_depth=0.0,
        bedrock_Vs=760.0, bedrock_density=2.4, damping=0.02,
        scale_factor=1.0, n_elem_per_layer=n_elem_per_layer,
        motion_name="test", plans=plans,
        rayleigh_freqs_hz=rayleigh_freqs_hz)
    return ops, result


class TestWiring:

    def test_default_backbone_unchanged(self):
        ops, _ = _run_fake([SAND33])
        (args,) = ops.named("nDMaterial")
        a = dict(zip(PDMY02_ARG_NAMES, args))
        assert len(args) == len(PDMY02_ARG_NAMES)
        assert a["noYieldSurf"] == 20 and a["contrac2"] == 5.0

    def test_user_backbone_pairs_follow_a_negative_noYieldSurf(self):
        ops, _ = _run_fake([dict(SAND33, curve="darendeli")])
        (args,) = ops.named("nDMaterial")
        n = args[15]
        assert isinstance(n, int) and n < 0
        pairs = args[16:16 - 2 * n]
        rest = args[16 - 2 * n:]
        # the named arguments after the pairs sit where they belong
        assert rest == (5.0, 3.0, 1.0, 0.0, 0.77, 0.9, 0.02, 0.7, 101.0)
        g_max, p_ref = args[4], args[8]
        tau_last = g_max * pairs[-2] * pairs[-1]
        assert tau_last == pytest.approx(C.sand_strength(33.0, p_ref))

    def test_clay_user_backbone_ends_at_su(self):
        ops, _ = _run_fake([dict(CLAY60, curve="darendeli",
                                 plas_index=30)])
        (args,) = ops.named("nDMaterial")
        n = args[11]
        assert n < 0
        pairs = args[12:]
        assert len(pairs) == -2 * n
        assert args[4] * pairs[-2] * pairs[-1] == pytest.approx(60.0)

    def test_sand_constants_nu_and_element_count(self):
        dense = dict(SAND33, contrac1=0.01, dilat1=0.4, nu=0.25, n_elem=5)
        ops, r = _run_fake([dense, dict(CLAY60, nu=0.45)])
        mats = {a[0]: a for a in ops.named("nDMaterial")}
        sand = mats["PressureDependMultiYield02"]
        clay = mats["PressureIndependMultiYield"]
        s = dict(zip(PDMY02_ARG_NAMES, sand))
        assert (s["contrac1"], s["dilat1"]) == (0.01, 0.4)
        assert (s["contrac3"], s["dilat3"]) == (0.23, 0.27)
        G = s["refShearModul"]
        assert s["refBulkModul"] == pytest.approx(
            2 * G * 1.25 / (3 * 0.5))
        Gc = clay[4]
        assert clay[5] == pytest.approx(2 * Gc * 1.45 / (3 * 0.1))
        elements = [a for a in ops.named("element") if a[0] == "SSPquadUP"]
        assert len(elements) == 5 + 2 and r.n_elements == 7

    def test_rayleigh_frequencies(self):
        ops, r = _run_fake([SAND33], rayleigh_freqs_hz=[1.0, 10.0])
        (a0, a1, *_), = ops.named("rayleigh")
        w1, w2 = 2 * math.pi, 20 * math.pi
        assert a0 == pytest.approx(2 * 0.02 * w1 * w2 / (w1 + w2))
        assert a1 == pytest.approx(2 * 0.02 / (w1 + w2))
        assert r.rayleigh_freqs_hz == [1.0, 10.0]
        _, r = _run_fake([SAND33])
        assert r.rayleigh_freqs_hz == pytest.approx([2.5, 12.5])

    def test_steps_counted_by_algorithm(self):
        _, r = _run_fake([SAND33], fail_first=("Newton",))
        st = r.solver_steps
        assert st["n_steps"] == 49
        assert st["by_algorithm"] == {"Newton": 48, "KrylovNewton": 1,
                                      "ModifiedNewton": 0}
        assert st["fallback_steps"] == 1 and st["substepped_steps"] == 0

    def test_substepped_steps_counted_and_named(self):
        _, r = _run_fake([SAND33], fail_first_full_step=True)
        st = r.solver_steps
        assert st["substepped_steps"] == 1
        assert st["substepped_times_s"] == [0.01]
        assert "substeps" in st["note"]
        assert sum(st["by_algorithm"].values()) == 48


# ===========================================================================
# Input checks and choices (no openseespy needed for the checks)
# ===========================================================================

def _check(layer, **kw):
    _validate_site_response_inputs([layer], 0.0, 760.0, 2.4, 0.02, 1.0, 4,
                                   kw.get("rayleigh_freqs_hz"))


class TestInputs:

    @pytest.mark.parametrize("layer, match", [
        (dict(SAND33, curve="seed_idriss"), "curve"),
        (dict(SAND33, curve="menq", diam_mean=0.4), "uniformity_coeff"),
        (dict(SAND33, curve="user"), "G_Gmax_curve"),
        (dict(SAND33, plas_index=10), "darendeli"),
        (dict(SAND33, n_cycles=10), "curve='darendeli'"),
        (dict(CLAY60, contrac1=0.1), "sand"),
        (dict(SAND33, n_elem=0), "n_elem"),
        (dict(SAND33, nu=0.5), "nu"),
        (dict(SAND33, dilat1=-1), "dilat1"),
        # GM-79
        (dict(SAND33, permeability=0.0), "permeability"),
        (dict(SAND33, K0=-0.5), "K0"),
        (dict(SAND33, e_init=0.0), "e_init"),
        (dict(SAND33, PTAng=0.0), "PTAng"),
        (dict(SAND33, n_surf=0), "n_surf"),
        (dict(SAND33, n_surf=20, curve="darendeli"), "pairs set the yield"),
    ])
    def test_refused(self, layer, match):
        with pytest.raises(ValueError, match=match):
            _check(layer)

    @pytest.mark.parametrize("freqs", [[5.0, 1.0], [0, 2.0], [1.0]])
    def test_rayleigh_frequencies_refused(self, freqs):
        with pytest.raises(ValueError, match="rayleigh_freqs_hz"):
            _check(SAND33, rayleigh_freqs_hz=freqs)

    def test_layer_curve_report(self):
        (p,) = _layer_plans([dict(SAND33, curve="menq",
                                  uniformity_coeff=2, diam_mean=0.4)],
                            50.0, 4)
        rep = p["report"]
        assert rep["curve"] == "menq"
        assert len(rep["G_Gmax"]) == len(rep["damping_pct"]) \
            == len(rep["backbone_masing_damping_pct"]) == 4
        assert rep["G_Gmax"][0] > 0.95 > rep["G_Gmax"][-1]


def _choices(layers, **kw):
    from opensees_agent.site_response import _site_response_choices
    plans = _layer_plans(layers, 50.0, 4)
    return {c["what"]: c for c in _site_response_choices(
        plans, 0.02, kw.get("rayleigh_freqs_hz"), 4, False)}


class TestChoices:
    """What the column was built with, given or default (GM-115)."""

    def test_darendeli_inputs_left_at_default_are_named(self):
        ch = _choices([dict(CLAY60, curve="darendeli", plas_index=30)])
        c = ch["layers[0] (clay) darendeli curve inputs"]
        assert c["chosen"] == "PI 30, OCR 1, 1 Hz, 10 cycles"
        assert c["how"] == "given"
        assert "left at default: OCR 1, 1 Hz, 10 cycles" in c["basis"]
        c = _choices([dict(CLAY60, curve="darendeli")])[
            "layers[0] (clay) darendeli curve inputs"]
        assert c["how"] == "default" and "PI 0" in c["basis"]

    def test_how_the_backbone_meets_the_strength(self):
        ch = _choices([dict(SAND33, curve="darendeli")])
        c = ch["layers[0] (sand) backbone at large strain"]
        assert c["how"] == "module"
        assert "stiffer segment" in c["chosen"] and "91.2 kPa" in c["chosen"]
        c = _choices([dict(SAND33, curve="menq", uniformity_coeff=2,
                           diam_mean=0.4)])[
            "layers[0] (sand) backbone at large strain"]
        assert c["chosen"].startswith("the curve to the layer's strength")
        assert "layers[0] (sand) backbone at large strain" not in \
            _choices([SAND33])

    def test_clay_poissons_ratio_reports_the_cap(self):
        c = _choices([dict(CLAY60, nu=0.499)])[
            "layers[0] (clay) Poisson's ratio"]
        assert c["chosen"] == pytest.approx(0.495, abs=1e-4)
        assert "capped" in c["basis"]
        c = _choices([CLAY60])["layers[0] (clay) Poisson's ratio"]
        assert c["chosen"] == 0.495 and c["how"] == "default"

    def test_k0_places_only_the_reference_pressure(self):
        """GM-79. Dry sand, mid-layer sigma'v 186.4 kPa: Jaky K0 0.455
        gives the reference pressure 118.7 kPa; the column's own
        horizontal stress after gravity is nu / (1 - nu) = 0.429 of the
        vertical (measured 0.4286 in OpenSees), a mean stress of 115.4
        kPa, so its Vs at mid-layer is (115.4 / 118.7)^0.25 - 1 = -0.7 %
        off the layer's."""
        c = _choices([SAND33])["layers[0] (sand) K0 and reference pressure"]
        assert c["chosen"] == "K0 0.455: reference pressure 118.7 kPa"
        assert c["how"] == "default" and c["basis"].startswith("Jaky")
        assert "(0.429 at Poisson's ratio 0.3)" in c["basis"]
        assert "115.4 kPa" in c["basis"] and "-0.7 %" in c["basis"]
        assert "nu = K0 / (1 + K0) (0.313)" in c["alternatives"][1]
        c = _choices([dict(SAND33, K0=0.5)])[
            "layers[0] (sand) K0 and reference pressure"]
        assert c["how"] == "given" and c["chosen"].startswith("K0 0.500")

    def test_phase_transformation_void_ratio_surfaces_permeability(self):
        ch = _choices([dict(SAND33, phi=24.0)])
        c = ch["layers[0] (sand) phase transformation angle"]
        assert c["chosen"] == "24 deg" and c["how"] == "default"
        assert "26 capped at phi" in c["basis"]
        c = ch["layers[0] (sand) void ratio"]
        assert c["chosen"] == 0.77 and "Dr = 40 %" in c["basis"]
        assert ch["layers[0] (sand) yield surfaces"]["chosen"] == 20
        c = ch["layers[0] (sand) permeability"]
        assert c["chosen"] == ("0.0001 m/s (the element takes k / (rho_f g) "
                               "= 1.02e-05)")
        assert c["how"] == "default" and "1e-3 to 1e-1 cm/s" in c["basis"]
        ch = _choices([dict(CLAY60, permeability=1e-9, e_init=1.1,
                            n_surf=25)])
        assert ch["layers[0] (clay) void ratio"]["chosen"] == 1.1
        assert ch["layers[0] (clay) yield surfaces"]["chosen"] == 25
        c = ch["layers[0] (clay) permeability"]
        assert c["how"] == "given" and c["chosen"].startswith("1e-09 m/s")
        assert "layers[0] (clay) phase transformation angle" not in ch
        # a fitted curve's pairs set the surfaces; a clay's reference
        # pressure is reported only where a curve is evaluated at it
        ch = _choices([dict(CLAY60, curve="darendeli")])
        c = ch["layers[0] (clay) yield surfaces"]
        assert c["how"] == "module" and 2 < c["chosen"] <= 31
        c = ch["layers[0] (clay) reference pressure"]
        assert c["chosen"].startswith("0.67 sigma'v")
        assert "layers[0] (clay) reference pressure" not in _choices([CLAY60])

    def test_layer_inputs_a_call_did_not_get_from_the_user_are_assumed(self):
        """GM-180: a nested value reads 'assumed' when the call says its
        layers are not the user's; a module never writes 'given' itself."""
        from geotech_common.choices import call_inputs
        layers = [dict(SAND33, curve="darendeli", nu=0.25, contrac1=0.01)]
        with call_inputs(["layers"], assumed=["layers"]):
            ch = _choices(layers)
        for what in ("stress-strain curve", "Poisson's ratio",
                     "contraction / dilation constants"):
            assert ch[f"layers[0] (sand) {what}"]["how"] == "assumed"
        assert ch["layers[0] (sand) darendeli curve inputs"]["how"] \
            == "default"


# ===========================================================================
# GM-80: the GM-77 dry sand column at 0.30 g with each curve
# ===========================================================================

DRY_SAND = dict(SAND33)  # 20 m, Vs 200 m/s, 1.9 Mg/m3, phi 33; dry


@requires_opensees
class TestAmplificationByCurve:
    """Synthetic pulse at 0.30 g, water below the column, 10 elements.
    Measured 2026-10-20 (nonlinear / pystrata equivalent-linear with the
    same curve at the same mean effective stress, 118.7 kPa):

    ==========================  ==========  ==================
    curve                       nonlinear   equivalent-linear
    ==========================  ==========  ==================
    opensees (default)          1.817       --
    darendeli, PI 0             0.637       0.939
    menq, Cu 10, D50 5 mm       0.676       0.972
    menq, Cu 2, D50 0.4 mm      1.640       1.407
    ==========================  ==========  ==================

    The default backbone is about twice as stiff as the laboratory
    curves (GM-80). With Darendeli and Menq Cu 10 the nonlinear column
    amplifies about 30 % less than the equivalent-linear one (peak strain
    0.35 %); with the stiffer Menq Cu 2 curve about 17 % more."""

    def _run(self, **extra):
        return analyze_site_response(
            [dict(DRY_SAND, **extra)], motion="synthetic_pulse",
            target_pga_g=0.30, gwt_depth=50.0, n_elem_per_layer=10,
            time_limit_s=TEST_TIME_LIMIT_S)

    def test_default_curve_stiffer_than_darendeli_and_menq(self):
        default = self._run()
        dar = self._run(curve="darendeli")
        menq = self._run(curve="menq", uniformity_coeff=10, diam_mean=5)
        assert default.amplification_factor == pytest.approx(1.817,
                                                             rel=0.05)
        assert dar.amplification_factor == pytest.approx(0.637, rel=0.05)
        assert menq.amplification_factor == pytest.approx(0.676, rel=0.05)
        assert default.amplification_factor \
            > 2.0 * dar.amplification_factor
        d = dar.to_dict()
        (curve,) = [c for c in d["choices"]
                    if c["what"].endswith("stress-strain curve")]
        assert curve["chosen"] == "darendeli" and curve["how"] == "given"
        assert d["layer_curves"][0]["curve"] == "darendeli"
        assert d["solver_steps"]["substepped_steps"] == 0
        dd = default.to_dict()
        (curve,) = [c for c in dd["choices"]
                    if c["what"].endswith("stress-strain curve")]
        assert curve["chosen"] == "opensees" and curve["how"] == "default"

    def test_user_table_of_darendeli_matches_darendeli(self):
        (plan,) = _layer_plans([dict(DRY_SAND, curve="darendeli")],
                               50.0, 10)
        strains = np.logspace(-4, 1, 25)
        g, d = C.modified_hyperbolic(
            strains / 100.0, *C.darendeli_params(plan["p_ref"]))
        user = self._run(curve="user", G_Gmax_curve={
            "strain_pct": strains.tolist(), "G_Gmax": g.tolist()})
        assert user.amplification_factor == pytest.approx(0.637, rel=0.08)

    @requires_pystrata
    def test_against_equivalent_linear(self):
        import pystrata_agent
        (plan,) = _layer_plans([dict(DRY_SAND, curve="darendeli")],
                               50.0, 10)
        eql_layers = [
            {"thickness": 20.0, "Vs": 200.0, "unit_wt": 1.9 * 9.81,
             "soil_model": "darendeli", "plas_index": 0,
             "stress_mean": plan["p_ref"]},
            {"thickness": 0, "Vs": 760.0, "unit_wt": 2.4 * 9.81,
             "soil_model": "linear", "damping": 0.0}]
        eql = pystrata_agent.analyze_eql_site_response(
            eql_layers, motion="synthetic_pulse", target_pga_g=0.30)
        assert eql.amplification_factor == pytest.approx(0.939, rel=0.05)
        nl = self._run(curve="darendeli")
        assert 0.5 * eql.amplification_factor < nl.amplification_factor \
            < eql.amplification_factor
