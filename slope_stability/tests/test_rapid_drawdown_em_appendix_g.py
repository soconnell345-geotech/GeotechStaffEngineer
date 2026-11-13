"""Rapid drawdown as EM 1110-2-1902 (31 Oct 2003) Appendix G prints it
(BACKLOG GM-140 step 9).

The manual's worked example (G-4, pp. G-9 to G-22; Figs. G-5, G-6, G-10b,
G-11b) is in US customary units; the module is SI, so every input is
converted here (ft, psf, pcf) and every result converted back (ksf, kips
per ft) before it is set beside the printed value.

The section: toe at El. 0; 3 horizontal to 1 vertical up to El. 74; 2.5 to
1 up to the crest at El. 110; one soil, 135 pcf; c' = 0, phi' = 30 deg;
c_R = 1,200 psf, phi_R = 16 deg. Pool from El. 103 to El. 24, level
piezometric lines. Slip circle of radius 210 ft, centre 169.5 ft from the
toe and 210 ft above El. 0. Twelve slices, numbered from the crest.
"""

import math

import pytest

from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.analysis import rapid_drawdown_fos
from slope_stability import rapid_drawdown as rd

FT = 0.3048                 # m per ft
PSF = 0.04788026            # kPa per psf
KSF = 1000.0 * PSF          # kPa per ksf
PCF = 0.15708746            # kN/m3 per pcf
KIP_FT = 14.5939            # kN/m per kip/ft

# The manual's slice edges, feet from the toe, crest first (recovered from
# the printed widths, Fig. G-10).
EDGES_FT = [354.2, 350.2, 330.0, 312.0, 294.5, 267.0, 245.0, 222.0, 200.0,
            169.5, 135.0, 105.0, 72.0]

# Fig. G-10b (p. G-16) and Fig. G-11b (p. G-19), slice 1 at the crest:
# sigma'_c, tau_c, K_c, tau_ff(K_c = 1), tau_ff(K_c = K_f), tau_ff (ksf);
# N of the second stage (kips); u after drawdown, sigma'_d, s_d (ksf);
# the strength used in the third stage.
FIG_G10B_G11B = [
    (0.32, 0.08, 1.71, 1.49, 0.19, 1.02, -6, 0.00, -0.72, 0.00, "drained"),
    (1.44, 0.38, 1.71, 1.85, 0.83, 1.49, 52, 0.00, 1.48, 0.86, "drained"),
    (2.90, 0.76, 1.71, 2.33, 1.67, 2.10, 120, 0.00, 4.61, 2.66, "undrained"),
    (3.64, 0.95, 1.71, 2.58, 2.10, 2.41, 148, 0.00, 6.52, 3.76, "undrained"),
    (4.05, 1.06, 1.71, 2.71, 2.34, 2.58, 250, 0.00, 7.69, 4.44, "undrained"),
    (4.45, 1.17, 1.71, 2.84, 2.57, 2.75, 204, 0.31, 8.13, 4.69, "undrained"),
    (4.57, 1.20, 1.71, 2.88, 2.64, 2.80, 208, 0.85, 7.77, 4.49, "undrained"),
    (4.54, 1.19, 1.71, 2.87, 2.62, 2.78, 191, 1.22, 7.30, 4.22, "undrained"),
    (4.31, 1.13, 1.71, 2.80, 2.49, 2.69, 246, 1.43, 6.61, 3.81, "undrained"),
    (3.66, 0.96, 1.71, 2.58, 2.11, 2.42, 235, 1.41, 5.38, 3.11, "undrained"),
    (2.60, 0.68, 1.71, 2.23, 1.50, 1.97, 149, 1.09, 3.75, 2.16, "undrained"),
    (1.01, 0.27, 1.71, 1.71, 0.59, 1.31, 73, 0.43, 1.61, 0.93, "drained"),
]
PHI = 30.0
C_R, PHI_R = 1200.0, 16.0           # psf, deg
KF = 3.00                           # printed, every slice (Eq. G-7)


def _dam(**layer):
    kw = dict(name="fill", top_elevation=110 * FT, bottom_elevation=-20 * FT,
              gamma=135 * PCF, phi=PHI, c_prime=0.0,
              R_c=C_R * PSF, R_phi=PHI_R)
    kw.update(layer)
    return SlopeGeometry(
        surface_points=[(-100 * FT, 0.0), (0.0, 0.0), (222 * FT, 74 * FT),
                        (312 * FT, 110 * FT), (420 * FT, 110 * FT)],
        soil_layers=[SlopeSoilLayer(**kw)])


def _run(method="duncan_3stage", geom=None, n_slices=50, **kw):
    return rapid_drawdown_fos(
        geom or _dam(), 103 * FT, 24 * FT, xc=169.5 * FT, yc=210 * FT,
        radius=210 * FT, method=method, n_slices=n_slices, **kw)


def _manual_slices(method="duncan_3stage", stage_solver="bishop", **kw):
    """The procedure on the manual's own twelve slices."""
    return _run(method, n_slices=12, stage_solver=stage_solver,
                extra_boundaries=[x * FT for x in EDGES_FT[1:-1]], **kw)


def _circle_z(x_ft):
    return 210.0 - math.sqrt(210.0 ** 2 - (x_ft - 169.5) ** 2)


# ---------------------------------------------------------------------------
# The printed equations against the printed table
# ---------------------------------------------------------------------------

def test_d_and_psi_from_the_r_envelope_eqs_g5_g6():
    """EM p. G-6: d = c_R cos(phi_R) cos(phi') / (1 - sin(phi_R)) (Eq. G-5)
    and psi = arctan[sin(phi_R) cos(phi') / (1 - sin(phi_R))] (Eq. G-6).
    Fig. G-5 prints both pairs: 1,200 psf and 16 deg become 1,379 psf and
    18.2 deg."""
    d, psi = rd.r_envelope_d_psi(C_R, PHI_R, PHI)
    assert d == pytest.approx(1379.0, abs=0.5)
    assert psi == pytest.approx(18.2, abs=0.05)


def test_kc_is_eq_g11_on_every_slice_of_the_table():
    """EM Eq. G-11 (p. G-7): K_c = [sigma'_c + tau_c (sin phi' + 1) /
    cos phi'] / [sigma'_c + tau_c (sin phi' - 1) / cos phi']. Fig. G-10b
    prints 1.71 on all twelve slices; from its sigma'_c and tau_c, printed
    to 0.01 ksf, the equation gives 1.71 within what that rounding allows
    (0.04 on the thinnest slice)."""
    for sig, tau, kc, *_ in FIG_G10B_G11B:
        assert rd.consolidation_stress_ratio(sig, tau, PHI, KF) == \
            pytest.approx(kc, abs=0.04)
    # with c' = 0, tau_c / sigma'_c = tan(phi') / F on every slice: the
    # printed first-stage F = 2.20 gives the printed 1.71
    ratio = math.tan(math.radians(PHI)) / 2.20
    assert rd.consolidation_stress_ratio(1.0, ratio, PHI, KF) == \
        pytest.approx(1.71, abs=0.005)


def test_kc_is_held_between_one_and_kf():
    """EM G-3b(2)(c), p. G-7: K_c is held between 1 and K_f."""
    assert rd.consolidation_stress_ratio(100.0, 0.0, PHI, KF) == 1.0
    assert rd.consolidation_stress_ratio(100.0, 90.0, PHI, KF) == KF
    # the minor principal stress not above zero: beyond K_f, so K_f
    assert rd.consolidation_stress_ratio(10.0, 40.0, PHI, KF) == KF


def test_strengths_are_eqs_g9_g10_g12_on_every_slice_of_the_table():
    """EM Eq. G-9 (p. G-7): tau_ff(K_c = 1) = d + sigma'_c tan(psi);
    Eq. G-10: tau_ff(K_c = K_f) = c' + sigma'_c tan(phi'); Eq. G-12
    (p. G-8): tau_ff = [(K_f - K_c) tau_ff(K_c = 1) + (K_c - 1)
    tau_ff(K_c = K_f)] / (K_f - 1). Every row of Fig. G-10b, to its
    rounding."""
    d, psi = rd.r_envelope_d_psi(C_R / 1000.0, PHI_R, PHI)      # ksf
    for sig, _tau, kc, t1, tf, tff, *_ in FIG_G10B_G11B:
        t_kc1 = d + sig * math.tan(math.radians(psi))
        t_kcf = sig * math.tan(math.radians(PHI))
        assert t_kc1 == pytest.approx(t1, abs=0.011)
        assert t_kcf == pytest.approx(tf, abs=0.011)
        assert rd.second_stage_strength(kc, KF, t1, tf) == \
            pytest.approx(tff, abs=0.011)


def test_drained_check_is_eqs_g13_to_g15_from_the_second_stage_n():
    """EM G-3d (p. G-8): sigma = N / dl with N from the second-stage
    computation (Eq. G-13), sigma'_d = sigma - u after drawdown (G-14),
    s_d = c' + sigma'_d tan(phi') (G-15). Every row of Fig. G-11b from its
    printed N (to 1 kip) on the slice's straight base; a negative sigma'_d
    gives s_d = c' (the manual prints 0.00 for slice 1, c' = 0)."""
    for k, row in enumerate(FIG_G10B_G11B):
        N, u, sig_d, s_d = row[6], row[7], row[8], row[9]
        xa, xb = EDGES_FT[k + 1], EDGES_FT[k]
        dl = math.hypot(xb - xa, _circle_z(xb) - _circle_z(xa))
        got_sig, got_s = rd.drained_check(N, dl, u, 0.0, PHI)
        tol = 0.5 / dl + 0.011            # N to 1 kip, stresses to 0.01
        assert got_sig == pytest.approx(sig_d, abs=tol)
        assert got_s == pytest.approx(s_d, abs=tol)
    assert rd.drained_check(-6.0, 8.0, 0.0, 0.0, PHI)[1] == 0.0
    assert rd.drained_check(-6.0, 8.0, 0.0, 5.0, PHI)[1] == 5.0


# ---------------------------------------------------------------------------
# The manual's example, stage by stage
# ---------------------------------------------------------------------------

def test_em_example_by_bishop_on_the_manuals_twelve_slices():
    """EM pp. G-15, G-18 and G-20: the improved procedure by the Simplified
    Bishop Method gives 2.20, 1.52 and 1.44, with the drained strength on
    slices 1, 2 and 12 in the third stage (Fig. G-11b)."""
    r = _manual_slices()
    assert r.n_slices == 12
    # GM-140 step 9: 2.194 / 1.372 / 1.235 before (the account's D38);
    # printed 2.20 / 1.52 / 1.44; now -0.3 %, -0.4 %, -0.1 %
    assert r.stage1_fos == pytest.approx(2.20, rel=0.01)
    assert r.stage2_fos == pytest.approx(1.52, rel=0.01)
    assert r.stage3_fos == pytest.approx(1.44, rel=0.01)
    assert r.answer_stage == 3 and r.FOS == r.stage3_fos
    rows = list(reversed(r.slice_table()))          # slice 1 at the crest
    for k, (row, printed) in enumerate(zip(rows, FIG_G10B_G11B), start=1):
        assert row["Kf"] == pytest.approx(KF, abs=1e-9)
        assert row["Kc"] == pytest.approx(printed[2], abs=0.015)
        # The printed weights are those of slices with straight bases; the
        # module weighs down to the arc at the slice's middle, 5 to 7 %
        # more on the thin end slices (2 and 12) and under 1.5 % elsewhere.
        # So tau_ff is within 0.03 ksf of the print on every slice, not
        # within its 0.005 rounding (the equations themselves are pinned to
        # the rounding by the tests above).
        assert row["tau_ff"] / KSF == pytest.approx(printed[5], abs=0.03), k
        assert row["strength_used"] == printed[10], k
        assert row["u_after"] / KSF == pytest.approx(printed[7], abs=0.06), k
    assert r.n_drained_substituted == 3
    # the second stage's N of Eq. G-13, slice by slice (kips; the print's
    # are to 1 kip on its lighter slices)
    for row, printed in zip(rows, FIG_G10B_G11B):
        assert row["N_stage2"] / KIP_FT == pytest.approx(
            printed[6], abs=max(5.0, 0.03 * abs(printed[6])))


def test_em_example_by_spencer_on_the_modules_usual_slices():
    """The same example by Spencer's method, 50 slices. The manual's
    Modified Swedish solution at Spencer's side-force inclinations prints
    2.23, 1.52 and 1.44 (pp. G-20 to G-22); its 2.23 rests on whole-degree
    base angles, and Spencer's method agrees with Bishop's 2.20 here."""
    r = _run()
    # GM-140 step 9: 2.194 / 1.372 / 1.235 before; printed 2.20 (Bishop;
    # 2.23 Modified Swedish) / 1.52 / 1.44; now 0.0 %, -0.5 %, -0.4 %
    assert r.stage1_fos == pytest.approx(2.20, rel=0.01)
    assert r.stage2_fos == pytest.approx(1.52, rel=0.01)
    assert r.stage3_fos == pytest.approx(1.44, rel=0.01)
    assert r.FOS == r.stage3_fos and r.answer_stage == 3
    assert "Spencer" in r.stage_method


def test_second_stage_is_reported_and_is_lowe_and_karafiaths_answer():
    """EM G-3c (p. G-8): the second stage is solved with tau_ff as cohesion
    and phi = 0. Without the drained check its factor of safety is the
    answer: 1.52 printed (p. G-18)."""
    lk = _run("lowe_karafiath")
    assert lk.stage2_fos == pytest.approx(1.52, rel=0.01)
    assert lk.FOS == lk.stage2_fos and lk.answer_stage == 2
    assert lk.stage3_fos is None and lk.n_drained_substituted == 0
    assert lk.sigma_d == [] and lk.s_d == []
    d = lk.to_dict()
    assert d["stage2_FOS"] == pytest.approx(lk.stage2_fos, abs=1e-4)
    assert d["stage3_FOS"] is None and d["answer_stage"] == 2
    three = _run("duncan_3stage")
    assert three.stage2_fos == pytest.approx(lk.stage2_fos, abs=1e-9)
    assert three.tau_ff == pytest.approx(lk.tau_ff, abs=1e-9)


def test_no_third_stage_when_no_slice_needs_it():
    """EM G-3d (p. G-8): where the drained strength is not below tau_ff on
    any slice, the second stage's factor of safety is the answer and no
    third stage is computed. A made-up fill with a low R envelope and the
    pool drawn down below the slip surface."""
    g = _dam(c_prime=20.0, phi=35.0, R_c=5.0, R_phi=12.0)
    r = rapid_drawdown_fos(g, 103 * FT, -15 * FT, xc=169.5 * FT,
                           yc=210 * FT, radius=210 * FT)
    assert r.method == "duncan_3stage"
    assert all(s >= t for s, t in zip(r.s_d, r.tau_ff))
    assert r.n_drained_substituted == 0
    assert r.stage3_fos is None and r.answer_stage == 2
    assert r.FOS == r.stage2_fos
    assert set(r.strength_used) == {"undrained"}


def test_third_stage_puts_c_phi_and_u_on_the_drained_slices(monkeypatch):
    """EM G-3e (pp. G-8 to G-9): the third stage is the second with c',
    phi' and the pore pressure after drawdown on the slices whose drained
    strength is lower; the others keep tau_ff with phi = 0."""
    seen = []
    real = rd._spencer_solution

    def spy(slices, slip, f_interslice, tol, reinf):
        seen.append([(s.c, s.phi, s.pore_pressure) for s in slices])
        return real(slices, slip, f_interslice, tol, reinf)

    monkeypatch.setattr(rd, "_spencer_solution", spy)
    r = _run()
    assert len(seen) == 3
    for i, (c, phi, u) in enumerate(seen[2]):
        if r.strength_used[i] == "drained":
            assert (c, phi) == (0.0, PHI)
            assert u == pytest.approx(r.u_after[i])
        else:
            assert c == pytest.approx(r.tau_ff[i]) and phi == 0.0 and u == 0.0
    assert all(phi == 0.0 and u == 0.0 for _c, phi, u in seen[1])


def test_a_first_stage_without_a_solution_gives_no_number(monkeypatch):
    """GM-140 step 9: sigma'_c and tau_c (Eqs. G-1, G-2) come from the
    first stage's own solution. Without one there is no factor of safety:
    no Ordinary Method of Slices estimate stands in."""
    from slope_stability.gle import GLEResult
    monkeypatch.setattr(
        rd, "gle_fos",
        lambda *a, **k: GLEResult(fos=1.7, converged=False,
                                  warnings=["no F_m = F_f crossing found "
                                            "in lambda range"]))
    r = _run()
    assert r.FOS is None and r.stage1_fos is None and r.stage2_fos is None
    assert r.answer_stage is None
    assert "first stage" in r.no_solution
    d = r.to_dict()
    assert d["FOS"] is None and "first stage" in d["no_solution"]
    assert "none" in r.summary()


def test_a_trial_without_a_solution_is_not_scored_by_the_search(monkeypatch):
    """The same procedure scores every trial of the drawdown search: a
    trial on which a stage has no solution is left out, and counted
    (GM-140 step 2)."""
    from slope_stability.rapid_drawdown import search_rapid_drawdown
    monkeypatch.setattr(
        rd, "_spencer_solution",
        lambda *a, **k: rd.StageSolution(reason="made to fail"))
    r = search_rapid_drawdown(_dam(), 103 * FT, 24 * FT, nx=2, ny=2,
                              n_slices=12, toe_circles=False,
                              infinite_slope=False, refine=False)
    assert r.no_surface_found and r.to_dict()["FOS"] is None
    counted = [rec for rec in r.no_solution if rec["method"] == "duncan_3stage"]
    assert counted and counted[0]["n"] > 0


def test_the_search_scores_a_trial_as_the_single_surface_does():
    from slope_stability.rapid_drawdown import search_rapid_drawdown
    r = search_rapid_drawdown(
        _dam(), 103 * FT, 24 * FT, method="duncan_3stage",
        surface_type="entry_exit", nx=3, ny=3, n_slices=20,
        x_entry_range=(40 * FT, 100 * FT), x_exit_range=(320 * FT, 380 * FT),
        toe_circles=False, infinite_slope=False)
    crit = r.critical
    one = rapid_drawdown_fos(_dam(), 103 * FT, 24 * FT, xc=crit.xc,
                             yc=crit.yc, radius=crit.radius, n_slices=20)
    assert r.FOS == pytest.approx(one.FOS, abs=1e-6)
    assert r.drawdown.stage2_fos == pytest.approx(one.stage2_fos, abs=1e-6)
    assert "stage2_FOS" in r.to_dict()["drawdown_detail"]


# ---------------------------------------------------------------------------
# The Corps of Engineers' 1970 procedure
# ---------------------------------------------------------------------------

def test_corps_1970_example_by_force_equilibrium_at_19_4_degrees():
    """EM G-5 (pp. G-9 to G-15): the 1970 procedure's example, by the
    Modified Swedish Method with the side forces at 19.4 deg, prints 3.49
    for the first stage and 1.35 for the second (Slide2 prints 1.347). The
    library's force-equilibrium solution at that fixed inclination
    (``fixed_inclination_solution``) reproduces both."""
    r = _run("corps_2stage", stage_solver=("force", 19.4))
    # printed 3.49 and 1.35; -0.1 % and -0.7 %
    assert r.stage1_fos == pytest.approx(3.49, rel=0.01)
    assert r.stage2_fos == pytest.approx(1.35, rel=0.01)
    assert r.FOS == r.stage2_fos and r.stage3_fos is None
    assert "19.4" in r.stage_method


def test_corps_1970_keeps_spencer_and_says_what_the_example_used():
    """The lead's call (GM-140 step 9): the module's two-stage procedure
    keeps Spencer's method, which the manual recommends (G-5, p. G-9), and
    says that the manual's example used the Modified Swedish Method at a
    fixed side-force inclination."""
    r = _run("corps_2stage")
    # GM-140 step 9: unchanged by this step (1.239). Printed 1.35 by the
    # Modified Swedish Method at 19.4 deg (EM p. G-15): -8.2 %, the
    # method of slices, as the test above shows.
    assert r.FOS == pytest.approx(1.239, abs=0.01)
    assert r.stage1_fos == pytest.approx(2.20, rel=0.01)
    assert "Spencer" in r.stage_method
    assert any("Modified Swedish" in n for n in r.notes)
    assert r.warnings == []
    # c_R and phi_R as entered: the composite envelope, not Eqs. G-5, G-6
    i = len(r.tau_ff) // 2
    sc = r.sigma_fc[i]
    assert r.tau_ff[i] == pytest.approx(
        min(C_R * PSF + sc * math.tan(math.radians(PHI_R)),
            sc * math.tan(math.radians(PHI))))


def test_fixed_inclination_at_zero_moment_is_bishop():
    from slope_stability.methods import bishop_fos
    from slope_stability.slices import build_slices
    from slope_stability.slip_surface import CircularSlipSurface
    slip = CircularSlipSurface(169.5 * FT, 210 * FT, 210 * FT)
    sl = build_slices(rd._pool_geometry(_dam(), 103 * FT), slip, 30)
    sol = rd.fixed_inclination_solution(sl, slip, 0.0, "moment")
    # the GLE's moment equation takes each normal force's own arm about
    # the centre on the slices' straight bases; `bishop_fos` takes R sin
    # (alpha): 0.1 % apart here
    assert sol.fos == pytest.approx(bishop_fos(sl, slip), rel=2e-3)
    with pytest.raises(ValueError, match="theta_deg = 0"):
        rd.fixed_inclination_solution(sl, slip, 5.0, "moment")


# ---------------------------------------------------------------------------
# Inputs: measured d and psi, the water, the removed input, the slices
# ---------------------------------------------------------------------------

def test_a_layer_may_give_d_and_psi_as_measured():
    """EM G-3b(1)(a) and (b), pp. G-4 to G-5: d and psi may be measured
    directly. Given as Eqs. G-5 and G-6 derive them, the stages are the
    same; the 1970 procedure, which uses c_R and phi_R as entered, refuses
    a layer that gives only d and psi."""
    d, psi = rd.r_envelope_d_psi(C_R * PSF, PHI_R, PHI)
    measured = _dam(R_c=0.0, R_phi=None, R_d=d, R_psi=psi)
    a, b = _run(geom=measured), _run()
    assert a.n_undrained_slices == a.n_slices
    assert a.stage2_fos == pytest.approx(b.stage2_fos, abs=1e-9)
    assert a.FOS == pytest.approx(b.FOS, abs=1e-9)
    with pytest.raises(ValueError, match="R_c and R_phi"):
        _run("corps_2stage", geom=measured)


def test_water_before_and_after_is_stated():
    """EM G-4b (p. G-9): the example assumes a level piezometric line at
    the maximum pool and another at the lowered pool 'for simplicity'. The
    result names that assumption where no line is given."""
    r = _run()
    assert "level pool" in r.water_before and "assumed" in r.water_before
    assert "level pool" in r.water_after and "assumed" in r.water_after
    whats = {c["what"]: c for c in r.choices}
    assert whats["water before drawdown"]["how"] == "default"
    assert whats["water after drawdown"]["how"] == "default"
    assert "G-4b" in whats["water before drawdown"]["basis"]
    given = _run(
        stage1_phreatic_points=[(-200 * FT, 103 * FT), (500 * FT, 103 * FT)],
        drawdown_phreatic_points=[(-200 * FT, 24 * FT), (500 * FT, 24 * FT)])
    assert "given" in given.water_before and "given" in given.water_after
    # the level lines given are the default's own
    assert given.FOS == pytest.approx(r.FOS, abs=1e-6)


def test_a_line_after_drawdown_acts_in_the_drained_check():
    """Eq. G-14: sigma'_d = sigma - u with the pore pressure after
    drawdown. A line left higher inside the fill than the lowered pool
    lowers the drained strengths, so the three-stage answer; the second
    stage (phi = 0, no pore pressure) does not move."""
    r = _run()
    high = _run(drawdown_phreatic_points=[
        (-200 * FT, 24 * FT), (72 * FT, 24 * FT), (300 * FT, 70 * FT),
        (500 * FT, 70 * FT)])
    # (the line's own crossings add slice edges: 0.02 % on the second stage)
    assert high.stage2_fos == pytest.approx(r.stage2_fos, abs=1e-3)
    assert max(high.u_after) > max(r.u_after)
    assert high.n_drained_substituted >= r.n_drained_substituted
    assert high.FOS < r.FOS


def test_the_removed_stage3_basis_is_refused_with_the_rule():
    """OWN-17 settled: the drained check has one printed basis, the second
    stage's N (Eq. G-13). A caller who names an old value is told the
    rule."""
    for old in ("fellenius", "gle"):
        with pytest.raises(ValueError, match="G-13"):
            _run(stage3_effective_normal=old)
    from slope_stability.rapid_drawdown import search_rapid_drawdown
    with pytest.raises(ValueError, match="G-13"):
        search_rapid_drawdown(_dam(), 103 * FT, 24 * FT,
                              stage3_effective_normal="gle")


def test_one_set_of_slices_for_every_stage(monkeypatch):
    """GM-135: the stages are paired slice by slice; a surface on which
    the slices before and after drawdown differ is refused."""
    r = _run(n_slices=20)
    cols = (r.x_left, r.sigma_fc, r.tau_ff, r.N_stage2, r.u_after, r.s_d,
            r.strength_used)
    assert {len(c) for c in cols} == {r.n_slices}
    real = rd.build_slices
    calls = []

    def dropping(*a, **k):
        calls.append(1)
        out = real(*a, **k)
        return out[:-1] if len(calls) == 2 else out

    monkeypatch.setattr(rd, "build_slices", dropping)
    with pytest.raises(ValueError, match="one set of slices"):
        _run(n_slices=20)


def test_a_base_in_tension_before_drawdown_is_named():
    """The manual prints no rule for sigma'_c below zero. The module takes
    sigma'_c as zero there, uses the lower envelope strength and says
    so."""
    from slope_stability.gle import gle_fos as real

    def tensioned(slices, slip, **k):
        res = real(slices, slip, **k)
        res.base_normal_eff[0] = -5.0
        return res

    mp = pytest.MonkeyPatch()
    mp.setattr(rd, "gle_fos", tensioned)
    try:
        r = _run("lowe_karafiath", n_slices=12)
    finally:
        mp.undo()
    d, _psi = rd.r_envelope_d_psi(C_R * PSF, PHI_R, PHI)
    assert r.tau_ff[0] == pytest.approx(min(d, 0.0))
    assert any("prints no rule" in w for w in r.warnings)


def test_the_first_stage_m_alpha_is_checked_and_reported(monkeypatch):
    """GM-130's drawdown item: m_alpha was looked at on the drawn-down
    stage only. The first stage's least m_alpha at its own factor of
    safety is reported too, and a given surface below 0.2 says so."""
    r = _run(n_slices=20)
    assert 0.2 < r.m_alpha_min_stage1 <= 1.5
    assert r.to_dict()["m_alpha_min_stage1"] == pytest.approx(
        r.m_alpha_min_stage1, abs=1e-3)
    assert not any("First stage" in w for w in r.warnings)
    real = rd._m_alpha_min
    calls = []

    def low_first(slices, fos, slip, method, rule):
        calls.append(1)
        return 0.1 if len(calls) == 1 else real(slices, fos, slip, method,
                                                rule)

    monkeypatch.setattr(rd, "_m_alpha_min", low_first)
    monkeypatch.setattr("slope_stability.analysis.m_alpha_statement",
                        lambda *a, **k: "m_alpha is 0.10 on a slice")
    flagged = _run(n_slices=20)
    assert flagged.m_alpha_min_stage1 == 0.1
    assert any(w.startswith("First stage (before drawdown): ")
               for w in flagged.warnings)
    assert flagged.FOS == pytest.approx(r.FOS, abs=1e-9)
