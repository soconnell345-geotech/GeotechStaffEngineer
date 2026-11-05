"""Footing bearing-capacity convenience — tests (v5.4 E11).

Fast tests cover the closed-form bearing-capacity factors and input validation
(no FEM). The slow tests drive the real load-control collapse and pin the
Prandtl anchor q_ult = (2 + pi) c (Nc = 5.14) for T6, and the CST locking that
makes CST unusable for collapse loads (VALIDATION.md §3).
"""

import math

import pytest

from fem2d import analyze_footing_capacity, bearing_capacity_factors


class TestBearingCapacityFactors:
    def test_phi_zero_prandtl(self):
        Nc, Nq, Ng = bearing_capacity_factors(0.0)
        assert Nc == pytest.approx(2.0 + math.pi, rel=1e-12)   # 5.142
        assert Nq == pytest.approx(1.0, rel=1e-12)
        assert Ng == pytest.approx(0.0, abs=1e-12)

    def test_phi_30_vesic_table(self):
        Nc, Nq, Ng = bearing_capacity_factors(30.0)
        # standard published values (Vesic): Nc=30.14, Nq=18.40, Ngamma=22.40
        assert Nc == pytest.approx(30.14, abs=0.1)
        assert Nq == pytest.approx(18.40, abs=0.1)
        assert Ng == pytest.approx(22.40, abs=0.1)


class TestValidation:
    def test_bad_width(self):
        with pytest.raises(ValueError, match="B must be positive"):
            analyze_footing_capacity(B=0.0, c=100.0)

    def test_bad_element_type(self):
        with pytest.raises(ValueError, match="element_type"):
            analyze_footing_capacity(B=2.0, c=100.0, element_type="q8")

    def test_cohesionless_weightless_needs_qmax(self):
        with pytest.raises(ValueError, match="auto-size q_max"):
            analyze_footing_capacity(B=2.0, c=0.0, phi=30.0, gamma=0.0)


@pytest.mark.slow
class TestPrandtlCollapse:
    def test_t6_reproduces_prandtl_nc(self):
        """Weightless phi=0 strip footing: q_ult = (2+pi)c, Nc = 5.14. T6
        load-control collapse lands within ~2% (VALIDATION.md §3 band)."""
        r = analyze_footing_capacity(B=2.0, c=100.0, phi=0.0, gamma=0.0,
                                     nx=40, ny=20, element_type='t6',
                                     q_applied=200.0)
        print(f"\n  Footing T6: Nc={r.Nc_backfigured:.3f} (exact 5.14), "
              f"q_ult={r.q_ult_kPa:.0f} kPa")
        assert r.collapse_bracketed
        assert 4.8 <= r.Nc_backfigured <= 5.5      # house Prandtl band (~2%)
        # FOS against a 200 kPa working pressure
        assert r.bearing_FOS == pytest.approx(r.q_ult_kPa / 200.0, rel=1e-9)

    def test_cst_locks_and_does_not_collapse(self):
        """CST cannot represent the isochoric Prandtl mechanism: it locks and
        carries the full ramp without collapsing (VALIDATION.md §3)."""
        r = analyze_footing_capacity(B=2.0, c=100.0, phi=0.0, gamma=0.0,
                                     nx=20, ny=10, element_type='cst',
                                     n_load_steps=30)
        print(f"\n  Footing CST: Nc~{r.Nc_backfigured:.2f}, "
              f"bracketed={r.collapse_bracketed}")
        assert not r.collapse_bracketed            # never collapsed
        assert r.Nc_backfigured > 7.0              # locked well above 5.14


class TestSurchargeValidation:
    def test_qmax_must_exceed_surcharge(self):
        with pytest.raises(ValueError, match="must exceed the surcharge"):
            analyze_footing_capacity(B=2.0, c=50.0, surcharge=36.0,
                                     q_max=30.0, nx=10, ny=5)


@pytest.mark.slow
class TestSurchargeResists:
    """GM-11: the surcharge beside the footing is applied and resists through
    the Nq mechanism. Gross q_ult = c Nc + q Nq (+ 0.5 gamma B Ngamma), GEC-6
    (FHWA-SA-02-054) Eqs. 5-1, 5-2, 5-4 and 5-7, pp. 51-53; for phi = 0,
    UFC 3-220-20 Eq. 5-17 (q_ult = su Nc + sigma_zD). Made-up values."""

    def test_undrained_weightless_c_nc_plus_q(self):
        # 5.14 x 50 + 36 = 293.1 kPa (Nq = 1)
        r = analyze_footing_capacity(B=2.0, c=50.0, surcharge=36.0)
        assert r.collapse_bracketed
        assert r.q_ult_kPa == pytest.approx(293.1, rel=0.03)
        assert r.surcharge_kPa == 36.0

    def test_undrained_with_weight_unchanged_by_weight(self):
        # phi = 0: Ngamma = 0, so the soil's weight adds nothing (Eq. 5-17)
        r = analyze_footing_capacity(B=2.0, c=50.0, gamma=18.0,
                                     surcharge=36.0, nu=0.49)
        assert r.collapse_bracketed
        assert r.q_ult_kPa == pytest.approx(293.1, rel=0.03)

    def test_weightless_sand_q_nq(self):
        # Associated flow (psi = phi), the Prandtl-Reissner basis of Eq. 5-2:
        # Nq(20) = 6.40, q Nq = 20 x 6.40 = 128.0 kPa. Before GM-11: 0.
        r = analyze_footing_capacity(B=2.0, c=0.0, phi=20.0, psi=20.0,
                                     surcharge=20.0)
        assert r.collapse_bracketed
        assert r.q_ult_kPa == pytest.approx(128.0, rel=0.04)

    def test_no_surcharge_path_unchanged(self):
        r = analyze_footing_capacity(B=2.0, c=50.0)
        assert r.surcharge_kPa == 0.0
        assert r.q_ult_kPa == pytest.approx(411.3 * 28 / 45, rel=1e-3)


@pytest.mark.slow
class TestWeightFirstStage:
    """GM-304: the soil's weight is a first stage whenever gamma > 0, with
    or without a surcharge. Ramped with the footing load, a cohesionless
    soil failed at the first load step (c = 0, phi = 30, gamma = 18 gave
    q_ult = 0).
    Coarse T6 20 x 10 mesh; made-up values."""

    def test_frictional_heavy_soil_carries_load(self):
        # Before GM-304: 0.0 kPa (collapse at the first step).
        r = analyze_footing_capacity(B=2.0, c=0.0, phi=30.0, psi=30.0,
                                     gamma=18.0, nx=20, ny=10)
        assert r.collapse_bracketed
        assert r.q_ult_kPa > 0.0
        assert r.q_ult_kPa == pytest.approx(86.03, rel=1e-3)

    @pytest.mark.parametrize("c, phi", [(0.0, 30.0), (20.0, 25.0)])
    def test_no_surcharge_agrees_with_tiny_surcharge(self, c, phi):
        kw = dict(B=2.0, c=c, phi=phi, psi=phi if c == 0 else 0.0,
                  gamma=18.0, nx=20, ny=10)
        r0 = analyze_footing_capacity(**kw)
        r1 = analyze_footing_capacity(surcharge=0.01, **kw)
        # Within one load step, (q_max - q) / n_load_steps.
        assert abs(r0.q_ult_kPa - r1.q_ult_kPa) <= r0.q_max_kPa / 45
        assert r0.n_steps_converged == r1.n_steps_converged
        # q_ult set by the mesh is warned of: c = 0 with no surcharge only.
        assert any("depends on the mesh" in w for w in r0.warnings) == (c == 0)
        assert not any("depends on the mesh" in w for w in r1.warnings)

    def test_c_phi_heavy_soil(self):
        # c = 20, phi = 25, gamma = 18: 499.0 kPa on 20 x 10 (before
        # GM-304 the one-ramp path gave less; see module_work/gm304).
        r = analyze_footing_capacity(B=2.0, c=20.0, phi=25.0, gamma=18.0,
                                     nx=20, ny=10)
        assert r.q_ult_kPa == pytest.approx(498.99, rel=1e-3)

    def test_weightless_undrained_unchanged_to_last_digit(self):
        # gamma = 0, no surcharge keeps the one-ramp path: master's value.
        r = analyze_footing_capacity(B=2.0, c=50.0)
        assert r.q_ult_kPa == 255.9370565342475


class TestRigidFootingInputs:
    """GM-305: footing='rigid' pushes the footing down as one body; the
    inputs of one footing model are refused for the other."""

    def test_bad_footing(self):
        with pytest.raises(ValueError, match="footing must be"):
            analyze_footing_capacity(B=2.0, c=50.0, footing="stiff")

    def test_bad_base(self):
        with pytest.raises(ValueError, match="base must be"):
            analyze_footing_capacity(B=2.0, c=50.0, footing="rigid",
                                     base="sticky")

    def test_rigid_refuses_q_max(self):
        with pytest.raises(ValueError, match="q_max"):
            analyze_footing_capacity(B=2.0, c=50.0, footing="rigid",
                                     q_max=400.0)

    @pytest.mark.parametrize("kw", [dict(base="smooth"),
                                    dict(settlement_max=0.05)])
    def test_flexible_refuses_rigid_inputs(self, kw):
        with pytest.raises(ValueError, match="footing='rigid'"):
            analyze_footing_capacity(B=2.0, c=50.0, **kw)

    def test_settlement_max_positive(self):
        with pytest.raises(ValueError, match="settlement_max"):
            analyze_footing_capacity(B=2.0, c=50.0, footing="rigid",
                                     settlement_max=0.0)

    def test_weightless_cohesionless_needs_settlement_max(self):
        with pytest.raises(ValueError, match="settlement_max"):
            analyze_footing_capacity(B=2.0, c=0.0, phi=30.0,
                                     footing="rigid")

    def test_choices(self):
        from fem2d.analysis import _footing_choices
        flex = _footing_choices("flexible", "rough")
        assert [c["what"] for c in flex] == ["footing model", "footing mesh"]
        assert [c["chosen"] for c in flex] == ["flexible", "uniform"]
        rig = _footing_choices("rigid", "rough")
        assert [c["chosen"] for c in rig] == ["rigid", "rough", "uniform"]
        assert "p. 55" in rig[0]["basis"]
        assert "p. 99" in rig[1]["basis"]


class TestGradedMeshInputs:
    """GM-306: mesh='graded' grades the same nx x ny elements towards the
    footing's edges and the surface; the default stays uniform."""

    def test_bad_mesh(self):
        with pytest.raises(ValueError, match="mesh must be"):
            analyze_footing_capacity(B=2.0, c=50.0, mesh="fine")

    @pytest.mark.parametrize("nx,ny", [(21, 10), (2, 10), (20, 1)])
    def test_graded_needs_even_nx(self, nx, ny):
        with pytest.raises(ValueError, match="even nx"):
            analyze_footing_capacity(B=2.0, c=50.0, mesh="graded",
                                     nx=nx, ny=ny)

    @pytest.mark.parametrize("nx,ny", [(20, 10), (40, 20), (80, 40)])
    def test_graded_grid(self, nx, ny):
        import numpy as np
        from fem2d.analysis import _footing_mesh, _GRADED_EDGE_FRACTION
        nu, eu = _footing_mesh(2.0, 10.0, 10.0, nx, ny, "uniform")
        ng, eg = _footing_mesh(2.0, 10.0, 10.0, nx, ny, "graded")
        assert ng.shape == nu.shape and eg.shape == eu.shape
        x = np.unique(ng[:, 0])
        y = np.unique(ng[:, 1])
        assert len(x) == nx + 1 and len(y) == ny + 1
        for xe in (-1.0, 0.0, 1.0):          # the footing's edges and centre
            assert np.isclose(x, xe, atol=1e-12).any()
        assert x[0] == -10.0 and x[-1] == 10.0
        assert y[0] == -10.0 and y[-1] == 0.0
        dx = np.diff(x)
        h_u = 20.0 / nx
        i = int(np.argmin(np.abs(x - 1.0)))
        # the elements either side of the edge are the smallest, 1/4 size
        assert dx[i - 1] == pytest.approx(_GRADED_EDGE_FRACTION * h_u)
        assert dx[i] == pytest.approx(_GRADED_EDGE_FRACTION * h_u)
        assert dx.min() == pytest.approx(_GRADED_EDGE_FRACTION * h_u)
        assert np.diff(y)[-1] == pytest.approx(
            _GRADED_EDGE_FRACTION * 10.0 / ny)
        np.testing.assert_allclose(x, -x[::-1], atol=1e-12)   # symmetric

    def test_geometric_spacing(self):
        from fem2d.mesh import geometric_spacing
        p = geometric_spacing(10.0, 5, 0.5)
        assert p[0] == 0.0 and p[-1] == 10.0 and len(p) == 6
        d = p[1:] - p[:-1]
        assert d[0] == pytest.approx(0.5)
        assert d[1:] / d[:-1] == pytest.approx([d[1] / d[0]] * 4)
        assert geometric_spacing(1.0, 4, 0.25) == pytest.approx(
            [0.0, 0.25, 0.5, 0.75, 1.0])

    def test_rect_mesh_coords_checked(self):
        from fem2d.mesh import generate_rect_mesh
        with pytest.raises(ValueError, match="x_coords"):
            generate_rect_mesh(0, 1, 0, 1, 2, 2, x_coords=[0, 0.7, 0.5])
        n, _ = generate_rect_mesh(0, 1, 0, 1, 2, 1,
                                  x_coords=[0, 0.2, 1.0])
        assert sorted(set(n[:, 0])) == [0.0, 0.2, 1.0]

    def test_choices_mesh(self):
        from fem2d.analysis import _footing_choices
        g = _footing_choices("flexible", "rough", "graded")
        assert g[-1]["what"] == "footing mesh"
        assert g[-1]["chosen"] == "graded"
        assert "module's choice" in g[-1]["basis"]


@pytest.mark.slow
class TestRigidFooting:
    """GM-305: the rigid footing on the coarse T6 20 x 10 mesh, B = 2 m.
    Made-up values. The mesh study (fem2d VALIDATION.md section 3) is why
    the default stays flexible: these values fall as the mesh is refined
    (Prandtl rough 347.9 -> 306.0 kPa from 20 x 10 to 40 x 20)."""

    def test_prandtl_rough(self):
        r = analyze_footing_capacity(B=2.0, c=50.0, footing="rigid",
                                     nx=20, ny=10)
        assert r.footing == "rigid" and r.base == "rough"
        assert r.collapse_bracketed                 # q levelled off
        assert r.q_ult_kPa == pytest.approx(347.94, rel=1e-3)
        # above the flexible footing on the same mesh (274.2) and above
        # (2 + pi) c = 257.1: the rigid footing reads high on this mesh
        assert r.q_ult_kPa > (2.0 + math.pi) * 50.0
        s = [p[0] for p in r.pressure_settlement]
        assert s == sorted(s) and s[-1] == pytest.approx(r.settlement_max_m)
        assert r.q_max_kPa is None
        assert r.Nc_backfigured == pytest.approx(r.q_ult_kPa / 50.0)
        assert [c["chosen"] for c in r.choices] == ["rigid", "rough"]
        assert any("falls as the mesh is refined" in w for w in r.warnings)

    def test_prandtl_smooth_below_rough(self):
        r = analyze_footing_capacity(B=2.0, c=50.0, footing="rigid",
                                     base="smooth", nx=20, ny=10)
        assert r.collapse_bracketed
        assert r.q_ult_kPa == pytest.approx(322.69, rel=1e-3)
        assert r.base == "smooth"

    def test_surcharge_starts_q_at_the_surcharge(self):
        # stage 1 brings weight and q into equilibrium; the strip's q is
        # carried by the footing's reactions, so q starts near 36 kPa
        r = analyze_footing_capacity(B=2.0, c=50.0, surcharge=36.0,
                                     footing="rigid", nx=20, ny=10)
        assert r.collapse_bracketed
        assert r.surcharge_kPa == 36.0
        assert r.q_ult_kPa > 36.0 + (2.0 + math.pi) * 50.0
        assert r.pressure_settlement[0][1] > 36.0

    def test_cohesionless_without_plateau_is_a_lower_bound(self):
        # c = 0, phi = 30, gamma = 18: the run stops before q levels off
        r = analyze_footing_capacity(B=2.0, c=0.0, phi=30.0, gamma=18.0,
                                     footing="rigid", nx=20, ny=10)
        assert not r.collapse_bracketed
        assert r.collapse_load_fraction < 1.0
        assert any("did not level off" in w for w in r.warnings)
        # the flexible footing's mesh warning is not the rigid one's
        assert not any("depends on the mesh" in w for w in r.warnings)


class TestFlexibleCollapseCriterion:
    """GM-306: a flexible-footing step that reaches max_iter with its
    displacements standing still is slow convergence, not collapse. On the
    80 x 40 mesh the c = 20, phi = 25 footing stopped at 173.6 kPa that way
    (step 9 at 1000 iterations, residual 1.8e-4 and falling, growth ratio
    0.095). Reproduced here, fast, with a low max_iter on 20 x 10."""

    KW = dict(B=2.0, c=20.0, phi=25.0, gamma=18.0, nx=20, ny=10,
              max_iter=100)

    def test_settled_steps_are_not_collapse(self, monkeypatch):
        import fem2d.analysis as A
        r = analyze_footing_capacity(**self.KW)
        assert r.n_steps_settled > 0
        monkeypatch.setattr(A, "_FLEX_SETTLED_RATIO", None)
        r_old = analyze_footing_capacity(**self.KW)
        assert r_old.n_steps_settled == 0
        # 282.0 kPa read as collapse before GM-306; 368.8 after; 499.0
        # with the default 1000 iterations
        assert r_old.q_ult_kPa == pytest.approx(282.0, abs=0.5)
        assert r.q_ult_kPa == pytest.approx(368.8, abs=0.5)
        assert r.q_ult_kPa > r_old.q_ult_kPa

    def test_step_that_converges_is_unchanged(self, monkeypatch):
        # settled_early=False: a step that meets tol on its own ends where
        # it did with settled_ratio=None, to the last digit.
        import numpy as np
        import fem2d.analysis as A
        kw = dict(B=2.0, c=50.0, nx=20, ny=10, n_load_steps=12)
        r = analyze_footing_capacity(**kw)
        monkeypatch.setattr(A, "_FLEX_SETTLED_RATIO", None)
        r_old = analyze_footing_capacity(**kw)
        assert r.n_steps_settled == 0
        assert r.q_ult_kPa == r_old.q_ult_kPa
        assert r.max_displacement_m == r_old.max_displacement_m
