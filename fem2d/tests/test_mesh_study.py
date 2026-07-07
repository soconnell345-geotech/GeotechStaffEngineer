"""SRM mesh-refinement (mesh-consistency) study — tests (v5.4 E7 / B2e).

Fast tests exercise the study bookkeeping and the Richardson extrapolation math
with synthetic data (no FEM). The slow tests drive the real SRM at increasing
densities and pin the documented mesh-convergence behaviour:

- Griffiths & Lane (1999) Example 1: the FE strength-reduction FOS stays
  mesh-consistent (a tight band around the published FE value 1.4) as the mesh
  is refined (VALIDATION.md §1).
- shared-geometry Bishop cross-check: the SRM FOS converges monotonically
  toward the limit-equilibrium value as the mesh is refined, with shrinking
  steps and a Richardson estimate (VALIDATION.md §6).
"""

import pytest

from fem2d import srm_mesh_refinement_study, MeshRefinementResult
from fem2d.mesh_study import _richardson


# ── fast: extrapolation math + bookkeeping (no FEM) ─────────────────────
class TestRichardsonMath:

    def test_second_order_recovers_exact(self):
        """Synthetic p=2 sequence at refinement ratio r=2 (n x4 per level):
        f_exact=1.40, errors 0.16/0.04/0.01 → f1=1.24, f2=1.36, f3=1.39.
        Richardson must recover p≈2.0 and f_exact≈1.40."""
        levels = [
            {"FOS": 1.24, "n_elements": 100},
            {"FOS": 1.36, "n_elements": 400},
            {"FOS": 1.39, "n_elements": 1600},
        ]
        f_exact, p = _richardson(levels)
        assert p == pytest.approx(2.0, abs=1e-6)
        assert f_exact == pytest.approx(1.40, abs=1e-6)

    def test_non_monotonic_returns_none(self):
        levels = [
            {"FOS": 1.30, "n_elements": 100},
            {"FOS": 1.34, "n_elements": 400},
            {"FOS": 1.31, "n_elements": 1600},
        ]
        assert _richardson(levels) == (None, None)

    def test_growing_steps_return_none(self):
        """Diverging / not-settling steps (|d23| >= |d12|) → no estimate."""
        levels = [
            {"FOS": 1.40, "n_elements": 100},
            {"FOS": 1.36, "n_elements": 400},
            {"FOS": 1.28, "n_elements": 1600},
        ]
        assert _richardson(levels) == (None, None)

    def test_fewer_than_three_levels(self):
        assert _richardson([{"FOS": 1.3, "n_elements": 100},
                            {"FOS": 1.32, "n_elements": 400}]) == (None, None)


class TestStudyBookkeeping:

    def test_requires_two_meshes(self):
        with pytest.raises(ValueError, match="at least two meshes"):
            srm_mesh_refinement_study([(0, 0)], [], meshes=[(10, 5)])

    def test_result_summary_and_to_dict(self):
        """Build a result directly and check summary()/to_dict() render."""
        levels = [
            {"nx": 24, "ny": 12, "n_elements": 593, "n_nodes": 1300,
             "FOS": 1.531, "fos_basis": "nonconvergence", "converged": True,
             "n_srf_trials": 12, "wall_time_s": 9.1, "rel_change": None,
             "error_vs_published": 0.305},
            {"nx": 32, "ny": 16, "n_elements": 1046, "n_nodes": 2200,
             "FOS": 1.456, "fos_basis": "nonconvergence", "converged": True,
             "n_srf_trials": 13, "wall_time_s": 20.6, "rel_change": -0.049,
             "error_vs_published": 0.241},
            {"nx": 40, "ny": 20, "n_elements": 1623, "n_nodes": 3400,
             "FOS": 1.419, "fos_basis": "nonconvergence", "converged": True,
             "n_srf_trials": 14, "wall_time_s": 42.2, "rel_change": -0.025,
             "error_vs_published": 0.210},
        ]
        r = MeshRefinementResult(
            levels=levels, fos_finest=1.419, fos_coarsest=1.531,
            fos_richardson=1.376, observed_order=2.81, converged=True,
            conv_tol=0.03, published=1.173, element_type="t6",
            srm_field="c_phi")
        s = r.summary()
        assert "Mesh-Refinement" in s and "Richardson" in s
        assert r.fos_estimate == 1.376        # richardson wins when present
        d = r.to_dict()
        assert d["fos_finest"] == 1.419 and d["fos_richardson"] == 1.376
        assert len(d["levels"]) == 3


# ── slow: real SRM convergence on the benchmarks ────────────────────────
GL_H = 10.0
GL_GAMMA = 20.0
EX1_SURF = [(0.0, GL_H), (1.2 * GL_H, GL_H), (3.2 * GL_H, 0.0),
            (4.4 * GL_H, 0.0)]
EX1_LAYER = [{'name': 'soil', 'bottom_elevation': -10.0, 'E': 1e5, 'nu': 0.3,
              'c': 10.0, 'phi': 20.0, 'psi': 0.0, 'gamma': GL_GAMMA}]

XCHK_SURF = [(0, 0), (10, 0), (30, 10), (50, 10)]
XCHK_LAYER = [{'name': 'clay', 'bottom_elevation': -10, 'E': 30000, 'nu': 0.3,
               'c': 10.0, 'phi': 15.0, 'psi': 0, 'gamma': 18.0}]


@pytest.mark.slow
class TestSRMMeshConsistency:

    def test_gl99_ex1_mesh_consistent_near_published(self):
        """GL99 Ex1 (published FE 1.4): the SRM FOS is mesh-consistent — every
        refinement level sits in a tight band just below 1.4 (the D=1 base
        sliver makes it noisy rather than strictly monotonic, VALIDATION.md
        §1 / Known limitations), and the finest mesh is within ~6% of 1.4."""
        study = srm_mesh_refinement_study(
            EX1_SURF, EX1_LAYER, meshes=[(24, 10), (32, 12), (40, 16)],
            depth=0.5, x_extend=0.0, element_type='t6', srf_tol=0.02,
            published=1.4)
        print("\n" + study.summary())
        assert len(study.levels) == 3
        for lv in study.levels:
            assert 1.28 <= lv['FOS'] <= 1.44, lv          # bounded near 1.4
            assert lv['fos_basis'] == 'nonconvergence'
        # finest mesh within ~6% of the published FE value
        assert abs(study.fos_finest - 1.4) / 1.4 < 0.06

    def test_xcheck_converges_toward_limit_equilibrium(self):
        """Shared-geometry Bishop cross-check (LE FOS ≈ 1.173): the SRM FOS
        decreases MONOTONICALLY toward the LE value as the mesh is refined,
        with shrinking steps, and Richardson-extrapolates below the finest
        mesh. 40x20 reproduces the documented 1.419 (VALIDATION.md §6)."""
        study = srm_mesh_refinement_study(
            XCHK_SURF, XCHK_LAYER, meshes=[(24, 12), (32, 16), (40, 20)],
            depth=5.0, x_extend=0.0, element_type='t6', srf_tol=0.02,
            published=1.173)
        print("\n" + study.summary())
        f = [lv['FOS'] for lv in study.levels]
        # monotonic decreasing toward the LE value
        assert f[0] > f[1] > f[2]
        # error vs LE shrinks every level
        errs = [abs(lv['error_vs_published']) for lv in study.levels]
        assert errs[0] > errs[1] > errs[2]
        # settling: the last step is smaller than the first
        assert abs(study.levels[-1]['rel_change']) < abs(study.levels[1]['rel_change'])
        assert study.converged                      # last step < 3%
        # documented finest value
        assert study.fos_finest == pytest.approx(1.419, abs=0.03)
        # Richardson estimate produced, below the finest, toward LE
        assert study.fos_richardson is not None
        assert 1.25 <= study.fos_richardson < study.fos_finest
        assert 1.5 <= study.observed_order <= 4.0
