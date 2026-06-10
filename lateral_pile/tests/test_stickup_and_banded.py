"""
Regression tests for the v5.1 lateral_pile fixes.

LP-3: native above-ground free (stickup) length in the FD solver/analysis.
LP-2: banded (scipy.linalg.solve_banded) solve replacing the dense
      O(n^3) np.linalg.solve — must be numerically identical.

Closed-form oracle for the stickup (LP-3): for a free-headed pile with
Q = 0, adding a stickup e is statically equivalent — for the EMBEDDED
response — to applying M = Mt + Vt*e (with the same Vt) at grade.
"""

import numpy as np
import pytest

from lateral_pile.pile import Pile
from lateral_pile.soil import SoilLayer
from lateral_pile.py_curves import SoftClayMatlock
from lateral_pile.analysis import LateralPileAnalysis
from lateral_pile.solver import (
    solve_lateral_pile, hetenyi_solution, _assemble_banded, _BAND,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EI_UNIF = 1.0e6        # kN-m^2
ES_UNIF = 20000.0      # kN/m^2  (beta*L ~ 5: "long" pile)
L_EMBED = 20.0         # m
STICKUP = 2.0          # m


def _elastic_py(n_nodes, Es=ES_UNIF):
    """Uniform linear-elastic p-y functions (p = Es*y) for each node."""
    return [lambda y, z, b: Es * y] * n_nodes


def _make_clay_analysis():
    pile = Pile(length=L_EMBED, diameter=0.6, thickness=0.012, E=200e6)
    layers = [SoilLayer(top=0.0, bottom=L_EMBED,
                        py_model=SoftClayMatlock(c=40.0, gamma=9.0,
                                                 eps50=0.01, J=0.5))]
    return LateralPileAnalysis(pile, layers)


# ---------------------------------------------------------------------------
# LP-3: stickup
# ---------------------------------------------------------------------------

class TestStickup:

    def test_stickup_zero_identical(self):
        """stickup=0 must reproduce existing behavior exactly (regression)."""
        analysis = _make_clay_analysis()
        r_default = analysis.solve(Vt=100.0, Mt=50.0, Q=200.0)
        r_zero = analysis.solve(Vt=100.0, Mt=50.0, Q=200.0, stickup=0.0)

        assert np.array_equal(r_default.z, r_zero.z)
        assert np.array_equal(r_default.deflection, r_zero.deflection)
        assert np.array_equal(r_default.moment, r_zero.moment)
        assert np.array_equal(r_default.shear, r_zero.shear)
        assert np.array_equal(r_default.slope, r_zero.slope)
        assert np.array_equal(r_default.soil_reaction, r_zero.soil_reaction)
        assert r_default.iterations == r_zero.iterations

    def test_stickup_negative_raises(self):
        analysis = _make_clay_analysis()
        with pytest.raises(ValueError, match="stickup"):
            analysis.solve(Vt=100.0, stickup=-1.0)
        with pytest.raises(ValueError, match="stickup"):
            solve_lateral_pile(
                pile_length=10.0, EI_values=np.array([EI_UNIF]),
                py_functions=_elastic_py(101), Vt=10.0, stickup=-0.5,
            )

    def test_stickup_mesh_and_zero_soil_above_grade(self):
        """Mesh extends to z = -stickup; p = 0 and Es = 0 above grade."""
        n = 110  # h = 0.2 m, grade node at index 10
        result = solve_lateral_pile(
            pile_length=L_EMBED, EI_values=np.array([EI_UNIF]),
            py_functions=_elastic_py(n + 1), Vt=50.0,
            n_elements=n, stickup=STICKUP,
        )
        assert result.z[0] == pytest.approx(-STICKUP)
        assert result.z[-1] == pytest.approx(L_EMBED)
        above = result.z < -1e-9
        assert above.sum() == 10
        assert np.all(result.soil_reaction[above] == 0.0)
        assert np.all(result.Es[above] == 0.0)

    def test_stickup_equivalent_moment_elastic(self):
        """Free head, Q=0: stickup e == equivalent (Vt, Mt + Vt*e) at grade
        for the embedded response (deflection/slope/moment at grade and the
        max embedded moment)."""
        Vt, Mt = 50.0, 20.0
        n_stick = 110   # total 22 m -> h = 0.2
        n_equiv = 100   # total 20 m -> h = 0.2 (same spacing)

        r_stick = solve_lateral_pile(
            pile_length=L_EMBED, EI_values=np.array([EI_UNIF]),
            py_functions=_elastic_py(n_stick + 1), Vt=Vt, Mt=Mt,
            n_elements=n_stick, stickup=STICKUP,
        )
        r_equiv = solve_lateral_pile(
            pile_length=L_EMBED, EI_values=np.array([EI_UNIF]),
            py_functions=_elastic_py(n_equiv + 1),
            Vt=Vt, Mt=Mt + Vt * STICKUP,
            n_elements=n_equiv,
        )

        i0 = 10  # grade node in the stickup run
        assert r_stick.z[i0] == pytest.approx(0.0, abs=1e-9)

        # Deflection and slope at grade
        assert r_stick.y[i0] == pytest.approx(r_equiv.y[0], rel=1e-3)
        assert r_stick.slope[i0] == pytest.approx(r_equiv.slope[0], rel=1e-3)

        # Moment transfer: M(grade) = Mt + Vt*e exactly; the discrete shear
        # at the grade node carries an O(h) artifact from the soil-reaction
        # kink there (V' jumps at z=0), so only bound it loosely.
        assert r_stick.moment[i0] == pytest.approx(Mt + Vt * STICKUP, rel=1e-3)
        assert r_stick.shear[i0] == pytest.approx(Vt, rel=0.06)

        # Max moment over the embedded length matches the equivalent run
        m_stick = np.max(np.abs(r_stick.moment[i0:]))
        m_equiv = np.max(np.abs(r_equiv.moment))
        assert m_stick == pytest.approx(m_equiv, rel=5e-3)

        # Embedded profiles match
        assert np.allclose(r_stick.y[i0:], r_equiv.y, rtol=5e-3,
                           atol=1e-9)

    def test_stickup_vs_hetenyi_closed_form(self):
        """Grade deflection with stickup matches the Hetenyi closed form
        with the statically equivalent head moment."""
        Vt = 50.0
        n_stick = 110
        r_stick = solve_lateral_pile(
            pile_length=L_EMBED, EI_values=np.array([EI_UNIF]),
            py_functions=_elastic_py(n_stick + 1), Vt=Vt,
            n_elements=n_stick, stickup=STICKUP,
        )
        oracle = hetenyi_solution(
            pile_length=L_EMBED, EI=EI_UNIF, Es_constant=ES_UNIF,
            Vt=Vt, Mt=Vt * STICKUP,
        )
        assert r_stick.y[10] == pytest.approx(oracle.y[0], rel=0.02)

    def test_stickup_full_analysis_matlock(self):
        """Nonlinear full-analysis oracle: stickup vs equivalent moment at
        grade in Matlock soft clay (free head, Q=0)."""
        Vt = 100.0
        analysis = _make_clay_analysis()
        r_stick = analysis.solve(Vt=Vt, n_elements=110, stickup=STICKUP)
        r_equiv = analysis.solve(Vt=Vt, Mt=Vt * STICKUP, n_elements=100)

        # Deflection at grade
        assert r_stick.y_ground == pytest.approx(r_equiv.y_top, rel=0.02)
        # Max moment (occurs below grade in both runs)
        assert r_stick.max_moment == pytest.approx(r_equiv.max_moment,
                                                   rel=0.02)
        # Bookkeeping
        assert r_stick.stickup == STICKUP
        assert r_stick.to_dict()["stickup_m"] == STICKUP

    def test_stickup_head_moves_more_than_grade(self):
        """The free head atop the stickup deflects more than the grade
        section (rigid-body rotation + column bending)."""
        analysis = _make_clay_analysis()
        r = analysis.solve(Vt=100.0, n_elements=110, stickup=STICKUP)
        assert r.y_top > r.y_ground > 0


# ---------------------------------------------------------------------------
# LP-2: banded solve numerically identical to the dense path
# ---------------------------------------------------------------------------

class TestBandedSolve:

    @pytest.mark.parametrize("head,Kr,Q", [
        ("free", 0.0, 0.0),
        ("free", 0.0, 500.0),
        ("fixed", 0.0, 0.0),
        ("partial", 5.0e4, 250.0),
    ])
    def test_banded_matches_dense(self, head, Kr, Q):
        """solve_banded on the banded assembly == np.linalg.solve on the
        equivalent dense matrix, to machine precision."""
        from scipy.linalg import solve_banded

        n = 100
        h = 0.2
        rng = np.random.default_rng(42)
        EI = np.full(n + 1, EI_UNIF) * (1.0 + 0.1 * rng.random(n + 1))
        Es = ES_UNIF * rng.random(n + 1)

        ab, F = _assemble_banded(n, h, EI, Es, Q, 75.0, 30.0, head, Kr)

        # Reconstruct the dense matrix from the banded storage
        N = n + 5
        K = np.zeros((N, N))
        for j in range(N):
            for k in range(2 * _BAND + 1):
                i = k - _BAND + j
                if 0 <= i < N:
                    K[i, j] = ab[k, j]

        Y_banded = solve_banded((_BAND, _BAND), ab, F)
        Y_dense = np.linalg.solve(K, F)

        assert np.allclose(Y_banded, Y_dense, rtol=1e-9, atol=1e-14)

    def test_full_solution_unchanged_vs_hetenyi(self):
        """End-to-end check that the banded path reproduces the closed-form
        Hetenyi solution (same accuracy bar as the legacy validation suite)."""
        n = 100
        result = solve_lateral_pile(
            pile_length=L_EMBED, EI_values=np.array([EI_UNIF]),
            py_functions=_elastic_py(n + 1), Vt=100.0,
            n_elements=n,
        )
        oracle = hetenyi_solution(L_EMBED, EI_UNIF, ES_UNIF, Vt=100.0)
        assert result.y[0] == pytest.approx(oracle.y[0], rel=0.02)
        assert np.max(np.abs(result.moment)) == pytest.approx(
            np.max(np.abs(oracle.moment)), rel=0.02)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
