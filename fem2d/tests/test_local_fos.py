"""Local factor-of-safety field — tests (v5.4 F5).

Fast tests hand-check the Mohr-Coulomb local FOS on synthetic stress states
(no FEM). The slow tests run the real SRM on the Griffiths & Lane slope and pin
the validation the exhibit rests on: at the critical SRF the minimum local FOS
tracks the global SRM FOS, and a stronger (stable) slope has local FOS > 1
everywhere.
"""

import math

import numpy as np
import pytest

from fem2d import local_fos_field, LocalFOSField
from fem2d.analysis import analyze_slope_srm


class _Res:
    """Minimal result stand-in: nodes / elements / element stresses (+FOS)."""
    def __init__(self, stresses, FOS=None):
        # one CST triangle per stress row, laid out trivially
        n = len(stresses)
        self.nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
                              + [[float(i + 2), 0.0] for i in range(n)])
        self.elements = np.array([[0, 1, i + 3] for i in range(n)]
                                 if n > 1 else [[0, 1, 2]])
        if n == 1:
            self.nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
            self.elements = np.array([[0, 1, 2]])
        self.stresses = np.asarray(stresses, dtype=float)
        self.FOS = FOS


class TestMohrCoulombHandChecks:
    def test_pure_shear_at_yield_is_one(self):
        # c=10, phi=0: tau_avail=10; pure shear txy=10 -> q=10 -> FOS=1
        lf = local_fos_field(_Res([[0.0, 0.0, 10.0]]), c=10.0, phi=0.0)
        assert lf.values[0] == pytest.approx(1.0, rel=1e-12)

    def test_half_mobilized_is_two(self):
        # q=5, tau_avail=10 -> FOS=2
        lf = local_fos_field(_Res([[0.0, 0.0, 5.0]]), c=10.0, phi=0.0)
        assert lf.values[0] == pytest.approx(2.0, rel=1e-12)

    def test_frictional_with_confinement(self):
        # c=10, phi=20, sxx=syy=-100 (compression), txy=20
        # p=-100; tau_avail = 10 cos20 + 100 sin20; q=20
        c, phi = 10.0, 20.0
        tau_av = c * math.cos(math.radians(phi)) + 100.0 * math.sin(math.radians(phi))
        lf = local_fos_field(_Res([[-100.0, -100.0, 20.0]]), c=c, phi=phi)
        assert lf.values[0] == pytest.approx(tau_av / 20.0, rel=1e-12)

    def test_near_zero_mobilization_caps(self):
        # isotropic stress, no shear -> q~0 -> capped
        lf = local_fos_field(_Res([[-100.0, -100.0, 0.0]]), c=10.0, phi=20.0,
                             cap=7.0)
        assert lf.values[0] == pytest.approx(7.0, rel=1e-12)
        assert lf.min_fos <= 7.0

    def test_per_element_strengths(self):
        res = _Res([[0.0, 0.0, 10.0], [0.0, 0.0, 10.0]])
        lf = local_fos_field(res, c=[10.0, 20.0], phi=[0.0, 0.0])
        assert lf.values[0] == pytest.approx(1.0, rel=1e-12)   # 10/10
        assert lf.values[1] == pytest.approx(2.0, rel=1e-12)   # 20/10

    def test_summary_and_to_dict(self):
        lf = local_fos_field(_Res([[0.0, 0.0, 5.0]], FOS=2.1), c=10.0, phi=0.0)
        assert isinstance(lf, LocalFOSField)
        assert "Local Factor-of-Safety" in lf.summary()
        d = lf.to_dict()
        assert d["min_local_fos"] == pytest.approx(2.0, rel=1e-9)
        assert d["global_fos"] == pytest.approx(2.1, rel=1e-12)

    def test_missing_stresses_raises(self):
        class _Bare:
            nodes = np.zeros((3, 2)); elements = np.array([[0, 1, 2]])
            stresses = None
            FOS = None
        with pytest.raises(ValueError, match="no element stresses"):
            local_fos_field(_Bare(), c=1.0, phi=0.0)


# ── slow: real SRM validation (fem2d/VALIDATION.md sec 8) ────────────────
GL_H = 10.0
GL_GAMMA = 20.0
EX1_SURF = [(0.0, GL_H), (1.2 * GL_H, GL_H), (3.2 * GL_H, 0.0),
            (4.4 * GL_H, 0.0)]


def _gl99(c, phi):
    layer = [{'name': 's', 'bottom_elevation': -10.0, 'E': 1e5, 'nu': 0.3,
              'c': c, 'phi': phi, 'psi': 0.0, 'gamma': GL_GAMMA}]
    return analyze_slope_srm(surface_points=EX1_SURF, soil_layers=layer,
                             depth=0.5, nx=32, ny=12, srf_tol=0.02,
                             x_extend=0.0, element_type='t6',
                             compute_local_fos=True)


@pytest.mark.slow
class TestSRMLocalFOS:
    def test_min_local_fos_tracks_global(self):
        """GL99 Ex1 at the critical SRF: minimum local FOS ~ the global SRM
        FOS (within ~10%), and no element is inadmissible (local FOS >= 1)."""
        srm = _gl99(10.0, 20.0)
        lf = srm.local_fos
        print(f"\n  GL99: global FOS={srm.FOS:.3f}, min local FOS={lf.min_fos:.3f}, "
              f"min/global={lf.min_fos / srm.FOS:.3f}")
        assert lf.frac_below_1 == 0.0
        assert 0.85 <= lf.min_fos / srm.FOS <= 1.05
        # the near-critical mobilized mass is a large fraction of the section
        assert lf.frac_below_1_5 > 0.3

    def test_stable_slope_all_above_one(self):
        """A stronger slope (higher FOS) has local FOS > 1 everywhere; its
        minimum still tracks the (higher) global FOS."""
        srm = _gl99(25.0, 25.0)
        lf = srm.local_fos
        print(f"\n  stable: global FOS={srm.FOS:.3f}, min local FOS={lf.min_fos:.3f}")
        assert lf.min_fos > 1.0
        assert lf.frac_below_1 == 0.0
        assert 0.85 <= lf.min_fos / srm.FOS <= 1.05
