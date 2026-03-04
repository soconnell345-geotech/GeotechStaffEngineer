"""Tests for SliceForces dataclass and compute_slice_forces()."""

import math
import pytest

from slope_stability.slices import Slice, SliceForces, compute_slice_forces


def _make_slice(**overrides):
    """Create a Slice with sensible defaults, overriding as needed."""
    defaults = dict(
        x_left=0.0, x_right=1.0, x_mid=0.5,
        width=1.0, z_top=10.0, z_base=5.0, height=5.0,
        alpha=math.radians(30.0),
        base_length=1.0 / math.cos(math.radians(30.0)),
        weight=90.0,  # 5m * 18 kN/m3 * 1m width
        pore_pressure=0.0,
        c=10.0, phi=25.0,
        surcharge_force=0.0, seismic_force=0.0,
        z_centroid=7.5,
    )
    defaults.update(overrides)
    return Slice(**defaults)


class TestComputeSliceForces:
    """Tests for compute_slice_forces()."""

    def test_basic_forces(self):
        """Verify W, N', S, T for a standard slice with no pore pressure."""
        s = _make_slice()
        f = compute_slice_forces(s)

        alpha = math.radians(30.0)
        dl = 1.0 / math.cos(alpha)

        assert f.W == pytest.approx(90.0)
        assert f.U == pytest.approx(0.0)
        assert f.N_prime == pytest.approx(90.0 * math.cos(alpha))
        assert f.S_mobilized == pytest.approx(90.0 * math.sin(alpha))
        assert f.T_available == pytest.approx(
            10.0 * dl + 90.0 * math.cos(alpha) * math.tan(math.radians(25.0))
        )
        assert f.alpha_deg == pytest.approx(30.0)

    def test_zero_pore_pressure(self):
        """With u=0, U=0 and N' = W*cos(alpha)."""
        s = _make_slice(pore_pressure=0.0)
        f = compute_slice_forces(s)

        assert f.U == pytest.approx(0.0)
        assert f.N_prime == pytest.approx(
            f.W * math.cos(s.alpha), abs=1e-6
        )

    def test_with_pore_pressure(self):
        """Pore pressure reduces N' by u*dl."""
        alpha = math.radians(20.0)
        dl = 1.0 / math.cos(alpha)
        s = _make_slice(
            alpha=alpha,
            base_length=dl,
            pore_pressure=30.0,  # kPa
        )
        f = compute_slice_forces(s)

        expected_U = 30.0 * dl
        expected_N = 90.0 * math.cos(alpha) - expected_U

        assert f.U == pytest.approx(expected_U)
        assert f.N_prime == pytest.approx(expected_N, abs=1e-6)

    def test_cohesion_only(self):
        """phi=0: T = c*dl (undrained)."""
        alpha = math.radians(15.0)
        dl = 1.0 / math.cos(alpha)
        s = _make_slice(
            alpha=alpha, base_length=dl,
            c=50.0, phi=0.0, pore_pressure=0.0,
        )
        f = compute_slice_forces(s)

        assert f.T_available == pytest.approx(50.0 * dl)

    def test_friction_only(self):
        """c=0: T = N'*tan(phi)."""
        alpha = math.radians(25.0)
        dl = 1.0 / math.cos(alpha)
        s = _make_slice(
            alpha=alpha, base_length=dl,
            c=0.0, phi=30.0, pore_pressure=0.0,
        )
        f = compute_slice_forces(s)

        expected_N = 90.0 * math.cos(alpha)
        expected_T = expected_N * math.tan(math.radians(30.0))

        assert f.T_available == pytest.approx(expected_T, abs=1e-6)

    def test_with_surcharge(self):
        """Surcharge adds to W in force decomposition."""
        s = _make_slice(surcharge_force=20.0)
        f = compute_slice_forces(s)

        assert f.W == pytest.approx(90.0 + 20.0)
        assert f.surcharge == pytest.approx(20.0)
        assert f.S_mobilized == pytest.approx(110.0 * math.sin(s.alpha))

    def test_with_seismic(self):
        """Seismic force passed through from slice."""
        s = _make_slice(seismic_force=9.0)
        f = compute_slice_forces(s)

        assert f.seismic == pytest.approx(9.0)

    def test_n_prime_clamped_to_zero(self):
        """N' clamped to zero when pore pressure exceeds W*cos(alpha)."""
        alpha = math.radians(10.0)
        dl = 1.0 / math.cos(alpha)
        # High pore pressure: u*dl >> W*cos(alpha)
        s = _make_slice(
            alpha=alpha, base_length=dl,
            weight=10.0,  # small weight
            pore_pressure=200.0,  # very high u
        )
        f = compute_slice_forces(s)

        assert f.N_prime == 0.0
        # T_available with N'=0 is just c*dl
        assert f.T_available == pytest.approx(10.0 * dl)

    def test_alpha_deg_conversion(self):
        """alpha_deg matches degrees conversion of slice alpha."""
        for deg in [0, 15, 30, 45, -10]:
            s = _make_slice(alpha=math.radians(deg))
            f = compute_slice_forces(s)
            assert f.alpha_deg == pytest.approx(deg, abs=1e-10)

    def test_horizontal_base(self):
        """alpha=0: S_mobilized=0, N'=W."""
        s = _make_slice(
            alpha=0.0, base_length=1.0,
            pore_pressure=0.0,
        )
        f = compute_slice_forces(s)

        assert f.S_mobilized == pytest.approx(0.0)
        assert f.N_prime == pytest.approx(f.W)
