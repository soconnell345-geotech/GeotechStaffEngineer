"""
Cross-validation tests: fem2d vs analytical/reference modules.

Compares FEM results against:
- Analytical closed-form solutions (gravity stresses, K0 theory)
- settlement module (Boussinesq stress distribution)
- slope_stability module (Bishop FOS vs SRM FOS)
- sheet_pile / soe earth pressure theory

Tolerances reflect inherent differences between numerical FEM
and analytical/LEM solutions, plus CST mesh discretization effects.

References:
    Griffiths & Lane (1999) — SRM vs Bishop comparison
    Potts & Zdravkovic (1999) — FEM in Geotechnical Engineering
    Poulos & Davis (1974) — Elastic Solutions for Soil and Rock Mechanics
"""

import math
import numpy as np
import pytest

from fem2d.analysis import analyze_gravity, analyze_foundation, analyze_slope_srm
from settlement.stress_distribution import boussinesq_center_rectangular
from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.analysis import search_critical_surface
from sheet_pile.earth_pressure import rankine_Ka


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _element_centroids(result):
    """Compute (n_elements, 2) array of element centroid coordinates."""
    return result.nodes[result.elements].mean(axis=1)


def _interior_mask(centroids, x_lo, x_hi):
    """Boolean mask for elements with centroid x between x_lo and x_hi."""
    return (centroids[:, 0] >= x_lo) & (centroids[:, 0] <= x_hi)


def _elements_near_x(centroids, x_target, x_tol):
    """Boolean mask for elements near a vertical line x=x_target."""
    return np.abs(centroids[:, 0] - x_target) < x_tol


def _avg_stress_at_depth(result, x_target, y_target, x_tol, y_tol):
    """Average FEM stress [sxx, syy, txy] for elements near (x, y)."""
    centroids = _element_centroids(result)
    mask = (
        (np.abs(centroids[:, 0] - x_target) < x_tol) &
        (np.abs(centroids[:, 1] - y_target) < y_tol)
    )
    if not np.any(mask):
        return None
    return result.stresses[mask].mean(axis=0)


# ---------------------------------------------------------------------------
# 1. Gravity Stress vs Analytical
# ---------------------------------------------------------------------------

class TestGravityStressVsAnalytical:
    """Gravity column: FEM vs sigma_yy = gamma*z, sigma_xx = K0*gamma*z."""

    WIDTH = 40.0
    DEPTH = 10.0
    GAMMA = 18.0
    E = 30000.0
    NU = 0.3

    @pytest.fixture(scope="class")
    def gravity_result(self):
        return analyze_gravity(
            width=self.WIDTH, depth=self.DEPTH, gamma=self.GAMMA,
            E=self.E, nu=self.NU, nx=40, ny=10)

    def test_vertical_stress_profile(self, gravity_result):
        """FEM sigma_yy should match gamma*z (tension-positive: negative).

        Uses depth-averaged comparison to smooth CST element-to-element
        variation, and skips shallow elements where discretization error
        is largest relative to the small stress magnitude.
        """
        centroids = _element_centroids(gravity_result)
        # Interior elements (middle 40%)
        x_lo = 0.3 * self.WIDTH
        x_hi = 0.7 * self.WIDTH
        mask = _interior_mask(centroids, x_lo, x_hi)

        depths = centroids[mask, 1]  # negative values (below surface)
        syy_fem = gravity_result.stresses[mask, 1]

        # Analytical: sigma_yy = gamma * y (y<0 → compression → negative)
        syy_analytical = self.GAMMA * depths

        # Average comparison over depth bands (skip shallow < 2m)
        errors = []
        for i in range(len(depths)):
            if abs(depths[i]) > 2.0:
                err = abs(syy_fem[i] - syy_analytical[i]) / abs(syy_analytical[i])
                errors.append(err)

        mean_err = np.mean(errors)
        assert mean_err < 0.10, (
            f"Mean vertical stress error={mean_err:.1%} (threshold 10%)")

    def test_horizontal_stress_K0(self, gravity_result):
        """FEM sigma_xx / sigma_yy should match elastic K0 = nu/(1-nu)."""
        centroids = _element_centroids(gravity_result)
        x_lo = 0.3 * self.WIDTH
        x_hi = 0.7 * self.WIDTH
        mask = _interior_mask(centroids, x_lo, x_hi)

        sxx_fem = gravity_result.stresses[mask, 0]
        syy_fem = gravity_result.stresses[mask, 1]

        K0_elastic = self.NU / (1.0 - self.NU)  # 0.4286 for nu=0.3

        # Only check elements with significant stress (not near surface)
        deep_mask = np.abs(syy_fem) > 10.0
        ratios = sxx_fem[deep_mask] / syy_fem[deep_mask]
        mean_ratio = np.mean(ratios)

        assert abs(mean_ratio - K0_elastic) / K0_elastic < 0.10, (
            f"K0 mismatch: FEM ratio={mean_ratio:.4f}, "
            f"elastic K0={K0_elastic:.4f}")

    def test_gravity_settlement_vs_analytical(self, gravity_result):
        """Max vertical displacement vs gamma*H^2 / (2*M)."""
        M = self.E * (1.0 - self.NU) / (
            (1.0 + self.NU) * (1.0 - 2.0 * self.NU))
        w_analytical = self.GAMMA * self.DEPTH ** 2 / (2.0 * M)

        w_fem = gravity_result.max_displacement_y_m

        err = abs(w_fem - w_analytical) / w_analytical
        assert err < 0.15, (
            f"Settlement mismatch: FEM={w_fem:.5f}m, "
            f"analytical={w_analytical:.5f}m, error={err:.1%}")


# ---------------------------------------------------------------------------
# 2. Foundation Settlement vs Settlement Module
# ---------------------------------------------------------------------------

class TestFoundationSettlementVsSettlementModule:
    """Strip foundation: FEM stress distribution vs Boussinesq."""

    B = 2.0
    Q = 100.0
    DEPTH = 20.0
    E = 30000.0
    NU = 0.3

    @pytest.fixture(scope="class")
    def foundation_result(self):
        return analyze_foundation(
            B=self.B, q=self.Q, depth=self.DEPTH,
            E=self.E, nu=self.NU, gamma=0.0, nx=40, ny=20)

    def test_stress_distribution_below_footing(self, foundation_result):
        """FEM sigma_yy below footing center vs Boussinesq (plane strain approx)."""
        depths_below = [2.0, 3.0, 5.0, 8.0]
        L_ps = 1000.0  # large L to approximate plane strain

        passed = 0
        for z in depths_below:
            y_coord = -z  # FEM coord (y<0 is below surface)
            stress = _avg_stress_at_depth(
                foundation_result, x_target=0.0, y_target=y_coord,
                x_tol=self.B * 0.3, y_tol=1.0)
            if stress is None:
                continue

            syy_fem = -stress[1]  # flip sign: tension-positive → compression-positive
            syy_bouss = boussinesq_center_rectangular(self.Q, self.B, L_ps, z)

            if syy_bouss > 1.0:
                err = abs(syy_fem - syy_bouss) / syy_bouss
                if err < 0.40:  # 40% tolerance for CST mesh vs analytical
                    passed += 1

        assert passed >= 2, (
            f"Only {passed}/{len(depths_below)} depth points within tolerance")

    def test_settlement_increases_with_load(self):
        """Settlement should double when load doubles (linear elastic)."""
        r1 = analyze_foundation(
            B=2, q=50, depth=15, E=30000, nu=0.3, gamma=0.0, nx=30, ny=15)
        r2 = analyze_foundation(
            B=2, q=100, depth=15, E=30000, nu=0.3, gamma=0.0, nx=30, ny=15)

        ratio = r2.max_displacement_y_m / r1.max_displacement_y_m
        assert abs(ratio - 2.0) < 0.10, (
            f"Linearity check: ratio={ratio:.3f}, expected 2.0")

    def test_settlement_increases_with_footing_width(self):
        """Wider footing should produce more settlement at same pressure."""
        r_narrow = analyze_foundation(
            B=1, q=100, depth=15, E=30000, nu=0.3, gamma=0.0, nx=30, ny=15)
        r_wide = analyze_foundation(
            B=3, q=100, depth=15, E=30000, nu=0.3, gamma=0.0, nx=30, ny=15)

        assert r_wide.max_displacement_y_m > r_narrow.max_displacement_y_m, (
            f"Wide footing ({r_wide.max_displacement_y_m:.5f}m) should settle "
            f"more than narrow ({r_narrow.max_displacement_y_m:.5f}m)")


# ---------------------------------------------------------------------------
# 3. Slope Stability FOS vs Bishop
# ---------------------------------------------------------------------------

class TestSlopeStabilityFOSVsBishop:
    """SRM FOS vs slope_stability Bishop FOS.

    Note on tolerances: CST (3-node) elements suffer from volumetric locking
    which makes the material appear stiffer than it really is. This causes
    SRM to systematically overestimate FOS compared to Bishop, especially
    on coarse meshes. Published SRM results (Griffiths & Lane 1999) use
    6-noded triangles or 8-noded quads for better accuracy.

    These tests validate:
    1. SRM produces a valid FOS (not capped at the upper bracket)
    2. Both methods agree on relative stability (trend validation)
    3. SRM FOS is in a physically reasonable range
    """

    SURFACE = [(0, 0), (10, 0), (30, 10), (50, 10)]

    def test_srm_produces_valid_fos(self):
        """SRM should produce a valid FOS in the bisection range."""
        result = analyze_slope_srm(
            surface_points=self.SURFACE,
            soil_layers=[{
                'name': 'clay', 'bottom_elevation': -20,
                'E': 30000, 'nu': 0.3,
                'c': 10, 'phi': 15, 'psi': 0, 'gamma': 18,
            }],
            nx=20, ny=10, srf_tol=0.05, n_load_steps=10)

        assert result.FOS is not None, "SRM did not produce a FOS"
        assert result.FOS > 0.5, f"FOS too low: {result.FOS:.3f}"
        assert result.n_srf_trials >= 2, "Should need multiple SRF trials"

    def test_higher_cohesion_gives_higher_fos(self):
        """Both SRM and Bishop should rank stronger soil as more stable.

        Uses c=10 kPa and c=20 kPa on the same slope geometry.
        Both methods should give higher FOS for the stronger soil.
        """
        fos_srm = {}
        fos_bishop = {}

        for c_val in [10, 20]:
            r = analyze_slope_srm(
                surface_points=self.SURFACE,
                soil_layers=[{
                    'name': 'clay', 'bottom_elevation': -20,
                    'E': 30000, 'nu': 0.3,
                    'c': c_val, 'phi': 15, 'psi': 0, 'gamma': 18,
                }],
                nx=20, ny=10, srf_tol=0.05, n_load_steps=10)
            fos_srm[c_val] = r.FOS

            geom = SlopeGeometry(
                surface_points=self.SURFACE,
                soil_layers=[SlopeSoilLayer(
                    name='clay', top_elevation=10, bottom_elevation=-20,
                    gamma=18, phi=15, c_prime=c_val,
                )],
            )
            search = search_critical_surface(geom, method='bishop', nx=15, ny=15)
            fos_bishop[c_val] = search.critical.FOS

        # Both methods should agree: c=20 is more stable than c=10
        assert fos_srm[20] > fos_srm[10], (
            f"SRM trend wrong: FOS(c=20)={fos_srm[20]:.3f} <= "
            f"FOS(c=10)={fos_srm[10]:.3f}")
        assert fos_bishop[20] > fos_bishop[10], (
            f"Bishop trend wrong: FOS(c=20)={fos_bishop[20]:.3f} <= "
            f"FOS(c=10)={fos_bishop[10]:.3f}")

    def test_srm_and_bishop_both_indicate_stability(self):
        """Both methods should agree the slope is stable (FOS > 1.0).

        Uses a moderate slope with c=10 kPa, phi=15 deg where
        Bishop FOS ≈ 1.17 (clearly stable). SRM should also
        produce FOS > 1.0 despite CST overestimation.
        """
        result_fem = analyze_slope_srm(
            surface_points=self.SURFACE,
            soil_layers=[{
                'name': 'clay', 'bottom_elevation': -20,
                'E': 30000, 'nu': 0.3,
                'c': 10, 'phi': 15, 'psi': 0, 'gamma': 18,
            }],
            nx=20, ny=10, srf_tol=0.05, n_load_steps=10)

        geom = SlopeGeometry(
            surface_points=self.SURFACE,
            soil_layers=[SlopeSoilLayer(
                name='clay', top_elevation=10, bottom_elevation=-20,
                gamma=18, phi=15, c_prime=10,
            )],
        )
        search = search_critical_surface(geom, method='bishop', nx=15, ny=15)

        # Both methods agree the slope is stable
        assert result_fem.FOS > 1.0, (
            f"SRM says unstable: FOS={result_fem.FOS:.3f}")
        assert search.critical.FOS > 1.0, (
            f"Bishop says unstable: FOS={search.critical.FOS:.3f}")


# ---------------------------------------------------------------------------
# 4. Earth Pressure vs Retaining Walls Theory
# ---------------------------------------------------------------------------

class TestEarthPressureVsRetainingWalls:
    """FEM lateral stress vs K0 / Ka earth pressure theory."""

    @pytest.fixture(scope="class")
    def gravity_result(self):
        return analyze_gravity(
            width=40, depth=10, gamma=18, E=30000, nu=0.3, nx=40, ny=10)

    def test_K0_lateral_stress_elastic(self, gravity_result):
        """FEM sigma_xx/sigma_yy should match elastic K0 = nu/(1-nu)."""
        nu = 0.3
        K0_elastic = nu / (1.0 - nu)

        centroids = _element_centroids(gravity_result)
        mask = _interior_mask(centroids, 12.0, 28.0)

        sxx = gravity_result.stresses[mask, 0]
        syy = gravity_result.stresses[mask, 1]

        deep = np.abs(syy) > 20.0
        ratios = sxx[deep] / syy[deep]
        mean_ratio = np.mean(ratios)

        assert abs(mean_ratio - K0_elastic) / K0_elastic < 0.10, (
            f"K0 ratio: FEM={mean_ratio:.4f}, expected={K0_elastic:.4f}")

    def test_K0_varies_with_poisson(self):
        """K0 should change with Poisson's ratio: nu=0.2 → 0.25, nu=0.4 → 0.667."""
        for nu, K0_expected in [(0.2, 0.25), (0.4, 2.0 / 3.0)]:
            result = analyze_gravity(
                width=40, depth=10, gamma=18, E=30000, nu=nu, nx=30, ny=8)
            centroids = _element_centroids(result)
            mask = _interior_mask(centroids, 12.0, 28.0)

            sxx = result.stresses[mask, 0]
            syy = result.stresses[mask, 1]
            deep = np.abs(syy) > 20.0
            ratios = sxx[deep] / syy[deep]
            mean_ratio = np.mean(ratios)

            assert abs(mean_ratio - K0_expected) / K0_expected < 0.15, (
                f"nu={nu}: K0 ratio={mean_ratio:.4f}, expected={K0_expected:.4f}")

    def test_lateral_pressure_distribution_linear(self, gravity_result):
        """sigma_xx should increase linearly with depth (R^2 > 0.95)."""
        centroids = _element_centroids(gravity_result)
        # Narrow vertical strip at domain center
        mask = _elements_near_x(centroids, 20.0, 2.0)

        depths = -centroids[mask, 1]  # positive depths
        sxx_mag = -gravity_result.stresses[mask, 0]  # positive magnitudes

        # Only deep elements
        deep = depths > 1.0
        depths = depths[deep]
        sxx_mag = sxx_mag[deep]

        # Linear regression
        coeffs = np.polyfit(depths, sxx_mag, 1)
        sxx_fit = np.polyval(coeffs, depths)
        ss_res = np.sum((sxx_mag - sxx_fit) ** 2)
        ss_tot = np.sum((sxx_mag - np.mean(sxx_mag)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        assert r_squared > 0.95, (
            f"Lateral stress not linear with depth: R^2={r_squared:.4f}")


# ---------------------------------------------------------------------------
# 5. Foundation Stress Distribution
# ---------------------------------------------------------------------------

class TestFoundationStressDistribution:
    """Stress bulb under strip footing: FEM vs Boussinesq."""

    @pytest.fixture(scope="class")
    def foundation_result(self):
        return analyze_foundation(
            B=3, q=150, depth=20, E=30000, nu=0.3, gamma=0.0, nx=40, ny=20)

    def test_stress_bulb_below_strip_footing(self, foundation_result):
        """Vertical stress decreases with depth, roughly matching Boussinesq."""
        B = 3.0
        q = 150.0
        L_ps = 1000.0  # plane-strain approximation
        depths = [1.5, 3.0, 5.0, 8.0]

        stresses_fem = []
        stresses_bouss = []

        for z in depths:
            stress = _avg_stress_at_depth(
                foundation_result, x_target=0.0, y_target=-z,
                x_tol=B * 0.25, y_tol=1.0)
            if stress is not None:
                syy_fem = -stress[1]  # tension-positive → magnitude
                syy_bouss = boussinesq_center_rectangular(q, B, L_ps, z)
                stresses_fem.append(syy_fem)
                stresses_bouss.append(syy_bouss)

        assert len(stresses_fem) >= 3, "Not enough depth points sampled"

        # At least half should be within 50%
        matches = sum(
            1 for f, b in zip(stresses_fem, stresses_bouss)
            if b > 1.0 and abs(f - b) / b < 0.50
        )
        assert matches >= len(stresses_fem) // 2, (
            f"Only {matches}/{len(stresses_fem)} points match Boussinesq "
            f"within 50%")

    def test_stress_attenuates_with_depth(self, foundation_result):
        """Vertical stress under the footing decreases with depth."""
        depths = [1.0, 3.0, 5.0, 8.0, 12.0]
        stresses = []

        for z in depths:
            stress = _avg_stress_at_depth(
                foundation_result, x_target=0.0, y_target=-z,
                x_tol=1.5, y_tol=1.5)
            if stress is not None:
                stresses.append(-stress[1])  # compressive magnitude

        assert len(stresses) >= 3, "Not enough depth points"

        # Check monotonic decrease (allowing small noise)
        decreasing = sum(
            1 for i in range(len(stresses) - 1)
            if stresses[i] > stresses[i + 1] - 2.0  # 2 kPa noise tolerance
        )
        assert decreasing >= len(stresses) - 2, (
            f"Stress not monotonically decreasing: {stresses}")


# ---------------------------------------------------------------------------
# 6. SOE Apparent Pressure Comparison
# ---------------------------------------------------------------------------

class TestSOEApparentPressureComparison:
    """FEM lateral stress vs SOE Ka/K0 bounds."""

    def test_lateral_stress_bounds_for_sand(self):
        """FEM sigma_xx should lie between Ka*gamma*z and K0*gamma*z."""
        phi = 30.0
        gamma = 18.0
        nu = 0.3
        depth = 10.0

        result = analyze_gravity(
            width=40, depth=depth, gamma=gamma, E=30000, nu=nu, nx=40, ny=10)

        Ka = rankine_Ka(phi)  # tan^2(45-15) = 0.333
        K0_elastic = nu / (1.0 - nu)  # 0.4286

        centroids = _element_centroids(result)
        mask = _interior_mask(centroids, 12.0, 28.0)

        sxx = result.stresses[mask, 0]
        syy = result.stresses[mask, 1]

        # Check elements at significant depth
        deep = np.abs(syy) > 20.0
        ratios = sxx[deep] / syy[deep]

        # Elastic FEM with rollers → K0_elastic, which should be >= Ka
        mean_ratio = np.mean(ratios)
        assert mean_ratio > Ka * 0.9, (
            f"K0 ratio ({mean_ratio:.4f}) below Ka ({Ka:.4f})")
        assert mean_ratio < 1.0, (
            f"K0 ratio ({mean_ratio:.4f}) exceeds 1.0 (passive)")

        # Should be close to K0_elastic
        assert abs(mean_ratio - K0_elastic) / K0_elastic < 0.15, (
            f"K0 ratio ({mean_ratio:.4f}) not close to elastic K0 "
            f"({K0_elastic:.4f})")
