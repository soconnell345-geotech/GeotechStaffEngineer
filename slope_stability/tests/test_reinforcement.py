"""
P4 tests: reinforcement (nails / anchors / geosynthetics) in the FOS
equations, GEC-7-style.

Hand-calc validation: for an undrained (phi=0) circular case the methods
have no normal-force feedback, so the reinforced FOS has the closed form

    FOS_reinf = M_R / (M_D - T*d_perp)       (active convention)

which the tests reproduce independently from the slice sums.
"""

import math

import pytest

from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.slip_surface import CircularSlipSurface
from slope_stability.slices import build_slices
from slope_stability.methods import fellenius_fos, bishop_fos
from slope_stability.gle import gle_fos, janbu_fos
from slope_stability.nails import SoilNail
from slope_stability.reinforcement import (
    Geosynthetic, Anchor, compute_reinforcement_forces,
    moment_reduction, horizontal_reduction,
)
from slope_stability.analysis import analyze_slope


def _clay_slope(**kw):
    """Simple undrained slope, crest right (standard orientation)."""
    layer = SlopeSoilLayer(
        name="clay", top_elevation=20.0, bottom_elevation=-15.0,
        gamma=18.0, cu=40.0, analysis_mode="undrained",
    )
    geom = SlopeGeometry(
        surface_points=[(0.0, 10.0), (20.0, 10.0), (40.0, 20.0),
                        (70.0, 20.0)],
        soil_layers=[layer], **kw,
    )
    slip = CircularSlipSurface(xc=30.0, yc=32.0, radius=26.0)
    return geom, slip


def _drained_slope(**kw):
    layer = SlopeSoilLayer(
        name="soil", top_elevation=20.0, bottom_elevation=-15.0,
        gamma=19.0, phi=25.0, c_prime=8.0,
    )
    geom = SlopeGeometry(
        surface_points=[(0.0, 10.0), (20.0, 10.0), (40.0, 20.0),
                        (70.0, 20.0)],
        soil_layers=[layer], **kw,
    )
    slip = CircularSlipSurface(xc=30.0, yc=32.0, radius=26.0)
    return geom, slip


class TestGeosyntheticHandCalc:

    def test_fellenius_undrained_closed_form(self):
        """Geosynthetic at z=12: FOS_new = M_R/(M_D - T*d_perp/R) exactly
        for phi=0 Fellenius."""
        T = 60.0
        z_g = 12.0
        geom, slip = _clay_slope(geosynthetics=[
            Geosynthetic(elevation=z_g, T_allow=T)])
        x_entry, x_exit = slip.find_entry_exit(geom)
        slices = build_slices(geom, slip, 40)

        fos_0 = fellenius_fos(slices, slip)
        forces = compute_reinforcement_forces(geom, slip, x_entry, x_exit)
        assert len(forces) == 1
        f = forces[0]
        assert f.kind == "geosynthetic"
        assert f.z == pytest.approx(z_g)
        assert abs(f.dir_z) < 1e-12  # horizontal

        fos_r = fellenius_fos(slices, slip, reinf_forces=forces)

        # closed form: same resisting; driving reduced by T*d_perp/R
        # horizontal force at z_g -> d_perp = yc - z_g
        d_perp = slip.yc - z_g
        driving_0 = sum(
            (s.weight + s.surcharge_force) * (s.x_mid - slip.xc) / slip.radius
            for s in slices)
        driving_0 = abs(driving_0)
        resisting = fos_0 * driving_0
        fos_hand = resisting / (driving_0 - T * d_perp / slip.radius)
        assert fos_r == pytest.approx(fos_hand, rel=1e-9)
        assert fos_r > fos_0

    def test_bishop_undrained_matches_fellenius(self):
        """phi=0: Bishop == Fellenius, with or without reinforcement."""
        geom, slip = _clay_slope(geosynthetics=[
            Geosynthetic(elevation=12.0, T_allow=60.0)])
        x_entry, x_exit = slip.find_entry_exit(geom)
        slices = build_slices(geom, slip, 40)
        forces = compute_reinforcement_forces(geom, slip, x_entry, x_exit)
        fb = bishop_fos(slices, slip, reinf_forces=forces)
        ff = fellenius_fos(slices, slip, reinf_forces=forces)
        assert fb == pytest.approx(ff, rel=1e-6)


class TestNails:

    def test_nail_capacity_switch(self):
        """Short embedment -> pullout-controlled; long -> tensile."""
        geom, slip = _drained_slope()
        x_entry, x_exit = slip.find_entry_exit(geom)

        short = SoilNail(x_head=25.0, z_head=12.5, length=18.0,
                         inclination=15.0, bond_stress=60.0,
                         bar_diameter=32.0, spacing_h=1.5)
        long_ = SoilNail(x_head=25.0, z_head=12.5, length=30.0,
                         inclination=15.0, bond_stress=200.0,
                         bar_diameter=16.0, spacing_h=1.5)
        geom.nails = [short, long_]
        forces = compute_reinforcement_forces(geom, slip, x_entry, x_exit)
        kinds = {f.index: f for f in forces if f.kind == "nail"}
        assert 0 in kinds and 1 in kinds
        assert kinds[0].controlled_by == "pullout"
        assert kinds[1].controlled_by == "tensile"
        # tensile cap: fy*A/spacing
        t_cap = 420.0 * math.pi * (16.0 / 2) ** 2 / 1000.0 / 1.5
        assert kinds[1].T == pytest.approx(t_cap, rel=1e-6)

    def test_nail_pullout_value(self):
        """Pullout = bond * pi * D_dh * L_behind / s_h (GEC-7)."""
        geom, slip = _drained_slope()
        x_entry, x_exit = slip.find_entry_exit(geom)
        nail = SoilNail(x_head=25.0, z_head=12.5, length=22.0,
                        inclination=10.0, bond_stress=80.0,
                        drill_hole_diameter=150.0, spacing_h=2.0)
        geom.nails = [nail]
        forces = compute_reinforcement_forces(geom, slip, x_entry, x_exit)
        assert len(forces) == 1
        f = forces[0]
        # geometric L_behind from the crossing parameter
        # verify with independent line-circle solve
        beta = math.radians(10.0)
        # crossing satisfies |head + t*dir - centre| = R
        dxh = nail.x_head - slip.xc
        dzh = nail.z_head - slip.yc
        bq = 2 * (dxh * math.cos(beta) - dzh * math.sin(beta))
        cq = dxh ** 2 + dzh ** 2 - slip.radius ** 2
        disc = bq * bq - 4 * cq
        t1 = (-bq - math.sqrt(disc)) / 2
        t2 = (-bq + math.sqrt(disc)) / 2
        t_cross = t1 if t1 > 0 else t2
        L_behind = nail.length - t_cross
        T_hand = 80.0 * math.pi * 0.150 * L_behind / 2.0
        assert f.T == pytest.approx(T_hand, rel=1e-3)

    def test_nails_raise_fos_all_methods(self):
        geom, slip = _drained_slope()
        x_entry, x_exit = slip.find_entry_exit(geom)
        slices = build_slices(geom, slip, 40)

        nails = [
            SoilNail(x_head=24.0, z_head=12.0, length=22.0, inclination=15.0,
                     bond_stress=100.0, spacing_h=1.5),
            SoilNail(x_head=30.0, z_head=15.0, length=22.0, inclination=15.0,
                     bond_stress=100.0, spacing_h=1.5),
            SoilNail(x_head=36.0, z_head=18.0, length=18.0, inclination=15.0,
                     bond_stress=100.0, spacing_h=1.5),
        ]
        geom.nails = nails
        forces = compute_reinforcement_forces(geom, slip, x_entry, x_exit)
        assert len(forces) >= 2

        f0_fell = fellenius_fos(slices, slip)
        f1_fell = fellenius_fos(slices, slip, reinf_forces=forces)
        assert f1_fell > f0_fell

        f0_b = bishop_fos(slices, slip)
        f1_b = bishop_fos(slices, slip, reinf_forces=forces)
        assert f1_b > f0_b

        r0 = gle_fos(slices, slip)
        # fresh slices: gle mutates rh/rv on its normalized copies only,
        # but reinforcement assignment happens per-call — safe to reuse
        r1 = gle_fos(slices, slip, reinf_forces=forces)
        assert r1.converged
        assert r1.fos > r0.fos

        j0 = janbu_fos(slices, slip)
        j1 = janbu_fos(slices, slip, reinf_forces=forces)
        assert j1[0] > j0[0]

    def test_nail_outside_surface_no_contribution(self):
        geom, slip = _drained_slope()
        x_entry, x_exit = slip.find_entry_exit(geom)
        # nail entirely inside the sliding mass (too short to cross)
        nail = SoilNail(x_head=30.0, z_head=15.0, length=3.0,
                        inclination=15.0)
        geom.nails = [nail]
        forces = compute_reinforcement_forces(geom, slip, x_entry, x_exit)
        assert forces == []


class TestAnchor:

    def test_anchor_contributes_T_allow(self):
        geom, slip = _drained_slope()
        x_entry, x_exit = slip.find_entry_exit(geom)
        geom.anchors = [Anchor(x_head=28.0, z_head=14.0, length=20.0,
                               inclination=20.0, T_allow=150.0)]
        forces = compute_reinforcement_forces(geom, slip, x_entry, x_exit)
        assert len(forces) == 1
        assert forces[0].T == pytest.approx(150.0)
        assert forces[0].kind == "anchor"


class TestAnalyzeSlopeIntegration:

    def test_analyze_slope_with_nails(self):
        geom, slip = _drained_slope()
        res0 = analyze_slope(geom, xc=30.0, yc=32.0, radius=26.0,
                             method="bishop", n_slices=40)
        geom.nails = [
            SoilNail(x_head=24.0, z_head=12.0, length=22.0, inclination=15.0,
                     bond_stress=100.0, spacing_h=1.5),
            SoilNail(x_head=32.0, z_head=16.0, length=20.0, inclination=15.0,
                     bond_stress=100.0, spacing_h=1.5),
        ]
        res1 = analyze_slope(geom, xc=30.0, yc=32.0, radius=26.0,
                             method="bishop", n_slices=40)
        assert res1.FOS > res0.FOS
        assert res1.reinforcements is not None
        d = res1.to_dict()
        assert d["n_reinforcements_active"] == len(res1.reinforcements)
        assert d["total_reinforcement_kN_per_m"] > 0
        assert "Reinforcement" in res1.summary()

    def test_legacy_reinforcement_force_field(self):
        """SlopeGeometry.reinforcement_force/elevation now actually acts."""
        geom0, slip = _clay_slope()
        res0 = analyze_slope(geom0, xc=30.0, yc=32.0, radius=26.0,
                             method="fellenius", n_slices=40)
        geom1, _ = _clay_slope(reinforcement_force=80.0,
                               reinforcement_elevation=12.0)
        res1 = analyze_slope(geom1, xc=30.0, yc=32.0, radius=26.0,
                             method="fellenius", n_slices=40)
        assert res1.FOS > res0.FOS

    def test_gle_with_nails_via_analyze(self):
        geom, slip = _drained_slope()
        geom.nails = [
            SoilNail(x_head=24.0, z_head=12.0, length=22.0, inclination=15.0,
                     bond_stress=120.0, spacing_h=1.5),
        ]
        res = analyze_slope(geom, xc=30.0, yc=32.0, radius=26.0,
                            method="gle", n_slices=40)
        assert res.FOS > 0
        assert res.reinforcements


class TestStabilizingPile:
    """B2d: single-row stabilizing piles (specified shear or Ito-Matsui)."""

    def test_explicit_shear_closed_form(self):
        """A pile with a specified shear capacity reduces the driving moment by
        (shear_capacity/spacing)*d_perp/R exactly, for phi=0 Fellenius."""
        from slope_stability.reinforcement import StabilizingPile
        shear, spacing, x_pile = 80.0, 2.0, 30.0
        geom, slip = _clay_slope(stabilizing_piles=[
            StabilizingPile(x=x_pile, shear_capacity=shear, spacing=spacing)])
        x_entry, x_exit = slip.find_entry_exit(geom)
        slices = build_slices(geom, slip, 40)
        fos_0 = fellenius_fos(slices, slip)
        forces = compute_reinforcement_forces(geom, slip, x_entry, x_exit)
        assert len(forces) == 1 and forces[0].kind == "pile"
        f = forces[0]
        assert f.T == pytest.approx(shear / spacing)     # per metre
        assert abs(f.dir_z) < 1e-12                       # horizontal
        fos_r = fellenius_fos(slices, slip, reinf_forces=forces)
        d_perp = slip.yc - f.z                            # horizontal force arm
        driving_0 = abs(sum((s.weight + s.surcharge_force)
                            * (s.x_mid - slip.xc) / slip.radius for s in slices))
        resisting = fos_0 * driving_0
        fos_hand = resisting / (driving_0 - (shear / spacing) * d_perp
                                / slip.radius)
        assert fos_r == pytest.approx(fos_hand, rel=1e-9)
        assert fos_r > fos_0

    def test_pile_off_surface_has_no_effect(self):
        from slope_stability.reinforcement import StabilizingPile
        geom, slip = _clay_slope(stabilizing_piles=[
            StabilizingPile(x=-4.0, shear_capacity=50.0)])   # left of entry
        x_entry, x_exit = slip.find_entry_exit(geom)
        forces = compute_reinforcement_forces(geom, slip, x_entry, x_exit)
        assert forces == []

    def test_ito_matsui_increases_fos_and_drops_with_spacing(self):
        from slope_stability.reinforcement import (
            StabilizingPile, ito_matsui_lateral_force)
        geom, slip = _drained_slope()
        res0 = analyze_slope(geom, xc=30.0, yc=32.0, radius=26.0,
                             method="bishop", n_slices=40)
        geom.stabilizing_piles = [StabilizingPile(
            x=30.0, spacing=1.6, ito_matsui=True, diameter=0.8)]
        res1 = analyze_slope(geom, xc=30.0, yc=32.0, radius=26.0,
                             method="bishop", n_slices=40)
        assert res1.FOS > res0.FOS
        assert res1.reinforcements
        # per-metre Ito-Matsui force decreases as spacing widens
        f = [ito_matsui_lateral_force(8.0, 25.0, 19.0, D1, D1 - 0.8,
                                      z_top=6.0, z_bot=0.0) / D1
             for D1 in (1.6, 2.4, 3.2, 4.8)]
        assert f == sorted(f, reverse=True)

    def test_ito_matsui_pressure_positive_and_grows_with_depth(self):
        from slope_stability.reinforcement import ito_matsui_pressure
        p = [ito_matsui_pressure(10.0, 20.0, 18.0, z, 1.6, 0.8)
             for z in (1.0, 3.0, 5.0)]
        assert all(v > 0 for v in p) and p == sorted(p)

    def test_validation_errors(self):
        from slope_stability.reinforcement import StabilizingPile
        with pytest.raises(ValueError):
            StabilizingPile(x=1.0)                       # no resistance given
        with pytest.raises(ValueError):
            StabilizingPile(x=1.0, ito_matsui=True)      # no diameter
        with pytest.raises(ValueError):
            StabilizingPile(x=1.0, ito_matsui=True, diameter=2.0, spacing=1.0)
